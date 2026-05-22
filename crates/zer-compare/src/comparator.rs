use rayon::prelude::*;
use zer_core::{
    comparison::{ComparisonBatch, ComparisonLevel, ComparisonVector},
    field_mapping::{FieldMapping, NullPolicy},
    record::Record,
    record_pool::RecordPool,
    schema::{FieldKind, Schema},
    traits::Comparator,
};

use crate::{
    discretize::LevelThresholds,
    similarity::{default_fns_for, SimilarityFn},
};

/// Pairwise field comparator that applies similarity functions to produce a field-major `ComparisonBatch`.
pub struct FieldComparator {
    field_fns:  Vec<Vec<Box<dyn SimilarityFn>>>,
    thresholds: Vec<LevelThresholds>,
}

impl FieldComparator {
    pub fn from_schema(schema: &Schema) -> Self {
        let field_fns = schema.fields.iter()
            .map(|f| default_fns_for(f.kind))
            .collect();
        let thresholds = schema.fields.iter()
            .map(|f| LevelThresholds::for_kind(f.kind))
            .collect();
        Self { field_fns, thresholds }
    }

    /// Build a comparator for cross-schema linkage from an explicit field-mapping list.
    ///
    /// Field kinds are inferred from `a_schema` by looking up each `a_field`.
    /// Fields not found in `a_schema` default to `FieldKind::Categorical`.
    pub fn from_mapping(mappings: &[FieldMapping], a_schema: &Schema) -> Self {
        let kind_of = |name: &str| {
            a_schema.fields.iter()
                .find(|f| f.name == name)
                .map(|f| f.kind)
                .unwrap_or(FieldKind::Categorical)
        };
        let (field_fns, thresholds): (Vec<_>, Vec<_>) = mappings.iter()
            .map(|m| { let k = kind_of(&m.a_field); (default_fns_for(k), LevelThresholds::for_kind(k)) })
            .unzip();
        Self { field_fns, thresholds }
    }

    /// Compare a cross-schema pair using an explicit field-mapping list.
    ///
    /// For each mapping, looks up `a_field` in record `a` and `b_field` in
    /// record `b`.  When a field is missing the `NullPolicy` decides the level:
    /// `Skip` gives `Null` (EM ignores it), `PenaliseAbsence` gives `None` (hard fail).
    pub fn compare_pair_mapped(
        &self,
        a:        &Record,
        b:        &Record,
        mappings: &[FieldMapping],
    ) -> ComparisonVector {
        let levels: Vec<ComparisonLevel> = mappings.iter().enumerate()
            .map(|(i, m)| {
                let va = a.fields.get(&m.a_field);
                let vb = b.fields.get(&m.b_field);
                match (va, vb, &m.null_policy) {
                    (Some(va), Some(vb), _) => {
                        let sim = self.field_fns[i].iter()
                            .map(|f| f.similarity(va, vb))
                            .fold(0.0_f32, f32::max);
                        self.thresholds[i].apply(sim)
                    }
                    (_, _, NullPolicy::PenaliseAbsence) => ComparisonLevel::None,
                    (_, _, NullPolicy::Skip)             => ComparisonLevel::Null,
                }
            })
            .collect();
        ComparisonVector::new(a.id, b.id, levels)
    }

    /// Batch comparison for cross-schema linkage using explicit field mappings.
    ///
    /// Equivalent to calling `compare_pair_mapped` per pair then assembling the
    /// field-major `ComparisonBatch`.  `n_fields = mappings.len()`.
    pub fn compare_batch_mapped(
        &self,
        records: &[Record],
        indices: &[(usize, usize)],
        mappings: &[FieldMapping],
    ) -> ComparisonBatch {
        let n_pairs  = indices.len();
        let n_fields = mappings.len();

        if n_pairs == 0 {
            return ComparisonBatch::new(0, n_fields, vec![]);
        }

        let pair_ids_and_levels: Vec<((u64, u64), Vec<u8>)> = indices
            .par_iter()
            .map(|&(i, j)| {
                let ids    = (records[i].id, records[j].id);
                let cv     = self.compare_pair_mapped(&records[i], &records[j], mappings);
                let levels = cv.levels.iter().map(|&l| l as u8).collect();
                (ids, levels)
            })
            .collect();

        Self::scatter_to_batch(n_pairs, n_fields, pair_ids_and_levels)
    }

    fn scatter_to_batch(
        n_pairs:  usize,
        n_fields: usize,
        pair_ids_and_levels: Vec<((u64, u64), Vec<u8>)>,
    ) -> ComparisonBatch {
        let pair_ids: Vec<(u64, u64)> =
            pair_ids_and_levels.iter().map(|(ids, _)| *ids).collect();
        let mut levels = vec![0u8; n_fields * n_pairs];
        for f in 0..n_fields {
            let field_slice = &mut levels[f * n_pairs..(f + 1) * n_pairs];
            for (p, (_, pair_lvls)) in pair_ids_and_levels.iter().enumerate() {
                field_slice[p] = pair_lvls[f];
            }
        }
        ComparisonBatch { n_pairs, n_fields, pair_ids, levels }
    }

    pub fn with_thresholds(mut self, field_idx: usize, thresholds: LevelThresholds) -> Self {
        self.thresholds[field_idx] = thresholds;
        self
    }

    pub fn with_fns(mut self, field_idx: usize, fns: Vec<Box<dyn SimilarityFn>>) -> Self {
        self.field_fns[field_idx] = fns;
        self
    }

    fn compare_pair(&self, a: &Record, b: &Record, schema: &Schema) -> ComparisonVector {
        let levels: Vec<ComparisonLevel> = schema.fields.iter().enumerate()
            .map(|(i, field)| {
                let va = a.fields.get(&field.name);
                let vb = b.fields.get(&field.name);
                match (va, vb) {
                    (Some(va), Some(vb)) => {
                        let sim = self.field_fns[i].iter()
                            .map(|f| f.similarity(va, vb))
                            .fold(0.0_f32, f32::max);
                        self.thresholds[i].apply(sim)
                    }
                    _ => ComparisonLevel::None,
                }
            })
            .collect();
        ComparisonVector::new(a.id, b.id, levels)
    }

    /// Compare field `f` using the zero-alloc `similarity_str` hot path.
    #[inline]
    fn compare_pool_field(&self, f: usize, a_str: &str, b_str: &str) -> u8 {
        if a_str.is_empty() || b_str.is_empty() {
            return ComparisonLevel::None as u8;
        }
        let sim = self.field_fns[f].iter()
            .map(|fn_| fn_.similarity_str(a_str, b_str))
            .fold(0.0_f32, f32::max);
        self.thresholds[f].apply(sim) as u8
    }

    /// Pool-native batch comparison, the primary hot path.
    ///
    /// Reads `RecordPool` columns directly: zero HashMap lookups, no
    /// `Record::clone()`.  Uses Rayon for parallel per-pair comparison
    /// into a flat pair-major buffer (zero per-pair heap allocations), then
    /// transposes to the field-major `ComparisonBatch` layout required by
    /// all GPU EM kernels (CUDA/Vulkan/AVX2): `levels[f * n_pairs + p]`.
    pub fn compare_batch_from_pool(
        &self,
        pool:    &RecordPool,
        indices: &[(usize, usize)],
        schema:  &Schema,
    ) -> ComparisonBatch {
        let n_pairs  = indices.len();
        let n_fields = schema.fields.len();

        if n_pairs == 0 {
            return ComparisonBatch::new(0, n_fields, vec![]);
        }

        // Pre-compute pair IDs (cheap, serial).
        let pair_ids: Vec<(u64, u64)> = indices.iter()
            .map(|&(i, j)| (pool.ids[i], pool.ids[j]))
            .collect();

        // Phase 1: parallel pair-major fill.  Each pair owns a contiguous
        // n_fields-byte slice, no per-pair allocation.
        // pair_major[p * n_fields + f] = level for pair p, field f.
        let mut pair_major = vec![0u8; n_pairs * n_fields];
        pair_major
            .par_chunks_mut(n_fields)
            .zip(indices.par_iter())
            .for_each(|(chunk, &(i, j))| {
                for f in 0..n_fields {
                    chunk[f] = self.compare_pool_field(f, pool.get(f, i), pool.get(f, j));
                }
            });

        // Phase 2: transpose pair-major → field-major.
        // Output: levels[f * n_pairs + p]  (required by GPU EM kernels).
        let mut levels = vec![0u8; n_fields * n_pairs];
        for (p, chunk) in pair_major.chunks_exact(n_fields).enumerate() {
            for (f, &lvl) in chunk.iter().enumerate() {
                levels[f * n_pairs + p] = lvl;
            }
        }

        ComparisonBatch { n_pairs, n_fields, pair_ids, levels }
    }
}

impl Comparator for FieldComparator {
    fn compare(&self, a: &Record, b: &Record, schema: &Schema) -> ComparisonVector {
        self.compare_pair(a, b, schema)
    }

    fn compare_batch_from_pool(
        &self,
        pool:    &RecordPool,
        indices: &[(usize, usize)],
        schema:  &Schema,
    ) -> ComparisonBatch {
        self.compare_batch_from_pool(pool, indices, schema)
    }
}

#[cfg(test)]
mod tests {
    use zer_core::{
        comparison::ComparisonLevel,
        record::FieldValue,
        record_pool::RecordPool,
        schema::{FieldKind, SchemaBuilder},
    };

    use super::*;

    fn person_schema() -> Schema {
        SchemaBuilder::new()
            .field("voornamen",     FieldKind::Name)
            .field("achternaam",    FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .field("postcode",      FieldKind::Id)
            .build()
            .unwrap()
    }

    fn make_record(id: u64, voornamen: &str, achternaam: &str, dob: &str, postcode: &str) -> Record {
        Record::new(id)
            .insert("voornamen",     FieldValue::Text(voornamen.into()))
            .insert("achternaam",    FieldValue::Text(achternaam.into()))
            .insert("geboortedatum", FieldValue::Text(dob.into()))
            .insert("postcode",      FieldValue::Text(postcode.into()))
    }

    #[test]
    fn compare_returns_correct_field_count() {
        let schema = person_schema();
        let cmp    = FieldComparator::from_schema(&schema);
        let a      = make_record(1, "Jan", "Jansen", "1990-06-15", "1011AB");
        let b      = make_record(2, "Jan", "Jansen", "1990-06-15", "1011AB");
        let cv     = cmp.compare(&a, &b, &schema);
        assert_eq!(cv.levels.len(), schema.len());
    }

    #[test]
    fn identical_records_score_exact_on_all_fields() {
        let schema = person_schema();
        let cmp    = FieldComparator::from_schema(&schema);
        let a      = make_record(1, "Jan", "Jansen", "1990-06-15", "1011AB");
        let b      = make_record(2, "Jan", "Jansen", "1990-06-15", "1011AB");
        let cv     = cmp.compare(&a, &b, &schema);
        assert!(cv.levels.iter().all(|&l| l == ComparisonLevel::Exact),
            "identical records should have all Exact levels: {:?}", cv.levels);
    }

    #[test]
    fn completely_different_records_score_none_or_low() {
        let schema = person_schema();
        let cmp    = FieldComparator::from_schema(&schema);
        let a      = make_record(1, "Jan", "Jansen", "1990-06-15", "1011AB");
        let b      = make_record(2, "Maria", "Bakker", "1955-12-01", "3001XY");
        let cv     = cmp.compare(&a, &b, &schema);
        let n_none = cv.levels.iter().filter(|&&l| l == ComparisonLevel::None).count();
        assert!(n_none >= 2, "dissimilar records should have several None levels: {:?}", cv.levels);
    }

    #[test]
    fn missing_field_produces_none() {
        let schema = person_schema();
        let cmp    = FieldComparator::from_schema(&schema);
        let a = make_record(1, "Jan", "Jansen", "1990-06-15", "1011AB");
        let b = Record::new(2)
            .insert("voornamen",     FieldValue::Text("Jan".into()))
            .insert("achternaam",    FieldValue::Text("Jansen".into()))
            .insert("geboortedatum", FieldValue::Text("1990-06-15".into()));
        let cv = cmp.compare(&a, &b, &schema);
        assert_eq!(cv.levels[3], ComparisonLevel::None,
            "missing postcode should yield None, got {:?}", cv.levels[3]);
    }

    #[test]
    fn compare_batch_field_major_layout() {
        let schema   = person_schema();
        let cmp      = FieldComparator::from_schema(&schema);
        let n_fields = schema.len();

        let records: Vec<Record> = (0..5).flat_map(|i| vec![
            make_record(i * 2,     "Jan", "Jansen", "1990-06-15", "1011AB"),
            make_record(i * 2 + 1, "Jan", "Jansen", "1990-06-15", "1011AB"),
        ]).collect();
        let pool    = RecordPool::from_records(&records, &schema);
        let indices: Vec<(usize, usize)> = (0..5).map(|i| (i * 2, i * 2 + 1)).collect();

        let batch = cmp.compare_batch_from_pool(&pool, &indices, &schema);

        assert_eq!(batch.n_pairs,  5);
        assert_eq!(batch.n_fields, n_fields);
        assert_eq!(batch.levels.len(), n_fields * 5);

        // All identical → all Exact
        for f in 0..n_fields {
            for p in 0..5 {
                assert_eq!(
                    batch.level(f, p),
                    ComparisonLevel::Exact,
                    "field {f} pair {p} should be Exact"
                );
            }
        }
    }

    #[test]
    fn compare_batch_from_pool_matches_individual_compare() {
        let schema  = person_schema();
        let cmp     = FieldComparator::from_schema(&schema);
        let records: Vec<Record> = (0..20).flat_map(|i| vec![
            make_record(i * 2,     "Jan", "Jansen", "1990-06-15", "1011AB"),
            make_record(i * 2 + 1, "Jan", "Jansen", "1990-06-15", "1011AB"),
        ]).collect();
        let pool    = RecordPool::from_records(&records, &schema);
        let indices: Vec<(usize, usize)> = (0..20).map(|i| (i * 2, i * 2 + 1)).collect();

        let batch = cmp.compare_batch_from_pool(&pool, &indices, &schema);
        for (p, &(i, j)) in indices.iter().enumerate() {
            let single = cmp.compare(&records[i], &records[j], &schema);
            for (f, &expected) in single.levels.iter().enumerate() {
                assert_eq!(
                    batch.level(f, p), expected,
                    "batch and individual disagree at field {f} pair {p}"
                );
            }
        }
    }

    #[test]
    fn empty_batch_is_valid() {
        let schema = person_schema();
        let cmp    = FieldComparator::from_schema(&schema);
        let pool   = RecordPool::new(schema.fields.len());
        let batch  = cmp.compare_batch_from_pool(&pool, &[], &schema);
        assert_eq!(batch.n_pairs, 0);
        assert!(batch.levels.is_empty());
    }

}
