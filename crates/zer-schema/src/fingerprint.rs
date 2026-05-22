use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use zer_core::{
    field_mapping::FieldMapping,
    record::{FieldValue, Record},
    schema::{FieldKind, Schema},
};

/// Per-field statistics collected from a sample of records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FieldStats {
    pub name: String,
    pub kind: FieldKind,
    /// Fraction of records where this field is absent or null.
    pub null_rate: f32,
    /// Approximate number of distinct values (exact for samples ≤ 1 M).
    pub cardinality: usize,
    /// Up to 10 most-common values, most frequent first.
    pub top_k: Vec<String>,
}

/// Fingerprint that identifies a schema structure plus its data distribution.
///
/// Two `SchemaFingerprint`s with equal `schema_hash` are structurally identical
/// (same field names and kinds, regardless of order). The `field_stats` carry
/// distribution information used by the nearest-neighbor warm-start heuristic.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchemaFingerprint {
    /// SHA-256 of sorted (name, kind) pairs, stable across field-ordering.
    pub schema_hash: [u8; 32],
    /// One entry per schema field; empty when built with [`Self::from_schema`].
    pub field_stats: Vec<FieldStats>,
    /// Number of records in the sample; 0 when built with [`Self::from_schema`].
    pub record_count: u64,
    /// Unix timestamp (seconds) when this fingerprint was created.
    pub created_at: u64,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute a deterministic SHA-256 hash from a schema's field names and kinds,
/// optionally including explicit field mappings for cross-schema runs.
///
/// Fields are sorted by name before hashing so that field order doesn't affect
/// the resulting hash.  When `mappings` is non-empty its `a_field:b_field` pairs
/// are also hashed (sorted by `a_field`) so that a BRP-only warm-start is never
/// mistakenly reused for a BRP↔SIS cross-schema run.
fn compute_schema_hash(schema: &Schema, mappings: &[FieldMapping]) -> [u8; 32] {
    let mut sorted: Vec<_> = schema.fields.iter().collect();
    sorted.sort_by_key(|f| f.name.as_str());

    let mut hasher = Sha256::new();
    for field in sorted {
        // Serialize the FieldKind discriminant via bincode for a compact,
        // stable byte representation.
        let kind_bytes = bincode::serialize(&field.kind).unwrap_or_default();
        hasher.update(field.name.as_bytes());
        hasher.update(b":");
        hasher.update(&kind_bytes);
        hasher.update(b"|");
    }

    if !mappings.is_empty() {
        hasher.update(b"mappings:");
        let mut sorted_m: Vec<_> = mappings.iter().collect();
        sorted_m.sort_by(|a, b| a.a_field.cmp(&b.a_field));
        for m in sorted_m {
            hasher.update(m.a_field.as_bytes());
            hasher.update(b":");
            hasher.update(m.b_field.as_bytes());
            hasher.update(b"|");
        }
    }

    hasher.finalize().into()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Convert a `FieldValue` to a string for statistics gathering.
/// Returns `None` for null / empty text values.
fn field_value_to_string(v: &FieldValue) -> Option<String> {
    match v {
        FieldValue::Text(s) if !s.is_empty() => Some(s.clone()),
        FieldValue::Int(i) => Some(i.to_string()),
        FieldValue::Float(f) => Some(f.to_string()),
        FieldValue::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Compute [`FieldStats`] for a single field across all records.
fn compute_field_stats(name: &str, kind: FieldKind, records: &[Record]) -> FieldStats {
    let total = records.len();
    if total == 0 {
        return FieldStats {
            name: name.to_string(),
            kind,
            null_rate: 0.0,
            cardinality: 0,
            top_k: vec![],
        };
    }

    let mut null_count = 0usize;
    let mut freq: HashMap<String, usize> = HashMap::new();

    for record in records {
        match record.fields.get(name) {
            None | Some(FieldValue::Null) => null_count += 1,
            Some(v) => match field_value_to_string(v) {
                Some(s) => *freq.entry(s).or_insert(0) += 1,
                None => null_count += 1,
            },
        }
    }

    let null_rate = null_count as f32 / total as f32;
    let cardinality = freq.len();

    // Build top-k: sort by descending frequency, keep up to 10.
    let mut freq_vec: Vec<(String, usize)> = freq.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top_k = freq_vec.into_iter().take(10).map(|(s, _)| s).collect();

    FieldStats { name: name.to_string(), kind, null_rate, cardinality, top_k }
}

// ── Public API ────────────────────────────────────────────────────────────────

impl SchemaFingerprint {
    /// Build a structure-only fingerprint from a schema definition.
    ///
    /// `field_stats` is populated with zero-value statistics for each field so
    /// that the Jaccard similarity computation in [`crate::similarity`] can
    /// always access the field names and kinds, even without sample data.
    pub fn from_schema(schema: &Schema) -> Self {
        Self::from_schema_with_mappings(schema, &[])
    }

    /// Like [`Self::from_schema`] but includes cross-schema `mappings` in the hash
    /// so warm-start artifacts from same-schema runs are never reused.
    pub fn from_schema_with_mappings(schema: &Schema, mappings: &[FieldMapping]) -> Self {
        let schema_hash = compute_schema_hash(schema, mappings);
        let field_stats = schema
            .fields
            .iter()
            .map(|f| FieldStats {
                name: f.name.clone(),
                kind: f.kind,
                null_rate: 0.0,
                cardinality: 0,
                top_k: vec![],
            })
            .collect();
        Self { schema_hash, field_stats, record_count: 0, created_at: unix_now() }
    }

    /// Build a full fingerprint from a schema and a sample of records.
    ///
    /// 50–100 records per field is typically enough for reliable statistics.
    /// The `schema_hash` is identical to what [`Self::from_schema`] would produce for
    /// the same schema, so exact-hash lookups still work.
    pub fn from_sample(schema: &Schema, records: &[Record]) -> Self {
        Self::from_sample_with_mappings(schema, records, &[])
    }

    /// Like [`Self::from_sample`] but includes cross-schema `mappings` in the hash.
    pub fn from_sample_with_mappings(
        schema:   &Schema,
        records:  &[Record],
        mappings: &[FieldMapping],
    ) -> Self {
        let schema_hash = compute_schema_hash(schema, mappings);
        let field_stats = schema
            .fields
            .iter()
            .map(|f| compute_field_stats(&f.name, f.kind, records))
            .collect();
        Self {
            schema_hash,
            field_stats,
            record_count: records.len() as u64,
            created_at:   unix_now(),
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::schema::SchemaBuilder;

    fn make_schema_ab() -> zer_core::schema::Schema {
        SchemaBuilder::new()
            .field("alpha", FieldKind::Name)
            .field("beta", FieldKind::Date)
            .build()
            .unwrap()
    }

    #[test]
    fn same_schema_same_hash() {
        let s1 = make_schema_ab();
        let s2 = make_schema_ab();
        assert_eq!(
            compute_schema_hash(&s1, &[]),
            compute_schema_hash(&s2, &[]),
            "identical schemas must produce identical hashes"
        );
    }

    #[test]
    fn reordered_fields_same_hash() {
        let s1 = SchemaBuilder::new()
            .field("alpha", FieldKind::Name)
            .field("beta", FieldKind::Date)
            .build()
            .unwrap();
        let s2 = SchemaBuilder::new()
            .field("beta", FieldKind::Date)
            .field("alpha", FieldKind::Name)
            .build()
            .unwrap();

        assert_eq!(
            compute_schema_hash(&s1, &[]),
            compute_schema_hash(&s2, &[]),
            "field order must not affect schema hash"
        );
    }

    #[test]
    fn different_kinds_different_hash() {
        let s1 = SchemaBuilder::new()
            .field("alpha", FieldKind::Name)
            .build()
            .unwrap();
        let s2 = SchemaBuilder::new()
            .field("alpha", FieldKind::Date)
            .build()
            .unwrap();

        assert_ne!(
            compute_schema_hash(&s1, &[]),
            compute_schema_hash(&s2, &[]),
            "same field name with different kinds must produce different hashes"
        );
    }

    #[test]
    fn from_schema_populates_field_names() {
        let schema = make_schema_ab();
        let fp = SchemaFingerprint::from_schema(&schema);

        assert_eq!(fp.field_stats.len(), 2);
        assert_eq!(fp.record_count, 0);
        let names: Vec<&str> = fp.field_stats.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn from_sample_computes_cardinality_and_null_rate() {
        use zer_core::record::Record;

        let schema = SchemaBuilder::new()
            .field("name", FieldKind::Name)
            .build()
            .unwrap();

        let records = vec![
            Record::new(1).insert("name", FieldValue::Text("Alice".into())),
            Record::new(2).insert("name", FieldValue::Text("Bob".into())),
            Record::new(3).insert("name", FieldValue::Text("Alice".into())),
            Record::new(4), // missing field → null
        ];

        let fp = SchemaFingerprint::from_sample(&schema, &records);

        assert_eq!(fp.record_count, 4);
        let stats = fp.field_stats.iter().find(|f| f.name == "name").unwrap();
        assert_eq!(stats.cardinality, 2, "Alice and Bob are 2 distinct values");
        assert!(
            (stats.null_rate - 0.25).abs() < 1e-6,
            "1 out of 4 records is null"
        );
        assert_eq!(stats.top_k[0], "Alice", "Alice appears twice, so it should be first");
    }

    #[test]
    fn from_schema_and_from_sample_same_hash_for_same_schema() {
        let schema = make_schema_ab();
        let records = vec![
            Record::new(1)
                .insert("alpha", FieldValue::Text("x".into()))
                .insert("beta", FieldValue::Text("2024-01-01".into())),
        ];
        let fp_s = SchemaFingerprint::from_schema(&schema);
        let fp_r = SchemaFingerprint::from_sample(&schema, &records);

        assert_eq!(
            fp_s.schema_hash, fp_r.schema_hash,
            "from_schema and from_sample must yield the same hash for the same schema"
        );
    }
}
