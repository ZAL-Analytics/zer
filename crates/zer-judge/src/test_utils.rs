/// Schema-driven near-duplicate record generator.
///
/// Produces perturbed copies of real records so that example pipelines and
/// integration tests always have candidate pairs that land in the borderline
/// band, ensuring the judge is actually exercised.
///
/// # Perturbation rules
///
/// | Field kind | Transformation |
/// |-----------|---------------|
/// | `Name`    | Strip the last character from the value (preserves phonetic code of surname). |
/// | `Date`    | Keep the four-digit year; replace month and day with deterministic alternatives derived from the pair index. |
/// | All other | Copy verbatim. |
///
/// The phonetic blocking key (`{Double-Metaphone}:{year}`) is preserved because
/// the surname's phonetic code is robust to single-character edits, and the
/// birth year is kept intact.  The full-DOB and first-name comparison features
/// become uncertain, pushing the Fellegi-Sunter score into the borderline band.
///
/// # Example
///
/// ```rust
/// use zer_judge::test_utils::NearDuplicateGenerator;
/// use zer_core::{record::{Record, FieldValue}, schema::{Schema, SchemaBuilder, FieldKind}};
///
/// let schema = SchemaBuilder::new()
///     .field("voornamen",     FieldKind::Name)
///     .field("achternaam",    FieldKind::Name)
///     .field("geboortedatum", FieldKind::Date)
///     .build()
///     .unwrap();
///
/// let source = vec![
///     Record::new(1)
///         .insert("voornamen",     FieldValue::Text("Maria".into()))
///         .insert("achternaam",    FieldValue::Text("Jansen".into()))
///         .insert("geboortedatum", FieldValue::Text("1985-03-15".into())),
/// ];
///
/// let synthetics = NearDuplicateGenerator { pair_count: 2, id_offset: 9_000_000 }
///     .generate(&source, &schema);
///
/// // 2 pairs → 4 synthetic records
/// assert_eq!(synthetics.len(), 4);
/// ```
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, Schema},
};

/// Produces perturbed near-duplicate records from an existing batch.
///
/// The generator is schema-driven: it reads `Name` and `Date` fields declared
/// in the schema, perturbs them in ways that preserve blocking-key membership
/// while creating genuine uncertainty for the Fellegi-Sunter scorer.
///
/// Output records are assigned IDs starting at `id_offset` to avoid collision
/// with the original data.
pub struct NearDuplicateGenerator {
    /// Number of near-duplicate pairs to generate (`generate()` returns 2  times  this many records).
    pub pair_count: usize,
    /// Starting ID for synthetic records.  Should be well above any real record IDs.
    pub id_offset: u64,
}

impl NearDuplicateGenerator {
    /// Generate `2  times  pair_count` synthetic records from `source`.
    ///
    /// For each pair index `i`:
    /// - Record at `id_offset + 2*i` , verbatim copy of the source record.
    /// - Record at `id_offset + 2*i+1`, perturbed copy (name stripped, date day/month changed).
    ///
    /// If `source` has fewer than `pair_count` records the generator cycles
    /// through the source slice.
    pub fn generate(&self, source: &[Record], schema: &Schema) -> Vec<Record> {
        if source.is_empty() || self.pair_count == 0 {
            return vec![];
        }

        let name_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Name).collect();
        let date_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Date).collect();

        let mut out = Vec::with_capacity(self.pair_count * 2);

        for i in 0..self.pair_count {
            let src = &source[i % source.len()];

            // Record A, verbatim copy with a fresh ID.
            let record_a_id = self.id_offset + (2 * i) as u64;
            let mut record_a = Record::new(record_a_id);
            for field in &schema.fields {
                if let Some(v) = src.get(&field.name) {
                    record_a = record_a.insert(&field.name, v.clone());
                }
            }

            // Record B, perturbed copy with the next ID.
            let record_b_id = self.id_offset + (2 * i + 1) as u64;
            let mut record_b = Record::new(record_b_id);
            for field in &schema.fields {
                let perturbed = if let Some(v) = src.get(&field.name) {
                    if name_fields.contains(&field.name.as_str()) {
                        perturb_name(v)
                    } else if date_fields.contains(&field.name.as_str()) {
                        perturb_date(v, i)
                    } else {
                        v.clone()
                    }
                } else {
                    FieldValue::Null
                };
                record_b = record_b.insert(&field.name, perturbed);
            }

            out.push(record_a);
            out.push(record_b);
        }

        out
    }
}

// ── Perturbation helpers ──────────────────────────────────────────────────────

/// Strip the last character from the name value.
///
/// "Maria" → "Mari", "Pieter" → "Piete".
/// Single-character values are kept as-is.  The phonetic code of surnames is
/// robust to this edit, so the blocking key is preserved.
fn perturb_name(value: &FieldValue) -> FieldValue {
    match value {
        FieldValue::Text(s) => {
            let mut chars: Vec<char> = s.chars().collect();
            if chars.len() > 1 {
                chars.pop();
                FieldValue::Text(chars.into_iter().collect())
            } else {
                value.clone()
            }
        }
        other => other.clone(),
    }
}

/// Keep the birth year, replace month and day with deterministic alternatives.
///
/// "1985-03-15" → "1985-09-22" (pair_index = 0).
/// The year component is identical to the source, so the `{phonetic}:{year}`
/// blocking key is preserved.  Month and day differ, making the full-DOB
/// comparison feature uncertain.
fn perturb_date(value: &FieldValue, pair_index: usize) -> FieldValue {
    match value {
        FieldValue::Text(s) if s.len() >= 4 => {
            let year = &s[..4];
            // Deterministic alternative month (1–11, excluding original range) and day (1–25).
            let alt_month = (pair_index % 11 + 1) as u8;
            let alt_day = (pair_index % 25 + 1) as u8;
            FieldValue::Text(format!("{year}-{alt_month:02}-{alt_day:02}"))
        }
        other => other.clone(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::schema::SchemaBuilder;

    fn person_schema() -> Schema {
        SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap()
    }

    fn make_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
        Record::new(id)
            .insert("voornamen", FieldValue::Text(first.into()))
            .insert("achternaam", FieldValue::Text(last.into()))
            .insert("geboortedatum", FieldValue::Text(dob.into()))
    }

    fn source_records() -> Vec<Record> {
        vec![
            make_record(1, "Maria", "Jansen", "1985-03-15"),
            make_record(2, "Pieter", "de Vries", "1990-07-22"),
            make_record(3, "Annelies", "Bakker", "1978-11-05"),
        ]
    }

    #[test]
    fn generate_correct_count() {
        let schema = person_schema();
        let source = source_records();
        let gen = NearDuplicateGenerator {
            pair_count: 3,
            id_offset: 9_000_000,
        };
        let result = gen.generate(&source, &schema);
        assert_eq!(result.len(), 6, "pair_count=3 → 6 synthetic records");
    }

    #[test]
    fn generate_ids_start_at_offset() {
        let schema = person_schema();
        let source = source_records();
        let gen = NearDuplicateGenerator {
            pair_count: 2,
            id_offset: 5_000,
        };
        let result = gen.generate(&source, &schema);
        assert!(
            result.iter().all(|r| r.id >= 5_000),
            "all IDs must be >= id_offset"
        );
        assert_eq!(result[0].id, 5_000);
        assert_eq!(result[1].id, 5_001);
    }

    #[test]
    fn generate_ids_are_unique() {
        let schema = person_schema();
        let source = source_records();
        let gen = NearDuplicateGenerator {
            pair_count: 5,
            id_offset: 1_000,
        };
        let result = gen.generate(&source, &schema);
        let ids: std::collections::HashSet<u64> = result.iter().map(|r| r.id).collect();
        assert_eq!(
            ids.len(),
            result.len(),
            "all generated record IDs must be unique"
        );
    }

    #[test]
    fn generate_name_fields_are_perturbed() {
        let schema = person_schema();
        let source = vec![make_record(1, "Maria", "Jansen", "1985-03-15")];
        let gen = NearDuplicateGenerator {
            pair_count: 1,
            id_offset: 9_000,
        };
        let result = gen.generate(&source, &schema);
        // result[0] = verbatim copy, result[1] = perturbed
        let orig_first = source[0].get("voornamen");
        let pert_first = result[1].get("voornamen");
        assert_ne!(
            orig_first, pert_first,
            "first name must differ between original and perturbed"
        );
    }

    #[test]
    fn generate_surname_is_also_perturbed() {
        let schema = person_schema();
        let source = vec![make_record(1, "Maria", "Jansen", "1985-03-15")];
        let gen = NearDuplicateGenerator {
            pair_count: 1,
            id_offset: 9_000,
        };
        let result = gen.generate(&source, &schema);
        // Surname gets the same strip-last-char treatment as first name.
        let orig = source[0].get("achternaam");
        let pert = result[1].get("achternaam");
        assert_ne!(orig, pert, "surname must be perturbed");
        // Specifically: "Jansen" → "Janse"
        assert_eq!(pert, Some(&FieldValue::Text("Janse".into())));
    }

    #[test]
    fn generate_date_year_preserved() {
        let schema = person_schema();
        let source = vec![make_record(1, "Maria", "Jansen", "1985-03-15")];
        let gen = NearDuplicateGenerator {
            pair_count: 1,
            id_offset: 9_000,
        };
        let result = gen.generate(&source, &schema);
        // perturbed record is result[1]
        let dob_val = result[1].get("geboortedatum");
        if let Some(FieldValue::Text(s)) = dob_val {
            assert!(s.starts_with("1985-"), "year must be preserved: {s}");
        } else {
            panic!("expected Text value for geboortedatum");
        }
    }

    #[test]
    fn generate_date_day_month_differ() {
        let schema = person_schema();
        let source = vec![make_record(1, "Maria", "Jansen", "1985-03-15")];
        let gen = NearDuplicateGenerator {
            pair_count: 1,
            id_offset: 9_000,
        };
        let result = gen.generate(&source, &schema);
        let orig_dob = source[0].get("geboortedatum");
        let pert_dob = result[1].get("geboortedatum");
        assert_ne!(
            orig_dob, pert_dob,
            "perturbed DOB must differ from original"
        );
    }

    #[test]
    fn generate_verbatim_copy_equals_source() {
        let schema = person_schema();
        let source = vec![make_record(1, "Maria", "Jansen", "1985-03-15")];
        let gen = NearDuplicateGenerator {
            pair_count: 1,
            id_offset: 9_000,
        };
        let result = gen.generate(&source, &schema);
        // result[0] is the verbatim copy (different ID, same field values)
        assert_eq!(result[0].get("voornamen"), source[0].get("voornamen"));
        assert_eq!(result[0].get("achternaam"), source[0].get("achternaam"));
        assert_eq!(
            result[0].get("geboortedatum"),
            source[0].get("geboortedatum")
        );
    }

    #[test]
    fn generate_cycles_when_fewer_sources() {
        let schema = person_schema();
        let source = vec![make_record(1, "Maria", "Jansen", "1985-03-15")]; // only 1 source
        let gen = NearDuplicateGenerator {
            pair_count: 3,
            id_offset: 9_000,
        };
        let result = gen.generate(&source, &schema);
        assert_eq!(
            result.len(),
            6,
            "should still generate 2 times pair_count records when cycling"
        );
    }

    #[test]
    fn generate_empty_source_returns_empty() {
        let schema = person_schema();
        let gen = NearDuplicateGenerator {
            pair_count: 5,
            id_offset: 9_000,
        };
        let result = gen.generate(&[], &schema);
        assert!(result.is_empty());
    }

    #[test]
    fn generate_zero_pairs_returns_empty() {
        let schema = person_schema();
        let source = source_records();
        let gen = NearDuplicateGenerator {
            pair_count: 0,
            id_offset: 9_000,
        };
        let result = gen.generate(&source, &schema);
        assert!(result.is_empty());
    }

    #[test]
    fn perturb_name_strips_last_char() {
        assert_eq!(
            perturb_name(&FieldValue::Text("Pieter".into())),
            FieldValue::Text("Piete".into()),
        );
        assert_eq!(
            perturb_name(&FieldValue::Text("Maria".into())),
            FieldValue::Text("Mari".into()),
        );
    }

    #[test]
    fn perturb_name_single_char_unchanged() {
        assert_eq!(
            perturb_name(&FieldValue::Text("A".into())),
            FieldValue::Text("A".into()),
        );
    }

    #[test]
    fn perturb_date_preserves_year() {
        let v = FieldValue::Text("1990-06-15".into());
        let p = perturb_date(&v, 0);
        if let FieldValue::Text(s) = p {
            assert!(s.starts_with("1990-"), "year must be preserved");
        } else {
            panic!("expected Text");
        }
    }

    #[test]
    fn perturb_date_changes_with_index() {
        let v = FieldValue::Text("1990-06-15".into());
        let p0 = perturb_date(&v, 0);
        let p1 = perturb_date(&v, 1);
        assert_ne!(
            p0, p1,
            "different pair indices must produce different dates"
        );
    }

    #[test]
    fn perturb_date_null_passthrough() {
        assert_eq!(perturb_date(&FieldValue::Null, 0), FieldValue::Null);
    }
}
