use std::collections::{HashMap, HashSet};
use std::path::Path;

use zer_core::{
    error::ZerError,
    record::{FieldValue, Record},
    schema::{FieldDef, FieldKind, Schema},
};

use crate::config::{NameHeuristics, ValuePatterns};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn text_samples<'a>(field_name: &str, records: &'a [Record], n: usize) -> Vec<&'a str> {
    records
        .iter()
        .filter_map(|r| match r.fields.get(field_name) {
            Some(FieldValue::Text(s)) if !s.is_empty() => Some(s.as_str()),
            _ => None,
        })
        .take(n)
        .collect()
}

fn collect_field_names(records: &[Record]) -> Vec<String> {
    let mut names: HashSet<String> = HashSet::new();
    for record in records {
        for name in record.fields.keys() {
            names.insert(name.clone());
        }
    }
    let mut sorted: Vec<String> = names.into_iter().collect();
    sorted.sort();
    sorted
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Automatic schema detector.
///
/// Samples column names and record values to produce a best-effort [`Schema`].
/// The inferred schema should be reviewed before use in production, call
/// individual field overrides for any column the heuristics might misclassify.
///
/// # Example
///
/// ```rust,no_run
/// # use zer_schema::infer::SchemaInferrer;
/// # use zer_core::schema::FieldKind;
/// # let records = vec![];
/// let schema = SchemaInferrer::new()
///     .override_field("internal_code", FieldKind::Id)
///     .override_field("notes",         FieldKind::FreeText)
///     .infer(&records)
///     .unwrap();
/// ```
pub struct SchemaInferrer {
    overrides: HashMap<String, FieldKind>,
    name_heuristics: NameHeuristics,
    value_patterns: ValuePatterns,
}

impl SchemaInferrer {
    /// Create a new inferrer loading heuristics from the embedded defaults
    /// (or from `ZER_NAME_HEURISTICS` / `ZER_VALUE_PATTERNS` env vars if set).
    pub fn new() -> Self {
        Self {
            overrides: HashMap::new(),
            name_heuristics: NameHeuristics::load_default(),
            value_patterns: ValuePatterns::load_default(),
        }
    }

    /// Override the name-based heuristics with rules loaded from a TOML file.
    ///
    /// Returns `Err` if the file cannot be read or parsed.
    pub fn with_name_heuristics_file(mut self, path: impl AsRef<Path>) -> Result<Self, ZerError> {
        self.name_heuristics = NameHeuristics::from_file(path.as_ref())?;
        Ok(self)
    }

    /// Override the value-pattern sampling with patterns loaded from a TOML file.
    ///
    /// Returns `Err` if the file cannot be read, parsed, or contains an invalid regex.
    pub fn with_value_patterns_file(mut self, path: impl AsRef<Path>) -> Result<Self, ZerError> {
        self.value_patterns = ValuePatterns::from_file(path.as_ref())?;
        Ok(self)
    }

    /// Force a specific `FieldKind` for one field, bypassing inference.
    ///
    /// This always takes precedence over both name-based and value-based
    /// heuristics.
    pub fn override_field(mut self, name: impl Into<String>, kind: FieldKind) -> Self {
        self.overrides.insert(name.into(), kind);
        self
    }

    /// Infer a [`Schema`] from a sample of records.
    ///
    /// 50–100 non-null values per field is enough for reliable inference.
    ///
    /// Returns `Err(ZerError::EmptySchema)` when `records` is empty (no
    /// field names can be discovered).
    pub fn infer(&self, records: &[Record]) -> Result<Schema, ZerError> {
        let field_names = collect_field_names(records);
        if field_names.is_empty() {
            return Err(ZerError::EmptySchema);
        }

        let fields: Vec<FieldDef> = field_names
            .into_iter()
            .map(|name| {
                let kind = self.overrides.get(&name).copied().unwrap_or_else(|| {
                    self.name_heuristics.infer_kind(&name).unwrap_or_else(|| {
                        let samples = text_samples(&name, records, 50);
                        self.value_patterns.infer_kind(&samples)
                    })
                });
                FieldDef { name, kind }
            })
            .collect();

        Ok(Schema { fields })
    }
}

impl Default for SchemaInferrer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn text_record(id: u64, fields: &[(&str, &str)]) -> Record {
        let mut r = Record::new(id);
        for (k, v) in fields {
            r = r.insert(*k, FieldValue::Text(v.to_string()));
        }
        r
    }

    // Thin helpers so existing test bodies don't need rewiring.
    fn infer_name(col: &str) -> Option<FieldKind> {
        NameHeuristics::load_default().infer_kind(col)
    }

    fn infer_values(field: &str, records: &[Record]) -> FieldKind {
        let samples = text_samples(field, records, 50);
        ValuePatterns::load_default().infer_kind(&samples)
    }

    // ── Name-heuristic tests ──────────────────────────────────────────────────

    #[test]
    fn infer_common_name_fields() {
        let cases = [
            ("first_name", FieldKind::Name),
            ("last_name", FieldKind::Name),
            ("voornamen", FieldKind::Name),
            ("achternaam", FieldKind::Name),
            ("surname", FieldKind::Name),
        ];
        for (col, expected) in cases {
            assert_eq!(
                infer_name(col),
                Some(expected),
                "'{col}' should infer as {expected:?}"
            );
        }
    }

    #[test]
    fn infer_date_fields_by_name() {
        for col in ["dob", "geboortedatum", "birth_date", "created_at"] {
            assert_eq!(
                infer_name(col),
                Some(FieldKind::Date),
                "'{col}' should infer as Date"
            );
        }
    }

    #[test]
    fn infer_phone_fields_by_name() {
        for col in ["phone", "tel", "mobile", "msisdn"] {
            assert_eq!(
                infer_name(col),
                Some(FieldKind::Phone),
                "'{col}' should infer as Phone"
            );
        }
    }

    #[test]
    fn infer_address_fields_by_name() {
        for col in ["straatnaam", "postcode", "woonplaats", "huisnummer"] {
            assert_eq!(
                infer_name(col),
                Some(FieldKind::Address),
                "'{col}' should infer as Address"
            );
        }
    }

    #[test]
    fn infer_id_fields_by_name() {
        for col in ["bsn", "imsi", "iccid", "document_nummer", "passport_id"] {
            let result = infer_name(col);
            assert_eq!(
                result,
                Some(FieldKind::Id),
                "'{col}' should infer as Id, got {result:?}"
            );
        }
    }

    // ── Value-pattern tests ───────────────────────────────────────────────────

    #[test]
    fn infer_date_from_iso_values() {
        let records: Vec<Record> = (0..20)
            .map(|i| text_record(i, &[("col_1", "2024-03-15")]))
            .collect();
        assert_eq!(infer_values("col_1", &records), FieldKind::Date);
    }

    #[test]
    fn infer_numeric_from_number_values() {
        let records: Vec<Record> = (0..20)
            .map(|i| text_record(i, &[("col_1", &i.to_string())]))
            .collect();
        assert_eq!(infer_values("col_1", &records), FieldKind::Numeric);
    }

    #[test]
    fn infer_categorical_from_low_cardinality_values() {
        let values = ["M", "V", "M", "V", "M", "V", "M", "V", "M", "V"];
        let records: Vec<Record> = values
            .iter()
            .enumerate()
            .map(|(i, v)| text_record(i as u64, &[("geslacht", v)]))
            .collect();
        assert_eq!(infer_values("geslacht", &records), FieldKind::Categorical);
    }

    #[test]
    fn infer_falls_back_to_freetext_for_empty_field() {
        let records = vec![Record::new(1)];
        assert_eq!(infer_values("col_1", &records), FieldKind::FreeText);
    }

    // ── Override tests ────────────────────────────────────────────────────────

    #[test]
    fn override_takes_precedence_over_name_heuristic() {
        let records = vec![text_record(1, &[("dob", "1990-01-01")])];
        let schema = SchemaInferrer::new()
            .override_field("dob", FieldKind::Id)
            .infer(&records)
            .unwrap();

        let dob = schema.fields.iter().find(|f| f.name == "dob").unwrap();
        assert_eq!(
            dob.kind,
            FieldKind::Id,
            "override must win over name heuristic"
        );
    }

    #[test]
    fn override_takes_precedence_over_value_pattern() {
        let records: Vec<Record> = (0..20)
            .map(|i| text_record(i, &[("col_x", "2024-01-01")]))
            .collect();
        let schema = SchemaInferrer::new()
            .override_field("col_x", FieldKind::FreeText)
            .infer(&records)
            .unwrap();

        let field = schema.fields.iter().find(|f| f.name == "col_x").unwrap();
        assert_eq!(field.kind, FieldKind::FreeText);
    }

    // ── Custom config file tests ──────────────────────────────────────────────

    #[test]
    fn with_name_heuristics_file_overrides_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("names.toml");
        std::fs::write(
            &path,
            r#"
[[rules]]
kind  = "Id"
exact = ["custom_col"]
"#,
        )
        .unwrap();

        let records = vec![text_record(1, &[("custom_col", "ABC123")])];
        let schema = SchemaInferrer::new()
            .with_name_heuristics_file(&path)
            .unwrap()
            .infer(&records)
            .unwrap();

        let f = schema
            .fields
            .iter()
            .find(|f| f.name == "custom_col")
            .unwrap();
        assert_eq!(f.kind, FieldKind::Id);
    }

    #[test]
    fn with_value_patterns_file_overrides_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("values.toml");
        std::fs::write(
            &path,
            r#"
[[patterns]]
kind      = "Phone"
regex     = '^\+31\d{9}$'
threshold = 0.8

[fallback]
default_kind = "FreeText"
"#,
        )
        .unwrap();

        let records: Vec<Record> = (0..20)
            .map(|i| text_record(i, &[("col", "+31612345678")]))
            .collect();
        let schema = SchemaInferrer::new()
            .with_value_patterns_file(&path)
            .unwrap()
            .infer(&records)
            .unwrap();

        let f = schema.fields.iter().find(|f| f.name == "col").unwrap();
        assert_eq!(f.kind, FieldKind::Phone);
    }

    #[test]
    fn with_name_heuristics_file_missing_returns_error() {
        let result =
            SchemaInferrer::new().with_name_heuristics_file("/nonexistent/path/names.toml");
        assert!(result.is_err());
    }

    // ── Full-inference integration tests ─────────────────────────────────────

    #[test]
    fn infer_brp_like_records() {
        let records: Vec<Record> = (0..10)
            .map(|i| {
                text_record(
                    i,
                    &[
                        ("voornamen", "Erik"),
                        ("achternaam", "Hendriks"),
                        ("geboortedatum", "1980-06-15"),
                        ("postcode", "1234AB"),
                        ("nationaliteit", "Nederland"),
                    ],
                )
            })
            .collect();

        let schema = SchemaInferrer::new().infer(&records).unwrap();
        let kind_of = |n: &str| schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);

        assert_eq!(kind_of("voornamen"), Some(FieldKind::Name));
        assert_eq!(kind_of("achternaam"), Some(FieldKind::Name));
        assert_eq!(kind_of("geboortedatum"), Some(FieldKind::Date));
    }

    #[test]
    fn infer_empty_records_returns_error() {
        let result = SchemaInferrer::new().infer(&[]);
        assert!(
            matches!(result, Err(ZerError::EmptySchema)),
            "empty record slice must return EmptySchema"
        );
    }

    #[test]
    fn infer_record_with_no_fields_returns_error() {
        let records = vec![Record::new(1), Record::new(2)];
        let result = SchemaInferrer::new().infer(&records);
        assert!(
            matches!(result, Err(ZerError::EmptySchema)),
            "records with no fields must return EmptySchema"
        );
    }

    #[test]
    fn infer_handles_null_values_gracefully() {
        let mut records = vec![];
        for i in 0..10u64 {
            let mut r = Record::new(i);
            if i % 2 == 0 {
                r = r.insert("col", FieldValue::Text("2024-01-01".into()));
            } else {
                r = r.insert("col", FieldValue::Null);
            }
            records.push(r);
        }
        let schema = SchemaInferrer::new().infer(&records).unwrap();
        assert_eq!(schema.len(), 1);
    }

    #[test]
    fn infer_field_names_sorted_deterministically() {
        let records = vec![text_record(1, &[("zzz", "a"), ("aaa", "b"), ("mmm", "c")])];
        let schema = SchemaInferrer::new().infer(&records).unwrap();
        let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["aaa", "mmm", "zzz"]);
    }
}
