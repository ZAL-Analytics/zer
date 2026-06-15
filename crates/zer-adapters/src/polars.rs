use polars_core::prelude::{AnyValue, DataFrame};
use zer_core::record::{FieldValue, Record};

use crate::DatasetConfig;

// ── AnyValue → FieldValue conversion ─────────────────────────────────────────

/// Convert a Polars `AnyValue` to a `FieldValue`.
pub fn anyvalue_to_field_value(av: AnyValue<'_>) -> FieldValue {
    match av {
        AnyValue::Null => FieldValue::Null,
        AnyValue::Boolean(b) => FieldValue::Bool(b),
        AnyValue::String(s) => FieldValue::Text(s.to_owned()),
        AnyValue::StringOwned(s) => FieldValue::Text(s.to_string()),
        AnyValue::Int8(i) => FieldValue::Int(i as i64),
        AnyValue::Int16(i) => FieldValue::Int(i as i64),
        AnyValue::Int32(i) => FieldValue::Int(i as i64),
        AnyValue::Int64(i) => FieldValue::Int(i),
        AnyValue::UInt8(u) => FieldValue::Int(u as i64),
        AnyValue::UInt16(u) => FieldValue::Int(u as i64),
        AnyValue::UInt32(u) => FieldValue::Int(u as i64),
        AnyValue::UInt64(u) => FieldValue::UInt(u),
        AnyValue::Float32(f) => FieldValue::Float(f as f64),
        AnyValue::Float64(f) => FieldValue::Float(f),
        AnyValue::Binary(b) => FieldValue::Bytes(b.to_vec()),
        AnyValue::BinaryOwned(b) => FieldValue::Bytes(b),
        // Temporal types: fall back to Polars' Display (ISO-8601 text)
        other => FieldValue::Text(format!("{other}")),
    }
}

/// Coerce an `AnyValue` to a string suitable for use as a natural key.
/// Falls back to `row_idx.to_string()` for Null and byte types.
fn key_from_anyvalue(av: AnyValue<'_>, row_idx: usize) -> String {
    match av {
        AnyValue::Null => row_idx.to_string(),
        AnyValue::String(s) => s.to_owned(),
        AnyValue::StringOwned(s) => s.to_string(),
        AnyValue::Int8(i) => i.to_string(),
        AnyValue::Int16(i) => i.to_string(),
        AnyValue::Int32(i) => i.to_string(),
        AnyValue::Int64(i) => i.to_string(),
        AnyValue::UInt8(u) => u.to_string(),
        AnyValue::UInt16(u) => u.to_string(),
        AnyValue::UInt32(u) => u.to_string(),
        AnyValue::UInt64(u) => u.to_string(),
        AnyValue::Float32(f) => f.to_string(),
        AnyValue::Float64(f) => f.to_string(),
        AnyValue::Boolean(b) => b.to_string(),
        _ => row_idx.to_string(),
    }
}

// ── PolarsIngest extension trait ──────────────────────────────────────────────

/// Extension trait that adds `into_records()` to a Polars [`DataFrame`].
///
/// # Example
///
/// ```rust,no_run
/// use zer_adapters::{PolarsIngest, DatasetConfig};
/// use polars_core::prelude::*;
///
/// let df = df! {
///     "bsn"  => ["893479421", "112233445"],
///     "name" => ["Alice", "Bob"],
///     "age"  => [30i64, 25i64],
/// }.unwrap();
///
/// let config = DatasetConfig::new("brp", "bsn");
/// let records = df.into_records(&config);
/// assert_eq!(records[0].key, "893479421");
/// assert_eq!(records[0].source.as_deref(), Some("brp"));
/// ```
pub trait PolarsIngest {
    /// Convert each row of the `DataFrame` into a [`Record`].
    ///
    /// The `key_column` field of `config` names the column whose values become
    /// each record's natural key.  If that column does not exist the row index
    /// is used as a fallback key.  The `source` label from `config` is attached
    /// to every record.
    fn into_records(self, config: &DatasetConfig) -> Vec<Record>;
}

impl PolarsIngest for DataFrame {
    fn into_records(self, config: &DatasetConfig) -> Vec<Record> {
        let height = self.height();
        let schema = self.schema();
        let col_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();

        // Find key column (optional. falls back to row index).
        let key_col = if col_names.contains(&config.key_column.as_str()) {
            self.column(&config.key_column).ok()
        } else {
            None
        };

        let mut records = Vec::with_capacity(height);

        for row_idx in 0..height {
            let key = if let Some(col) = key_col {
                let av = col.get(row_idx).expect("row index must be in range");
                key_from_anyvalue(av, row_idx)
            } else {
                row_idx.to_string()
            };

            let mut record = Record::from_key(&config.source, &key);
            for col_name in &col_names {
                let col = self.column(col_name).expect("column must exist");
                let av = col.get(row_idx).expect("row index must be in range");
                record = record.insert(*col_name, anyvalue_to_field_value(av));
            }
            records.push(record);
        }

        records
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use polars_core::prelude::*;
    use zer_core::record::derive_record_id;

    #[test]
    fn anyvalue_null_maps_to_null() {
        assert_eq!(anyvalue_to_field_value(AnyValue::Null), FieldValue::Null);
    }

    #[test]
    fn anyvalue_bool_roundtrip() {
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Boolean(true)),
            FieldValue::Bool(true)
        );
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Boolean(false)),
            FieldValue::Bool(false)
        );
    }

    #[test]
    fn anyvalue_integer_widening() {
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Int8(1)),
            FieldValue::Int(1)
        );
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Int16(2)),
            FieldValue::Int(2)
        );
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Int32(3)),
            FieldValue::Int(3)
        );
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Int64(-99)),
            FieldValue::Int(-99)
        );
        assert_eq!(
            anyvalue_to_field_value(AnyValue::UInt8(4)),
            FieldValue::Int(4)
        );
        assert_eq!(
            anyvalue_to_field_value(AnyValue::UInt16(5)),
            FieldValue::Int(5)
        );
        assert_eq!(
            anyvalue_to_field_value(AnyValue::UInt32(6)),
            FieldValue::Int(6)
        );
        // u64 must NOT lose precision → UInt
        assert_eq!(
            anyvalue_to_field_value(AnyValue::UInt64(u64::MAX)),
            FieldValue::UInt(u64::MAX),
        );
    }

    #[test]
    fn anyvalue_float_widening() {
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Float32(1.5)),
            FieldValue::Float(1.5f32 as f64)
        );
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Float64(2.25)),
            FieldValue::Float(2.25)
        );
    }

    #[test]
    fn anyvalue_string_owned_and_borrowed() {
        assert_eq!(
            anyvalue_to_field_value(AnyValue::String("hello")),
            FieldValue::Text("hello".into()),
        );
    }

    #[test]
    fn anyvalue_binary_to_bytes() {
        assert_eq!(
            anyvalue_to_field_value(AnyValue::Binary(&[1u8, 2, 3])),
            FieldValue::Bytes(vec![1, 2, 3]),
        );
    }

    #[test]
    fn dataframe_into_records_keys_and_source() {
        let df = df! {
            "bsn"  => ["111", "222"],
            "name" => ["Alice", "Bob"],
            "age"  => [30i64, 25i64],
        }
        .unwrap();

        let config = DatasetConfig::new("brp", "bsn");
        let records = df.into_records(&config);

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].key, "111");
        assert_eq!(records[1].key, "222");
        assert_eq!(records[0].source.as_deref(), Some("brp"));
        assert_eq!(records[1].source.as_deref(), Some("brp"));
        assert_eq!(records[0].id, derive_record_id("brp", "111"));
        assert_eq!(records[1].id, derive_record_id("brp", "222"));
        assert_eq!(
            records[0].get("name"),
            Some(&FieldValue::Text("Alice".into()))
        );
        assert_eq!(records[0].get("age"), Some(&FieldValue::Int(30)));
        assert_eq!(
            records[1].get("name"),
            Some(&FieldValue::Text("Bob".into()))
        );
        assert_eq!(records[1].get("age"), Some(&FieldValue::Int(25)));
    }

    #[test]
    fn dataframe_into_records_missing_key_column_falls_back_to_row_index() {
        let df = df! {
            "name" => ["Alice", "Bob"],
        }
        .unwrap();

        let config = DatasetConfig::new("src", "nonexistent");
        let records = df.into_records(&config);
        assert_eq!(records[0].key, "0");
        assert_eq!(records[1].key, "1");
    }

    #[test]
    fn dataframe_into_records_float_column() {
        let df = df! {
            "id"  => ["r1", "r2"],
            "lat" => [52.37f64, 51.92f64],
            "lon" => [4.90f64,  4.48f64],
        }
        .unwrap();

        let config = DatasetConfig::new("src", "id");
        let records = df.into_records(&config);
        assert_eq!(records[0].field_as::<f64>("lat"), Some(52.37));
        assert_eq!(records[1].field_as::<f64>("lon"), Some(4.48));
    }

    #[test]
    fn dataframe_into_records_mixed_nulls() {
        let df = df! {
            "id"   => [Some("k1"), Some("k2")],
            "name" => [Some("Alice"), None::<&str>],
            "age"  => [Some(30i64),  None::<i64>],
        }
        .unwrap();

        let config = DatasetConfig::new("src", "id");
        let records = df.into_records(&config);
        assert_eq!(
            records[0].get("name"),
            Some(&FieldValue::Text("Alice".into()))
        );
        assert_eq!(records[1].get("name"), Some(&FieldValue::Null));
        assert_eq!(records[1].get("age"), Some(&FieldValue::Null));
    }

    #[test]
    fn dataframe_into_records_uint64_preserved() {
        let df = df! {
            "id" => [u64::MAX],
        }
        .unwrap();

        let config = DatasetConfig::new("src", "id");
        let records = df.into_records(&config);
        assert_eq!(records[0].get("id"), Some(&FieldValue::UInt(u64::MAX)));
        assert_eq!(records[0].key, u64::MAX.to_string());
    }
}
