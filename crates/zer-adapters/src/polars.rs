use polars_core::prelude::{AnyValue, DataFrame};
use zer_core::record::{FieldValue, Record, RecordId};

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

// ── PolarsIngest extension trait ──────────────────────────────────────────────

/// Extension trait that adds `into_records()` to a Polars [`DataFrame`].
///
/// # Example
///
/// ```rust,no_run
/// use zer_adapters::PolarsIngest;
/// use polars_core::prelude::*;
///
/// let df = df! {
///     "name" => ["Alice", "Bob"],
///     "age"  => [30i64, 25i64],
/// }.unwrap();
///
/// let records = df.into_records(1);  // IDs start at 1
/// ```
pub trait PolarsIngest {
    /// Convert each row of the `DataFrame` into a [`Record`].
    ///
    /// `id_start` is the [`RecordId`] assigned to the first row; subsequent
    /// rows receive `id_start + 1`, `id_start + 2`, …
    fn into_records(self, id_start: RecordId) -> Vec<Record>;
}

impl PolarsIngest for DataFrame {
    fn into_records(self, id_start: RecordId) -> Vec<Record> {
        let height = self.height();
        let schema = self.schema();
        let col_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
        let mut records = Vec::with_capacity(height);

        for row_idx in 0..height {
            let id = id_start + row_idx as RecordId;
            let mut record = Record::new(id);
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
    fn dataframe_into_records_ids_and_values() {
        let df = df! {
            "name" => ["Alice", "Bob"],
            "age"  => [30i64, 25i64],
        }
        .unwrap();

        let records = df.into_records(10);

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, 10);
        assert_eq!(records[1].id, 11);
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
    fn dataframe_into_records_float_column() {
        let df = df! {
            "lat" => [52.37f64, 51.92f64],
            "lon" => [4.90f64,  4.48f64],
        }
        .unwrap();

        let records = df.into_records(1);
        assert_eq!(records[0].field_as::<f64>("lat"), Some(52.37));
        assert_eq!(records[1].field_as::<f64>("lon"), Some(4.48));
    }

    #[test]
    fn dataframe_into_records_mixed_nulls() {
        let df = df! {
            "name" => [Some("Alice"), None::<&str>],
            "age"  => [Some(30i64),  None::<i64>],
        }
        .unwrap();

        let records = df.into_records(1);
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

        let records = df.into_records(1);
        assert_eq!(records[0].get("id"), Some(&FieldValue::UInt(u64::MAX)));
    }
}
