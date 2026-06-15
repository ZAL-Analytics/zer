use arrow_array::{
    array::{
        Array, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
        Int64Array, Int8Array, LargeBinaryArray, LargeStringArray, StringArray, UInt16Array,
        UInt32Array, UInt64Array, UInt8Array,
    },
    RecordBatch,
};
use arrow_schema::DataType;
use zer_core::record::{FieldValue, Record};

use crate::DatasetConfig;

// ── Column-cell → FieldValue ──────────────────────────────────────────────────

/// Extract one cell from an Arrow array column and convert it to a [`FieldValue`].
///
/// Returns `FieldValue::Null` for null cells or unrecognised column types.
pub fn arrow_cell_to_field_value(col: &dyn Array, row: usize) -> FieldValue {
    if col.is_null(row) {
        return FieldValue::Null;
    }
    match col.data_type() {
        DataType::Boolean => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            FieldValue::Bool(arr.value(row))
        }
        DataType::Int8 => {
            let arr = col.as_any().downcast_ref::<Int8Array>().unwrap();
            FieldValue::Int(arr.value(row) as i64)
        }
        DataType::Int16 => {
            let arr = col.as_any().downcast_ref::<Int16Array>().unwrap();
            FieldValue::Int(arr.value(row) as i64)
        }
        DataType::Int32 => {
            let arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
            FieldValue::Int(arr.value(row) as i64)
        }
        DataType::Int64 => {
            let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
            FieldValue::Int(arr.value(row))
        }
        DataType::UInt8 => {
            let arr = col.as_any().downcast_ref::<UInt8Array>().unwrap();
            FieldValue::Int(arr.value(row) as i64)
        }
        DataType::UInt16 => {
            let arr = col.as_any().downcast_ref::<UInt16Array>().unwrap();
            FieldValue::Int(arr.value(row) as i64)
        }
        DataType::UInt32 => {
            let arr = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            FieldValue::Int(arr.value(row) as i64)
        }
        DataType::UInt64 => {
            let arr = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            FieldValue::UInt(arr.value(row))
        }
        DataType::Float32 => {
            let arr = col.as_any().downcast_ref::<Float32Array>().unwrap();
            FieldValue::Float(arr.value(row) as f64)
        }
        DataType::Float64 => {
            let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
            FieldValue::Float(arr.value(row))
        }
        DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
            FieldValue::Text(arr.value(row).to_owned())
        }
        DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
            FieldValue::Text(arr.value(row).to_owned())
        }
        DataType::Binary => {
            let arr = col.as_any().downcast_ref::<BinaryArray>().unwrap();
            FieldValue::Bytes(arr.value(row).to_vec())
        }
        DataType::LargeBinary => {
            let arr = col.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            FieldValue::Bytes(arr.value(row).to_vec())
        }
        // All other types (Date32, Timestamp, List, …) fall back to Debug text.
        other => FieldValue::Text(format!("{other:?}")),
    }
}

/// Extract the key string from a cell, coercing scalars to their string form.
/// Falls back to `row_idx.to_string()` for null or unhandled types.
fn key_from_cell(col: &dyn Array, row: usize, row_idx: usize) -> String {
    if col.is_null(row) {
        return row_idx.to_string();
    }
    match arrow_cell_to_field_value(col, row) {
        FieldValue::Text(s) => s,
        FieldValue::Int(i) => i.to_string(),
        FieldValue::UInt(u) => u.to_string(),
        FieldValue::Float(f) => f.to_string(),
        FieldValue::Bool(b) => b.to_string(),
        _ => row_idx.to_string(),
    }
}

// ── ArrowIngest extension trait ───────────────────────────────────────────────

/// Extension trait that adds `into_records()` to an Arrow [`RecordBatch`].
///
/// # Example
///
/// ```rust,no_run
/// use zer_adapters::{ArrowIngest, DatasetConfig};
/// use arrow_array::{RecordBatch, Int64Array, StringArray};
/// use arrow_schema::{DataType, Field, Schema};
/// use std::sync::Arc;
///
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("bsn",  DataType::Utf8,  false),
///     Field::new("name", DataType::Utf8,  false),
///     Field::new("age",  DataType::Int64, false),
/// ]));
/// let batch = RecordBatch::try_new(schema, vec![
///     Arc::new(StringArray::from(vec!["123456789", "987654321"])),
///     Arc::new(StringArray::from(vec!["Alice", "Bob"])),
///     Arc::new(Int64Array::from(vec![30i64, 25i64])),
/// ]).unwrap();
///
/// let config = DatasetConfig::new("brp", "bsn");
/// let records = batch.into_records(&config);
/// assert_eq!(records[0].key, "123456789");
/// assert_eq!(records[0].source.as_deref(), Some("brp"));
/// ```
pub trait ArrowIngest {
    /// Convert each row of the `RecordBatch` into a [`Record`].
    ///
    /// The `key_column` field of `config` names the column whose values
    /// become each record's natural key.  If that column does not exist in the
    /// batch the row index is used as a fallback key.  The `source` label from
    /// `config` is attached to every record.
    fn into_records(self, config: &DatasetConfig) -> Vec<Record>;
}

impl ArrowIngest for RecordBatch {
    fn into_records(self, config: &DatasetConfig) -> Vec<Record> {
        let schema = self.schema();
        let n_rows = self.num_rows();
        let n_cols = self.num_columns();

        // Find key column by name (optional. falls back to row index).
        let key_col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == &config.key_column);

        let mut records = Vec::with_capacity(n_rows);

        for row in 0..n_rows {
            let key = if let Some(idx) = key_col_idx {
                key_from_cell(self.column(idx).as_ref(), row, row)
            } else {
                row.to_string()
            };

            let mut record = Record::from_key(&config.source, &key);
            for col_idx in 0..n_cols {
                let field_name = schema.field(col_idx).name().clone();
                let col = self.column(col_idx).as_ref();
                let value = arrow_cell_to_field_value(col, row);
                record = record.insert(field_name, value);
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
    use arrow_array::{
        BinaryArray, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    use zer_core::record::derive_record_id;

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
            Field::new("score", DataType::Float64, true),
            Field::new("active", DataType::Boolean, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    Some("k001"),
                    Some("k002"),
                    None::<&str>,
                ])),
                Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob"), None])),
                Arc::new(Int64Array::from(vec![Some(30i64), Some(25i64), None])),
                Arc::new(Float64Array::from(vec![Some(0.9f64), Some(0.7f64), None])),
                Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn batch_into_records_count_and_keys() {
        let config = DatasetConfig::new("src", "id");
        let records = make_batch().into_records(&config);
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].key, "k001");
        assert_eq!(records[1].key, "k002");
        // Null key → row index fallback
        assert_eq!(records[2].key, "2");
    }

    #[test]
    fn batch_into_records_ids_derived_from_key() {
        let config = DatasetConfig::new("src", "id");
        let records = make_batch().into_records(&config);
        assert_eq!(records[0].id, derive_record_id("src", "k001"));
        assert_eq!(records[1].id, derive_record_id("src", "k002"));
    }

    #[test]
    fn batch_into_records_source_attached() {
        let config = DatasetConfig::new("brp", "id");
        let records = make_batch().into_records(&config);
        for r in &records {
            assert_eq!(r.source.as_deref(), Some("brp"));
        }
    }

    #[test]
    fn batch_missing_key_column_falls_back_to_row_index() {
        let config = DatasetConfig::new("src", "nonexistent");
        let records = make_batch().into_records(&config);
        assert_eq!(records[0].key, "0");
        assert_eq!(records[1].key, "1");
        assert_eq!(records[2].key, "2");
    }

    #[test]
    fn batch_into_records_string_column() {
        let config = DatasetConfig::new("src", "id");
        let records = make_batch().into_records(&config);
        assert_eq!(
            records[0].get("name"),
            Some(&FieldValue::Text("Alice".into()))
        );
        assert_eq!(records[2].get("name"), Some(&FieldValue::Null));
    }

    #[test]
    fn batch_into_records_int64_column() {
        let config = DatasetConfig::new("src", "id");
        let records = make_batch().into_records(&config);
        assert_eq!(records[0].get("age"), Some(&FieldValue::Int(30)));
        assert_eq!(records[2].get("age"), Some(&FieldValue::Null));
    }

    #[test]
    fn batch_into_records_float64_column() {
        let config = DatasetConfig::new("src", "id");
        let records = make_batch().into_records(&config);
        assert_eq!(records[0].get("score"), Some(&FieldValue::Float(0.9)));
    }

    #[test]
    fn batch_into_records_boolean_column() {
        let config = DatasetConfig::new("src", "id");
        let records = make_batch().into_records(&config);
        assert_eq!(records[0].get("active"), Some(&FieldValue::Bool(true)));
        assert_eq!(records[1].get("active"), Some(&FieldValue::Bool(false)));
    }

    #[test]
    fn batch_uint64_preserved() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "uid",
            DataType::UInt64,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![u64::MAX]))])
            .unwrap();
        let config = DatasetConfig::new("src", "uid");
        let records = batch.into_records(&config);
        assert_eq!(records[0].get("uid"), Some(&FieldValue::UInt(u64::MAX)));
        // Key is the string representation of u64::MAX
        assert_eq!(records[0].key, u64::MAX.to_string());
    }

    #[test]
    fn batch_int32_widened_to_int64() {
        let schema = Arc::new(Schema::new(vec![Field::new("val", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![42i32]))]).unwrap();
        let config = DatasetConfig::new("src", "nonexistent");
        let records = batch.into_records(&config);
        assert_eq!(records[0].get("val"), Some(&FieldValue::Int(42)));
    }

    #[test]
    fn batch_binary_column() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "blob",
            DataType::Binary,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BinaryArray::from_vec(vec![&[1u8, 2u8, 3u8][..]]))],
        )
        .unwrap();
        let config = DatasetConfig::new("src", "nonexistent");
        let records = batch.into_records(&config);
        assert_eq!(
            records[0].get("blob"),
            Some(&FieldValue::Bytes(vec![1, 2, 3]))
        );
    }
}
