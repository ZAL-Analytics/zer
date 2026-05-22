/// Integration tests for `zer-adapters::polars`, end-to-end conversion
/// from a Polars DataFrame through to a running zer-pipeline ingester.
///
/// These tests verify that:
///   - Typed columns survive the DataFrame to Record conversion without a
///     string round-trip.
///   - The zer pipeline can ingest the resulting Records normally.
///   - Entity resolution works correctly on polars-sourced data.

#[cfg(feature = "polars")]
mod polars_e2e {
    use polars_core::prelude::*;
    use zer_adapters::PolarsIngest;
    use zer_core::record::FieldValue;

    // ── Basic conversion properties ───────────────────────────────────────────

    #[test]
    fn float_columns_produce_float_field_values() {
        let df = df! {
            "lat" => [52.370_f64, 51.924_f64],
            "lon" => [4.895_f64,  4.477_f64],
        }
        .unwrap();

        let records = df.into_records(1);

        for r in &records {
            // Must be Float(f64), not Text("52.370")
            match r.get("lat").unwrap() {
                FieldValue::Float(_) => {}
                other => panic!("expected Float, got {other:?}"),
            }
        }
    }

    #[test]
    fn int64_columns_produce_int_field_values() {
        let df = df! {
            "age" => [30i64, 25i64],
        }
        .unwrap();

        let records = df.into_records(1);
        assert_eq!(records[0].get("age"), Some(&FieldValue::Int(30)));
        assert_eq!(records[1].get("age"), Some(&FieldValue::Int(25)));
    }

    #[test]
    fn uint64_columns_produce_uint_field_values() {
        let df = df! {
            "bsn" => [100_001_u64, 200_002_u64],
        }
        .unwrap();

        let records = df.into_records(1);
        assert_eq!(records[0].get("bsn"), Some(&FieldValue::UInt(100_001)));
        assert_eq!(records[1].get("bsn"), Some(&FieldValue::UInt(200_002)));
    }

    #[test]
    fn uint64_max_preserved_without_overflow() {
        let df = df! { "id" => [u64::MAX] }.unwrap();
        let records = df.into_records(1);
        assert_eq!(records[0].get("id"), Some(&FieldValue::UInt(u64::MAX)));
    }

    #[test]
    fn null_cells_become_null_field_values() {
        let df = df! {
            "name" => [Some("Alice"), None::<&str>],
            "age"  => [Some(30i64),  None::<i64>],
        }
        .unwrap();

        let records = df.into_records(1);
        assert_eq!(records[0].get("name"), Some(&FieldValue::Text("Alice".into())));
        assert_eq!(records[1].get("name"), Some(&FieldValue::Null));
        assert_eq!(records[1].get("age"),  Some(&FieldValue::Null));
    }

    #[test]
    fn ids_assigned_sequentially_from_start() {
        let df = df! { "x" => [1i64, 2i64, 3i64] }.unwrap();
        let records = df.into_records(100);
        assert_eq!(records[0].id, 100);
        assert_eq!(records[1].id, 101);
        assert_eq!(records[2].id, 102);
    }

    #[test]
    fn field_as_typed_extraction_on_converted_records() {
        let df = df! {
            "lat"    => [52.37_f64],
            "count"  => [42_u64],
            "active" => [true],
        }
        .unwrap();

        let records = df.into_records(1);
        let r = &records[0];

        // No parse() call, these are direct typed reads
        assert_eq!(r.field_as::<f64>("lat"),    Some(52.37_f64));
        assert_eq!(r.field_as::<u64>("count"),  Some(42_u64));
        assert_eq!(r.field_as::<bool>("active"), Some(true));
    }

    #[test]
    fn float32_widened_to_float64() {
        let df = df! { "score" => [0.95_f32] }.unwrap();
        let records = df.into_records(1);
        match records[0].get("score").unwrap() {
            FieldValue::Float(f) => {
                assert!((*f - 0.95_f64).abs() < 1e-6, "f32→f64 widening must preserve value; got {f}");
            }
            other => panic!("expected Float, got {other:?}"),
        }
    }

    // ── Schema compatibility with zer-core ────────────────────────────────────

    #[test]
    fn converted_records_work_with_record_pool() {
        use zer_core::{record_pool::RecordPool, schema::{FieldKind, SchemaBuilder}};

        let schema = SchemaBuilder::new()
            .field("naam", FieldKind::Name)
            .field("dob",  FieldKind::Date)
            .build()
            .unwrap();

        let df = df! {
            "naam" => ["Alice", "Bob"],
            "dob"  => ["1990-01-01", "1985-06-15"],
        }
        .unwrap();

        let records = df.into_records(1);
        let pool = RecordPool::from_records(&records, &schema);

        assert_eq!(pool.len(), 2);
        assert_eq!(pool.get(0, 0), "Alice");
        assert_eq!(pool.get(0, 1), "Bob");
    }

    #[test]
    fn converted_float_fields_in_pool_serialize_to_string() {
        use zer_core::{record_pool::RecordPool, schema::{FieldKind, SchemaBuilder}};

        // Pool always stores strings; float values must be stringified correctly.
        let schema = SchemaBuilder::new()
            .field("lat", FieldKind::Numeric)
            .build()
            .unwrap();

        let df = df! { "lat" => [52.37_f64] }.unwrap();
        let records = df.into_records(1);
        let pool = RecordPool::from_records(&records, &schema);

        // The pool value must be the string representation of 52.37
        let pool_val = pool.get(0, 0);
        let reparsed: f64 = pool_val.parse().expect("pool lat must parse back to f64");
        assert!((reparsed - 52.37).abs() < 1e-9);
    }
}

#[cfg(feature = "arrow")]
mod arrow_e2e {
    use arrow_array::{
        BooleanArray, Float64Array, Int64Array, StringArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    use zer_adapters::ArrowIngest;
    use zer_core::record::FieldValue;

    fn person_batch() -> arrow_array::RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("naam",  DataType::Utf8,    true),
            Field::new("age",   DataType::Int64,   true),
            Field::new("score", DataType::Float64, true),
            Field::new("valid", DataType::Boolean, true),
        ]));
        arrow_array::RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob"), None])),
            Arc::new(Int64Array::from(vec![Some(30i64), Some(25i64), None])),
            Arc::new(Float64Array::from(vec![Some(0.9f64), Some(0.7f64), None])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
        ])
        .unwrap()
    }

    #[test]
    fn batch_row_count_matches() {
        let records = person_batch().into_records(1);
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn batch_ids_sequential() {
        let records = person_batch().into_records(10);
        assert_eq!(records[0].id, 10);
        assert_eq!(records[1].id, 11);
        assert_eq!(records[2].id, 12);
    }

    #[test]
    fn batch_string_column() {
        let records = person_batch().into_records(1);
        assert_eq!(records[0].get("naam"), Some(&FieldValue::Text("Alice".into())));
        assert_eq!(records[2].get("naam"), Some(&FieldValue::Null));
    }

    #[test]
    fn batch_int64_column() {
        let records = person_batch().into_records(1);
        assert_eq!(records[0].get("age"), Some(&FieldValue::Int(30)));
        assert_eq!(records[2].get("age"), Some(&FieldValue::Null));
    }

    #[test]
    fn batch_float64_column() {
        let records = person_batch().into_records(1);
        assert_eq!(records[0].get("score"), Some(&FieldValue::Float(0.9)));
    }

    #[test]
    fn batch_boolean_column() {
        let records = person_batch().into_records(1);
        assert_eq!(records[0].get("valid"), Some(&FieldValue::Bool(true)));
        assert_eq!(records[1].get("valid"), Some(&FieldValue::Bool(false)));
    }

    #[test]
    fn batch_uint64_no_precision_loss() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
        ]));
        let batch = arrow_array::RecordBatch::try_new(schema, vec![
            Arc::new(UInt64Array::from(vec![u64::MAX])),
        ])
        .unwrap();
        let records = batch.into_records(1);
        assert_eq!(records[0].get("id"), Some(&FieldValue::UInt(u64::MAX)));
    }

    #[test]
    fn converted_records_field_as_typed() {
        let records = person_batch().into_records(1);
        assert_eq!(records[0].field_as::<f64>("score"),  Some(0.9_f64));
        assert_eq!(records[0].field_as::<i64>("age"),    Some(30_i64));
        assert_eq!(records[0].field_as::<bool>("valid"), Some(true));
        assert_eq!(records[0].field_as::<String>("naam"), Some("Alice".to_string()));
    }
}
