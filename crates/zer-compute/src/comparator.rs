//! `DeviceComparator`, implements the `Comparator` trait.
//!
//! `compare_batch_from_pool` always routes to the CPU (Rayon parallel) path.
//! String comparison (Jaro-Winkler) is branch-heavy and dominated by PCIe
//! transfer overhead on GPU, making CPU faster for all observed batch sizes.

use std::sync::Arc;

use zer_core::{
    comparison::{ComparisonBatch, ComparisonVector},
    record::Record,
    record_pool::RecordPool,
    schema::Schema,
    traits::Comparator,
};

use crate::{
    backend::{cpu::CpuFallbackComparator, DeviceBackend},
    error::GpuError,
};

pub struct DeviceComparator {
    backend: Arc<DeviceBackend>,
    cpu_fallback: CpuFallbackComparator,
}

impl DeviceComparator {
    pub fn new(backend: Arc<DeviceBackend>, schema: &Schema) -> Result<Self, GpuError> {
        let cpu_fallback = CpuFallbackComparator::from_schema(schema);
        Ok(Self {
            backend,
            cpu_fallback,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }
}

impl Comparator for DeviceComparator {
    fn compare(&self, a: &Record, b: &Record, schema: &Schema) -> ComparisonVector {
        self.cpu_fallback.compare(a, b, schema)
    }

    fn compare_batch_from_pool(
        &self,
        pool: &RecordPool,
        indices: &[(usize, usize)],
        schema: &Schema,
    ) -> ComparisonBatch {
        if indices.is_empty() {
            return ComparisonBatch::new(0, schema.fields.len(), vec![]);
        }

        // compare_batch always runs on CPU: string comparison (Jaro-Winkler) is branch-heavy
        // and dominated by PCIe transfer overhead on GPU for all observed batch sizes.
        self.cpu_fallback
            .compare_batch_from_pool(pool, indices, schema)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{
        comparison::ComparisonLevel,
        record::{FieldValue, Record},
        record_pool::RecordPool,
        schema::{FieldKind, SchemaBuilder},
    };

    fn test_schema() -> Schema {
        SchemaBuilder::new()
            .field("naam", FieldKind::Name)
            .field("datum", FieldKind::Date)
            .field("kenteken", FieldKind::LicensePlate)
            .build()
            .unwrap()
    }

    fn make_record(id: u64) -> Record {
        Record::new(id)
            .insert("naam", FieldValue::Text("Alice de Vries".into()))
            .insert("datum", FieldValue::Text("1990-03-15".into()))
            .insert("kenteken", FieldValue::Text("12-ABC-3".into()))
    }

    fn make_record_b(id: u64) -> Record {
        Record::new(id)
            .insert("naam", FieldValue::Text("Alicia de Vrees".into()))
            .insert("datum", FieldValue::Text("1990-03-15".into()))
            .insert("kenteken", FieldValue::Text("12-ABC-3".into()))
    }

    #[test]
    fn single_pair_uses_cpu_path() {
        let schema = test_schema();
        let backend = Arc::new(DeviceBackend::cpu());
        let cmp = DeviceComparator::new(backend, &schema).unwrap();

        let a = make_record(1);
        let b = make_record_b(2);
        let vec = cmp.compare(&a, &b, &schema);

        assert_eq!(vec.record_a, 1);
        assert_eq!(vec.record_b, 2);
        assert_eq!(vec.levels.len(), 3);
    }

    #[test]
    fn small_batch_uses_cpu_fallback() {
        let schema = test_schema();
        let backend = Arc::new(DeviceBackend::cpu());
        let cmp = DeviceComparator::new(backend, &schema).unwrap();

        let records: Vec<Record> = (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    make_record(i)
                } else {
                    make_record_b(i)
                }
            })
            .collect();
        let pool = RecordPool::from_records(&records, &schema);
        let indices: Vec<(usize, usize)> = (0..10).map(|i| (i * 2, i * 2 + 1)).collect();

        let batch = cmp.compare_batch_from_pool(&pool, &indices, &schema);
        assert_eq!(batch.n_pairs, 10);
        assert_eq!(batch.n_fields, 3);
        assert_eq!(batch.levels.len(), 3 * 10);
    }

    #[test]
    fn empty_batch_returns_empty() {
        let schema = test_schema();
        let backend = Arc::new(DeviceBackend::cpu());
        let cmp = DeviceComparator::new(backend, &schema).unwrap();
        let pool = RecordPool::new(schema.fields.len());
        let batch = cmp.compare_batch_from_pool(&pool, &[], &schema);
        assert_eq!(batch.n_pairs, 0);
        assert!(batch.levels.is_empty());
    }

    #[test]
    fn exact_match_produces_exact_levels() {
        let schema = test_schema();
        let backend = Arc::new(DeviceBackend::cpu());
        let cmp = DeviceComparator::new(backend, &schema).unwrap();

        let r = Record::new(1)
            .insert("naam", FieldValue::Text("Jan Jansen".into()))
            .insert("datum", FieldValue::Text("1980-01-01".into()))
            .insert("kenteken", FieldValue::Text("AB-123-C".into()));
        let vec = cmp.compare(&r.clone(), &r, &schema);

        for level in &vec.levels {
            assert_eq!(
                *level,
                ComparisonLevel::Exact,
                "identical records should give Exact"
            );
        }
    }

    #[test]
    fn completely_different_records_produce_none_levels() {
        let schema = test_schema();
        let backend = Arc::new(DeviceBackend::cpu());
        let cmp = DeviceComparator::new(backend, &schema).unwrap();

        let a = Record::new(1)
            .insert("naam", FieldValue::Text("Henk".into()))
            .insert("datum", FieldValue::Text("1950-01-01".into()))
            .insert("kenteken", FieldValue::Text("XX-000-X".into()));
        let b = Record::new(2)
            .insert("naam", FieldValue::Text("Zäzä".into()))
            .insert("datum", FieldValue::Text("2010-12-31".into()))
            .insert("kenteken", FieldValue::Text("YY-999-Y".into()));

        let vec = cmp.compare(&a, &b, &schema);
        for level in &vec.levels {
            assert!(
                matches!(level, ComparisonLevel::None | ComparisonLevel::Partial),
                "very different records should produce None or Partial levels"
            );
        }
    }

    fn synthetic_records(n: usize, schema: &Schema) -> Vec<Record> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let names = [
            "Alice",
            "Alicia",
            "Bob",
            "Robert",
            "Eva",
            "Eva-Marie",
            "Jan",
            "Johan",
            "Petra",
            "Pietra",
            "Lena",
            "Lena-Marie",
        ];
        let dates = [
            "1990-01-15",
            "1990-01-16",
            "1985-06-20",
            "1975-03-03",
            "2000-12-31",
            "2001-01-01",
            "1960-07-07",
            "1970-11-22",
        ];
        let plates = [
            "12-ABC-3", "12-ABD-3", "45-XYZ-6", "46-XYZ-6", "AB-123-C", "AB-124-C", "ZZ-999-Z",
            "ZZ-998-Z",
        ];

        let fields: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();

        (0..n)
            .map(|i| {
                let mut r = Record::new(i as u64);
                for field in &fields {
                    let val = match *field {
                        "naam" => names[rng.gen_range(0..names.len())],
                        "datum" => dates[rng.gen_range(0..dates.len())],
                        "kenteken" => plates[rng.gen_range(0..plates.len())],
                        _ => "unknown",
                    };
                    r = r.insert(*field, FieldValue::Text(val.into()));
                }
                r
            })
            .collect()
    }

    #[test]
    fn large_batch_auto_detect_returns_correct_count() {
        let schema = test_schema();
        let backend = Arc::new(DeviceBackend::auto_detect());
        let cmp = DeviceComparator::new(backend, &schema).unwrap();

        let records = synthetic_records(4_000, &schema);
        let pool = RecordPool::from_records(&records, &schema);
        let indices: Vec<(usize, usize)> = (0..2_000).map(|i| (i * 2, i * 2 + 1)).collect();

        let batch = cmp.compare_batch_from_pool(&pool, &indices, &schema);

        assert_eq!(
            batch.n_pairs, 2_000,
            "compare_batch_from_pool must return one row per pair"
        );
        assert_eq!(
            batch.n_fields, 3,
            "each batch must have one column per field"
        );
    }
}
