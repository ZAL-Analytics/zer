/// CPU vs CUDA parity tests for the `compare_score_kernel`.
///
/// Each test builds a set of record pairs, runs them through both the CPU
/// reference path (`FieldComparator` from `zer-compare`) and the CUDA kernel
/// (`DeviceComparator` with the CUDA backend), then asserts that every
/// `ComparisonLevel` in every field position is identical.
///
/// Run with `--features=cuda` to activate the CUDA path; the tests are skipped
/// gracefully when CUDA is not compiled in or no GPU is present.
///
/// Tests are organised by field kind so regressions are easy to localise.
use std::sync::Arc;

use zer_compare::FieldComparator;
use zer_compute::{BackendPreference, DeviceBackend, DeviceComparator};
use zer_core::{
    comparison::ComparisonLevel,
    record::{FieldValue, Record},
    record_pool::RecordPool,
    schema::{FieldKind, SchemaBuilder},
    traits::Comparator,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a two-field schema with the given kinds for quick synthetic tests.
fn schema2(k0: FieldKind, k1: FieldKind) -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("f0", k0)
        .field("f1", k1)
        .build()
        .unwrap()
}

/// Build a schema with a single field of the given kind.
fn schema1(k: FieldKind) -> zer_core::schema::Schema {
    SchemaBuilder::new().field("f0", k).build().unwrap()
}

fn tv(s: &str) -> FieldValue {
    FieldValue::Text(s.into())
}

/// Create a CUDA `DeviceComparator`, or return `None` if CUDA is unavailable.
/// Tests that call this will be skipped (not failed) on machines without a GPU.
fn try_cuda_cmp(schema: &zer_core::schema::Schema) -> Option<DeviceComparator> {
    let backend = DeviceBackend::from_preference(BackendPreference::Cuda).ok()?;
    DeviceComparator::new(Arc::new(backend), schema).ok()
}

fn cpu_cmp(schema: &zer_core::schema::Schema) -> DeviceComparator {
    DeviceComparator::new(Arc::new(DeviceBackend::cpu()), schema).unwrap()
}

/// Run both CPU and CUDA comparators on `pairs` and assert level-identical output.
/// Returns the number of pairs compared, or 0 if CUDA was unavailable.
fn assert_parity(pairs: &[(Record, Record)], schema: &zer_core::schema::Schema) -> usize {
    let cuda_cmp = match try_cuda_cmp(schema) {
        Some(c) => c,
        None => {
            println!("CUDA unavailable, skipping parity check");
            return 0;
        }
    };
    let ref_cmp = FieldComparator::from_schema(schema);

    let pool = RecordPool::from_pairs(pairs, schema);
    let indices: Vec<(usize, usize)> = (0..pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let cuda_batch = cuda_cmp.compare_batch_from_pool(&pool, &indices, schema);
    let ref_batch = ref_cmp.compare_batch_from_pool(&pool, &indices, schema);

    assert_eq!(
        cuda_batch.n_pairs, ref_batch.n_pairs,
        "output length mismatch"
    );

    let n_fields = schema.fields.len();
    for p in 0..cuda_batch.n_pairs {
        for f in 0..n_fields {
            let cuda_level = cuda_batch.level(f, p);
            let ref_level = ref_batch.level(f, p);
            assert_eq!(
                cuda_level, ref_level,
                "pair {p} field {f}: CUDA level {cuda_level:?} ≠ CPU level {ref_level:?}",
            );
        }
    }
    pairs.len()
}

// ── Synthetic pair builders ───────────────────────────────────────────────────

fn pair(id_a: u64, id_b: u64, field: &str, va: &str, vb: &str) -> (Record, Record) {
    let a = Record::new(id_a).insert(field, tv(va));
    let b = Record::new(id_b).insert(field, tv(vb));
    (a, b)
}

fn pair2(id_a: u64, id_b: u64, va0: &str, vb0: &str, va1: &str, vb1: &str) -> (Record, Record) {
    let a = Record::new(id_a)
        .insert("f0", tv(va0))
        .insert("f1", tv(va1));
    let b = Record::new(id_b)
        .insert("f0", tv(vb0))
        .insert("f1", tv(vb1));
    (a, b)
}

// ── Tests: Name / Jaro-Winkler ────────────────────────────────────────────────

#[test]
fn parity_name_exact_match() {
    let schema = schema1(FieldKind::Name);
    let pairs = vec![
        pair(1, 2, "f0", "Johannes", "Johannes"),
        pair(3, 4, "f0", "Maria", "Maria"),
        pair(5, 6, "f0", "Alexander", "Alexander"),
    ];
    assert_parity(&pairs, &schema);
}

#[test]
fn parity_name_close_match() {
    let schema = schema1(FieldKind::Name);
    let pairs = vec![
        pair(1, 2, "f0", "Johannes", "Johanness"), // one extra char
        pair(3, 4, "f0", "Maria", "Mariya"),       // common variant
        pair(5, 6, "f0", "Pieter", "Peter"),
        pair(7, 8, "f0", "Willem", "William"),
    ];
    assert_parity(&pairs, &schema);
}

#[test]
fn parity_name_no_match() {
    let schema = schema1(FieldKind::Name);
    let pairs = vec![
        pair(1, 2, "f0", "Johannes", "Fatima"),
        pair(3, 4, "f0", "Hendrik", "Xiaoming"),
        pair(5, 6, "f0", "Anna", "Boris"),
    ];
    assert_parity(&pairs, &schema);
}

#[test]
fn parity_name_empty_strings() {
    let schema = schema1(FieldKind::Name);
    let pairs = vec![
        (
            Record::new(1).insert("f0", FieldValue::Null),
            Record::new(2).insert("f0", tv("Johannes")),
        ),
        (
            Record::new(3).insert("f0", FieldValue::Null),
            Record::new(4).insert("f0", FieldValue::Null),
        ),
        pair(5, 6, "f0", "", ""),
    ];
    assert_parity(&pairs, &schema);
}

#[test]
fn parity_name_long_string_truncation() {
    // Strings > STRING_STRIDE (64 bytes) are truncated, both sides must agree.
    let schema = schema1(FieldKind::Name);
    let long_a = "A".repeat(100);
    let long_b = "A".repeat(100);
    let long_c = "B".repeat(100);
    let pairs = vec![
        pair(1, 2, "f0", &long_a, &long_b), // same after truncation → Exact
        pair(3, 4, "f0", &long_a, &long_c), // different after truncation → None
    ];
    assert_parity(&pairs, &schema);
}

// ── Tests: Date ───────────────────────────────────────────────────────────────

#[test]
fn parity_date_all_levels() {
    let schema = schema1(FieldKind::Date);
    let pairs = vec![
        pair(1, 2, "f0", "1990-06-15", "1990-06-15"),  // exact
        pair(3, 4, "f0", "1990-06-15", "1990-06-20"),  // same yr+mo
        pair(5, 6, "f0", "1990-06-15", "1990-09-01"),  // same yr
        pair(7, 8, "f0", "1990-06-15", "1991-06-15"),  // adjacent yr
        pair(9, 10, "f0", "1990-06-15", "1985-01-01"), // far
        pair(11, 12, "f0", "bad-date", "1990-01-01"),  // parse fail
    ];
    assert_parity(&pairs, &schema);
}

// ── Tests: ID / Phone / LicensePlate ─────────────────────────────────────────

#[test]
fn parity_id_hamming() {
    let schema = schema1(FieldKind::Id);
    let pairs = vec![
        pair(1, 2, "f0", "12345678", "12345678"), // exact
        pair(3, 4, "f0", "12345678", "12345679"), // hamming-1 → Close
        pair(5, 6, "f0", "12345678", "12345699"), // hamming-2 → None
        pair(7, 8, "f0", "ABC123", "XYZ999"),     // totally different
    ];
    assert_parity(&pairs, &schema);
}

#[test]
fn parity_licenseplate_ocr_pairs() {
    let schema = schema1(FieldKind::LicensePlate);
    let pairs = vec![
        pair(1, 2, "f0", "AB-123-C", "AB-123-C"), // exact
        pair(3, 4, "f0", "AB-123-C", "AB-123-G"), // B/G confusion → Close
        pair(5, 6, "f0", "AB-123-C", "ZZ-999-X"), // unrelated
    ];
    assert_parity(&pairs, &schema);
}

// ── Tests: Categorical ────────────────────────────────────────────────────────

#[test]
fn parity_categorical() {
    let schema = schema1(FieldKind::Categorical);
    let pairs = vec![
        pair(1, 2, "f0", "NL", "NL"),
        pair(3, 4, "f0", "NL", "BE"),
        pair(5, 6, "f0", "MALE", "MALE"),
        pair(7, 8, "f0", "MALE", "FEMALE"),
    ];
    assert_parity(&pairs, &schema);
}

// ── Tests: Numeric ────────────────────────────────────────────────────────────

#[test]
fn parity_numeric_buckets() {
    let schema = schema1(FieldKind::Numeric);
    let pairs = vec![
        pair(1, 2, "f0", "100.0", "100.0"),   // exact
        pair(3, 4, "f0", "100.0", "104.9"),   // ≤5% → very close
        pair(5, 6, "f0", "100.0", "115.0"),   // ≤20%
        pair(7, 8, "f0", "100.0", "140.0"),   // ≤50%
        pair(9, 10, "f0", "100.0", "300.0"),  // >50% → None
        pair(11, 12, "f0", "-50.5", "-50.5"), // negative equal
    ];
    assert_parity(&pairs, &schema);
}

// ── Tests: Address / FreeText (Jaro-Winkler variant) ─────────────────────────

#[test]
fn parity_address() {
    let schema = schema1(FieldKind::Address);
    let pairs = vec![
        pair(1, 2, "f0", "Keizersgracht 100", "Keizersgracht 100"),
        pair(3, 4, "f0", "Keizersgracht 100", "Keizersgracht 101"),
        pair(5, 6, "f0", "Prinsengracht 44 A", "Prinsengracht 44"),
        pair(7, 8, "f0", "Herengracht 500 huis", "Singel 12"),
    ];
    assert_parity(&pairs, &schema);
}

// ── Tests: multi-field schema ─────────────────────────────────────────────────

#[test]
fn parity_multi_field_mixed_kinds() {
    let schema = schema2(FieldKind::Name, FieldKind::Date);
    let pairs = vec![
        pair2(1, 2, "Jan", "Jan", "1990-01-01", "1990-01-01"), // both exact
        pair2(3, 4, "Jan", "Jon", "1990-01-01", "1990-01-02"), // name close, date close
        pair2(5, 6, "Alice", "Boris", "1985-06-15", "1985-06-15"), // name none, date exact
        pair2(7, 8, "Alice", "Alicia", "1985-06-15", "1975-12-31"), // name close, date none
    ];
    assert_parity(&pairs, &schema);
}

#[test]
fn parity_all_field_kinds_in_one_schema() {
    let schema = SchemaBuilder::new()
        .field("naam", FieldKind::Name)
        .field("datum", FieldKind::Date)
        .field("postcode", FieldKind::Id)
        .field("land", FieldKind::Categorical)
        .field("straat", FieldKind::Address)
        .field("nr", FieldKind::Numeric)
        .field("kenteken", FieldKind::LicensePlate)
        .build()
        .unwrap();

    let mut pairs = Vec::new();
    let mut id: u64 = 1;
    for (name_a, name_b, date_a, date_b) in [
        ("Jan de Vries", "Jan de Vries", "1985-03-12", "1985-03-12"),
        ("Anna Bakker", "Anna B.", "1990-07-04", "1990-07-05"),
        ("Pieter Jansen", "Piet Janssen", "1978-11-22", "1979-11-22"),
        ("Mohammed Al", "Mohammed Al", "2000-01-01", "2001-01-01"),
    ] {
        let a = Record::new(id)
            .insert("naam", tv(name_a))
            .insert("datum", tv(date_a))
            .insert("postcode", tv("1234AB"))
            .insert("land", tv("NL"))
            .insert("straat", tv("Hoofdstraat 1"))
            .insert("nr", tv("42.5"))
            .insert("kenteken", tv("AB-123-C"));
        let b = Record::new(id + 1)
            .insert("naam", tv(name_b))
            .insert("datum", tv(date_b))
            .insert("postcode", tv("1234AB"))
            .insert("land", tv("NL"))
            .insert("straat", tv("Hoofdstraat 2"))
            .insert("nr", tv("43.0"))
            .insert("kenteken", tv("AB-123-D"));
        pairs.push((a, b));
        id += 2;
    }

    assert_parity(&pairs, &schema);
}

// ── Tests: large batch (stress test for warp boundary correctness) ────────────

#[test]
fn parity_large_batch_crosses_warp_boundaries() {
    // 512 pairs, exercises multiple CUDA blocks (blockDim=256) and ensures
    // warp boundary (32-thread) alignment is correct in the new interleaved layout.
    let schema = schema1(FieldKind::Name);
    let names_a = [
        "Jan", "Piet", "Klaas", "Marie", "Anna", "Dirk", "Henk", "Lena",
    ];
    let names_b = [
        "Jan", "Piet", "Klaas", "Maria", "Anne", "Dick", "Henk", "Lena",
    ];

    let pairs: Vec<_> = (0..512u64)
        .map(|i| {
            let na = names_a[(i as usize) % names_a.len()];
            let nb = names_b[(i as usize) % names_b.len()];
            pair(i * 2, i * 2 + 1, "f0", na, nb)
        })
        .collect();

    let n = assert_parity(&pairs, &schema);
    if n > 0 {
        println!("large batch: {n} pairs compared, all levels match");
    }
}

// ── Tests: CPU DeviceComparator vs FieldComparator (always runs) ──────────────
//
// These don't need CUDA; they verify the CPU path of DeviceComparator agrees
// with FieldComparator across all field kinds, including the new interleaved
// GpuStringBuffer packing path used by the CUDA launch code.

#[test]
fn cpu_device_comparator_matches_field_comparator_name() {
    let schema = schema1(FieldKind::Name);
    let pairs = vec![
        pair(1, 2, "f0", "Johannes", "Johanness"),
        pair(3, 4, "f0", "Maria", "Mariya"),
        pair(5, 6, "f0", "Anna", "Anna"),
        pair(7, 8, "f0", "Hans", "Boris"),
    ];

    let cpu_device = cpu_cmp(&schema);
    let ref_cmp = FieldComparator::from_schema(&schema);

    let pool = RecordPool::from_pairs(&pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let device_batch = cpu_device.compare_batch_from_pool(&pool, &indices, &schema);
    let ref_batch = ref_cmp.compare_batch_from_pool(&pool, &indices, &schema);

    let n_fields = schema.fields.len();
    for p in 0..device_batch.n_pairs {
        for f in 0..n_fields {
            let d = device_batch.level(f, p);
            let r = ref_batch.level(f, p);
            assert_eq!(
                d, r,
                "pair {p} field {f}: DeviceComparator(cpu) {d:?} ≠ FieldComparator {r:?}",
            );
        }
    }
}

#[test]
fn cpu_device_comparator_matches_field_comparator_all_kinds() {
    let schema = SchemaBuilder::new()
        .field("naam", FieldKind::Name)
        .field("datum", FieldKind::Date)
        .field("postcode", FieldKind::Id)
        .field("land", FieldKind::Categorical)
        .field("straat", FieldKind::Address)
        .field("nr", FieldKind::Numeric)
        .field("kenteken", FieldKind::LicensePlate)
        .build()
        .unwrap();

    let make_rec = |id: u64, suffix: &str| {
        Record::new(id)
            .insert("naam", tv(&format!("Jan{suffix}")))
            .insert("datum", tv("1990-01-01"))
            .insert("postcode", tv("1234AB"))
            .insert("land", tv("NL"))
            .insert("straat", tv("Hoofdstraat 1"))
            .insert("nr", tv("42.5"))
            .insert("kenteken", tv("AB-123-C"))
    };

    let pairs: Vec<_> = (0..64u64)
        .map(|i| (make_rec(i * 2, "A"), make_rec(i * 2 + 1, "B")))
        .collect();

    let pool = RecordPool::from_pairs(&pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let device_batch = cpu_cmp(&schema).compare_batch_from_pool(&pool, &indices, &schema);
    let ref_batch =
        FieldComparator::from_schema(&schema).compare_batch_from_pool(&pool, &indices, &schema);

    let n_fields = schema.fields.len();
    for p in 0..device_batch.n_pairs {
        for f in 0..n_fields {
            let d = device_batch.level(f, p);
            let r = ref_batch.level(f, p);
            assert_eq!(
                d, r,
                "pair {p} field {f}: DeviceComparator(cpu) {d:?} ≠ FieldComparator {r:?}",
            );
        }
    }
}

// ── Test: level values are in the valid range ─────────────────────────────────

#[test]
fn cuda_output_levels_are_valid_variants() {
    let schema = schema1(FieldKind::Name);
    let pairs = vec![
        pair(1, 2, "f0", "Test", "Test"),
        pair(3, 4, "f0", "Test", "Different"),
    ];

    let cuda_cmp = match try_cuda_cmp(&schema) {
        Some(c) => c,
        None => return,
    };

    let pool = RecordPool::from_pairs(&pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let batch = cuda_cmp.compare_batch_from_pool(&pool, &indices, &schema);
    let n_fields = schema.fields.len();
    for p in 0..batch.n_pairs {
        for f in 0..n_fields {
            let level = batch.level(f, p);
            assert!(
                matches!(
                    level,
                    ComparisonLevel::None
                        | ComparisonLevel::Partial
                        | ComparisonLevel::Close
                        | ComparisonLevel::Exact
                ),
                "invalid comparison level: {level:?}"
            );
        }
    }
}
