//! Integration tests for the unified `zer` API.
//!
//! These tests import *only* from `zer`, no `zer-compare`, `zer-compute`,
//! or `zer-core` imports.  That is the point: a user who adds `zer` to
//! their Cargo.toml should need nothing else for the common case.
//!
//! Run with `--features=cuda` to exercise the GPU path.

use std::sync::{Arc, OnceLock};

use zer::prelude::*;

// Initialised once per test binary; all auto_detect tests share this backend.
static BACKEND: OnceLock<Arc<Backend>> = OnceLock::new();
fn shared_backend() -> Arc<Backend> {
    Arc::clone(BACKEND.get_or_init(|| Arc::new(Backend::auto_detect())))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn two_field_schema() -> Schema {
    SchemaBuilder::new()
        .field("naam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .unwrap()
}

fn exact_record(id: u64, naam: &str, datum: &str) -> Record {
    Record::new(id)
        .insert("naam", FieldValue::Text(naam.into()))
        .insert("geboortedatum", FieldValue::Text(datum.into()))
}

fn separable_vectors(n_matches: usize, n_nonmatches: usize, n_fields: usize) -> ComparisonBatch {
    let mut v = Vec::with_capacity(n_matches + n_nonmatches);
    for i in 0..n_matches as u64 {
        v.push(ComparisonVector::new(
            i * 2,
            i * 2 + 1,
            vec![ComparisonLevel::Exact; n_fields],
        ));
    }
    let off = (n_matches as u64) * 2;
    for i in 0..n_nonmatches as u64 {
        v.push(ComparisonVector::new(
            off + i * 2,
            off + i * 2 + 1,
            vec![ComparisonLevel::None; n_fields],
        ));
    }
    ComparisonBatch::from_vectors(&v)
}

// ── Backend ───────────────────────────────────────────────────────────────────

#[test]
fn backend_auto_detect_returns_valid_name() {
    let b = shared_backend();
    assert!(
        matches!(b.name(), "cpu" | "cuda" | "avx2"),
        "unexpected backend name: {}",
        b.name()
    );
}

#[test]
fn backend_cpu_is_not_gpu() {
    let b = Backend::cpu();
    assert_eq!(b.name(), "cpu");
    assert!(!b.is_gpu());
}

#[test]
fn backend_auto_detect_does_not_panic() {
    // Just verify it completes without panicking, GPU may or may not be present.
    let _ = shared_backend();
}

// ── Comparator ────────────────────────────────────────────────────────────────

#[test]
fn comparator_backend_name_matches_backend() {
    let schema = two_field_schema();
    let b = Backend::cpu();
    let c = Comparator::new(&schema, &b);
    assert_eq!(c.backend_name(), b.name());
}

#[test]
fn comparator_identical_records_produce_exact_levels() {
    let schema = two_field_schema();
    let b = shared_backend();
    let c = Comparator::new(&schema, &*b);

    let a = exact_record(1, "Jan de Vries", "1990-01-15");
    let r = exact_record(2, "Jan de Vries", "1990-01-15");

    let v = c.compare(&a, &r, &schema);
    assert_eq!(v.levels.len(), 2, "expected one level per field");
    assert_eq!(
        v.levels[0],
        ComparisonLevel::Exact,
        "identical names should be Exact"
    );
}

#[test]
fn comparator_batch_length_matches_input() {
    let schema = two_field_schema();
    let b = shared_backend();
    let c = Comparator::new(&schema, &*b);

    let pairs: Vec<(Record, Record)> = (0..10u64)
        .map(|i| {
            (
                exact_record(i * 2, "Alice", "2000-01-01"),
                exact_record(i * 2 + 1, "Alice", "2000-01-01"),
            )
        })
        .collect();

    let pool = RecordPool::from_pairs(&pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors = c.compare_batch_from_pool(&pool, &indices, &schema);
    assert_eq!(vectors.n_pairs, 10);
}

// ── Scorer ────────────────────────────────────────────────────────────────────

#[test]
fn scorer_backend_name_matches_backend() {
    let b = Backend::cpu();
    let s = Scorer::new(&b);
    assert_eq!(s.backend_name(), b.name());
}

#[test]
fn scorer_comparator_backend_names_agree() {
    // Both are built from the same auto-detected Backend, they must report
    // the same name, verifying that the feature-flag backend selection is
    // consistent across Comparator and Scorer.
    let schema = two_field_schema();
    let b = shared_backend();
    let c = Comparator::new(&schema, &*b);
    let s = Scorer::new(&*b);
    assert_eq!(c.backend_name(), s.backend_name());
    assert_eq!(c.backend_name(), b.name());
}

#[test]
fn scorer_exact_match_gives_high_probability() {
    let b = Backend::cpu();
    let scorer = Scorer::new(&b);
    let params = ModelParams {
        m: vec![vec![0.05, 0.10, 0.15, 0.70]; 2],
        u: vec![vec![0.70, 0.15, 0.10, 0.05]; 2],
        log_prior_odds: 0.0,
        upper_threshold: 0.9,
        lower_threshold: 0.1,
    };
    let v = ComparisonVector::new(1, 2, vec![ComparisonLevel::Exact; 2]);
    let pair = scorer.score(&v, &params);
    assert!(
        pair.match_probability > 0.9,
        "all-Exact should score > 0.9, got {}",
        pair.match_probability
    );
    assert_eq!(pair.band, MatchBand::AutoMatch);
}

#[test]
fn scorer_none_levels_give_low_probability() {
    let b = Backend::cpu();
    let scorer = Scorer::new(&b);
    let params = ModelParams {
        m: vec![vec![0.05, 0.10, 0.15, 0.70]; 2],
        u: vec![vec![0.70, 0.15, 0.10, 0.05]; 2],
        log_prior_odds: 0.0,
        upper_threshold: 0.9,
        lower_threshold: 0.1,
    };
    let v = ComparisonVector::new(3, 4, vec![ComparisonLevel::None; 2]);
    let pair = scorer.score(&v, &params);
    assert!(
        pair.match_probability < 0.1,
        "all-None should score < 0.1, got {}",
        pair.match_probability
    );
    assert_eq!(pair.band, MatchBand::AutoReject);
}

// ── EM parameter estimation ───────────────────────────────────────────────────

#[test]
fn estimate_params_converges_on_separable_data() {
    let b = shared_backend();
    let scorer = Scorer::new(&*b);
    let vectors = separable_vectors(200, 800, 3);

    let params = scorer
        .estimate_params(&vectors, None, 30)
        .expect("EM must not fail on separable data");

    for f in 0..3 {
        assert!(
            params.m[f][3] > params.u[f][3],
            "field {f}: m[Exact] must exceed u[Exact] after EM"
        );
        assert!(
            params.m[f][0] < params.u[f][0],
            "field {f}: m[None] must be below u[None] after EM"
        );
    }
}

#[test]
fn estimate_params_returns_error_on_empty_input() {
    let b = Backend::cpu();
    let scorer = Scorer::new(&b);
    assert!(
        scorer
            .estimate_params(&ComparisonBatch::new(0, 0, vec![]), None, 10)
            .is_err(),
        "empty input should return an error"
    );
}

#[test]
fn score_batch_matches_individual_scores() {
    let b = Backend::cpu();
    let scorer = Scorer::new(&b);
    let params = ModelParams {
        m: vec![vec![0.05, 0.10, 0.15, 0.70]; 3],
        u: vec![vec![0.70, 0.15, 0.10, 0.05]; 3],
        log_prior_odds: 0.0,
        upper_threshold: 0.9,
        lower_threshold: 0.1,
    };
    let raw_vectors = vec![
        ComparisonVector::new(1, 2, vec![ComparisonLevel::Exact; 3]),
        ComparisonVector::new(3, 4, vec![ComparisonLevel::None; 3]),
        ComparisonVector::new(
            5,
            6,
            vec![
                ComparisonLevel::Close,
                ComparisonLevel::Partial,
                ComparisonLevel::Exact,
            ],
        ),
    ];
    let comparison_batch = ComparisonBatch::from_vectors(&raw_vectors);

    let scored = scorer.score_batch(&comparison_batch, &params);
    for (v, br) in raw_vectors.iter().zip(scored.iter()) {
        let single = scorer.score(v, &params);
        assert!(
            (single.match_probability - br.match_probability).abs() < 1e-6,
            "batch and individual scores must agree"
        );
    }
}

// ── Separation proof: CPU-only path works without zer-compute ──────────────────

#[test]
fn cpu_only_path_uses_no_gpu_types() {
    // This test exists to document the invariant: when Backend::cpu() is used,
    // no GPU code path is entered.  We verify it by checking backend_name() is
    // "cpu" and the results are correct, same assertions as the GPU tests.
    let schema = two_field_schema();
    let backend = Backend::cpu();
    assert!(!backend.is_gpu(), "Backend::cpu() must not report is_gpu()");

    let comparator = Comparator::new(&schema, &backend);
    let scorer = Scorer::new(&backend);
    assert_eq!(comparator.backend_name(), "cpu");
    assert_eq!(scorer.backend_name(), "cpu");

    // Full pipeline on CPU
    let pairs: Vec<(Record, Record)> = vec![
        (
            exact_record(0, "Alice Jansen", "1985-03-10"),
            exact_record(1, "Alice Jansen", "1985-03-10"),
        ),
        (
            exact_record(2, "Bob Smit", "1970-07-22"),
            exact_record(3, "Carol Visser", "1995-12-01"),
        ),
    ];
    let pool = RecordPool::from_pairs(&pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors = comparator.compare_batch_from_pool(&pool, &indices, &schema);
    assert_eq!(vectors.n_pairs, 2);

    let params = scorer
        .estimate_params(&vectors, None, 20)
        .expect("EM on tiny CPU batch must succeed");
    let scored = scorer.score_batch(&vectors, &params);
    assert_eq!(scored.len(), 2);
    // Identical pair scores higher than mismatched pair
    assert!(scored[0].match_probability > scored[1].match_probability);
}

// ── Cross-backend correctness ─────────────────────────────────────────────────

/// CPU comparator and auto_detect comparator must produce identical
/// `ComparisonLevel` vectors for the same input pairs.
///
/// This is the user-API-level exit criterion for phase-04: when run with
/// `--features=cuda`, `auto_detect()` selects the CUDA backend and this test
/// verifies its output is bit-for-bit identical to the CPU reference.
#[test]
fn cpu_and_auto_detect_comparator_produce_identical_levels() {
    let schema = two_field_schema();

    let cpu_backend = Backend::cpu();
    let gpu_backend = shared_backend();
    let cpu_cmp = Comparator::new(&schema, &cpu_backend);
    let gpu_cmp = Comparator::new(&schema, &*gpu_backend);

    let pairs: Vec<(Record, Record)> = vec![
        (
            exact_record(0, "Alice Jansen", "1985-03-10"),
            exact_record(1, "Alice Jansen", "1985-03-10"),
        ),
        (
            exact_record(2, "Bob Smit", "1970-07-22"),
            exact_record(3, "Carol Visser", "1995-12-01"),
        ),
        (
            exact_record(4, "Jan de Vries", "2001-11-05"),
            exact_record(5, "Jan de Vries", "2001-11-05"),
        ),
        (
            exact_record(6, "Mohammed El A", "1990-06-15"),
            exact_record(7, "Mohamed El A", "1990-06-15"),
        ),
        (
            exact_record(8, "Fatima Yilmaz", "1978-02-28"),
            exact_record(9, "Fatimah Yilmaz", "1978-02-28"),
        ),
        (
            exact_record(10, "Pieter Bakker", "1965-09-03"),
            exact_record(11, "Piet Bakker", "1965-09-03"),
        ),
    ];

    let pool = RecordPool::from_pairs(&pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let cpu_vectors = cpu_cmp.compare_batch_from_pool(&pool, &indices, &schema);
    let gpu_vectors = gpu_cmp.compare_batch_from_pool(&pool, &indices, &schema);

    assert_eq!(cpu_vectors.n_pairs, gpu_vectors.n_pairs);
    let n_fields = schema.fields.len();
    for p in 0..cpu_vectors.n_pairs {
        for f in 0..n_fields {
            let c = cpu_vectors.level(f, p);
            let g = gpu_vectors.level(f, p);
            assert_eq!(
                c,
                g,
                "pair {p} field {f}: CPU and {} backends differ (cpu={c:?}, gpu={g:?})",
                gpu_backend.name(),
            );
        }
    }
}
