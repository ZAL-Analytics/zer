/// End-to-end inference tests, require real ONNX model files.
///
/// All tests in this file are marked `#[ignore]` and must be run explicitly:
///
/// ```sh
/// # Run all inference tests with the MiniLM model (fast, ~330 MB):
/// cargo test -p zer-judge --test test_inference -- --ignored
///
/// # Run with the DeBERTa-base model (~740 MB):
/// cargo test -p zer-judge --test test_inference deberta_base -- --ignored
/// ```
///
/// Tests skip gracefully when the model directory is absent rather than failing.

use std::sync::Arc;

use zer_core::{
    comparison::ComparisonVector,
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
    scoring::{MatchBand, ScoredPair},
    traits::{Judge, JudgeVerdict, RecordStore, VecRecordStore},
};
use zer_judge::{
    backend::JudgeBackend,
    judge::{DebertaJudge, DebertaJudgeConfig},
    spec::{DebertaBaseSpec, MiniLmSpec},
};

// ── Paths ─────────────────────────────────────────────────────────────────────

const MINILM_DIR:       &str = "../../models/fp16_fused/nli-minilm-onnx";
const DEBERTA_BASE_DIR: &str = "../../models/fp16_fused/nli-deberta-v3-base-onnx";

// ── Helpers ───────────────────────────────────────────────────────────────────

fn name_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("naam",          FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .unwrap()
}

fn scored_pair(a: u64, b: u64) -> ScoredPair {
    ScoredPair {
        record_a:          a,
        record_b:          b,
        match_weight:      0.0,
        match_probability: 0.55,
        vector:            ComparisonVector::new(a, b, vec![]),
        band:              MatchBand::Borderline,
    }
}

fn build_store(records: Vec<Record>) -> Arc<VecRecordStore> {
    let store = Arc::new(VecRecordStore::new());
    for r in records { store.insert(r); }
    store
}

// ── MiniLM tests ──────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires models/fp16_fused/nli-minilm-onnx/model.onnx (~330 MB)"]
fn minilm_empty_pairs_returns_empty_verdicts() {
    if !std::path::Path::new(MINILM_DIR).exists() { return; }

    let spec    = MiniLmSpec::from_dir(MINILM_DIR);
    let backend = JudgeBackend::cpu();
    let store   = build_store(vec![]);
    let judge   = DebertaJudge::new(&spec, &backend, store, name_schema(), DebertaJudgeConfig::default())
        .expect("judge construction failed");

    let verdicts = judge.adjudicate(&[]).expect("adjudicate failed");
    assert!(verdicts.is_empty());
}

#[test]
#[ignore = "requires models/fp16_fused/nli-minilm-onnx/model.onnx (~330 MB)"]
fn minilm_inference_runs_on_cpu() {
    if !std::path::Path::new(MINILM_DIR).exists() { return; }

    let schema = name_schema();
    let r_a = Record::new(1)
        .insert("naam",          FieldValue::Text("jan smits".into()))
        .insert("geboortedatum", FieldValue::Text("1981-04-02".into()));
    let r_b = Record::new(2)
        .insert("naam",          FieldValue::Text("jan smyts".into()))
        .insert("geboortedatum", FieldValue::Text("1981-04-02".into()));

    let store = build_store(vec![r_a, r_b]);
    let spec    = MiniLmSpec::from_dir(MINILM_DIR);
    let backend = JudgeBackend::cpu();

    let judge = DebertaJudge::new(&spec, &backend, store, schema, DebertaJudgeConfig::default())
        .expect("judge construction failed");

    let pairs    = vec![scored_pair(1, 2)];
    let verdicts = judge.adjudicate(&pairs).expect("adjudicate failed");

    assert_eq!(verdicts.len(), 1);
    assert!(
        matches!(verdicts[0], JudgeVerdict::IncreaseConfidence | JudgeVerdict::NoChange | JudgeVerdict::DecreaseConfidence),
        "verdict should be a known variant"
    );
}

#[test]
#[ignore = "requires models/fp16_fused/nli-minilm-onnx/model.onnx (~330 MB)"]
fn minilm_identical_records_produce_increase_confidence() {
    if !std::path::Path::new(MINILM_DIR).exists() { return; }

    let schema = name_schema();
    let record = Record::new(1)
        .insert("naam",          FieldValue::Text("anna de vries".into()))
        .insert("geboortedatum", FieldValue::Text("1990-06-15".into()));
    // Record 2 is an exact copy with a different id
    let record2 = Record::new(2)
        .insert("naam",          FieldValue::Text("anna de vries".into()))
        .insert("geboortedatum", FieldValue::Text("1990-06-15".into()));

    let store = build_store(vec![record, record2]);
    let spec    = MiniLmSpec::from_dir(MINILM_DIR);
    let backend = JudgeBackend::cpu();

    let judge = DebertaJudge::new(&spec, &backend, store, schema, DebertaJudgeConfig::default())
        .expect("judge construction failed");

    let verdicts = judge.adjudicate(&[scored_pair(1, 2)]).expect("adjudicate failed");
    // Identical records should score high → IncreaseConfidence
    assert!(
        matches!(verdicts[0], JudgeVerdict::IncreaseConfidence),
        "identical records should produce IncreaseConfidence, got {:?}", verdicts[0],
    );
}

#[test]
#[ignore = "requires models/fp16_fused/nli-minilm-onnx/model.onnx (~330 MB)"]
fn minilm_batch_returns_one_verdict_per_pair() {
    if !std::path::Path::new(MINILM_DIR).exists() { return; }

    let schema = name_schema();
    let records: Vec<Record> = (1..=6).map(|i| {
        Record::new(i).insert("naam", FieldValue::Text(format!("person {i}")))
    }).collect();

    let store   = build_store(records);
    let spec    = MiniLmSpec::from_dir(MINILM_DIR);
    let backend = JudgeBackend::cpu();
    let judge   = DebertaJudge::new(&spec, &backend, store, schema, DebertaJudgeConfig::default())
        .expect("judge construction failed");

    let pairs: Vec<ScoredPair> = (1..=5).map(|i| scored_pair(i, i + 1)).collect();
    let verdicts = judge.adjudicate(&pairs).expect("adjudicate failed");
    assert_eq!(verdicts.len(), 5, "one verdict per pair");
}

#[test]
#[ignore = "requires models/fp16_fused/nli-minilm-onnx/model.onnx (~330 MB)"]
fn minilm_clone_shares_worker_thread() {
    if !std::path::Path::new(MINILM_DIR).exists() { return; }

    let schema = name_schema();
    let r1 = Record::new(1).insert("naam", FieldValue::Text("alice".into()));
    let r2 = Record::new(2).insert("naam", FieldValue::Text("alice".into()));
    let store   = build_store(vec![r1, r2]);
    let spec    = MiniLmSpec::from_dir(MINILM_DIR);
    let backend = JudgeBackend::cpu();
    let judge   = DebertaJudge::new(&spec, &backend, store, schema, DebertaJudgeConfig::default())
        .expect("judge construction failed");

    let judge2 = judge.clone();

    // Both handles should produce valid verdicts, they share the same session
    let v1 = judge.adjudicate(&[scored_pair(1, 2)]).expect("adjudicate on original failed");
    let v2 = judge2.adjudicate(&[scored_pair(1, 2)]).expect("adjudicate on clone failed");
    assert_eq!(v1.len(), 1);
    assert_eq!(v2.len(), 1);
}

#[test]
#[ignore = "requires models/fp16_fused/nli-minilm-onnx/model.onnx (~330 MB)"]
fn minilm_adjudicate_from_tokio_via_spawn_blocking() {
    if !std::path::Path::new(MINILM_DIR).exists() { return; }

    // Verify that adjudicate() is safe to call from within tokio via spawn_blocking,     // this is the canonical async+sync integration pattern.
    let schema = name_schema();
    let r1 = Record::new(1).insert("naam", FieldValue::Text("bob".into()));
    let r2 = Record::new(2).insert("naam", FieldValue::Text("bob".into()));
    let store   = build_store(vec![r1, r2]);
    let spec    = MiniLmSpec::from_dir(MINILM_DIR);
    let backend = JudgeBackend::cpu();
    let judge   = Arc::new(
        DebertaJudge::new(&spec, &backend, store, schema, DebertaJudgeConfig::default())
            .expect("judge construction failed")
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let verdicts = rt.block_on(async {
        let judge = Arc::clone(&judge);
        tokio::task::spawn_blocking(move || {
            judge.adjudicate(&[scored_pair(1, 2)])
        })
        .await
        .expect("spawn_blocking panicked")
        .expect("adjudicate failed")
    });

    assert_eq!(verdicts.len(), 1);
}

// ── DeBERTa-base tests ────────────────────────────────────────────────────────

#[test]
#[ignore = "requires models/fp16_fused/nli-deberta-v3-base-onnx/model.onnx (~740 MB)"]
fn deberta_base_inference_runs_on_cpu() {
    if !std::path::Path::new(DEBERTA_BASE_DIR).exists() { return; }

    let schema = name_schema();
    let r1 = Record::new(1)
        .insert("naam",          FieldValue::Text("pieter van dijk".into()))
        .insert("geboortedatum", FieldValue::Text("1975-03-12".into()));
    let r2 = Record::new(2)
        .insert("naam",          FieldValue::Text("pieter van dijk".into()))
        .insert("geboortedatum", FieldValue::Text("1975-03-12".into()));

    let store   = build_store(vec![r1, r2]);
    let spec    = DebertaBaseSpec::from_dir(DEBERTA_BASE_DIR);
    let backend = JudgeBackend::cpu();
    let judge   = DebertaJudge::new(&spec, &backend, store, schema, DebertaJudgeConfig::default())
        .expect("deberta-base judge construction failed");

    let verdicts = judge.adjudicate(&[scored_pair(1, 2)]).expect("adjudicate failed");
    assert_eq!(verdicts.len(), 1);
}

#[test]
#[ignore = "requires models/fp16_fused/nli-minilm-onnx/model.onnx (~330 MB)"]
fn minilm_missing_record_returns_error() {
    if !std::path::Path::new(MINILM_DIR).exists() { return; }

    let schema  = name_schema();
    let store   = build_store(vec![]); // empty, records 1 & 2 do not exist
    let spec    = MiniLmSpec::from_dir(MINILM_DIR);
    let backend = JudgeBackend::cpu();
    let judge   = DebertaJudge::new(&spec, &backend, store, schema, DebertaJudgeConfig::default())
        .expect("judge construction failed");

    let result = judge.adjudicate(&[scored_pair(1, 2)]);
    assert!(result.is_err(), "missing record must return an error");
}
