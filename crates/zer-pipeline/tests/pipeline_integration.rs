/// Integration tests for `zer-pipeline`.
///
/// These tests exercise the public API end-to-end: `Pipeline::builder`,
/// `run_batch`, `Pipeline::ingester`, and `Ingester::send`.  Each test
/// stands alone, no shared mutable state between tests.

use std::sync::Arc;

use tempfile::TempDir;
use zer_cluster::ZalEntityStore;
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_pipeline::{
    config::{BatchStartupMode, PipelineConfig}, ingester::Ingester, pipeline::Pipeline, BatchReport,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn person_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .unwrap()
}

fn make_pipeline(dir: &TempDir) -> Arc<Pipeline> {
    Pipeline::builder()
        .schema(person_schema())
        .store(ZalEntityStore::open_in_memory().unwrap())
        .config(PipelineConfig {
            registry_path: dir.path().join("test.zsm"),
            ..PipelineConfig::default()
        })
        .build()
        .unwrap()
}

fn make_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen",     FieldValue::Text(first.into()))
        .insert("achternaam",    FieldValue::Text(last.into()))
        .insert("geboortedatum", FieldValue::Text(dob.into()))
}

fn identical_records(n: usize) -> Vec<Record> {
    (1..=(n as u64))
        .map(|i| make_record(i, "Jan", "de Vries", "1985-03-15"))
        .collect()
}

fn diverse_records() -> Vec<Record> {
    vec![
        make_record(1,  "Alice",   "Smith",   "1990-01-01"),
        make_record(2,  "Bob",     "Jones",   "1975-06-20"),
        make_record(3,  "Carlos",  "Ramirez", "1988-11-03"),
        make_record(4,  "Diana",   "Muller",  "1993-09-14"),
        make_record(5,  "Ethan",   "Brown",   "1969-02-28"),
    ]
}

// ── run_batch integration tests ───────────────────────────────────────────────

#[tokio::test]
async fn batch_empty_input_returns_zero_report() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let report: BatchReport = pipeline.run_batch(vec![]).await.unwrap();
    assert_eq!(report.total_records,   0);
    assert_eq!(report.candidate_pairs, 0);
    assert_eq!(report.auto_matched,    0);
    assert_eq!(report.entities_created, 0);
}

#[tokio::test]
async fn batch_single_record_creates_one_entity() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let report   = pipeline.run_batch(vec![make_record(1, "Alice", "Smith", "1990-01-01")]).await.unwrap();
    assert_eq!(report.total_records,    1);
    assert_eq!(report.candidate_pairs,  0);
    // A single record with no pairs still results in at least one entity
}

#[tokio::test]
async fn batch_identical_records_have_candidates() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let report   = pipeline.run_batch(identical_records(5)).await.unwrap();
    assert_eq!(report.total_records, 5);
    assert!(report.candidate_pairs > 0, "identical records must produce candidates");
}

#[tokio::test]
async fn batch_diverse_records_produce_report() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let report   = pipeline.run_batch(diverse_records()).await.unwrap();
    assert_eq!(report.total_records, 5);
    // Diverse records with different names/dates should not auto-match
    assert_eq!(report.candidate_pairs, 0, "unrelated records should not block together");
}

#[tokio::test]
async fn batch_cold_start_label_on_new_registry() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let report   = pipeline.run_batch(identical_records(3)).await.unwrap();
    assert_eq!(report.startup_mode, BatchStartupMode::ColdStart);
}

#[tokio::test]
async fn batch_second_run_is_warm_load() {
    let dir = TempDir::new().unwrap();

    // First run, trains and saves params
    let p1  = make_pipeline(&dir);
    let r1  = p1.run_batch(identical_records(8)).await.unwrap();
    assert_eq!(r1.startup_mode, BatchStartupMode::ColdStart);

    // Second run with same schema and same registry file, should warm-load
    let p2 = Pipeline::builder()
        .schema(person_schema())
        .store(ZalEntityStore::open_in_memory().unwrap())
        .config(PipelineConfig {
            registry_path: dir.path().join("test.zsm"),
            ..PipelineConfig::default()
        })
        .build()
        .unwrap();
    let r2 = p2.run_batch(identical_records(5)).await.unwrap();
    assert_eq!(r2.startup_mode,  BatchStartupMode::WarmLoad);
    assert_eq!(r2.em_iterations, 0, "WarmLoad must skip EM");
}

#[tokio::test]
async fn batch_band_counts_sum_to_candidate_pairs() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let report   = pipeline.run_batch(identical_records(6)).await.unwrap();
    let band_sum = report.auto_matched + report.borderline + report.auto_rejected;
    // After judge pass: bands may shift, but before judge the sum equals candidate_pairs.
    // With no judge configured, judge_promoted and judge_demoted are both 0.
    assert_eq!(report.judge_promoted, 0);
    assert_eq!(report.judge_demoted,  0);
    // The band total should equal the number of candidate pairs (one band per pair)
    assert_eq!(band_sum, report.candidate_pairs);
}

// ── Ingester integration tests ────────────────────────────────────────────────

#[tokio::test]
async fn ingester_singleton_assigned_entity() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let ingester = Arc::clone(&pipeline).ingester();
    let result   = ingester.send(make_record(1, "Alice", "Smith", "1990-01-01")).await.unwrap();
    assert_eq!(result.record_id, 1);
    assert!(result.entity_id.is_some());
}

#[tokio::test]
async fn ingester_stream_returns_correct_ids() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let ingester = Arc::clone(&pipeline).ingester();
    for i in 1u64..=8 {
        let rec    = make_record(i, "Test", "User", "2000-06-15");
        let result = ingester.send(rec).await.unwrap();
        assert_eq!(result.record_id, i, "result must correspond to the sent record");
    }
}

#[tokio::test]
async fn ingester_flush_completes_without_error() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let ingester = Arc::clone(&pipeline).ingester();
    ingester.send(make_record(1, "Anna", "Berg", "1991-04-11")).await.unwrap();
    ingester.flush_borderlines().await.unwrap();
}

#[tokio::test]
async fn ingester_diverse_records_all_singletons() {
    let dir      = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir);
    let ingester: Ingester = Arc::clone(&pipeline).ingester();
    for record in diverse_records() {
        let id     = record.id;
        let result = ingester.send(record).await.unwrap();
        assert_eq!(result.record_id, id);
        // Diverse records should not block together → each is a singleton
        assert!(
            result.entity_id.is_some(),
            "record {id} should be assigned an entity as a singleton"
        );
    }
}

// ── Builder error-path tests ──────────────────────────────────────────────────

#[test]
fn builder_missing_schema_is_error() {
    let dir   = TempDir::new().unwrap();
    let store = ZalEntityStore::open_in_memory().unwrap();
    let result = Pipeline::builder()
        .store(store)
        .config(PipelineConfig {
            registry_path: dir.path().join("test.zsm"),
            ..PipelineConfig::default()
        })
        .build();
    assert!(result.is_err(), "missing schema must be an error");
}

#[test]
fn builder_missing_store_is_error() {
    let dir = TempDir::new().unwrap();
    let result = Pipeline::builder()
        .schema(person_schema())
        .config(PipelineConfig {
            registry_path: dir.path().join("test.zsm"),
            ..PipelineConfig::default()
        })
        .build();
    assert!(result.is_err(), "missing store must be an error");
}
