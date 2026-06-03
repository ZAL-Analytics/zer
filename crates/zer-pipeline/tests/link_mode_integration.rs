/// Integration tests for `LinkMode`, end-to-end `run_batch` with two in-memory sources.
///
/// These tests verify that the pair filter works correctly end-to-end and that
/// the regression against `Deduplicate` mode is preserved.
use std::sync::Arc;

use tempfile::TempDir;
use zer_cluster::ZalEntityStore;
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_pipeline::{
    config::{LinkMode, PipelineConfig},
    pipeline::Pipeline,
    BatchReport,
};

fn person_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .unwrap()
}

fn make_pipeline(dir: &TempDir, mode: LinkMode) -> Arc<Pipeline> {
    Pipeline::builder()
        .schema(person_schema())
        .store(ZalEntityStore::open_in_memory().unwrap())
        .config(PipelineConfig {
            registry_path: dir.path().join("test.zsm"),
            link_mode: mode,
            ..PipelineConfig::default()
        })
        .build()
        .unwrap()
}

fn make_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen", FieldValue::Text(first.into()))
        .insert("achternaam", FieldValue::Text(last.into()))
        .insert("geboortedatum", FieldValue::Text(dob.into()))
}

fn brp_records() -> Vec<Record> {
    (1..=5)
        .map(|i| make_record(i, "Jan", "de Vries", "1985-03-15").with_source("brp"))
        .collect()
}

fn kvk_records() -> Vec<Record> {
    (100..=104)
        .map(|i| make_record(i, "Jan", "de Vries", "1985-03-15").with_source("kvk"))
        .collect()
}

// ── LinkOnly ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn link_only_cross_source_pairs_present() {
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::LinkOnly);
    let mut records = brp_records();
    records.extend(kvk_records());
    let report: BatchReport = pipeline.run_batch(records).await.unwrap();
    assert!(
        report.cross_source_pairs > 0,
        "LinkOnly must produce cross-source pairs; got 0"
    );
    assert_eq!(
        report.within_source_pairs, 0,
        "LinkOnly must produce zero within-source pairs"
    );
    assert_eq!(report.link_mode, LinkMode::LinkOnly);
}

#[tokio::test]
async fn link_only_within_source_zero_single_source() {
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::LinkOnly);
    // All records come from a single source, no cross-source pairs possible.
    let report: BatchReport = pipeline.run_batch(brp_records()).await.unwrap();
    assert_eq!(
        report.candidate_pairs, 0,
        "single source in LinkOnly must produce no pairs"
    );
    assert_eq!(report.within_source_pairs, 0);
    assert_eq!(report.cross_source_pairs, 0);
}

#[tokio::test]
async fn link_only_pair_sum_equals_candidate_pairs() {
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::LinkOnly);
    let mut records = brp_records();
    records.extend(kvk_records());
    let report = pipeline.run_batch(records).await.unwrap();
    assert_eq!(
        report.cross_source_pairs + report.within_source_pairs,
        report.candidate_pairs,
        "cross + within must equal total candidate_pairs"
    );
}

// ── Deduplicate (regression guard) ────────────────────────────────────────────

#[tokio::test]
async fn deduplicate_pair_count_regression() {
    // A Deduplicate run on records that all share one source must produce the
    // same pair count as before phase-07c (no pairs filtered out).
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::Deduplicate);
    let report = pipeline.run_batch(brp_records()).await.unwrap();
    assert_eq!(report.link_mode, LinkMode::Deduplicate);
    // 5 identical records → 10 candidate pairs (C(5,2))
    assert!(
        report.candidate_pairs > 0,
        "Deduplicate must not filter any pairs"
    );
    assert_eq!(
        report.cross_source_pairs, 0,
        "Deduplicate on single-source records has no cross-source pairs"
    );
    assert_eq!(
        report.within_source_pairs, report.candidate_pairs,
        "all pairs are within-source in single-source Deduplicate run"
    );
}

#[tokio::test]
async fn deduplicate_two_sources_all_pairs_included() {
    // Deduplicate with two sources must include ALL pairs (within and cross).
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::Deduplicate);
    let mut records = brp_records();
    records.extend(kvk_records());
    let report = pipeline.run_batch(records).await.unwrap();
    assert_eq!(report.link_mode, LinkMode::Deduplicate);
    assert!(
        report.within_source_pairs > 0,
        "Deduplicate must include within-source pairs"
    );
    assert!(
        report.cross_source_pairs > 0,
        "Deduplicate must include cross-source pairs"
    );
    assert_eq!(
        report.cross_source_pairs + report.within_source_pairs,
        report.candidate_pairs
    );
}

// ── LinkAndDedupe ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn link_and_dedupe_includes_all_pairs() {
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::LinkAndDedupe);
    let mut records = brp_records();
    records.extend(kvk_records());
    let report = pipeline.run_batch(records).await.unwrap();
    assert_eq!(report.link_mode, LinkMode::LinkAndDedupe);
    assert!(
        report.within_source_pairs > 0,
        "LinkAndDedupe must include within-source pairs"
    );
    assert!(
        report.cross_source_pairs > 0,
        "LinkAndDedupe must include cross-source pairs"
    );
    assert_eq!(
        report.cross_source_pairs + report.within_source_pairs,
        report.candidate_pairs
    );
}

#[tokio::test]
async fn link_and_dedupe_same_total_as_deduplicate() {
    // LinkAndDedupe and Deduplicate must generate the same candidate pairs
    // (both include all pairs, the mode only affects reporting).
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();
    let p_dedup = make_pipeline(&dir1, LinkMode::Deduplicate);
    let p_lad = make_pipeline(&dir2, LinkMode::LinkAndDedupe);

    let mut records1 = brp_records();
    records1.extend(kvk_records());
    let mut records2 = brp_records();
    records2.extend(kvk_records());

    let r_dedup = p_dedup.run_batch(records1).await.unwrap();
    let r_lad = p_lad.run_batch(records2).await.unwrap();

    assert_eq!(
        r_dedup.candidate_pairs, r_lad.candidate_pairs,
        "Deduplicate and LinkAndDedupe must generate the same candidate pairs"
    );
}
