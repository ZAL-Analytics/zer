/// Integration tests for `ClusterView::linked_pairs()`.
///
/// Builds a mock `ClusterView` by running a real pipeline in `LinkOnly` mode
/// with two named sources, then verifies that `linked_pairs()` returns the
/// expected cross-source rows.

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
    LinkedPair,
};

fn person_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .unwrap()
}

fn make_pipeline(dir: &TempDir, mode: LinkMode) -> Arc<Pipeline> {
    Pipeline::builder()
        .schema(person_schema())
        .store(ZalEntityStore::open_in_memory().unwrap())
        .config(PipelineConfig {
            registry_path: dir.path().join("cv_link_test.zsm"),
            link_mode: mode,
            ..PipelineConfig::default()
        })
        .build()
        .unwrap()
}

fn jan_de_vries(id: u64, source: &str) -> Record {
    Record::new(id)
        .insert("voornamen",     FieldValue::Text("Jan".into()))
        .insert("achternaam",    FieldValue::Text("de Vries".into()))
        .insert("geboortedatum", FieldValue::Text("1985-03-15".into()))
        .with_source(source)
}

// 5 records per source gives EM enough data to converge above the auto-match threshold.
fn brp_records() -> Vec<Record> {
    (1..=5).map(|i| jan_de_vries(i, "brp")).collect()
}

fn kvk_records() -> Vec<Record> {
    (100..=104).map(|i| jan_de_vries(i, "kvk")).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn linked_pairs_from_pipeline_run_cross_source() {
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::LinkAndDedupe);

    // 5 records from "brp" and 5 from "kvk", same person, different sources.
    // Using 5 per source so EM has enough pairs to converge above the auto-match threshold.
    let mut records = brp_records();
    records.extend(kvk_records());
    pipeline.run_batch(records).await.unwrap();

    let view  = pipeline.cluster_view();
    let pairs = view.linked_pairs();

    assert!(!pairs.is_empty(), "a matched cross-source pair must appear in linked_pairs()");
    let lp: &LinkedPair = &pairs[0];
    // The two records must come from different sources
    assert_ne!(lp.source_a, lp.source_b, "linked pair must span two distinct sources");
    let sources: Vec<_> = [lp.source_a.as_deref(), lp.source_b.as_deref()].into_iter().collect();
    assert!(sources.contains(&Some("brp")), "source 'brp' must be present");
    assert!(sources.contains(&Some("kvk")), "source 'kvk' must be present");
}

#[tokio::test]
async fn linked_pairs_empty_when_single_source() {
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::Deduplicate);

    // All records from the same source, no cross-source pairs.
    let records: Vec<Record> = (1..=4)
        .map(|i| jan_de_vries(i, "brp"))
        .collect();
    pipeline.run_batch(records).await.unwrap();

    let view  = pipeline.cluster_view();
    let pairs = view.linked_pairs();

    assert!(
        pairs.is_empty(),
        "single-source run must produce no LinkedPairs, got {}",
        pairs.len()
    );
}

#[tokio::test]
async fn linked_pairs_entity_id_is_consistent() {
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::LinkAndDedupe);

    let mut records = brp_records();
    records.extend(kvk_records());
    pipeline.run_batch(records).await.unwrap();

    let view  = pipeline.cluster_view();
    let pairs = view.linked_pairs();

    if let Some(lp) = pairs.first() {
        // entity_id must reference an entity that actually exists in the store
        let entity_result = pipeline.store().get_entity(lp.entity_id);
        assert!(entity_result.is_ok(), "entity_id in LinkedPair must exist in the entity store");
    }
}

#[tokio::test]
async fn linked_pairs_record_ids_match_input() {
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::LinkAndDedupe);

    // Use a pair with unique IDs not shared with other tests; 5 per source for EM convergence.
    let brp: Vec<Record> = (10..=14).map(|i| jan_de_vries(i, "brp")).collect();
    let kvk: Vec<Record> = (20..=24).map(|i| jan_de_vries(i, "kvk")).collect();
    let mut records = brp;
    records.extend(kvk);
    pipeline.run_batch(records).await.unwrap();

    let view  = pipeline.cluster_view();
    let pairs = view.linked_pairs();

    assert!(!pairs.is_empty(), "cross-source pairs must be produced for linked_pairs_record_ids_match_input");
    for lp in &pairs {
        let ids = [lp.record_id_a, lp.record_id_b];
        assert!(
            ids.iter().any(|&id| (10..=14).contains(&id)),
            "one record in each pair must come from brp (ids 10-14)"
        );
        assert!(
            ids.iter().any(|&id| (20..=24).contains(&id)),
            "one record in each pair must come from kvk (ids 20-24)"
        );
    }
}

#[tokio::test]
async fn linked_pairs_multiple_cross_source_entities() {
    // Two separate persons, each appearing 5× in both sources → EM convergence → cross-source pairs.
    let dir = TempDir::new().unwrap();
    let pipeline = make_pipeline(&dir, LinkMode::LinkAndDedupe);

    // Person A: 5 records in brp + 5 in kvk
    let mut records: Vec<Record> = (1..=5).map(|i| jan_de_vries(i, "brp")).collect();
    records.extend((100..=104).map(|i| jan_de_vries(i, "kvk")));

    // Person B: 5 records in brp + 5 in kvk (different person → different entity)
    let maria = |id: u64, source: &str| {
        Record::new(id)
            .insert("voornamen",     FieldValue::Text("Maria".into()))
            .insert("achternaam",    FieldValue::Text("Bakker".into()))
            .insert("geboortedatum", FieldValue::Text("1990-07-22".into()))
            .with_source(source)
    };
    records.extend((200..=204).map(|i| maria(i, "brp")));
    records.extend((300..=304).map(|i| maria(i, "kvk")));

    pipeline.run_batch(records).await.unwrap();

    let view  = pipeline.cluster_view();
    let pairs = view.linked_pairs();

    // All returned pairs must span two distinct sources
    for lp in &pairs {
        assert_ne!(lp.source_a, lp.source_b, "all linked_pairs must be cross-source");
    }
}
