/// Demonstrates `ClusterView::linked_pairs()`, iterating entity-level linkage
/// rows after a cross-source pipeline run.
///
/// `linked_pairs()` emits one `LinkedPair` per cross-source (source_a ≠ source_b)
/// member combination within each resolved entity.  This is the primary output
/// format for record linkage mode and mirrors splink's `predict()` pairs table.
///
/// Run with:  cargo run --example linked_pairs_output -p zer-pipeline
use tempfile::TempDir;
use zer_cluster::ZalEntityStore;
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_pipeline::{
    config::{LinkMode, PipelineConfig},
    label_source,
    pipeline::Pipeline,
    LinkedPair,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = TempDir::new()?;

    let schema = SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()?;

    let pipeline = Pipeline::builder()
        .schema(schema)
        .store(ZalEntityStore::open_in_memory()?)
        .config(PipelineConfig {
            registry_path: dir.path().join("linked_pairs.zsm"),
            link_mode: LinkMode::LinkAndDedupe,
            ..PipelineConfig::default()
        })
        .build()?;

    // ── Three persons, each appearing in two sources ──────────────────────────

    let brp = label_source(
        vec![
            make_record(1, "Jan", "de Vries", "1985-03-15"),
            make_record(2, "Maria", "Jansen", "1992-07-04"),
            make_record(3, "Ahmed", "El Amrani", "1978-11-20"),
        ],
        "brp",
    );

    let kvk = label_source(
        vec![
            make_record(101, "Jan", "de Vries", "1985-03-15"),
            make_record(102, "Maria", "Jansen", "1992-07-04"),
            make_record(103, "Ahmed", "El Amrani", "1978-11-20"),
        ],
        "kvk",
    );

    let all_records: Vec<Record> = [brp, kvk].concat();
    let report = pipeline.run_batch(all_records).await?;

    println!("Pipeline run complete.");
    println!("  total_records:      {}", report.total_records);
    println!("  candidate_pairs:    {}", report.candidate_pairs);
    println!("  cross_source_pairs: {}", report.cross_source_pairs);
    println!("  within_source_pairs:{}", report.within_source_pairs);

    // ── Iterate linked_pairs() ────────────────────────────────────────────────

    let view = pipeline.cluster_view();
    let pairs: Vec<LinkedPair> = view.linked_pairs();

    println!("\nLinked pairs ({}):", pairs.len());
    println!(
        "{:<10} {:<20} {:<8} {:<20} {:<8} {:<8}",
        "entity_id", "key_a", "src_a", "key_b", "src_b", "score"
    );
    println!("{}", "-".repeat(78));

    for lp in &pairs {
        println!(
            "{:<10} {:<20} {:<8} {:<20} {:<8} {:.4}",
            lp.entity_id,
            lp.record_key_a,
            lp.source_a.as_deref().unwrap_or("?"),
            lp.record_key_b,
            lp.source_b.as_deref().unwrap_or("?"),
            lp.score,
        );
    }

    // ── Verify output ─────────────────────────────────────────────────────────

    // All linked pairs must be cross-source
    for lp in &pairs {
        assert_ne!(
            lp.source_a, lp.source_b,
            "linked_pairs() must only emit cross-source pairs"
        );
    }
    println!("\nOK, all {} linked pairs are cross-source.", pairs.len());

    Ok(())
}

fn make_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen", FieldValue::Text(first.into()))
        .insert("achternaam", FieldValue::Text(last.into()))
        .insert("geboortedatum", FieldValue::Text(dob.into()))
}
