/// Demonstrates `LinkMode::LinkOnly`, finding matches across two named sources
/// while skipping all within-source pairs.
///
/// This is the canonical "record linkage" scenario: you have two curated
/// datasets (e.g. BRP and KvK) and want to find which records in dataset A
/// refer to the same real-world entity as records in dataset B, without
/// disturbing the internal integrity of either dataset.
///
/// Run with:  cargo run --example link_only -p zer-pipeline
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
            registry_path: dir.path().join("link_only.zsm"),
            link_mode: LinkMode::LinkOnly,
            ..PipelineConfig::default()
        })
        .build()?;

    // ── Two synthetic datasets ────────────────────────────────────────────────

    // Dataset A: BRP persons (source = "brp")
    let brp_raw = vec![
        make_record(1, "Jan", "de Vries", "1985-03-15"),
        make_record(2, "Maria", "Jansen", "1992-07-04"),
        make_record(3, "Ahmed", "El Amrani", "1978-11-20"),
        make_record(4, "Sophie", "van der Berg", "2000-01-08"),
    ];

    // Dataset B: KvK persons (source = "kvk")
    let kvk_raw = vec![
        make_record(101, "Jan", "de Vries", "1985-03-15"), // same as BRP 1
        make_record(102, "Maria", "Jansen", "1992-07-04"), // same as BRP 2
        make_record(103, "Peter", "Bakker", "1965-05-30"), // not in BRP
    ];

    let brp = label_source(brp_raw, "brp");
    let kvk = label_source(kvk_raw, "kvk");

    let all_records: Vec<Record> = [brp, kvk].concat();
    println!("total records: {}", all_records.len());

    // ── Run LinkOnly pipeline ─────────────────────────────────────────────────

    let report = pipeline.run_batch(all_records).await?;

    println!("link_mode:          {}", report.link_mode.as_str());
    println!("total_records:      {}", report.total_records);
    println!("candidate_pairs:    {}", report.candidate_pairs);
    println!("cross_source_pairs: {}", report.cross_source_pairs);
    println!("within_source_pairs:{}", report.within_source_pairs);
    println!("auto_matched:       {}", report.auto_matched);
    println!("borderline:         {}", report.borderline);
    println!("auto_rejected:      {}", report.auto_rejected);
    println!("elapsed_ms:         {}", report.elapsed_ms);

    assert_eq!(
        report.within_source_pairs, 0,
        "LinkOnly must produce zero within-source pairs"
    );
    println!("\nOK, only cross-source pairs were generated.");

    // ── Inspect linked pairs from the cluster view ────────────────────────────

    let view = pipeline.cluster_view();
    let pairs = view.linked_pairs();
    println!("\nLinked pairs ({}):", pairs.len());
    for lp in &pairs {
        println!(
            "  entity={} | key_a={} ({}) <-> key_b={} ({}) | score={:.3}",
            lp.entity_id,
            lp.record_key_a,
            lp.source_a.as_deref().unwrap_or("?"),
            lp.record_key_b,
            lp.source_b.as_deref().unwrap_or("?"),
            lp.score,
        );
    }

    Ok(())
}

fn make_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen", FieldValue::Text(first.into()))
        .insert("achternaam", FieldValue::Text(last.into()))
        .insert("geboortedatum", FieldValue::Text(dob.into()))
}
