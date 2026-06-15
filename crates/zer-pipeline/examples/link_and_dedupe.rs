/// Demonstrates `LinkMode::LinkAndDedupe`, simultaneously deduplicating within
/// each source AND finding cross-source matches.
///
/// This mode generates all candidate pairs (within-source and cross-source)
/// and reports both counts in `BatchReport`.  Use it when your datasets may
/// contain internal duplicates AND you also need cross-source linkage.
///
/// Run with:  cargo run --example link_and_dedupe -p zer-pipeline
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
            registry_path: dir.path().join("link_and_dedupe.zsm"),
            link_mode: LinkMode::LinkAndDedupe,
            ..PipelineConfig::default()
        })
        .build()?;

    // ── Two synthetic datasets ────────────────────────────────────────────────

    // Dataset A: BRP, contains an internal duplicate (records 1 and 2)
    let brp_raw = vec![
        make_record(1, "Jan", "de Vries", "1985-03-15"), // duplicate within BRP
        make_record(2, "Jan", "de Vries", "1985-03-15"), // duplicate within BRP
        make_record(3, "Maria", "Jansen", "1992-07-04"),
    ];

    // Dataset B: KvK, also has Jan de Vries (cross-source match)
    let kvk_raw = vec![
        make_record(101, "Jan", "de Vries", "1985-03-15"), // matches BRP 1 & 2
        make_record(102, "Peter", "Bakker", "1965-05-30"),
    ];

    let brp = label_source(brp_raw, "brp");
    let kvk = label_source(kvk_raw, "kvk");

    let all_records: Vec<Record> = [brp, kvk].concat();
    println!("total records: {}", all_records.len());

    // ── Run LinkAndDedupe pipeline ────────────────────────────────────────────

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

    assert!(
        report.within_source_pairs > 0,
        "LinkAndDedupe must include within-source pairs (BRP has a duplicate)"
    );
    assert!(
        report.cross_source_pairs > 0,
        "LinkAndDedupe must include cross-source pairs (BRP/KvK overlap)"
    );
    assert_eq!(
        report.cross_source_pairs + report.within_source_pairs,
        report.candidate_pairs,
        "cross + within must equal total candidate_pairs"
    );
    println!("\nOK, both within-source and cross-source pairs were generated.");

    // ── Inspect linked pairs from the cluster view ────────────────────────────

    let view = pipeline.cluster_view();
    let pairs = view.linked_pairs();
    println!("\nCross-source linked pairs ({}):", pairs.len());
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

    // Show full cluster view (including within-source dedupe clusters)
    println!("\nAll resolved clusters:");
    for (entity, records) in &view {
        let sources: Vec<_> = records.iter().filter_map(|r| r.source.as_deref()).collect();
        println!(
            "  entity={} | {} member(s) | sources: {:?}",
            entity.id,
            records.len(),
            sources
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
