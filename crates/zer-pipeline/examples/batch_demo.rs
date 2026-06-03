/// Demonstrates `Pipeline::run_batch` with a set of synthetic records.
///
/// Run with:  cargo run --example batch_demo -p zer-pipeline
use tempfile::TempDir;
use zer_cluster::ZalEntityStore;
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_pipeline::{config::PipelineConfig, pipeline::Pipeline};

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
            registry_path: dir.path().join("demo.zsm"),
            ..PipelineConfig::default()
        })
        .build()?;

    // Mix of true duplicates and unique individuals
    let records = vec![
        // Cluster A, same person, three records
        make_record(1, "Jan", "de Vries", "1985-03-15"),
        make_record(2, "Jan", "de Vries", "1985-03-15"),
        make_record(3, "Jan", "de Vries", "1985-03-15"),
        // Cluster B, another person, two records
        make_record(4, "Maria", "Jansen", "1992-07-04"),
        make_record(5, "Maria", "Jansen", "1992-07-04"),
        // Unique individuals
        make_record(6, "Carlos", "Ramirez", "1978-11-01"),
        make_record(7, "Alice", "Smith", "1990-01-01"),
    ];

    println!("Running batch with {} records…", records.len());
    let report = pipeline.run_batch(records).await?;

    println!("\n── Batch Report ─────────────────────────────────────");
    println!("  total_records:    {}", report.total_records);
    println!("  candidate_pairs:  {}", report.candidate_pairs);
    println!("  auto_matched:     {}", report.auto_matched);
    println!("  borderline:       {}", report.borderline);
    println!("  auto_rejected:    {}", report.auto_rejected);
    println!("  entities_created: {}", report.entities_created);
    println!("  entities_updated: {}", report.entities_updated);
    println!("  em_iterations:    {}", report.em_iterations);
    println!("  startup_mode:     {:?}", report.startup_mode);

    // Run a second batch to show warm-load behaviour
    println!("\nRunning second batch (warm-load expected)…");
    let pipeline2 = Pipeline::builder()
        .schema(
            SchemaBuilder::new()
                .field("voornamen", FieldKind::Name)
                .field("achternaam", FieldKind::Name)
                .field("geboortedatum", FieldKind::Date)
                .build()?,
        )
        .store(ZalEntityStore::open_in_memory()?)
        .config(PipelineConfig {
            registry_path: dir.path().join("demo.zsm"),
            ..PipelineConfig::default()
        })
        .build()?;
    let r2 = pipeline2
        .run_batch(vec![
            make_record(10, "Test", "Person", "2000-01-01"),
            make_record(11, "Test", "Person", "2000-01-01"),
        ])
        .await?;
    println!("  startup_mode: {:?}", r2.startup_mode);

    Ok(())
}

fn make_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen", FieldValue::Text(first.into()))
        .insert("achternaam", FieldValue::Text(last.into()))
        .insert("geboortedatum", FieldValue::Text(dob.into()))
}
