/// Demonstrates `Pipeline::ingester` for streaming record intake.
///
/// Records arrive one at a time through `Ingester::send`.  Per-record
/// resolution results (entity assignment, match band) are printed immediately.
///
/// Run with:  cargo run --example streaming_demo -p zer-pipeline
use std::sync::Arc;

use tempfile::TempDir;
use zer_cluster::ZalEntityStore;
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
    scoring::MatchBand,
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
            registry_path: dir.path().join("streaming.zsm"),
            ..PipelineConfig::default()
        })
        .build()?;

    let ingester = Arc::clone(&pipeline).ingester();

    println!("Streaming records into the pipeline…\n");

    // Send a mix of duplicates and unique individuals
    let stream: Vec<Record> = vec![
        make_record(1, "Jan", "de Vries", "1985-03-15"), // first occurrence
        make_record(2, "Alice", "Smith", "1990-01-01"),  // unique
        make_record(3, "Jan", "de Vries", "1985-03-15"), // dup of #1
        make_record(4, "Bob", "Brown", "1978-07-12"),    // unique
        make_record(5, "Jan", "de Vries", "1985-03-15"), // dup of #1 & #3
        make_record(6, "Alice", "Smith", "1990-01-01"),  // dup of #2
        make_record(7, "Carlos", "Ramirez", "1969-04-22"), // unique
        make_record(8, "Maria", "Jansen", "1992-07-04"), // unique
        make_record(9, "Maria", "Jansen", "1992-07-04"), // dup of #8
        make_record(10, "Diana", "Muller", "2003-11-19"), // unique
    ];

    let mut auto_matched = 0usize;
    let mut borderline = 0usize;
    let mut auto_rejected = 0usize;

    for record in stream {
        let id = record.id;
        let result = ingester.send(record).await?;

        let band_str = match result.band {
            MatchBand::AutoMatch => {
                auto_matched += 1;
                "AutoMatch "
            }
            MatchBand::Borderline => {
                borderline += 1;
                "Borderline"
            }
            MatchBand::AutoReject => {
                auto_rejected += 1;
                "AutoReject"
            }
        };
        let entity_str = result
            .entity_id
            .map(|e| format!("entity={e}"))
            .unwrap_or_else(|| "pending".into());

        println!("  record {:>2}  band={}  {}", id, band_str, entity_str);
    }

    ingester.flush_borderlines().await?;

    println!("\n── Stream Summary ───────────────────────────────────");
    println!("  auto_matched:   {auto_matched}");
    println!("  borderline:     {borderline}");
    println!("  auto_rejected:  {auto_rejected}");

    Ok(())
}

fn make_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen", FieldValue::Text(first.into()))
        .insert("achternaam", FieldValue::Text(last.into()))
        .insert("geboortedatum", FieldValue::Text(dob.into()))
}
