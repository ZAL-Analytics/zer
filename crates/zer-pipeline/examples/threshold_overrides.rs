//! Demonstrates `PipelineConfig::upper_threshold` and `lower_threshold`, //! optional overrides that pin the Fellegi-Sunter classification thresholds
//! instead of relying solely on the EM estimates.
//!
//! Two pipelines run over the same records:
//!   - **Default**, thresholds determined entirely by EM.
//!   - **Tightened**, upper=0.95 / lower=0.05 to force a narrower auto-match
//!     band and push more pairs into the borderline category.
//!
//! Run:
//!   cargo run --example threshold_overrides -p zer-pipeline

use std::sync::Arc;

use tempfile::TempDir;
use zer_cluster::ZalEntityStore;
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_pipeline::{batch::BatchReport, config::PipelineConfig, pipeline::Pipeline};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()?;

    let records = synthetic_records(50);
    println!("records: {}", records.len());
    println!();

    // ── Default thresholds (EM-estimated) ────────────────────────────────────
    let default_report =
        run_with_config(records.clone(), &schema, PipelineConfig::default()).await?;

    // ── Tightened thresholds ──────────────────────────────────────────────────
    let tightened_report = run_with_config(
        records.clone(),
        &schema,
        PipelineConfig {
            upper_threshold: Some(0.95),
            lower_threshold: Some(0.05),
            ..PipelineConfig::default()
        },
    )
    .await?;

    // ── Wide thresholds (for aggressive matching) ─────────────────────────────
    let wide_report = run_with_config(
        records.clone(),
        &schema,
        PipelineConfig {
            upper_threshold: Some(0.70),
            lower_threshold: Some(0.30),
            ..PipelineConfig::default()
        },
    )
    .await?;

    println!(
        "{:<20}  {:>10}  {:>10}  {:>10}  {:>10}",
        "config", "matched", "borderline", "rejected", "elapsed_ms"
    );
    println!("{}", "─".repeat(70));
    for (label, r) in [
        ("default (EM)", &default_report),
        ("tightened (0.95/0.05)", &tightened_report),
        ("wide (0.70/0.30)", &wide_report),
    ] {
        println!(
            "{:<20}  {:>10}  {:>10}  {:>10}  {:>10}",
            label, r.auto_matched, r.borderline, r.auto_rejected, r.elapsed_ms,
        );
    }

    println!();
    println!("Note: tightened thresholds push pairs into 'borderline' (judge territory).");
    println!("      wide thresholds auto-match more aggressively.");

    Ok(())
}

async fn run_with_config(
    records: Vec<Record>,
    schema: &zer_core::schema::Schema,
    config: PipelineConfig,
) -> Result<BatchReport, Box<dyn std::error::Error>> {
    let dir = TempDir::new()?;
    let pipeline = Arc::new(
        Pipeline::builder()
            .schema(schema.clone())
            .store(ZalEntityStore::open_in_memory()?)
            .config(PipelineConfig {
                registry_path: dir.path().join("demo.zsm"),
                ..config
            })
            .build()?,
    );
    Ok(pipeline.run_batch(records).await?)
}

fn synthetic_records(n: usize) -> Vec<Record> {
    // Pattern: 3 canonical records followed by 1 near-duplicate, cycling through
    // 5 persons.  Near-duplicates share the same name and surname but have a
    // different birth month (same year → same name-based blocking bucket; date
    // similarity is Close, not Exact → FS score < 1.0).
    //
    // With a tightened upper_threshold=0.95 these pairs score below 0.95 and
    // land in the borderline band rather than being auto-matched.
    let canonical: &[(&str, &str, &str)] = &[
        ("Jan", "Jansen", "1980-01-15"),
        ("Maria", "de Vries", "1985-06-15"),
        ("Pieter", "Bakker", "1990-03-22"),
        ("Anna", "Smit", "1975-11-08"),
        ("Kees", "Visser", "1992-09-30"),
    ];
    let near_dupes: &[(&str, &str, &str)] = &[
        ("Jan", "Jansen", "1980-08-03"),
        ("Maria", "de Vries", "1985-02-20"),
        ("Pieter", "Bakker", "1990-10-07"),
        ("Anna", "Smit", "1975-04-14"),
        ("Kees", "Visser", "1992-03-19"),
    ];

    let nc = canonical.len();
    (0..n)
        .map(|i| {
            let group = i / 4;
            let idx = group % nc;
            let (first, last, dob) = if i % 4 == 3 {
                near_dupes[idx]
            } else {
                canonical[idx]
            };
            Record::new(i as u64 + 1)
                .insert("voornamen", FieldValue::Text(first.into()))
                .insert("achternaam", FieldValue::Text(last.into()))
                .insert("geboortedatum", FieldValue::Text(dob.into()))
        })
        .collect()
}
