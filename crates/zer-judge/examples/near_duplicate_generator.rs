//! Demonstrates `NearDuplicateGenerator`, schema-driven production of near-duplicate
//! record pairs that land in the Fellegi-Sunter borderline band.
//!
//! The generator is useful for populating test pipelines and integration tests with
//! realistic borderline candidates so that the judge is always exercised, even on small
//! datasets.
//!
//! Run:
//!   cargo run --example near_duplicate_generator -p zer-judge

use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_judge::NearDuplicateGenerator;

fn main() {
    let schema = SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .expect("schema build failed");

    // Real BRP-style records as seed data.
    let source = vec![
        make_record(1, "Maria",    "Jansen",   "1985-03-15"),
        make_record(2, "Pieter",   "de Vries", "1990-07-22"),
        make_record(3, "Annelies", "Bakker",   "1978-11-05"),
    ];

    let gen = NearDuplicateGenerator { pair_count: 4, id_offset: 9_000_000 };
    let synthetics = gen.generate(&source, &schema);

    println!("source records  : {}", source.len());
    println!("pair_count      : {}", gen.pair_count);
    println!("synthetic records generated: {} (2  times  pair_count)", synthetics.len());
    println!();

    println!(
        "{:<12}  {:<10}  {:<14}  {:<12}  {}",
        "id", "voornamen", "achternaam", "geboortedatum", "note"
    );
    println!("{}", "─".repeat(76));

    for (i, r) in synthetics.iter().enumerate() {
        let first = field_str(r, "voornamen");
        let last  = field_str(r, "achternaam");
        let dob   = field_str(r, "geboortedatum");
        let note  = if i % 2 == 0 { "verbatim copy" } else { "perturbed copy" };
        println!("{:<12}  {:<10}  {:<14}  {:<12}  {}", r.id, first, last, dob, note);
    }

    println!();
    println!("Perturbation rules:");
    println!("  Name fields , last character stripped (preserves phonetic blocking key)");
    println!("  Date fields , year kept; month and day replaced with deterministic alternatives");
    println!("  Other fields, copied verbatim");
}

fn make_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen",     FieldValue::Text(first.into()))
        .insert("achternaam",    FieldValue::Text(last.into()))
        .insert("geboortedatum", FieldValue::Text(dob.into()))
}

fn field_str<'a>(r: &'a Record, field: &str) -> &'a str {
    match r.get(field) {
        Some(FieldValue::Text(s)) => s.as_str(),
        _ => "<null>",
    }
}
