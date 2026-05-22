//! Demonstrates the secondary `DateFragmentKey(YearMonth)` blocking key that
//! `BlockerFactory::from_schema()` adds alongside `PhoneticNameDobKey` when
//! both Name and Date fields are present.
//!
//! **Why it matters**: `PhoneticNameDobKey` groups records by surname phonetic
//! code + birth year.  It misses pairs whose surnames differ phonetically but
//! who share the same birth year-month, a common scenario in transcription
//! errors (e.g., "Jansen" vs "Janssen").  The secondary key closes that gap.
//!
//! The example indexes five records and shows which pairs each key surfaces.
//!
//! Run:
//!   cargo run --example secondary_blocking_key -p zer-blocking

use zer_blocking::{BlockerFactory, InvertedIndex};
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
    traits::Blocker,
};

fn main() {
    let schema = SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .expect("schema build failed");

    // Five records:
    //  r1  "Jansen"    1985-06-15 , baseline
    //  r2  "Jansen"    1985-06-22 , same surname+year-month, different day
    //  r3  "Janssen"   1985-06-03 , different phonetic code, same year-month → secondary key
    //  r4  "Pietersen" 1985-06-10 , completely different surname, same year-month → secondary key
    //  r5  "Jansen"    1990-12-01 , same surname, different year-month → NOT a candidate
    let records = vec![
        make_record(1, "Jan",      "Jansen",    "1985-06-15"),
        make_record(2, "Maria",    "Jansen",    "1985-06-22"),
        make_record(3, "Pieter",   "Janssen",   "1985-06-03"),
        make_record(4, "Annelies", "Pietersen", "1985-06-10"),
        make_record(5, "Kees",     "Jansen",    "1990-12-01"),
    ];

    let blocker = BlockerFactory::from_schema(&schema);

    // Index all records.
    let mut idx = InvertedIndex::new();
    for r in &records {
        blocker.index_record(r, &schema, &mut idx);
    }

    println!("Records:");
    println!(
        "  {:<4}  {:<10}  {:<12}  {}",
        "id", "voornamen", "achternaam", "geboortedatum"
    );
    println!("  {}", "─".repeat(50));
    for r in &records {
        println!(
            "  {:<4}  {:<10}  {:<12}  {}",
            r.id,
            field_str(r, "voornamen"),
            field_str(r, "achternaam"),
            field_str(r, "geboortedatum"),
        );
    }

    println!();
    println!("Blocking keys for record 1 (Jan Jansen, 1985-06-15):");
    for key in blocker.blocking_keys(&records[0], &schema) {
        println!("  {key}");
    }

    println!();
    println!("Candidates for record 1 (via all keys):");
    let cands = blocker.candidates(&records[0], &schema, &idx);
    let mut sorted: Vec<u64> = cands.into_iter().collect();
    sorted.sort();
    for id in &sorted {
        let r = records.iter().find(|r| r.id == *id).unwrap();
        let reason = candidate_reason(r);
        println!(
            "  id={id}  {} {} {}  ← {reason}",
            field_str(r, "voornamen"),
            field_str(r, "achternaam"),
            field_str(r, "geboortedatum"),
        );
    }

    println!();
    // Verify the key findings.
    assert!(sorted.contains(&2), "r2 same surname+year-month must be a candidate");
    assert!(sorted.contains(&3), "r3 Janssen/1985-06 must be a candidate via secondary key");
    assert!(sorted.contains(&4), "r4 Pietersen/1985-06 must be a candidate via secondary key");
    assert!(!sorted.contains(&5), "r5 different year-month must NOT be a candidate");

    println!("All assertions passed.");
    println!();
    println!("Observation:");
    println!("  r3 (Janssen) and r4 (Pietersen) appear only because of the secondary");
    println!("  DateFragmentKey(YearMonth).  Without it they would be missed entirely.");
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

fn candidate_reason(r: &Record) -> &'static str {
    match r.id {
        2 => "PhoneticNameDobKey (same surname phonetic + birth year)",
        3 => "DateFragmentKey(YearMonth), phonetic code differs (Jansen≠Janssen)",
        4 => "DateFragmentKey(YearMonth), phonetic code differs (Jansen≠Pietersen)",
        _ => "?",
    }
}
