//! Show the NLI cross-encoder input format produced by `serialize_pair`.
//!
//! Prints the serialized text for several sample record pairs so you can
//! inspect what the judge model actually receives.
//!
//! Run with:
//!   cargo run -p zer-judge --example serialize_pair

use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_judge::serialize::serialize_pair;

fn main() {
    let schema = SchemaBuilder::new()
        .field("naam",          FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("postcode",      FieldKind::FreeText)
        .build()
        .expect("schema build failed");

    // ── Example 1: near-duplicate pair ───────────────────────────────────────
    let a = Record::new(1)
        .insert("naam",          FieldValue::Text("Jan Smits".into()))
        .insert("geboortedatum", FieldValue::Text("1981-04-02".into()))
        .insert("postcode",      FieldValue::Text("1234 AB".into()));

    let b = Record::new(2)
        .insert("naam",          FieldValue::Text("Jan Smyts".into()))   // typo
        .insert("geboortedatum", FieldValue::Text("1981-04-02".into()))
        .insert("postcode",      FieldValue::Text("1234 AB".into()));

    println!("── Example 1: near-duplicate (name typo) ────────────────────────");
    println!("{}", serialize_pair(&a, &b, &schema));
    println!();

    // ── Example 2: clearly different people ──────────────────────────────────
    let c = Record::new(3)
        .insert("naam",          FieldValue::Text("Anna de Vries".into()))
        .insert("geboortedatum", FieldValue::Text("1990-06-15".into()))
        .insert("postcode",      FieldValue::Text("5678 CD".into()));

    println!("── Example 2: different people ──────────────────────────────────");
    println!("{}", serialize_pair(&a, &c, &schema));
    println!();

    // ── Example 3: missing fields ─────────────────────────────────────────────
    let d = Record::new(4)
        .insert("naam", FieldValue::Text("Pieter Jansen".into()));
        // geboortedatum and postcode intentionally absent

    println!("── Example 3: record with missing fields (absent = empty VAL) ───");
    println!("{}", serialize_pair(&a, &d, &schema));
    println!();

    // ── Example 4: identical records ─────────────────────────────────────────
    println!("── Example 4: identical records ─────────────────────────────────");
    println!("{}", serialize_pair(&a, &a, &schema));
    println!();

    println!("Format: [CLS] <left record fields> [SEP] <right record fields> [SEP]");
    println!("Each field: COL:<name> VAL:<value>");
    println!("Fields are emitted in schema declaration order regardless of insertion order.");
}
