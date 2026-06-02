/// Example: automatic schema inference with `SchemaInferrer`.
///
/// Demonstrates:
/// 1. Loading a sample of records from the BRP CSV
/// 2. Inferring a `Schema` entirely from column names and value patterns
/// 3. Printing the inferred field kinds for a quick sanity check
/// 4. Applying field-level overrides for columns the heuristics would misclassify
/// 5. Doing the same for the SIM subscriber dataset
use zer_core::{record::{FieldValue, Record}, schema::FieldKind};
use zer_schema::infer::SchemaInferrer;

fn brp_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "examples/brp_q1/brp_persons.csv")
}
fn sim_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "examples/sim/sim_subscribers.csv")
}

fn load_csv_sample(path: impl AsRef<std::path::Path>, limit: usize) -> Vec<Record> {
    let path = path.as_ref();
    let mut rdr = csv::Reader::from_path(path)
        .unwrap_or_else(|_| panic!("CSV not found: {}", path.display()));
    let headers = rdr.headers().unwrap().clone();
    let mut records = Vec::new();
    let mut id: u64 = 1;

    for result in rdr.records().take(limit) {
        let row = result.unwrap();
        let mut r = Record::new(id);
        for (i, header) in headers.iter().enumerate() {
            let v = row.get(i).unwrap_or("").trim();
            r = r.insert(
                header,
                if v.is_empty() { FieldValue::Null } else { FieldValue::Text(v.into()) },
            );
        }
        records.push(r);
        id += 1;
    }
    records
}

fn main() {
    println!("=== zer-schema: automatic schema inference example ===\n");

    // ── BRP population register ───────────────────────────────────────────────

    println!("── BRP Population Register (Q1 sample, 200 records) ──");
    let brp_records = load_csv_sample(brp_csv(), 200);

    let brp_schema = SchemaInferrer::new()
        .infer(&brp_records)
        .expect("BRP records must produce a valid schema");

    println!("  Inferred {} fields:", brp_schema.len());
    for field in &brp_schema.fields {
        println!("    {:<20} {:?}", field.name, field.kind);
    }

    // Verify a few critical fields
    let kind_of = |n: &str| brp_schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);
    assert_eq!(kind_of("voornamen"),     Some(FieldKind::Name),        "voornamen → Name");
    assert_eq!(kind_of("achternaam"),    Some(FieldKind::Name),        "achternaam → Name");
    assert_eq!(kind_of("geboortedatum"), Some(FieldKind::Date),        "geboortedatum → Date");
    assert_eq!(kind_of("bsn"),           Some(FieldKind::Id),          "bsn → Id");
    println!("  Name / Date / Id fields correctly inferred. ✓\n");

    // ── BRP with overrides ────────────────────────────────────────────────────

    println!("── BRP with explicit overrides ──");
    let brp_override_schema = SchemaInferrer::new()
        // Classify the civil status field as Categorical explicitly.
        // (It would likely be inferred as FreeText due to high cardinality strings.)
        .override_field("verblijfstitel", FieldKind::Categorical)
        .infer(&brp_records)
        .expect("override infer must succeed");

    let verblijf = brp_override_schema.fields.iter().find(|f| f.name == "verblijfstitel");
    if let Some(f) = verblijf {
        println!("  verblijfstitel overridden to {:?}. ✓", f.kind);
        assert_eq!(f.kind, FieldKind::Categorical);
    } else {
        println!("  (verblijfstitel not present in this sample, override silently ignored)");
    }
    println!();

    // ── SIM subscriber dataset ────────────────────────────────────────────────

    println!("── SIM Subscriber Snapshot (snap1 sample, 200 records) ──");
    let sim_records = load_csv_sample(sim_csv(), 200);

    let sim_schema = SchemaInferrer::new()
        .infer(&sim_records)
        .expect("SIM records must produce a valid schema");

    println!("  Inferred {} fields:", sim_schema.len());
    for field in &sim_schema.fields {
        println!("    {:<20} {:?}", field.name, field.kind);
    }

    let sim_kind_of = |n: &str| sim_schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);
    assert_eq!(sim_kind_of("msisdn"),       Some(FieldKind::Phone), "msisdn → Phone");
    assert_eq!(sim_kind_of("imsi"),         Some(FieldKind::Id),    "imsi → Id");
    assert_eq!(sim_kind_of("geboortedatum"),Some(FieldKind::Date),  "geboortedatum → Date");
    println!("  Phone / Id / Date fields correctly inferred. ✓\n");

    println!("Example completed successfully.");
}
