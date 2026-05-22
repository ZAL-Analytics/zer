/// Blocking explorer demo.
///
/// Loads the person registry dataset and compares several blocking strategies
/// side-by-side, showing block-size distributions and candidate-pair statistics.
///
/// Run first: python data_generator/generate_demo_persons.py
use std::{collections::HashMap, path::Path};

use demo_common::{init_tracing, print_block_histogram, section};
use zer_blocking::keys::{
    BlockingKey, DateFragmentKey, DateGranularity, ExactFieldKey, PhoneticAlgo,
    PhoneticNameDobKey, PhoneticNameDobInitialKey,
};
use zer_core::{
    record::Record,
    schema::{FieldKind, SchemaBuilder},
};

const DATA_DIR: &str = "data/demos/persons";

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct PersonRow {
    record_id:     u64,
    voornamen:     String,
    #[serde(default)] tussenvoegsel: String,
    achternaam:    String,
    geboortedatum: String,
    geslacht:      String,
    straatnaam:    String,
    huisnummer:    String,
    postcode:      String,
    woonplaats:    String,
    #[serde(default)] bsn: String,
    #[serde(default)] geboorteplaats: String,
}

fn load_records(path: &Path) -> Vec<Record> {
    let mut rdr = csv::Reader::from_path(path).expect("open records.csv");
    rdr.deserialize::<PersonRow>()
        .map(|r| {
            let row = r.expect("parse record row");
            Record::new(row.record_id)
                .insert("voornamen",     row.voornamen)
                .insert("tussenvoegsel", row.tussenvoegsel)
                .insert("achternaam",    row.achternaam)
                .insert("geboortedatum", row.geboortedatum)
                .insert("geslacht",      row.geslacht)
                .insert("straatnaam",    row.straatnaam)
                .insert("huisnummer",    row.huisnummer)
                .insert("postcode",      row.postcode)
                .insert("woonplaats",    row.woonplaats)
        })
        .collect()
}

/// Build a bucket-size histogram for a blocking key applied to a record set.
///
/// Returns `(block_sizes, candidate_pairs)` where `block_sizes` is a `Vec<usize>`
/// of per-block member counts (blocks with 1 record are singletons and excluded).
fn analyse_key(
    key: &dyn BlockingKey,
    records: &[Record],
    schema: &zer_core::schema::Schema,
) -> (Vec<usize>, usize) {
    let mut bucket_counts: HashMap<String, usize> = HashMap::new();

    for record in records {
        for k in key.extract(record, schema) {
            *bucket_counts.entry(k).or_insert(0) += 1;
        }
    }

    let block_sizes: Vec<usize> = bucket_counts
        .values()
        .copied()
        .filter(|&n| n >= 2)
        .collect();

    let candidates: usize = block_sizes.iter().map(|&n| n * (n - 1) / 2).sum();

    (block_sizes, candidates)
}

fn report_key(
    label: &str,
    key: &dyn BlockingKey,
    records: &[Record],
    schema: &zer_core::schema::Schema,
) {
    let (sizes, candidates) = analyse_key(key, records, schema);

    let coverage_pct = 100.0 * sizes.iter().sum::<usize>() as f64 / records.len() as f64;

    println!(
        "  {:<40} blocks: {:>5}  candidates: {:>8}  coverage: {:.1}%",
        label,
        sizes.len(),
        candidates,
        coverage_pct,
    );

    print_block_histogram(label, &sizes);
}

fn main() {
    init_tracing();

    let path = Path::new(DATA_DIR).join("records.csv");

    section("Loading dataset");

    if !path.exists() {
        eprintln!(
            "Dataset not found at {}\n  Run: python data_generator/generate_demo_persons.py",
            path.display()
        );
        std::process::exit(1);
    }

    let records = load_records(&path);
    println!("{} records loaded", records.len());

    let schema = SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("geslacht",      FieldKind::FreeText)
        .field("straatnaam",    FieldKind::FreeText)
        .field("postcode",      FieldKind::FreeText)
        .build()
        .expect("build schema");

    // ── Key-by-key analysis ───────────────────────────────────────────────────
    section("Exact gender block (worst case, sanity check)");
    report_key(
        "exact geslacht",
        &ExactFieldKey::new("geslacht"),
        &records,
        &schema,
    );

    section("Date-fragment blocking");
    report_key(
        "birth year",
        &DateFragmentKey::new("geboortedatum", DateGranularity::Year),
        &records,
        &schema,
    );
    report_key(
        "birth year+month",
        &DateFragmentKey::new("geboortedatum", DateGranularity::YearMonth),
        &records,
        &schema,
    );

    section("Phonetic blocking");
    report_key(
        "phonetic surname + DOB (metaphone)",
        &PhoneticNameDobKey::new("achternaam", "geboortedatum"),
        &records,
        &schema,
    );
    report_key(
        "phonetic surname + DOB (soundex)",
        &PhoneticNameDobKey::new("achternaam", "geboortedatum")
            .with_algo(PhoneticAlgo::Soundex),
        &records,
        &schema,
    );
    report_key(
        "phonetic surname + DOB + initial",
        &PhoneticNameDobInitialKey::new("voornamen", "achternaam", "geboortedatum"),
        &records,
        &schema,
    );

    section("Interpretation");
    println!("  Fewer, larger blocks → higher recall, lower precision (more comparisons).");
    println!("  Smaller, tighter blocks → higher precision, lower recall (fewer comparisons).");
    println!("  PhoneticNameDobInitialKey is the recommended default for person deduplication.");

    section("Done");
}
