/// Integration tests: SchemaFingerprint construction and distance metric.
///
/// Exercises `from_sample` against example CSV data to confirm that the
/// stats (null_rate, cardinality, top_k) are populated correctly, and that
/// the distance metric properly orders BRP vs SIM fingerprints.
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_schema::{
    fingerprint::SchemaFingerprint,
    similarity::{fingerprint_distance, WARM_START_THRESHOLD},
};

fn brp_q1_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "examples/brp_q1/brp_persons.csv")
}
fn brp_q2_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "examples/brp_q2/brp_persons.csv")
}
fn sim_snap1_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "examples/sim/sim_subscribers.csv")
}

fn load_records(path: impl AsRef<std::path::Path>) -> Vec<Record> {
    let path = path.as_ref();
    let mut rdr = csv::Reader::from_path(path)
        .unwrap_or_else(|_| panic!("CSV not found at {}", path.display()));
    let headers = rdr.headers().unwrap().clone();
    let mut records = Vec::new();
    let mut id: u64 = 1;
    for result in rdr.records() {
        let row = result.unwrap();
        let mut r = Record::new(id);
        for (i, header) in headers.iter().enumerate() {
            let v = row.get(i).unwrap_or("").trim();
            r = r.insert(
                header,
                if v.is_empty() {
                    FieldValue::Null
                } else {
                    FieldValue::Text(v.into())
                },
            );
        }
        records.push(r);
        id += 1;
    }
    records
}

fn brp_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("bsn", FieldKind::Id)
        .field("voornamen", FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("geboorteplaats", FieldKind::Categorical)
        .field("geboorteland", FieldKind::Categorical)
        .field("nationaliteit", FieldKind::Categorical)
        .field("geslacht", FieldKind::Categorical)
        .field("straatnaam", FieldKind::Address)
        .field("huisnummer", FieldKind::Address)
        .field("postcode", FieldKind::Id)
        .field("woonplaats", FieldKind::Address)
        .build()
        .unwrap()
}

fn sim_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("sim_id", FieldKind::Id)
        .field("msisdn", FieldKind::Phone)
        .field("imsi", FieldKind::Id)
        .field("iccid", FieldKind::Id)
        .field("carrier", FieldKind::Categorical)
        .field("contract_type", FieldKind::Categorical)
        .field("activatiedatum", FieldKind::Date)
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("nationaliteit", FieldKind::Categorical)
        .field("document_type", FieldKind::Categorical)
        .field("document_nummer", FieldKind::Id)
        .field("bsn", FieldKind::Id)
        .build()
        .unwrap()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn brp_q1_q2_same_schema_hash() {
    let q1 = load_records(brp_q1_csv());
    let q2 = load_records(brp_q2_csv());
    let schema = brp_schema();

    let fp1 = SchemaFingerprint::from_sample(&schema, &q1);
    let fp2 = SchemaFingerprint::from_sample(&schema, &q2);

    assert_eq!(
        fp1.schema_hash, fp2.schema_hash,
        "Q1 and Q2 share the same schema, hashes must be identical"
    );
    assert_eq!(fingerprint_distance(&fp1, &fp2), 0.0);
}

#[test]
fn brp_vs_sim_distance_exceeds_warm_start_threshold() {
    let brp_records = load_records(brp_q1_csv());
    let sim_records = load_records(sim_snap1_csv());

    let fp_brp = SchemaFingerprint::from_sample(&brp_schema(), &brp_records);
    let fp_sim = SchemaFingerprint::from_sample(&sim_schema(), &sim_records);

    let dist = fingerprint_distance(&fp_brp, &fp_sim);
    assert!(
        dist > WARM_START_THRESHOLD,
        "BRP vs SIM distance {dist:.4} should exceed warm-start threshold {WARM_START_THRESHOLD}"
    );
}

#[test]
fn from_sample_populates_stats_for_brp() {
    let records = load_records(brp_q1_csv());
    let schema = brp_schema();
    let fp = SchemaFingerprint::from_sample(&schema, &records);

    assert_eq!(fp.field_stats.len(), schema.len());
    // Q1 has ~6 600 records.
    assert!(fp.record_count > 1_000, "record_count should reflect CSV size");

    // geboortedatum should have exactly one value format (YYYY-MM-DD) →
    // cardinality should be high (many unique dates) and null_rate near 0.
    let dob = fp.field_stats.iter().find(|f| f.name == "geboortedatum").unwrap();
    assert!(
        dob.null_rate < 0.1,
        "geboortedatum null_rate should be near 0, got {}",
        dob.null_rate
    );
    assert!(
        dob.cardinality > 100,
        "geboortedatum should have many distinct values"
    );
}

#[test]
fn from_sample_top_k_is_ordered_by_frequency() {
    let records = load_records(brp_q1_csv());
    let schema = brp_schema();
    let fp = SchemaFingerprint::from_sample(&schema, &records);

    // geslacht (gender) should have low cardinality (M/V/X) and a non-empty top_k.
    let gender = fp.field_stats.iter().find(|f| f.name == "geslacht").unwrap();
    assert!(!gender.top_k.is_empty(), "geslacht top_k must not be empty");
    assert!(
        gender.cardinality <= 5,
        "geslacht should have at most 5 distinct values"
    );
}

#[test]
fn brp_extended_schema_warm_start_distance() {
    let records = load_records(brp_q1_csv());
    let base = brp_schema();

    let extended = SchemaBuilder::new()
        .field("bsn", FieldKind::Id)
        .field("voornamen", FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("geboorteplaats", FieldKind::Categorical)
        .field("geboorteland", FieldKind::Categorical)
        .field("nationaliteit", FieldKind::Categorical)
        .field("geslacht", FieldKind::Categorical)
        .field("straatnaam", FieldKind::Address)
        .field("huisnummer", FieldKind::Address)
        .field("postcode", FieldKind::Id)
        .field("woonplaats", FieldKind::Address)
        .field("verblijfstitel", FieldKind::Categorical) // added
        .build()
        .unwrap();

    let fp_base = SchemaFingerprint::from_sample(&base, &records);
    let fp_ext = SchemaFingerprint::from_schema(&extended);

    let dist = fingerprint_distance(&fp_base, &fp_ext);

    assert!(
        dist > 0.0,
        "schemas differ by one field, distance must be > 0"
    );
    assert!(
        dist <= WARM_START_THRESHOLD,
        "one extra field should stay within warm-start threshold, got {dist:.4}"
    );
}
