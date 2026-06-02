/// Integration tests: SchemaInferrer on example CSV data.
///
/// Verifies that the heuristics correctly classify the known BRP and SIM
/// column types without any explicit overrides.
use zer_core::{
    record::{FieldValue, Record},
    schema::FieldKind,
};
use zer_schema::infer::SchemaInferrer;

fn brp_q1_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "examples/brp_q1/brp_persons.csv")
}
fn sim_snap1_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "examples/sim/sim_subscribers.csv")
}

// ── CSV loader helpers ────────────────────────────────────────────────────────

fn load_csv_as_records(path: impl AsRef<std::path::Path>) -> Vec<Record> {
    let path = path.as_ref();
    let mut rdr = csv::Reader::from_path(path)
        .unwrap_or_else(|_| panic!("CSV not found at {}, run data generator first", path.display()));
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

// ── BRP inference tests ───────────────────────────────────────────────────────

#[test]
fn infer_brp_name_fields() {
    let records = load_csv_as_records(brp_q1_csv());
    let schema = SchemaInferrer::new().infer(&records).unwrap();

    let kind_of = |n: &str| schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);

    assert_eq!(kind_of("voornamen"), Some(FieldKind::Name), "voornamen should be Name");
    assert_eq!(kind_of("achternaam"), Some(FieldKind::Name), "achternaam should be Name");
}

#[test]
fn infer_brp_date_field() {
    let records = load_csv_as_records(brp_q1_csv());
    let schema = SchemaInferrer::new().infer(&records).unwrap();
    let kind_of = |n: &str| schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);
    assert_eq!(
        kind_of("geboortedatum"),
        Some(FieldKind::Date),
        "geboortedatum should be Date"
    );
}

#[test]
fn infer_brp_id_field() {
    let records = load_csv_as_records(brp_q1_csv());
    let schema = SchemaInferrer::new().infer(&records).unwrap();
    let kind_of = |n: &str| schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);
    assert_eq!(kind_of("bsn"), Some(FieldKind::Id), "bsn should be Id");
}

#[test]
fn infer_brp_address_fields() {
    let records = load_csv_as_records(brp_q1_csv());
    let schema = SchemaInferrer::new().infer(&records).unwrap();
    let kind_of = |n: &str| schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);

    assert_eq!(
        kind_of("straatnaam"),
        Some(FieldKind::Address),
        "straatnaam should be Address"
    );
    assert_eq!(
        kind_of("huisnummer"),
        Some(FieldKind::Address),
        "huisnummer should be Address"
    );
    assert_eq!(
        kind_of("postcode"),
        Some(FieldKind::Address),
        "postcode should be Address"
    );
    assert_eq!(
        kind_of("woonplaats"),
        Some(FieldKind::Address),
        "woonplaats should be Address"
    );
}

#[test]
fn infer_brp_produces_all_columns() {
    let records = load_csv_as_records(brp_q1_csv());
    let schema = SchemaInferrer::new().infer(&records).unwrap();

    // BRP Q1 has 14 columns.
    assert_eq!(
        schema.len(),
        14,
        "BRP schema should have 14 fields, got {}",
        schema.len()
    );
}

// ── SIM inference tests ───────────────────────────────────────────────────────

#[test]
fn infer_sim_phone_field() {
    let records = load_csv_as_records(sim_snap1_csv());
    let schema = SchemaInferrer::new().infer(&records).unwrap();
    let kind_of = |n: &str| schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);

    assert_eq!(kind_of("msisdn"), Some(FieldKind::Phone), "msisdn should be Phone");
}

#[test]
fn infer_sim_id_fields() {
    let records = load_csv_as_records(sim_snap1_csv());
    let schema = SchemaInferrer::new().infer(&records).unwrap();
    let kind_of = |n: &str| schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);

    assert_eq!(kind_of("imsi"), Some(FieldKind::Id), "imsi should be Id");
    assert_eq!(kind_of("iccid"), Some(FieldKind::Id), "iccid should be Id");
    assert_eq!(
        kind_of("document_nummer"),
        Some(FieldKind::Id),
        "document_nummer should be Id"
    );
}

#[test]
fn infer_sim_date_field() {
    let records = load_csv_as_records(sim_snap1_csv());
    let schema = SchemaInferrer::new().infer(&records).unwrap();
    let kind_of = |n: &str| schema.fields.iter().find(|f| f.name == n).map(|f| f.kind);

    assert_eq!(
        kind_of("activatiedatum"),
        Some(FieldKind::Date),
        "activatiedatum should be Date"
    );
    assert_eq!(
        kind_of("geboortedatum"),
        Some(FieldKind::Date),
        "geboortedatum should be Date"
    );
}

// ── Override test ─────────────────────────────────────────────────────────────

#[test]
fn override_beats_inference_on_real_data() {
    let records = load_csv_as_records(brp_q1_csv());
    // Force "geboortedatum" to Id even though heuristics would say Date.
    let schema = SchemaInferrer::new()
        .override_field("geboortedatum", FieldKind::Id)
        .infer(&records)
        .unwrap();

    let dob = schema.fields.iter().find(|f| f.name == "geboortedatum").unwrap();
    assert_eq!(
        dob.kind,
        FieldKind::Id,
        "override must win over heuristics on real BRP data"
    );
}
