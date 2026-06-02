/// Example: save a trained schema registry to a persistent `.zsm` file.
///
/// Loads BRP Q1 records, computes a realistic `SchemaFingerprint` from the
/// actual data distribution, simulates EM training, and writes the resulting
/// `ModelArtifact` to `data/examples/demo_registry.zsm`.
///
/// Run `registry_load` afterwards to verify the file survives a process
/// restart and that warm-start lookup returns the correct startup mode.
///
/// Run order:
///   cargo run --example registry_save -p zer-schema
///   cargo run --example registry_load -p zer-schema
use std::path::Path;

use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
    scoring::ModelParams,
};
use zer_schema::{
    artifact::ModelArtifact,
    fingerprint::SchemaFingerprint,
    registry::SchemaRegistry,
};

fn brp_q1_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "examples/brp_q1/brp_persons.csv")
}

const REGISTRY_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/examples/demo_registry.zsm"
);

fn brp_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("bsn",            FieldKind::Id)
        .field("voornamen",      FieldKind::Name)
        .field("tussenvoegsel",  FieldKind::Categorical)
        .field("achternaam",     FieldKind::Name)
        .field("geboortedatum",  FieldKind::Date)
        .field("geboorteplaats", FieldKind::Categorical)
        .field("geboorteland",   FieldKind::Categorical)
        .field("nationaliteit",  FieldKind::Categorical)
        .field("geslacht",       FieldKind::Categorical)
        .field("straatnaam",     FieldKind::Address)
        .field("huisnummer",     FieldKind::Address)
        .field("postcode",       FieldKind::Id)
        .field("woonplaats",     FieldKind::Address)
        .build()
        .unwrap()
}

fn load_records(path: impl AsRef<std::path::Path>) -> Vec<Record> {
    let path = path.as_ref();
    let mut rdr = csv::Reader::from_path(path)
        .unwrap_or_else(|_| panic!("CSV not found: {}", path.display()));
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
                if v.is_empty() { FieldValue::Null } else { FieldValue::Text(v.into()) },
            );
        }
        records.push(r);
        id += 1;
    }
    records
}

fn simulate_em(n_fields: usize) -> (ModelParams, usize) {
    let params = ModelParams {
        m: vec![vec![0.02, 0.06, 0.12, 0.80]; n_fields],
        u: vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
        log_prior_odds: -2.0,
        upper_threshold: 0.9,
        lower_threshold: 0.1,
    };
    (params, 25)
}

fn main() {
    println!("=== registry_save: writing .zsm file ===\n");

    let registry_path = Path::new(REGISTRY_PATH);
    println!("Output: {}", registry_path.display());

    // Load BRP Q1 data to compute a realistic fingerprint with data statistics.
    let csv_path = brp_q1_csv();
    println!("\nLoading BRP Q1 data from: {}", csv_path.display());
    let records = load_records(&csv_path);
    println!("  Loaded {} records.", records.len());

    let schema = brp_schema();
    let fingerprint = SchemaFingerprint::from_sample(&schema, &records);
    println!("  Schema hash: {}", hex_short(&fingerprint.schema_hash));
    println!("  Fields: {}, Record count: {}", schema.len(), fingerprint.record_count);

    // Simulate EM training.
    let (params, iterations) = simulate_em(schema.len());
    let artifact = ModelArtifact {
        fingerprint: fingerprint.clone(),
        params,
        tag: Some("brp_q1_demo".into()),
        trained_on: unix_now(),
        em_iterations: iterations,
    };

    let bytes = artifact.to_bytes().unwrap();
    println!("  Artifact size: {} bytes ({:.1} KB)", bytes.len(), bytes.len() as f64 / 1024.0);

    // Open (or create) the persistent registry and save.
    let registry = SchemaRegistry::open(registry_path).expect("failed to open registry");
    registry.save(&artifact).expect("save failed");

    // Confirm it can be read back in the same session.
    let loaded = registry.get_exact(&fingerprint).unwrap().unwrap();
    assert_eq!(loaded.tag.as_deref(), Some("brp_q1_demo"));
    println!("\nArtifact saved and verified in-session.");

    let file_size = std::fs::metadata(registry_path).map(|m| m.len()).unwrap_or(0);
    println!("File size on disk: {} bytes", file_size);

    println!("\nDone. Run `cargo run --example registry_load -p zer-schema` to verify the file.");
}

fn hex_short(bytes: &[u8]) -> String {
    bytes.iter().take(8).map(|b| format!("{b:02x}")).collect::<String>() + "…"
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
