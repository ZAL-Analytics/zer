/// Example: load a persisted schema registry from a `.zsm` file.
///
/// Reads `data/examples/demo_registry.zsm` written by `registry_save`, then:
/// 1. Lists every stored artifact.
/// 2. Recomputes the BRP Q1 fingerprint → expects **WarmLoad** (exact hash match).
/// 3. Recomputes a BRP Q2 fingerprint (same fields + `verblijfstitel`) → expects
///    **WarmStart** (similar schema, small distance).
///
/// Run order:
///   cargo run --example registry_save -p zer-schema
///   cargo run --example registry_load -p zer-schema
use std::path::Path;

use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_schema::{
    fingerprint::SchemaFingerprint,
    registry::{SchemaRegistry, StartupMode},
};

fn brp_q1_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(
        env!("CARGO_MANIFEST_DIR"),
        "examples/brp_q1/brp_persons.csv",
    )
}
fn brp_q2_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(
        env!("CARGO_MANIFEST_DIR"),
        "examples/brp_q2/brp_persons.csv",
    )
}

const REGISTRY_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/examples/demo_registry.zsm"
);

fn brp_q1_schema() -> zer_core::schema::Schema {
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

fn brp_q2_schema() -> zer_core::schema::Schema {
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
        .field("verblijfstitel", FieldKind::Categorical)
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

fn main() {
    println!("=== registry_load: reading .zsm file ===\n");

    let registry_path = Path::new(REGISTRY_PATH);
    if !registry_path.exists() {
        println!(
            "Registry file not found: {}\nRun `cargo run --example registry_save -p zer-schema` first.",
            registry_path.display()
        );
        std::process::exit(1);
    }

    println!("Reading: {}", registry_path.display());
    let registry = SchemaRegistry::open(registry_path).expect("failed to open registry");

    // ── Step 1: list all stored artifacts ────────────────────────────────────

    let all = registry.list_all().unwrap();
    println!("\nStored artifacts: {}", all.len());
    for art in &all {
        println!(
            "  tag={:?}  fields={}  trained_on={}  em_iters={}",
            art.tag.as_deref().unwrap_or("<none>"),
            art.fingerprint.field_stats.len(),
            art.trained_on,
            art.em_iterations,
        );
    }

    // ── Step 2: BRP Q1 → WarmLoad (exact match) ──────────────────────────────

    println!("\nLoading BRP Q1 data for fingerprint…");
    let q1_records = load_records(brp_q1_csv());
    let q1_schema = brp_q1_schema();
    let q1_fp = SchemaFingerprint::from_sample(&q1_schema, &q1_records);

    println!("  Q1 schema hash: {}", hex_short(&q1_fp.schema_hash));
    match registry.lookup_startup_mode(&q1_fp).unwrap() {
        StartupMode::WarmLoad(art) => {
            println!(
                "  → WarmLoad ✓  (tag={:?}, no EM needed)",
                art.tag.as_deref()
            );
        }
        StartupMode::WarmStart { artifact, distance } => {
            println!(
                "  → WarmStart  distance={distance:.4}  (tag={:?})",
                artifact.tag.as_deref()
            );
            println!("  WARNING: expected WarmLoad for exact Q1 fingerprint");
        }
        StartupMode::ColdStart => {
            println!("  → ColdStart  (artifact not found, did registry_save run?)");
        }
    }

    // ── Step 3: BRP Q2 → WarmStart (one extra field: verblijfstitel) ─────────

    println!("\nLoading BRP Q2 data for fingerprint…");
    let q2_records = load_records(brp_q2_csv());
    let q2_schema = brp_q2_schema();
    let q2_fp = SchemaFingerprint::from_sample(&q2_schema, &q2_records);

    println!("  Q2 schema hash: {}", hex_short(&q2_fp.schema_hash));
    println!(
        "  Q2 has {} fields vs Q1's {} fields (+verblijfstitel)",
        q2_schema.len(),
        q1_schema.len(),
    );

    match registry.lookup_startup_mode(&q2_fp).unwrap() {
        StartupMode::WarmLoad(art) => {
            println!("  → WarmLoad  (tag={:?})", art.tag.as_deref());
        }
        StartupMode::WarmStart { artifact, distance } => {
            println!(
                "  → WarmStart ✓  distance={distance:.4}  (tag={:?}, run 2–3 EM iters to fine-tune)",
                artifact.tag.as_deref()
            );
        }
        StartupMode::ColdStart => {
            println!("  → ColdStart  (schemas too different or no stored artifact)");
        }
    }

    println!("\nDone.");
}

fn hex_short(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        + "…"
}
