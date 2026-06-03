/// Example: schema registry warm-start workflow.
///
/// Demonstrates the full pipeline startup decision loop:
/// 1. Train on BRP Q1 → save `ModelArtifact` to registry
/// 2. Arrive with BRP Q2 (same schema) → WarmLoad (skip EM entirely)
/// 3. Arrive with BRP + one new field (verblijfstitel) → WarmStart (2–3 EM iters)
/// 4. Arrive with SIM subscriber schema → ColdStart (full EM from priors)
/// 5. Demonstrate `list_all` and `delete`
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
    scoring::ModelParams,
};
use zer_schema::{
    artifact::ModelArtifact,
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

fn brp_schema_base() -> zer_core::schema::Schema {
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

fn brp_schema_extended() -> zer_core::schema::Schema {
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
        .field("verblijfstitel", FieldKind::Categorical) // added field
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

/// Simulate EM training by building dummy params (in real code you'd call
/// `FellegiSunterScorer::estimate_params`).
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
    println!("=== zer-schema: registry warm-start example ===\n");

    // ── Setup: open a temporary registry ─────────────────────────────────────

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let registry =
        SchemaRegistry::open(&dir.path().join("demo.zsm")).expect("failed to open registry");

    // ── Step 1: Train on BRP Q1 and save artifact ─────────────────────────────

    println!("Step 1, Training on BRP Q1 ...");
    let q1_records = load_records(brp_q1_csv());
    let base_schema = brp_schema_base();
    let fp_q1 = SchemaFingerprint::from_sample(&base_schema, &q1_records);

    let (params, iterations) = simulate_em(base_schema.len());
    let artifact = ModelArtifact {
        fingerprint: fp_q1.clone(),
        params,
        tag: Some("brp_q1_2024".into()),
        trained_on: 0,
        em_iterations: iterations,
    };
    registry.save(&artifact).expect("save must succeed");
    println!(
        "  Saved artifact 'brp_q1_2024' ({} fields, {} EM iterations).",
        base_schema.len(),
        iterations
    );

    // Check serialized size
    let bytes = artifact.to_bytes().unwrap();
    println!(
        "  Serialized size: {} bytes ({:.1} KB).",
        bytes.len(),
        bytes.len() as f64 / 1024.0
    );
    assert!(bytes.len() < 10_240, "artifact should be under 10 KB");

    // ── Step 2: Arrive with BRP Q2, same schema ──────────────────────────────

    println!("\nStep 2, Arriving with BRP Q2 (same schema) ...");
    let q2_records = load_records(brp_q2_csv());
    let fp_q2 = SchemaFingerprint::from_sample(&base_schema, &q2_records);

    match registry.lookup_startup_mode(&fp_q2).expect("lookup failed") {
        StartupMode::WarmLoad(art) => {
            println!(
                "  → WarmLoad: loaded '{}', EM skipped entirely. ✓",
                art.tag.as_deref().unwrap_or("(no tag)")
            );
        }
        other => panic!(
            "expected WarmLoad, got {:?}",
            std::mem::discriminant(&other)
        ),
    }

    // ── Step 3: Arrive with extended schema (verblijfstitel added) ────────────

    println!("\nStep 3, Arriving with extended BRP schema (verblijfstitel added) ...");
    let fp_ext = SchemaFingerprint::from_schema(&brp_schema_extended());

    match registry
        .lookup_startup_mode(&fp_ext)
        .expect("lookup failed")
    {
        StartupMode::WarmStart {
            artifact: art,
            distance: d,
        } => {
            println!(
                "  → WarmStart: loaded '{}', distance={:.4}, run 2–3 EM iterations. ✓",
                art.tag.as_deref().unwrap_or("(no tag)"),
                d
            );
        }
        other => panic!(
            "expected WarmStart, got {:?}",
            std::mem::discriminant(&other)
        ),
    }

    // ── Step 4: Arrive with SIM subscriber schema ─────────────────────────────

    println!("\nStep 4, Arriving with SIM subscriber schema (incompatible) ...");
    let fp_sim = SchemaFingerprint::from_schema(&sim_schema());

    match registry
        .lookup_startup_mode(&fp_sim)
        .expect("lookup failed")
    {
        StartupMode::ColdStart => {
            println!("  → ColdStart: no suitable prior found, run full EM. ✓");
        }
        other => panic!(
            "expected ColdStart, got {:?}",
            std::mem::discriminant(&other)
        ),
    }

    // ── Step 5: list_all and delete ───────────────────────────────────────────

    println!("\nStep 5, Inspecting and cleaning registry ...");
    let all = registry.list_all().expect("list_all failed");
    println!("  Registry contains {} artifact(s):", all.len());
    for a in &all {
        println!(
            "    • {} (trained_on={})",
            a.tag.as_deref().unwrap_or("(no tag)"),
            a.trained_on
        );
    }

    let removed = registry
        .delete(&artifact.fingerprint.schema_hash)
        .expect("delete failed");
    assert!(removed, "delete must return true for existing artifact");
    println!(
        "  Deleted 'brp_q1_2024'. Registry now empty: {}",
        registry.list_all().unwrap().is_empty()
    );

    println!("\nExample completed successfully.");
}
