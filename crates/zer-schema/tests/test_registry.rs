/// Integration tests: SchemaRegistry warm-load / warm-start / cold-start behaviour.
///
/// Uses the synthetic BRP (population register) and SIM (subscriber) datasets
/// from `data/v1.1/examples/` to verify the three startup modes:
///
/// 1. `WarmLoad` , BRP Q1 artifact → lookup with Q2 (same schema)
/// 2. `WarmStart`, BRP Q1 artifact → lookup with Q2 + one extra field
/// 3. `ColdStart`, BRP artifact    → lookup with a SIM subscriber schema
///
/// And verifies that the nearest-neighbour search picks the correct artifact
/// when both BRP and SIM artifacts are stored.
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
    scoring::ModelParams,
};

use zer_schema::{
    artifact::ModelArtifact,
    fingerprint::SchemaFingerprint,
    registry::{SchemaRegistry, StartupMode},
    similarity::WARM_START_THRESHOLD,
};

// ── Shared schema definitions ─────────────────────────────────────────────────

/// The full 13-column BRP schema (both Q1 and Q2 have identical columns).
fn brp_schema_full() -> zer_core::schema::Schema {
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

/// BRP schema *without* `verblijfstitel`.  Used as the "trained" schema so that
/// the drift test (adding `verblijfstitel`) exercises the warm-start path.
fn brp_schema_base() -> zer_core::schema::Schema {
    brp_schema_full()
}

/// BRP schema with `verblijfstitel` added, simulates the Q2 schema drift test.
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

/// SIM subscriber schema, structurally different from BRP.
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

// ── CSV loaders ───────────────────────────────────────────────────────────────

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
fn sim_snap1_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(
        env!("CARGO_MANIFEST_DIR"),
        "examples/sim/sim_subscribers.csv",
    )
}

fn load_brp_records(path: impl AsRef<std::path::Path>) -> Vec<Record> {
    let mut rdr =
        csv::Reader::from_path(path).expect("BRP CSV not found, run data generator first");
    let headers = rdr.headers().unwrap().clone();
    let col = |n: &str| headers.iter().position(|h| h == n).unwrap_or(usize::MAX);

    let c_bsn = col("bsn");
    let c_voor = col("voornamen");
    let c_tuss = col("tussenvoegsel");
    let c_ach = col("achternaam");
    let c_dob = col("geboortedatum");
    let c_gpl = col("geboorteplaats");
    let c_gland = col("geboorteland");
    let c_nat = col("nationaliteit");
    let c_ges = col("geslacht");
    let c_str = col("straatnaam");
    let c_huis = col("huisnummer");
    let c_post = col("postcode");
    let c_woon = col("woonplaats");

    let mut records = Vec::new();
    let mut id: u64 = 1;

    for result in rdr.records() {
        let row = result.unwrap();
        let tv = |idx: usize| -> FieldValue {
            let v = row.get(idx).unwrap_or("").trim();
            if v.is_empty() {
                FieldValue::Null
            } else {
                FieldValue::Text(v.into())
            }
        };

        let r = Record::new(id)
            .with_source("brp")
            .insert("bsn", tv(c_bsn))
            .insert("voornamen", tv(c_voor))
            .insert("tussenvoegsel", tv(c_tuss))
            .insert("achternaam", tv(c_ach))
            .insert("geboortedatum", tv(c_dob))
            .insert("geboorteplaats", tv(c_gpl))
            .insert("geboorteland", tv(c_gland))
            .insert("nationaliteit", tv(c_nat))
            .insert("geslacht", tv(c_ges))
            .insert("straatnaam", tv(c_str))
            .insert("huisnummer", tv(c_huis))
            .insert("postcode", tv(c_post))
            .insert("woonplaats", tv(c_woon));

        records.push(r);
        id += 1;
    }
    records
}

fn load_sim_records(path: impl AsRef<std::path::Path>) -> Vec<Record> {
    let mut rdr =
        csv::Reader::from_path(path).expect("SIM CSV not found, run data generator first");
    let headers = rdr.headers().unwrap().clone();
    let col = |n: &str| headers.iter().position(|h| h == n).unwrap_or(usize::MAX);

    let c_sid = col("sim_id");
    let c_msisdn = col("msisdn");
    let c_imsi = col("imsi");
    let c_iccid = col("iccid");
    let c_car = col("carrier");
    let c_ctype = col("contract_type");
    let c_act = col("activatiedatum");
    let c_voor = col("voornamen");
    let c_ach = col("achternaam");
    let c_dob = col("geboortedatum");
    let c_nat = col("nationaliteit");
    let c_dtype = col("document_type");
    let c_dnum = col("document_nummer");
    let c_bsn = col("bsn");

    let mut records = Vec::new();
    let mut id: u64 = 100_000;

    for result in rdr.records() {
        let row = result.unwrap();
        let tv = |idx: usize| -> FieldValue {
            let v = row.get(idx).unwrap_or("").trim();
            if v.is_empty() {
                FieldValue::Null
            } else {
                FieldValue::Text(v.into())
            }
        };

        let r = Record::new(id)
            .with_source("sim")
            .insert("sim_id", tv(c_sid))
            .insert("msisdn", tv(c_msisdn))
            .insert("imsi", tv(c_imsi))
            .insert("iccid", tv(c_iccid))
            .insert("carrier", tv(c_car))
            .insert("contract_type", tv(c_ctype))
            .insert("activatiedatum", tv(c_act))
            .insert("voornamen", tv(c_voor))
            .insert("achternaam", tv(c_ach))
            .insert("geboortedatum", tv(c_dob))
            .insert("nationaliteit", tv(c_nat))
            .insert("document_type", tv(c_dtype))
            .insert("document_nummer", tv(c_dnum))
            .insert("bsn", tv(c_bsn));

        records.push(r);
        id += 1;
    }
    records
}

// ── Dummy model parameters ────────────────────────────────────────────────────

fn dummy_params(n_fields: usize) -> ModelParams {
    ModelParams {
        m: vec![vec![0.02, 0.06, 0.12, 0.80]; n_fields],
        u: vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
        log_prior_odds: -2.0,
        upper_threshold: 0.9,
        lower_threshold: 0.1,
    }
}

fn make_artifact(fingerprint: SchemaFingerprint, n_fields: usize, tag: &str) -> ModelArtifact {
    ModelArtifact {
        fingerprint,
        params: dummy_params(n_fields),
        tag: Some(tag.into()),
        trained_on: 0,
        em_iterations: 25,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// BRP Q1 artifact → look up with BRP Q2 → WarmLoad (identical schema hash).
#[test]
fn brp_q1_to_q2_is_warm_load() {
    let q1_records = load_brp_records(brp_q1_csv());
    let q2_records = load_brp_records(brp_q2_csv());

    assert!(!q1_records.is_empty(), "Q1 records must not be empty");
    assert!(!q2_records.is_empty(), "Q2 records must not be empty");

    let schema = brp_schema_base();
    let fp_q1 = SchemaFingerprint::from_sample(&schema, &q1_records);
    let fp_q2 = SchemaFingerprint::from_sample(&schema, &q2_records);

    // Same schema → same hash.
    assert_eq!(
        fp_q1.schema_hash, fp_q2.schema_hash,
        "Q1 and Q2 have the same schema, hashes must match"
    );

    let dir = tempfile::tempdir().unwrap();
    let registry = SchemaRegistry::open(&dir.path().join("model.zsm")).unwrap();
    let artifact = make_artifact(fp_q1, schema.len(), "brp_q1");
    registry.save(&artifact).unwrap();

    let mode = registry.lookup_startup_mode(&fp_q2).unwrap();
    assert!(
        matches!(mode, StartupMode::WarmLoad(_)),
        "same schema must return WarmLoad"
    );
}

/// BRP Q1 artifact → look up with extended schema (verblijfstitel added) → WarmStart.
#[test]
fn brp_extended_schema_is_warm_start() {
    let q1_records = load_brp_records(brp_q1_csv());

    let base_schema = brp_schema_base();
    let extended_schema = brp_schema_extended();

    let fp_base = SchemaFingerprint::from_sample(&base_schema, &q1_records);
    let fp_ext = SchemaFingerprint::from_schema(&extended_schema);

    let dir = tempfile::tempdir().unwrap();
    let registry = SchemaRegistry::open(&dir.path().join("model.zsm")).unwrap();
    registry
        .save(&make_artifact(fp_base, base_schema.len(), "brp_q1"))
        .unwrap();

    let mode = registry.lookup_startup_mode(&fp_ext).unwrap();
    match mode {
        StartupMode::WarmStart { distance, .. } => {
            assert!(
                distance <= WARM_START_THRESHOLD,
                "WarmStart distance {distance:.4} must be ≤ {WARM_START_THRESHOLD}"
            );
        }
        other => panic!(
            "expected WarmStart for schema with one added field, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// BRP artifact → look up with SIM subscriber schema → ColdStart.
#[test]
fn sim_schema_against_brp_artifact_is_cold_start() {
    let q1_records = load_brp_records(brp_q1_csv());

    let brp_schema = brp_schema_base();
    let fp_brp = SchemaFingerprint::from_sample(&brp_schema, &q1_records);

    let dir = tempfile::tempdir().unwrap();
    let registry = SchemaRegistry::open(&dir.path().join("model.zsm")).unwrap();
    registry
        .save(&make_artifact(fp_brp, brp_schema.len(), "brp_q1"))
        .unwrap();

    // Look up with SIM schema, structurally incompatible.
    let fp_sim = SchemaFingerprint::from_schema(&sim_schema());
    let mode = registry.lookup_startup_mode(&fp_sim).unwrap();

    assert!(
        matches!(mode, StartupMode::ColdStart),
        "SIM schema vs BRP artifact must return ColdStart"
    );
}

/// When both BRP and SIM artifacts are stored, a SIM-like query should match SIM.
#[test]
fn nearest_neighbour_picks_sim_over_brp_for_sim_query() {
    let q1_records = load_brp_records(brp_q1_csv());
    let sim_records = load_sim_records(sim_snap1_csv());

    let brp_schema = brp_schema_base();
    let sim_schema = sim_schema();

    let fp_brp = SchemaFingerprint::from_sample(&brp_schema, &q1_records);
    let fp_sim = SchemaFingerprint::from_sample(&sim_schema, &sim_records);

    let dir = tempfile::tempdir().unwrap();
    let registry = SchemaRegistry::open(&dir.path().join("model.zsm")).unwrap();
    registry
        .save(&make_artifact(fp_brp, brp_schema.len(), "brp"))
        .unwrap();
    registry
        .save(&make_artifact(fp_sim.clone(), sim_schema.len(), "sim"))
        .unwrap();

    // Query with an exact SIM fingerprint → exact match (WarmLoad for SIM).
    let mode = registry.lookup_startup_mode(&fp_sim).unwrap();
    assert!(
        matches!(mode, StartupMode::WarmLoad(_)),
        "exact SIM fingerprint must WarmLoad the SIM artifact"
    );
    if let StartupMode::WarmLoad(art) = mode {
        assert_eq!(art.tag.as_deref(), Some("sim"));
    }
}

/// Artifact roundtrip: saved artifact byte-for-byte matches the loaded one.
#[test]
fn artifact_roundtrip_through_registry() {
    let q1_records = load_brp_records(brp_q1_csv());
    let schema = brp_schema_base();
    let fingerprint = SchemaFingerprint::from_sample(&schema, &q1_records);
    let original = make_artifact(fingerprint.clone(), schema.len(), "roundtrip_test");

    let dir = tempfile::tempdir().unwrap();
    let registry = SchemaRegistry::open(&dir.path().join("model.zsm")).unwrap();
    registry.save(&original).unwrap();

    let loaded = registry.get_exact(&fingerprint).unwrap().unwrap();

    assert_eq!(original.tag, loaded.tag);
    assert_eq!(original.em_iterations, loaded.em_iterations);
    assert_eq!(
        original.params.upper_threshold,
        loaded.params.upper_threshold
    );
    assert_eq!(
        original.params.lower_threshold,
        loaded.params.lower_threshold
    );
    assert_eq!(original.params.m, loaded.params.m);
    assert_eq!(original.params.u, loaded.params.u);
    assert_eq!(
        original.fingerprint.schema_hash,
        loaded.fingerprint.schema_hash
    );
    assert_eq!(
        original.fingerprint.record_count,
        loaded.fingerprint.record_count
    );
}

/// `list_all` returns one entry per saved artifact.
#[test]
fn list_all_reflects_all_saved_artifacts() {
    let brp_records = load_brp_records(brp_q1_csv());
    let sim_records = load_sim_records(sim_snap1_csv());

    let dir = tempfile::tempdir().unwrap();
    let registry = SchemaRegistry::open(&dir.path().join("model.zsm")).unwrap();

    let brp = brp_schema_base();
    let sim = sim_schema();

    registry
        .save(&make_artifact(
            SchemaFingerprint::from_sample(&brp, &brp_records),
            brp.len(),
            "brp",
        ))
        .unwrap();
    registry
        .save(&make_artifact(
            SchemaFingerprint::from_sample(&sim, &sim_records),
            sim.len(),
            "sim",
        ))
        .unwrap();

    let all = registry.list_all().unwrap();
    assert_eq!(all.len(), 2, "registry must hold exactly 2 artifacts");

    let tags: std::collections::HashSet<Option<String>> = all.into_iter().map(|a| a.tag).collect();
    assert!(tags.contains(&Some("brp".into())));
    assert!(tags.contains(&Some("sim".into())));
}

/// Persisting an artifact to disk and re-opening the registry should still find it.
#[test]
fn registry_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let zsm = dir.path().join("model.zsm");
    let schema = brp_schema_base();
    let records = load_brp_records(brp_q1_csv());
    let fp = SchemaFingerprint::from_sample(&schema, &records);

    {
        let reg = SchemaRegistry::open(&zsm).unwrap();
        reg.save(&make_artifact(fp.clone(), schema.len(), "persistent"))
            .unwrap();
        // reg is dropped here; file already flushed on save
    }

    // Re-open the same .zsm file
    let reg2 = SchemaRegistry::open(&zsm).unwrap();
    let loaded = reg2.get_exact(&fp).unwrap();
    assert!(
        loaded.is_some(),
        "artifact should survive a registry close/reopen"
    );
    assert_eq!(loaded.unwrap().tag.as_deref(), Some("persistent"));
}
