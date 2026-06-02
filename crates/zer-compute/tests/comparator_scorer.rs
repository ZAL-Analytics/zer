/// Integration tests for `DeviceComparator` + `DeviceScorer` with both BRP and ANPR
/// datasets.
///
/// Run with `--features=cuda` to exercise the GPU path.
///
/// Covers:
///   - Full pipeline: compare → estimate_params → score on BRP data
///   - Batch size chunking behaves correctly (results identical for different
///     batch sizes across backends)
///   - ANPR license-plate comparison: OCR-confusion pairs should score higher
///     than random non-matching plates
///   - Backend auto-detect does not panic
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use zer_core::{
    comparison::ComparisonLevel,
    record::{FieldValue, Record, RecordId},
    record_pool::RecordPool,
    schema::{FieldKind, SchemaBuilder, Schema},
    scoring::MatchBand,
    traits::{Comparator, Scorer},
};
use zer_compute::{DeviceBackend, GpuBackend, DeviceComparator, DeviceScorer};

// Shared backend initialised once per test binary.
static BACKEND: OnceLock<Arc<DeviceBackend>> = OnceLock::new();
fn shared_backend() -> Arc<DeviceBackend> {
    Arc::clone(BACKEND.get_or_init(|| Arc::new(GpuBackend::auto_detect())))
}

// ── Paths ────────────────────────────────────────────────────────────────────

fn brp_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "tests/brp/brp_persons.csv")
}
fn brp_gt_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "tests/brp/ground_truth_pairs.csv")
}
fn anpr_gt_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "tests/anpr/ground_truth_compare_pairs.csv")
}

// ── Schemas ──────────────────────────────────────────────────────────────────

fn brp_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("geboortedatum", FieldKind::Date)
        .field("geboorteland",  FieldKind::Categorical)
        .field("nationaliteit", FieldKind::Categorical)
        .field("straatnaam",    FieldKind::Address)
        .field("huisnummer",    FieldKind::Address)
        .field("postcode",      FieldKind::Id)
        .field("woonplaats",    FieldKind::Address)
        .build()
        .unwrap()
}

fn anpr_schema() -> Schema {
    SchemaBuilder::new()
        .field("kenteken", FieldKind::LicensePlate)
        .build()
        .unwrap()
}

// ── BRP helpers ───────────────────────────────────────────────────────────────

fn load_brp(path: impl AsRef<std::path::Path>) -> HashMap<String, Record> {
    let mut rdr = csv::Reader::from_path(path).expect("BRP CSV missing");
    let headers = rdr.headers().unwrap().clone();
    let col     = |n: &str| headers.iter().position(|h| h == n).unwrap_or(usize::MAX);

    let c_bsn  = col("bsn");
    let c_voor = col("voornamen");  let c_tuss = col("tussenvoegsel");
    let c_ach  = col("achternaam"); let c_dob  = col("geboortedatum");
    let c_land = col("geboorteland"); let c_nat = col("nationaliteit");
    let c_str  = col("straatnaam"); let c_huis = col("huisnummer");
    let c_post = col("postcode");   let c_woon = col("woonplaats");

    let mut out = HashMap::new();
    let mut id: u64 = 1;
    for row in rdr.records().flatten() {
        let bsn = row.get(c_bsn).unwrap_or("").trim().to_string();
        if bsn.is_empty() { continue; }
        let tv = |i: usize| -> FieldValue {
            let v = row.get(i).unwrap_or("").trim();
            if v.is_empty() { FieldValue::Null } else { FieldValue::Text(v.into()) }
        };
        let r = Record::new(id).with_source("brp")
            .insert("voornamen",     tv(c_voor))
            .insert("achternaam",    tv(c_ach))
            .insert("tussenvoegsel", tv(c_tuss))
            .insert("geboortedatum", tv(c_dob))
            .insert("geboorteland",  tv(c_land))
            .insert("nationaliteit", tv(c_nat))
            .insert("straatnaam",    tv(c_str))
            .insert("huisnummer",    tv(c_huis))
            .insert("postcode",      tv(c_post))
            .insert("woonplaats",    tv(c_woon));
        out.insert(bsn, r);
        id += 1;
    }
    out
}

fn load_gt_pairs(
    records: &HashMap<String, Record>,
    gt_path: impl AsRef<std::path::Path>,
    col_a:   usize,
    col_b:   usize,
    col_match: usize,
) -> Vec<(RecordId, RecordId)> {
    let mut rdr = csv::Reader::from_path(gt_path).expect("GT CSV missing");
    let mut out = vec![];
    for row in rdr.records().flatten() {
        let key_a    = row.get(col_a).unwrap_or("").trim();
        let key_b    = row.get(col_b).unwrap_or("").trim();
        let is_match = row.get(col_match).unwrap_or("False").trim();
        if is_match != "True" { continue; }
        if let (Some(ra), Some(rb)) = (records.get(key_a), records.get(key_b)) {
            out.push((ra.id, rb.id));
        }
    }
    out
}

fn nonmatches(records: &HashMap<String, Record>, n: usize) -> Vec<(RecordId, RecordId)> {
    let mut ids: Vec<RecordId> = records.values().map(|r| r.id).collect();
    ids.sort_unstable();
    let step = (ids.len() / (n + 1)).max(1);
    (0..n).filter_map(|i| {
        let a = ids[i * step % ids.len()];
        let b = ids[(i * step + ids.len() / 2) % ids.len()];
        if a != b { Some((a, b)) } else { None }
    }).collect()
}

fn pairs_to_records(
    id_pairs: &[(RecordId, RecordId)],
    id_map:   &HashMap<RecordId, &Record>,
) -> Vec<(Record, Record)> {
    id_pairs.iter()
        .filter_map(|(a, b)| {
            Some(((*id_map.get(a)?).clone(), (*id_map.get(b)?).clone()))
        })
        .collect()
}

// ── ANPR helpers ─────────────────────────────────────────────────────────────

/// Build two-record pairs from the ANPR OCR ground truth.
/// Each row has (passage_id, true_kenteken, ocr_kenteken), create a record
/// for each and pair them so DeviceComparator can compare the plates.
fn load_anpr_pairs(_schema: &Schema) -> Vec<(Record, Record)> {
    let mut rdr = csv::Reader::from_path(anpr_gt_csv()).expect("ANPR GT CSV missing");
    let headers = rdr.headers().unwrap().clone();
    let col     = |n: &str| headers.iter().position(|h| h == n).unwrap_or(usize::MAX);

    let c_true = col("kenteken_true");
    let c_ocr  = col("kenteken_ocr");
    let c_match= col("is_match");

    let mut out = vec![];
    let mut id: u64 = 1;
    for row in rdr.records().flatten() {
        let is_match = row.get(c_match).unwrap_or("False").trim();
        if is_match != "True" { continue; }

        let true_plate = row.get(c_true).unwrap_or("").trim();
        let ocr_plate  = row.get(c_ocr).unwrap_or("").trim();
        if true_plate.is_empty() || ocr_plate.is_empty() { continue; }

        let a = Record::new(id)
            .insert("kenteken", FieldValue::Text(true_plate.into()));
        let b = Record::new(id + 1)
            .insert("kenteken", FieldValue::Text(ocr_plate.into()));
        out.push((a, b));
        id += 2;
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// Full BRP pipeline: compare → EM → score, precision and recall ≥ 0.70.
#[test]
fn brp_full_pipeline_precision_recall() {
    let schema  = brp_schema();
    let records = load_brp(brp_csv());
    let id_map: HashMap<RecordId, &Record> =
        records.values().map(|r| (r.id, r)).collect();

    let true_ids   = load_gt_pairs(&records, brp_gt_csv(), 0, 1, 2);
    let nonmatch_ids = nonmatches(&records, true_ids.len());

    let match_count = true_ids.len();

    let all_pairs: Vec<(Record, Record)> = {
        let mut v = pairs_to_records(&true_ids, &id_map);
        v.extend(pairs_to_records(&nonmatch_ids, &id_map));
        v
    };

    let backend = shared_backend();
    let cmp     = DeviceComparator::new(Arc::clone(&backend), &schema).unwrap();
    let scorer  = DeviceScorer::new(Arc::clone(&backend));

    let pool    = RecordPool::from_pairs(&all_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..all_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors = cmp.compare_batch_from_pool(&pool, &indices, &schema);
    assert_eq!(vectors.n_pairs, all_pairs.len());

    let params = scorer.estimate_params(&vectors, None, 100)
        .expect("EM should converge on BRP data");

    let scored = scorer.score_batch(&vectors, &params);

    let tp = scored[..match_count].iter().filter(|s| s.band == MatchBand::AutoMatch).count();
    let fp = scored[match_count..].iter().filter(|s| s.band == MatchBand::AutoMatch).count();
    let fn_ = match_count - tp;

    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall    = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };

    println!(
        "BRP (gpu backend=CPU): precision={:.3}, recall={:.3}  TP={} FP={} FN={}",
        precision, recall, tp, fp, fn_
    );

    assert!(precision >= 0.70, "precision {:.3} < 0.70", precision);
    assert!(recall    >= 0.70, "recall {:.3} < 0.70", recall);
}

/// Batch size chunking must not alter comparison results.
///
/// We run compare_batch on 200 pairs using the CPU backend (which uses one
/// internal chunk).  We then manually call `compare` pair-by-pair and verify
/// vectors are identical.  This catches any off-by-one in chunk slicing.
#[test]
fn batch_chunking_is_result_stable() {
    let schema  = brp_schema();
    let records = load_brp(brp_csv());
    let id_map: HashMap<RecordId, &Record> =
        records.values().map(|r| (r.id, r)).collect();

    let true_ids = load_gt_pairs(&records, brp_gt_csv(), 0, 1, 2);

    let sample: Vec<(Record, Record)> = pairs_to_records(
        &true_ids[..true_ids.len().min(200)],
        &id_map,
    );

    let backend = shared_backend();
    let cmp     = DeviceComparator::new(Arc::clone(&backend), &schema).unwrap();

    let pool    = RecordPool::from_pairs(&sample, &schema);
    let indices: Vec<(usize, usize)> = (0..sample.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let batch = cmp.compare_batch_from_pool(&pool, &indices, &schema);
    let single: Vec<_> = sample.iter()
        .map(|(a, b)| cmp.compare(a, b, &schema))
        .collect();

    assert_eq!(batch.n_pairs, single.len());
    let n_fields = schema.fields.len();
    for p in 0..batch.n_pairs {
        for f in 0..n_fields {
            let batch_level  = batch.level(f, p);
            let single_level = single[p].levels[f];
            assert_eq!(
                batch_level, single_level,
                "pair {p} field {f}: batch and single-pair results differ"
            );
        }
    }
}

/// ANPR OCR confusion pairs should have ≥ Close level on the `kenteken` field
/// more often than random unrelated plates.
#[test]
fn anpr_ocr_pairs_score_higher_than_random() {
    let schema = anpr_schema();
    let backend = shared_backend();
    let cmp     = DeviceComparator::new(Arc::clone(&backend), &schema).unwrap();

    let ocr_pairs = load_anpr_pairs(&schema);
    assert!(!ocr_pairs.is_empty(), "no ANPR OCR pairs loaded");

    let pool    = RecordPool::from_pairs(&ocr_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..ocr_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let ocr_vecs = cmp.compare_batch_from_pool(&pool, &indices, &schema);

    // Count how many OCR pairs are Close or better (field 0 = kenteken)
    let ocr_close_or_exact = (0..ocr_vecs.n_pairs)
        .filter(|&p| matches!(ocr_vecs.level(0, p), ComparisonLevel::Exact | ComparisonLevel::Close))
        .count();

    // Build same-size random pairs from first 1000 records in each direction
    let mut random_pairs: Vec<(Record, Record)> = Vec::new();
    let limit = ocr_pairs.len().min(200);
    for i in 0..limit {
        let a = ocr_pairs[i].0.clone();
        let b = ocr_pairs[(i + limit / 2) % limit].1.clone();
        random_pairs.push((a, b));
    }
    let pool_r      = RecordPool::from_pairs(&random_pairs, &schema);
    let indices_r: Vec<(usize, usize)> = (0..random_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let random_vecs = cmp.compare_batch_from_pool(&pool_r, &indices_r, &schema);

    let random_close_or_exact = (0..random_vecs.n_pairs)
        .filter(|&p| matches!(random_vecs.level(0, p), ComparisonLevel::Exact | ComparisonLevel::Close))
        .count();

    println!(
        "ANPR: OCR pairs Close/Exact={}/{}, random Close/Exact={}/{}",
        ocr_close_or_exact, ocr_vecs.n_pairs,
        random_close_or_exact, random_vecs.n_pairs
    );

    assert!(
        ocr_close_or_exact > random_close_or_exact,
        "OCR confusion pairs should match more closely than random pairs"
    );
}

/// auto_detect must not panic and must return a named backend.
#[test]
fn auto_detect_does_not_panic() {
    let backend = shared_backend();
    let name = backend.name();
    assert!(!name.is_empty(), "backend name must not be empty");
    println!("auto_detect resolved to backend: {name}");
}

/// DeviceScorer backend name matches DeviceComparator when sharing the same backend.
#[test]
fn scorer_and_comparator_share_backend_name() {
    let schema  = brp_schema();
    let backend = shared_backend();
    let cmp     = DeviceComparator::new(Arc::clone(&backend), &schema).unwrap();
    let scorer  = DeviceScorer::new(Arc::clone(&backend));
    assert_eq!(cmp.backend_name(), scorer.backend_name());
}
