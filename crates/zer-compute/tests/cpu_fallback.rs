/// Correctness tests for `zer-compute`, verifies that `DeviceComparator` and
/// `DeviceScorer` produce the same results as the bare `zer-compare` CPU
/// reference regardless of which backend `auto_detect()` selects.
///
/// Run with `--features=cuda` to exercise the GPU path.
///
/// Phase-04 data:
///   data/tests/brp/brp_persons.csv       , 2 000 person records
///   data/tests/brp/ground_truth_pairs.csv, 200 true-match pairs
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use zer_compare::{FellegiSunterScorer, FieldComparator};
use zer_compute::{DeviceBackend, DeviceComparator, DeviceScorer, GpuBackend};
use zer_core::{
    comparison::ComparisonLevel,
    record::{FieldValue, Record, RecordId},
    record_pool::RecordPool,
    schema::{FieldKind, Schema, SchemaBuilder},
    traits::{Comparator, Scorer},
};

static BACKEND: OnceLock<Arc<DeviceBackend>> = OnceLock::new();
fn shared_backend() -> Arc<DeviceBackend> {
    Arc::clone(BACKEND.get_or_init(|| Arc::new(GpuBackend::auto_detect())))
}

// ── Data paths ───────────────────────────────────────────────────────────────

fn brp_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "tests/brp/brp_persons.csv")
}
fn brp_gt_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(
        env!("CARGO_MANIFEST_DIR"),
        "tests/brp/ground_truth_pairs.csv",
    )
}

// ── Schema ───────────────────────────────────────────────────────────────────

fn brp_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("geboortedatum", FieldKind::Date)
        .field("geboorteland", FieldKind::Categorical)
        .field("nationaliteit", FieldKind::Categorical)
        .field("straatnaam", FieldKind::Address)
        .field("huisnummer", FieldKind::Address)
        .field("postcode", FieldKind::Id)
        .field("woonplaats", FieldKind::Address)
        .build()
        .unwrap()
}

// ── Loaders ───────────────────────────────────────────────────────────────────

fn load_brp_records() -> HashMap<String, Record> {
    let mut rdr =
        csv::Reader::from_path(brp_csv()).expect("BRP CSV not found, run data generator first");
    let headers = rdr.headers().unwrap().clone();
    let col = |name: &str| headers.iter().position(|h| h == name).unwrap_or(usize::MAX);

    let c_bsn = col("bsn");
    let c_voor = col("voornamen");
    let c_tuss = col("tussenvoegsel");
    let c_ach = col("achternaam");
    let c_dob = col("geboortedatum");
    let c_land = col("geboorteland");
    let c_nat = col("nationaliteit");
    let c_str = col("straatnaam");
    let c_huis = col("huisnummer");
    let c_post = col("postcode");
    let c_woon = col("woonplaats");

    let mut records = HashMap::new();
    let mut next_id: u64 = 1;

    for result in rdr.records() {
        let row = result.unwrap();
        let bsn = row.get(c_bsn).unwrap_or("").trim().to_string();
        if bsn.is_empty() {
            continue;
        }

        let tv = |idx: usize| -> FieldValue {
            let v = row.get(idx).unwrap_or("").trim();
            if v.is_empty() {
                FieldValue::Null
            } else {
                FieldValue::Text(v.into())
            }
        };

        let r = Record::new(next_id)
            .with_source("brp")
            .insert("voornamen", tv(c_voor))
            .insert("achternaam", tv(c_ach))
            .insert("tussenvoegsel", tv(c_tuss))
            .insert("geboortedatum", tv(c_dob))
            .insert("geboorteland", tv(c_land))
            .insert("nationaliteit", tv(c_nat))
            .insert("straatnaam", tv(c_str))
            .insert("huisnummer", tv(c_huis))
            .insert("postcode", tv(c_post))
            .insert("woonplaats", tv(c_woon));

        records.insert(bsn, r);
        next_id += 1;
    }

    records
}

fn load_true_pairs(records: &HashMap<String, Record>) -> Vec<(RecordId, RecordId)> {
    let mut rdr = csv::Reader::from_path(brp_gt_csv()).expect("BRP ground truth CSV not found");
    let mut pairs = vec![];

    for result in rdr.records() {
        let row = result.unwrap();
        let bsn_a = row.get(0).unwrap_or("").trim();
        let bsn_b = row.get(1).unwrap_or("").trim();
        let is_match = row.get(2).unwrap_or("False").trim();
        if is_match != "True" {
            continue;
        }

        if let (Some(ra), Some(rb)) = (records.get(bsn_a), records.get(bsn_b)) {
            pairs.push((ra.id, rb.id));
        }
    }
    pairs
}

fn nonmatch_pairs(records: &HashMap<String, Record>, n: usize) -> Vec<(RecordId, RecordId)> {
    let mut ids: Vec<RecordId> = records.values().map(|r| r.id).collect();
    ids.sort_unstable();
    let step = (ids.len() / (n + 1)).max(1);
    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        let a = ids[i * step % ids.len()];
        let b = ids[(i * step + ids.len() / 2) % ids.len()];
        if a != b {
            pairs.push((a, b));
        }
    }
    pairs
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// DeviceComparator on CPU backend should produce the same vectors as FieldComparator.
#[test]
fn gpu_comparator_cpu_matches_field_comparator() {
    let schema = brp_schema();
    let records = load_brp_records();
    let true_pairs = load_true_pairs(&records);
    assert!(!true_pairs.is_empty(), "no true pairs loaded");

    let id_to_record: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();

    // Take first 50 true-match pairs for a fast comparison
    let sample: Vec<(Record, Record)> = true_pairs
        .iter()
        .take(50)
        .filter_map(|(a_id, b_id)| {
            let a = (*id_to_record.get(a_id)?).clone();
            let b = (*id_to_record.get(b_id)?).clone();
            Some((a, b))
        })
        .collect();

    let cpu_cmp = FieldComparator::from_schema(&schema);
    let gpu_cmp = DeviceComparator::new(shared_backend(), &schema).unwrap();

    let pool = RecordPool::from_pairs(&sample, &schema);
    let indices: Vec<(usize, usize)> = (0..sample.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let cpu_vecs = cpu_cmp.compare_batch_from_pool(&pool, &indices, &schema);
    let gpu_vecs = gpu_cmp.compare_batch_from_pool(&pool, &indices, &schema);

    assert_eq!(cpu_vecs.n_pairs, gpu_vecs.n_pairs);
    let n_fields = schema.fields.len();
    for p in 0..cpu_vecs.n_pairs {
        for f in 0..n_fields {
            let c = cpu_vecs.level(f, p);
            let g = gpu_vecs.level(f, p);
            assert_eq!(
                c, g,
                "DeviceComparator (CPU backend) diverged from FieldComparator at pair {p} field {f}"
            );
        }
    }
}

/// DeviceScorer on CPU backend should produce the same scores as FellegiSunterScorer.
#[test]
fn gpu_scorer_cpu_matches_fellegi_sunter() {
    let schema = brp_schema();
    let records = load_brp_records();
    let true_pairs = load_true_pairs(&records);
    let nonmatches = nonmatch_pairs(&records, true_pairs.len().min(200));

    let id_to_record: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();

    let all_pairs: Vec<(Record, Record)> = true_pairs
        .iter()
        .take(200)
        .chain(nonmatches.iter().take(200))
        .filter_map(|(a_id, b_id)| {
            let a = (*id_to_record.get(a_id)?).clone();
            let b = (*id_to_record.get(b_id)?).clone();
            Some((a, b))
        })
        .collect();

    // Compare with the reference implementation
    let ref_cmp = FieldComparator::from_schema(&schema);
    let ref_scorer = FellegiSunterScorer;
    let pool = RecordPool::from_pairs(&all_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..all_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors = ref_cmp.compare_batch_from_pool(&pool, &indices, &schema);
    let params = ref_scorer
        .estimate_params(&vectors, None, 50)
        .expect("EM should converge on BRP data");

    let gpu_scorer = DeviceScorer::new(shared_backend());
    let gpu_scores = gpu_scorer.score_batch(&vectors, &params);
    let ref_scores = ref_scorer.score_batch(&vectors, &params);

    assert_eq!(gpu_scores.len(), ref_scores.len());
    for (g, r) in gpu_scores.iter().zip(ref_scores.iter()) {
        assert!(
            (g.match_probability - r.match_probability).abs() < 1e-5,
            "DeviceScorer and FellegiSunterScorer diverged: gpu={}, ref={}",
            g.match_probability,
            r.match_probability
        );
    }
}

/// DeviceScorer EM on CPU backend converges with plausible params on BRP data.
#[test]
fn gpu_scorer_em_converges_on_brp_data() {
    let schema = brp_schema();
    let records = load_brp_records();
    let true_pairs = load_true_pairs(&records);
    let nonmatches = nonmatch_pairs(&records, true_pairs.len());

    let id_to_record: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();

    let all_pairs: Vec<(Record, Record)> = true_pairs
        .iter()
        .chain(nonmatches.iter())
        .filter_map(|(a_id, b_id)| {
            let a = (*id_to_record.get(a_id)?).clone();
            let b = (*id_to_record.get(b_id)?).clone();
            Some((a, b))
        })
        .collect();

    let gpu_cmp = DeviceComparator::new(shared_backend(), &schema).unwrap();
    let gpu_scorer = DeviceScorer::new(shared_backend());
    let pool = RecordPool::from_pairs(&all_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..all_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors = gpu_cmp.compare_batch_from_pool(&pool, &indices, &schema);

    let params = gpu_scorer
        .estimate_params(&vectors, None, 100)
        .expect("EM should converge");

    // m[Exact] > u[Exact] for at least half the fields
    let exact_idx = ComparisonLevel::Exact as usize;
    let fields_m_gt_u: usize = (0..schema.len())
        .filter(|&f| params.m[f][exact_idx] > params.u[f][exact_idx])
        .count();

    assert!(
        fields_m_gt_u >= schema.len() / 2,
        "EM should recover m[Exact]>u[Exact] for ≥½ fields; got {}/{}: m={:?} u={:?}",
        fields_m_gt_u,
        schema.len(),
        params.m.iter().map(|m| m[exact_idx]).collect::<Vec<_>>(),
        params.u.iter().map(|u| u[exact_idx]).collect::<Vec<_>>(),
    );
}

/// True-match pairs should produce non-None levels on key fields.
#[test]
fn true_match_pairs_have_non_none_levels() {
    let schema = brp_schema();
    let records = load_brp_records();
    let pairs = load_true_pairs(&records);
    assert!(!pairs.is_empty());

    let id_to_record: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();

    let gpu_cmp = DeviceComparator::new(shared_backend(), &schema).unwrap();

    let mut all_non_none = 0usize;
    for (a_id, b_id) in pairs.iter().take(20) {
        if let (Some(a), Some(b)) = (id_to_record.get(a_id), id_to_record.get(b_id)) {
            let cv = gpu_cmp.compare(*a, *b, &schema);
            let non_none = cv
                .levels
                .iter()
                .filter(|&&l| l != ComparisonLevel::None)
                .count();
            all_non_none += non_none;
        }
    }

    assert!(
        all_non_none > 0,
        "true-match pairs must produce at least some non-None comparison levels"
    );
}
