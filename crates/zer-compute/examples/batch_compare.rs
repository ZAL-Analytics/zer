//! Example: GPU-accelerated batch comparison on BRP data.
//!
//! Demonstrates the full entity-resolution pipeline:
//!   1. Load BRP person records from CSV.
//!   2. Load ground-truth duplicate pairs.
//!   3. Run `DeviceComparator::compare_batch` on all pairs.
//!   4. Run `DeviceScorer::estimate_params` (EM) to learn m / u parameters.
//!   5. Score all pairs and print precision / recall.
//!
//! The CPU backend is used by default (no GPU toolchain required).  With the
//! `cuda` feature enabled, large batches will automatically dispatch to the GPU.
//!
//! Run with:
//!   cargo run -p zer-compute --example batch_compare
//!
//! Expected output (approximate, CPU backend):
//!   Loaded  10000 BRP records and 1000 true-match pairs
//!   compare_batch: 2000 pairs in ~Xms
//!   EM converged in ≤100 iterations
//!   precision=0.8xx  recall=0.8xx  (TP=xxx FP=xx FN=xxx)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use zer_compute::{DeviceComparator, DeviceScorer, GpuBackend};
use zer_core::{
    record::{FieldValue, Record, RecordId},
    record_pool::RecordPool,
    schema::{FieldKind, SchemaBuilder},
    scoring::MatchBand,
    traits::{Comparator, Scorer},
};

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

fn brp_schema() -> zer_core::schema::Schema {
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

// ── Helpers ───────────────────────────────────────────────────────────────────

fn load_records() -> HashMap<String, Record> {
    let mut rdr = csv::Reader::from_path(brp_csv())
        .expect("tests/brp/brp_persons.csv not found, run the data generator first");
    let headers = rdr.headers().unwrap().clone();
    let col = |n: &str| headers.iter().position(|h| h == n).unwrap_or(usize::MAX);

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

    let mut out = HashMap::new();
    let mut id: u64 = 1;
    for row in rdr.records().flatten() {
        let bsn = row.get(c_bsn).unwrap_or("").trim().to_string();
        if bsn.is_empty() {
            continue;
        }
        let tv = |i: usize| -> FieldValue {
            let v = row.get(i).unwrap_or("").trim();
            if v.is_empty() {
                FieldValue::Null
            } else {
                FieldValue::Text(v.into())
            }
        };
        let r = Record::new(id)
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
        out.insert(bsn, r);
        id += 1;
    }
    out
}

fn load_true_pairs(records: &HashMap<String, Record>) -> Vec<(RecordId, RecordId)> {
    let mut rdr =
        csv::Reader::from_path(brp_gt_csv()).expect("brp_small/ground_truth_pairs.csv not found");
    let mut out = vec![];
    for row in rdr.records().flatten() {
        let bsn_a = row.get(0).unwrap_or("").trim();
        let bsn_b = row.get(1).unwrap_or("").trim();
        let is_match = row.get(2).unwrap_or("False").trim();
        if is_match != "True" {
            continue;
        }
        if let (Some(ra), Some(rb)) = (records.get(bsn_a), records.get(bsn_b)) {
            out.push((ra.id, rb.id));
        }
    }
    out
}

/// Build N deterministic non-matching pairs from the record pool.
fn nonmatch_pairs(records: &HashMap<String, Record>, n: usize) -> Vec<(RecordId, RecordId)> {
    let mut ids: Vec<RecordId> = records.values().map(|r| r.id).collect();
    ids.sort_unstable();
    let step = (ids.len() / (n + 1)).max(1);
    (0..n)
        .filter_map(|i| {
            let a = ids[i * step % ids.len()];
            let b = ids[(i * step + ids.len() / 2) % ids.len()];
            if a != b {
                Some((a, b))
            } else {
                None
            }
        })
        .collect()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    // ── 1. Setup ─────────────────────────────────────────────────────────────
    let schema = brp_schema();
    let records = load_records();
    let id_map: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();

    let true_pairs = load_true_pairs(&records);
    let nonmatch_ids = nonmatch_pairs(&records, true_pairs.len());
    let match_count = true_pairs.len();

    println!(
        "Loaded {:>6} BRP records and {:>5} true-match pairs",
        records.len(),
        match_count
    );

    // Build pair slices (true matches first, non-matches after)
    let all_pairs: Vec<(Record, Record)> = true_pairs
        .iter()
        .chain(nonmatch_ids.iter())
        .filter_map(|(a_id, b_id)| {
            let a = (*id_map.get(a_id)?).clone();
            let b = (*id_map.get(b_id)?).clone();
            Some((a, b))
        })
        .collect();

    // ── 2. Build backend + comparator ────────────────────────────────────────
    let backend = Arc::new(GpuBackend::auto_detect());
    println!("Backend          : {}", backend.name());

    let comparator = DeviceComparator::new(Arc::clone(&backend), &schema).unwrap();
    let scorer = DeviceScorer::new(Arc::clone(&backend));

    // ── 3. Compare ───────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let pool = RecordPool::from_pairs(&all_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..all_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors = comparator.compare_batch_from_pool(&pool, &indices, &schema);
    let cmp_ms = t0.elapsed().as_millis();
    println!(
        "compare_batch    : {} pairs in {}ms  ({:.0} k-pairs/s)",
        vectors.n_pairs,
        cmp_ms,
        vectors.n_pairs as f64 / (cmp_ms as f64 + 1.0)
    );

    // ── 4. EM parameter estimation ───────────────────────────────────────────
    let t1 = Instant::now();
    let params = scorer
        .estimate_params(&vectors, None, 100)
        .expect("EM should converge");
    let em_ms = t1.elapsed().as_millis();
    println!("estimate_params  : {}ms", em_ms);

    // Show estimated m / u for the first two fields (names)
    for (f, field) in schema.fields.iter().enumerate().take(2) {
        println!(
            "  {} : m[Exact]={:.3}  u[Exact]={:.3}",
            field.name, params.m[f][3], params.u[f][3]
        );
    }

    // ── 5. Score and evaluate ─────────────────────────────────────────────────
    let scored = scorer.score_batch(&vectors, &params);

    let tp = scored[..match_count]
        .iter()
        .filter(|s| s.band == MatchBand::AutoMatch)
        .count();
    let fp = scored[match_count..]
        .iter()
        .filter(|s| s.band == MatchBand::AutoMatch)
        .count();
    let fn_ = match_count - tp;

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!(
        "\nprecision={:.3}  recall={:.3}  F1={:.3}  (TP={} FP={} FN={})",
        precision, recall, f1, tp, fp, fn_
    );
}
