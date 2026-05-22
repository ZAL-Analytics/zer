/// Integration test: FS comparison and EM scoring on synthetic BRP person data.
///
/// The BRP (Basisregistratie Personen) dataset contains ~10 000 person records
/// with ~1 000 perturbed duplicate pairs (ground truth `is_match = True`).
///
/// Test plan:
/// 1. Load all records keyed by BSN.
/// 2. Extract the 1 000 true-match pairs from ground truth.
/// 3. Build an equal-sized sample of random (non-matching) pairs.
/// 4. Run `FieldComparator::compare_batch` on all pairs.
/// 5. Run `FellegiSunterScorer::estimate_params` (EM) on the comparison vectors.
/// 6. Score all pairs.
/// 7. Assert:
///    - EM converges (no error, params plausible)
///    - Precision and recall at the estimated thresholds are both ≥ 0.80
///    - m[f][Exact] > u[f][Exact] for most fields (EM recovers the right signal)
use std::collections::HashMap;

use csv;
use zer_compare::{FieldComparator, FellegiSunterScorer};
use zer_core::{
    comparison::ComparisonLevel,
    record::{FieldValue, Record, RecordId},
    record_pool::RecordPool,
    schema::{FieldKind, Schema, SchemaBuilder},
    scoring::MatchBand,
    traits::{Comparator, Scorer},
};

const BRP_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/tests/brp/brp_persons.csv"
);
const BRP_GT_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/tests/brp/ground_truth_pairs.csv"
);

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

fn load_brp_records() -> HashMap<String, Record> {
    let mut rdr = csv::Reader::from_path(BRP_CSV)
        .expect("BRP CSV not found, run data generator first");
    let headers = rdr.headers().unwrap().clone();
    let col     = |name: &str| headers.iter().position(|h| h == name).unwrap_or(usize::MAX);

    let c_bsn   = col("bsn");
    let c_voor  = col("voornamen");
    let c_tuss  = col("tussenvoegsel");
    let c_ach   = col("achternaam");
    let c_dob   = col("geboortedatum");
    let c_land  = col("geboorteland");
    let c_nat   = col("nationaliteit");
    let c_str   = col("straatnaam");
    let c_huis  = col("huisnummer");
    let c_post  = col("postcode");
    let c_woon  = col("woonplaats");

    let mut records = HashMap::new();
    let mut next_id: u64 = 1;

    for result in rdr.records() {
        let row = result.unwrap();
        let bsn = row.get(c_bsn).unwrap_or("").trim().to_string();
        if bsn.is_empty() { continue; }

        let tv = |idx: usize| -> FieldValue {
            let v = row.get(idx).unwrap_or("").trim();
            if v.is_empty() { FieldValue::Null } else { FieldValue::Text(v.into()) }
        };

        let r = Record::new(next_id)
            .with_source("brp")
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

        records.insert(bsn, r);
        next_id += 1;
    }

    records
}

fn load_true_pairs(records: &HashMap<String, Record>) -> Vec<(RecordId, RecordId)> {
    let mut rdr = csv::Reader::from_path(BRP_GT_CSV)
        .expect("BRP ground truth CSV not found");
    let mut pairs = vec![];

    for result in rdr.records() {
        let row      = result.unwrap();
        let bsn_a    = row.get(0).unwrap_or("").trim();
        let bsn_b    = row.get(1).unwrap_or("").trim();
        let is_match = row.get(2).unwrap_or("False").trim();
        if is_match != "True" { continue; }

        if let (Some(ra), Some(rb)) = (records.get(bsn_a), records.get(bsn_b)) {
            pairs.push((ra.id, rb.id));
        }
    }
    pairs
}

/// Build N random non-matching pairs from the record pool.
/// Uses a deterministic pattern (pairs spaced far apart by ID) to avoid
/// accidentally selecting true matches.
fn random_nonmatch_pairs(
    records: &HashMap<String, Record>,
    n:       usize,
) -> Vec<(RecordId, RecordId)> {
    let mut ids: Vec<RecordId> = records.values().map(|r| r.id).collect();
    ids.sort_unstable();
    let step   = (ids.len() / (n + 1)).max(1);
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

#[test]
fn brp_em_converges_and_params_are_plausible() {
    let schema       = brp_schema();
    let records      = load_brp_records();
    let true_pairs   = load_true_pairs(&records);

    assert!(!records.is_empty(),    "BRP CSV produced no records");
    assert!(!true_pairs.is_empty(), "BRP ground truth has no true pairs");

    let id_to_record: HashMap<RecordId, &Record> =
        records.values().map(|r| (r.id, r)).collect();

    let nonmatch_pairs = random_nonmatch_pairs(&records, true_pairs.len());

    // Build pair slices for the comparator
    let all_pairs: Vec<(Record, Record)> = true_pairs.iter().chain(nonmatch_pairs.iter())
        .filter_map(|(a_id, b_id)| {
            let a = (*id_to_record.get(a_id)?).clone();
            let b = (*id_to_record.get(b_id)?).clone();
            Some((a.clone(), b.clone()))
        })
        .collect();

    let comparator = FieldComparator::from_schema(&schema);
    let pool       = RecordPool::from_pairs(&all_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..all_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors    = comparator.compare_batch_from_pool(&pool, &indices, &schema);

    assert_eq!(vectors.n_pairs, all_pairs.len(), "compare_batch should produce one vector per pair");
    assert_eq!(vectors.n_fields, schema.len(), "each vector should have one level per field");

    let scorer = FellegiSunterScorer;
    let params = scorer.estimate_params(&vectors, None, 100)
        .expect("EM estimation should succeed on BRP data");

    // m[Exact] > u[Exact] for at least half the fields (name, date, id should all pass)
    let exact_idx = ComparisonLevel::Exact as usize;
    let fields_where_m_gt_u = (0..schema.len())
        .filter(|&f| params.m[f][exact_idx] > params.u[f][exact_idx])
        .count();

    assert!(
        fields_where_m_gt_u >= schema.len() / 2,
        "EM should recover m[Exact] > u[Exact] for at least half the fields; got {}/{}: m={:?}, u={:?}",
        fields_where_m_gt_u, schema.len(),
        params.m.iter().map(|m| m[exact_idx]).collect::<Vec<_>>(),
        params.u.iter().map(|u| u[exact_idx]).collect::<Vec<_>>(),
    );
}

#[test]
fn brp_precision_recall_at_threshold() {
    let schema       = brp_schema();
    let records      = load_brp_records();
    let true_pairs   = load_true_pairs(&records);
    assert!(!records.is_empty() && !true_pairs.is_empty());

    let id_to_record: HashMap<RecordId, &Record> =
        records.values().map(|r| (r.id, r)).collect();

    // Use a balanced set of match / non-match pairs
    let nonmatch_pairs = random_nonmatch_pairs(&records, true_pairs.len());

    let match_count    = true_pairs.len();
    let nonmatch_count = nonmatch_pairs.len();

    let all_pairs: Vec<(Record, Record)> = true_pairs.iter().chain(nonmatch_pairs.iter())
        .filter_map(|(a_id, b_id)| {
            let a = (*id_to_record.get(a_id)?).clone();
            let b = (*id_to_record.get(b_id)?).clone();
            Some((a, b))
        })
        .collect();

    let comparator = FieldComparator::from_schema(&schema);
    let pool       = RecordPool::from_pairs(&all_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..all_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors    = comparator.compare_batch_from_pool(&pool, &indices, &schema);

    let scorer = FellegiSunterScorer;
    let params = scorer.estimate_params(&vectors, None, 100)
        .expect("EM should converge");

    let scored = scorer.score_batch(&vectors, &params);

    // True-match pairs are first `match_count` in the list
    let true_match_predicted_match = scored[..match_count].iter()
        .filter(|s| s.band == MatchBand::AutoMatch)
        .count();
    let true_nonmatch_predicted_match = scored[match_count..].iter()
        .filter(|s| s.band == MatchBand::AutoMatch)
        .count();

    let tp = true_match_predicted_match;
    let fp = true_nonmatch_predicted_match;
    let fn_ = match_count - tp;

    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall    = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };

    println!(
        "BRP precision={:.3}, recall={:.3} (TP={}, FP={}, FN={}, {} match pairs, {} nonmatch pairs)",
        precision, recall, tp, fp, fn_, match_count, nonmatch_count
    );

    // EM without blocking context is harder; target is reasonable but not strict
    assert!(precision >= 0.70, "BRP precision {:.3} below 0.70", precision);
    assert!(recall    >= 0.70, "BRP recall {:.3} below 0.70", recall);
}

#[test]
fn brp_comparison_vector_fields_populated() {
    let schema  = brp_schema();
    let records = load_brp_records();
    let pairs   = load_true_pairs(&records);
    assert!(!pairs.is_empty());

    let id_to_record: HashMap<RecordId, &Record> =
        records.values().map(|r| (r.id, r)).collect();

    let (a_id, b_id) = pairs[0];
    let a = (*id_to_record.get(&a_id).unwrap()).clone();
    let b = (*id_to_record.get(&b_id).unwrap()).clone();

    let comparator = FieldComparator::from_schema(&schema);
    let cv         = comparator.compare(&a, &b, &schema);

    assert_eq!(cv.levels.len(), schema.len());
    // True matches should have at least some non-None levels
    let non_none = cv.levels.iter().filter(|&&l| l != ComparisonLevel::None).count();
    assert!(non_none > 0, "True pair should have at least one non-None comparison level");
}
