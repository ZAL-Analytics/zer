/// Integration test: FS comparison and EM scoring on synthetic HKS criminal
/// records data.
///
/// The HKS dataset contains ~6 400 person records with ~1 400 pairs. Most
/// ground truth pairs represent the same individual appearing under an alias
/// name vs. their registered name (`alias_vs_registered_name`).
///
/// Key things tested beyond BRP:
/// - Alias field comparison (pipe-delimited multi-value `alias_namen`)
/// - Null BSN handling (some records have no BSN)
/// - Physical descriptor fields (`lengte` Numeric, `haarkleur`/`oogkleur` Categorical)
/// - EM still converges on alias-heavy data where name fields are less reliable
use std::collections::HashMap;

use csv;
use zer_compare::{FieldComparator, FellegiSunterScorer};
use zer_core::{
    comparison::ComparisonLevel,
    record::{FieldValue, Record, RecordId},
    record_pool::RecordPool,
    schema::{FieldKind, Schema, SchemaBuilder},
    traits::{Comparator, Scorer},
};

const HKS_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/tests/hks/hks_records.csv"
);
const HKS_GT_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/tests/hks/ground_truth_pairs.csv"
);

fn hks_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen",    FieldKind::Name)
        .field("achternaam",   FieldKind::Name)
        .field("alias_namen",  FieldKind::Alias)
        .field("geboortedatum", FieldKind::Date)
        .field("geboorteland", FieldKind::Categorical)
        .field("nationaliteit", FieldKind::Categorical)
        .field("geslacht",     FieldKind::Categorical)
        .field("bsn",          FieldKind::Id)
        .field("document_nummer", FieldKind::Id)
        .field("lengte",       FieldKind::Numeric)
        .field("haarkleur",    FieldKind::Categorical)
        .field("oogkleur",     FieldKind::Categorical)
        .build()
        .unwrap()
}

fn load_hks_records() -> HashMap<String, Record> {
    let mut rdr = csv::Reader::from_path(HKS_CSV)
        .expect("HKS CSV not found, run data generator first");
    let headers = rdr.headers().unwrap().clone();
    let col     = |name: &str| headers.iter().position(|h| h == name).unwrap_or(usize::MAX);

    let c_id    = col("hks_id");
    let c_voor  = col("voornamen");
    let c_ach   = col("achternaam");
    let c_alias = col("alias_namen");
    let c_dob   = col("geboortedatum");
    let c_land  = col("geboorteland");
    let c_nat   = col("nationaliteit");
    let c_gesl  = col("geslacht");
    let c_bsn   = col("bsn");
    let c_docnr = col("document_nummer");
    let c_len   = col("lengte");
    let c_haar  = col("haarkleur");
    let c_oog   = col("oogkleur");

    let mut records = HashMap::new();
    let mut next_id: u64 = 1;

    for result in rdr.records() {
        let row = result.unwrap();
        let hks_id = row.get(c_id).unwrap_or("").trim().to_string();
        if hks_id.is_empty() { continue; }

        let tv = |idx: usize| -> FieldValue {
            let v = row.get(idx).unwrap_or("").trim();
            if v.is_empty() { FieldValue::Null } else { FieldValue::Text(v.into()) }
        };

        let r = Record::new(next_id)
            .with_source("hks")
            .insert("voornamen",      tv(c_voor))
            .insert("achternaam",     tv(c_ach))
            .insert("alias_namen",    tv(c_alias))
            .insert("geboortedatum",  tv(c_dob))
            .insert("geboorteland",   tv(c_land))
            .insert("nationaliteit",  tv(c_nat))
            .insert("geslacht",       tv(c_gesl))
            .insert("bsn",            tv(c_bsn))
            .insert("document_nummer", tv(c_docnr))
            .insert("lengte",         tv(c_len))
            .insert("haarkleur",      tv(c_haar))
            .insert("oogkleur",       tv(c_oog));

        records.insert(hks_id, r);
        next_id += 1;
    }

    records
}

fn load_true_pairs(records: &HashMap<String, Record>) -> Vec<(RecordId, RecordId)> {
    let mut rdr = csv::Reader::from_path(HKS_GT_CSV)
        .expect("HKS ground truth CSV not found");
    let mut pairs = vec![];

    for result in rdr.records() {
        let row      = result.unwrap();
        let id_a     = row.get(0).unwrap_or("").trim();
        let id_b     = row.get(1).unwrap_or("").trim();
        let is_match = row.get(2).unwrap_or("False").trim();
        if is_match != "True" { continue; }

        if let (Some(ra), Some(rb)) = (records.get(id_a), records.get(id_b)) {
            pairs.push((ra.id, rb.id));
        }
    }
    pairs
}

fn random_nonmatch_pairs(records: &HashMap<String, Record>, n: usize) -> Vec<(RecordId, RecordId)> {
    let mut ids: Vec<RecordId> = records.values().map(|r| r.id).collect();
    ids.sort_unstable();
    let step = (ids.len() / (n + 1)).max(1);
    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        let a = ids[i * step % ids.len()];
        let b = ids[(i * step + ids.len() / 2) % ids.len()];
        if a != b { pairs.push((a, b)); }
    }
    pairs
}

#[test]
fn hks_comparison_vector_correct_field_count() {
    let schema  = hks_schema();
    let records = load_hks_records();
    let pairs   = load_true_pairs(&records);
    assert!(!pairs.is_empty(), "HKS ground truth has no true pairs");

    let id_to_rec: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();
    let (a_id, b_id) = pairs[0];
    let a = (*id_to_rec[&a_id]).clone();
    let b = (*id_to_rec[&b_id]).clone();

    let cmp = FieldComparator::from_schema(&schema);
    let cv  = cmp.compare(&a, &b, &schema);

    assert_eq!(cv.levels.len(), schema.len(),
        "comparison vector should have one level per schema field");
}

#[test]
fn hks_alias_field_contributes_non_none_for_alias_pairs() {
    let schema  = hks_schema();
    let records = load_hks_records();
    let pairs   = load_true_pairs(&records);
    assert!(!pairs.is_empty());

    let id_to_rec: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();
    let cmp        = FieldComparator::from_schema(&schema);

    // Find the alias_namen field index
    let alias_idx = schema.fields.iter().position(|f| f.name == "alias_namen").unwrap();

    // Count how many true pairs have a non-None alias comparison level
    let mut non_none_alias = 0usize;
    let sample = pairs.iter().take(200);
    let total  = sample.clone().count();

    for &(a_id, b_id) in sample {
        let a  = (*id_to_rec[&a_id]).clone();
        let b  = (*id_to_rec[&b_id]).clone();
        let cv = cmp.compare(&a, &b, &schema);
        if cv.levels[alias_idx] != ComparisonLevel::None {
            non_none_alias += 1;
        }
    }

    let alias_signal = non_none_alias as f64 / total as f64;
    println!(
        "HKS alias field non-None rate on true pairs: {:.3} ({}/{})",
        alias_signal, non_none_alias, total
    );

    // At least 20% of alias_vs_registered_name pairs should show alias overlap
    assert!(
        alias_signal >= 0.20,
        "alias field should contribute signal for at least 20% of true pairs, got {:.3}",
        alias_signal
    );
}

#[test]
fn hks_null_bsn_handled_gracefully() {
    let schema  = hks_schema();
    let records = load_hks_records();
    let pairs   = load_true_pairs(&records);
    assert!(!pairs.is_empty());

    let id_to_rec: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();
    let cmp = FieldComparator::from_schema(&schema);

    // Processing must not panic, even when bsn is Null in one or both records
    for &(a_id, b_id) in pairs.iter().take(50) {
        let a  = (*id_to_rec[&a_id]).clone();
        let b  = (*id_to_rec[&b_id]).clone();
        let cv = cmp.compare(&a, &b, &schema); // should not panic
        assert_eq!(cv.levels.len(), schema.len());
    }
}

#[test]
fn hks_em_converges_on_alias_data() {
    let schema       = hks_schema();
    let records      = load_hks_records();
    let true_pairs   = load_true_pairs(&records);
    assert!(!records.is_empty() && !true_pairs.is_empty());

    let id_to_rec: HashMap<RecordId, &Record> = records.values().map(|r| (r.id, r)).collect();
    let nonmatch     = random_nonmatch_pairs(&records, true_pairs.len());

    let all_pairs: Vec<(Record, Record)> = true_pairs.iter().chain(nonmatch.iter())
        .filter_map(|(a_id, b_id)| {
            let a = (*id_to_rec.get(a_id)?).clone();
            let b = (*id_to_rec.get(b_id)?).clone();
            Some((a, b))
        })
        .collect();

    let cmp     = FieldComparator::from_schema(&schema);
    let pool    = RecordPool::from_pairs(&all_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..all_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let vectors = cmp.compare_batch_from_pool(&pool, &indices, &schema);

    let scorer = FellegiSunterScorer;
    let params = scorer.estimate_params(&vectors, None, 100)
        .expect("EM should converge on HKS data");

    // At least the date and name fields should have m[Exact] > u[Exact]
    let exact_idx = ComparisonLevel::Exact as usize;
    let n_fields_ok = (0..schema.len())
        .filter(|&f| params.m[f][exact_idx] > params.u[f][exact_idx])
        .count();

    assert!(
        n_fields_ok >= schema.len() / 3,
        "EM should recover m[Exact] > u[Exact] for at least 1/3 of fields; got {}/{}: {:?}",
        n_fields_ok, schema.len(),
        (0..schema.len()).map(|f| (params.m[f][exact_idx], params.u[f][exact_idx])).collect::<Vec<_>>()
    );
}
