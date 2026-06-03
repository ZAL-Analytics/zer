/// Integration test: blocking recall on the synthetic KvK Director Extract.
///
/// Measures what fraction of known true-match pairs (same director appearing
/// in multiple company records) share at least one blocking key after indexing.
/// Target: recall ≥ 0.97.
use std::collections::HashMap;

use csv;
use zer_blocking::{BlockerFactory, CustomSchemaCategory, InvertedIndex};
use zer_core::{
    record::Record,
    schema::{FieldKind, Schema, SchemaBuilder},
    traits::Blocker,
};

fn kvk_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(
        env!("CARGO_MANIFEST_DIR"),
        "tests/kvk/kvk_director_flat.csv",
    )
}
fn gt_csv() -> std::path::PathBuf {
    zer_test_utils::dataset_path(
        env!("CARGO_MANIFEST_DIR"),
        "tests/kvk/ground_truth_pairs.csv",
    )
}

fn kvk_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("geboortedatum", FieldKind::Date)
        .field("woonplaats", FieldKind::Address)
        .field("straatnaam", FieldKind::Address)
        .field("postcode", FieldKind::Id)
        .field("kvkNummer", FieldKind::Id)
        .build()
        .unwrap()
}

fn load_kvk_records() -> HashMap<u64, Record> {
    let mut rdr =
        csv::Reader::from_path(kvk_csv()).expect("KvK CSV not found, run data generator first");
    let headers_row = rdr.headers().unwrap().clone();
    let col = |name: &str| headers_row.iter().position(|h| h == name);

    let c_kvk = col("kvkNummer").unwrap();
    let c_voor = col("voornamen").unwrap();
    let c_tuss = col("tussenvoegsel").unwrap();
    let c_ach = col("achternaam").unwrap();
    let c_dob = col("geboortedatum").unwrap();
    let c_woon = col("woonplaats").unwrap();
    let c_str = col("straatnaam").unwrap();
    let c_post = col("postcode").unwrap();

    let mut records = HashMap::new();
    for result in rdr.records() {
        let row = result.unwrap();
        let kvk: u64 = row[c_kvk].parse().unwrap();

        let opt = |idx: usize| -> Option<&str> {
            let v = row[idx].trim();
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        };

        let r = Record::new(kvk)
            .with_source("kvk")
            .insert("voornamen", opt(c_voor))
            .insert("tussenvoegsel", opt(c_tuss))
            .insert("achternaam", opt(c_ach))
            .insert("geboortedatum", opt(c_dob))
            .insert("woonplaats", opt(c_woon))
            .insert("straatnaam", opt(c_str))
            .insert("postcode", opt(c_post))
            .insert("kvkNummer", row[c_kvk].trim());

        records.insert(kvk, r);
    }
    records
}

fn load_true_pairs() -> Vec<(u64, u64)> {
    let mut rdr = csv::Reader::from_path(gt_csv()).expect("Ground truth CSV not found");
    let mut pairs = vec![];
    for result in rdr.records() {
        let row = result.unwrap();
        if row.get(2).unwrap_or("") == "True" {
            let a: u64 = row[0].parse().unwrap();
            let b: u64 = row[1].parse().unwrap();
            pairs.push((a, b));
        }
    }
    pairs
}

#[test]
fn blocking_recall_kvk_above_threshold() {
    let schema = kvk_schema();
    let records = load_kvk_records();
    let true_pairs = load_true_pairs();

    assert!(!true_pairs.is_empty(), "Ground truth must not be empty");

    let blocker = BlockerFactory::from_schema(&schema);
    let mut idx = InvertedIndex::new();

    for record in records.values() {
        blocker.index_record(record, &schema, &mut idx);
    }

    let mut found = 0usize;
    for (a_id, b_id) in &true_pairs {
        if let Some(rec_a) = records.get(a_id) {
            let candidates = blocker.candidates(rec_a, &schema, &idx);
            if candidates.contains(b_id) {
                found += 1;
            }
        }
    }

    let recall = found as f64 / true_pairs.len() as f64;
    println!(
        "KvK blocking recall: {:.4} ({}/{} pairs found)",
        recall,
        found,
        true_pairs.len()
    );

    assert!(
        recall >= 0.97,
        "Blocking recall {:.4} is below the 0.97 target ({}/{} pairs found)",
        recall,
        found,
        true_pairs.len()
    );
}

#[test]
fn blocking_reduction_ratio_kvk() {
    let schema = kvk_schema();
    let records = load_kvk_records();
    let n = records.len() as f64;

    let blocker = BlockerFactory::from_schema(&schema);
    let mut idx = InvertedIndex::new();

    for record in records.values() {
        blocker.index_record(record, &schema, &mut idx);
    }

    let total_candidates: usize = records
        .values()
        .map(|r| blocker.candidates(r, &schema, &idx).len())
        .sum::<usize>()
        / 2; // each pair counted twice

    let all_possible = (n * (n - 1.0) / 2.0) as usize;
    let reduction = 1.0 - total_candidates as f64 / all_possible as f64;

    println!(
        "KvK reduction ratio: {:.4} ({} candidate pairs from {} possible)",
        reduction, total_candidates, all_possible
    );

    assert!(
        reduction >= 0.90,
        "Reduction ratio {:.4} is below the 0.90 threshold on synthetic data",
        reduction
    );
}

/// Verify that a `CustomSchemaCategory` assembled to mirror `PersonRegistry`
/// achieves the same ≥ 0.97 recall as `BlockerFactory::from_schema`.
///
/// The KvK schema has: Name (voornamen, achternaam), Categorical (tussenvoegsel),
/// Date (geboortedatum), Address (woonplaats, straatnaam), Id (postcode, kvkNummer).
/// The equivalent custom category uses:
///   - with_phonetic_name_dob()  uses PhoneticNameDobKey(achternaam, geboortedatum)
///   - with_address_initial()    uses AddressInitialKey(woonplaats, voornamen)
///   - with_id_suffix(4)         uses SuffixKey(postcode, 4) + SuffixKey(kvkNummer, 4)
///   - with_exact_categorical()  uses ExactFieldKey(tussenvoegsel)
#[test]
fn blocking_recall_custom_category_kvk_above_threshold() {
    let schema = kvk_schema();
    let records = load_kvk_records();
    let true_pairs = load_true_pairs();

    assert!(!true_pairs.is_empty(), "Ground truth must not be empty");

    let category = CustomSchemaCategory::new()
        .with_phonetic_name_dob()
        .with_address_initial()
        .with_id_suffix(4)
        .with_exact_categorical();

    let blocker = BlockerFactory::from_custom_category(&schema, category);
    let mut idx = InvertedIndex::new();

    for record in records.values() {
        blocker.index_record(record, &schema, &mut idx);
    }

    let mut found = 0usize;
    for (a_id, b_id) in &true_pairs {
        if let Some(rec_a) = records.get(a_id) {
            let candidates = blocker.candidates(rec_a, &schema, &idx);
            if candidates.contains(b_id) {
                found += 1;
            }
        }
    }

    let recall = found as f64 / true_pairs.len() as f64;
    println!(
        "KvK custom-category blocking recall: {:.4} ({}/{} pairs found)",
        recall,
        found,
        true_pairs.len()
    );

    assert!(
        recall >= 0.97,
        "Custom-category recall {:.4} is below the 0.97 target ({}/{} pairs found)",
        recall,
        found,
        true_pairs.len()
    );
}

/// Verify that a `CustomSchemaCategory` with only `with_id_suffix(4)` still
/// produces no self-candidates (no record appears in its own candidate list).
#[test]
fn no_self_candidates_custom_category_kvk() {
    let schema = kvk_schema();
    let records = load_kvk_records();

    let category = CustomSchemaCategory::new().with_id_suffix(4);
    let blocker = BlockerFactory::from_custom_category(&schema, category);
    let mut idx = InvertedIndex::new();

    for record in records.values() {
        blocker.index_record(record, &schema, &mut idx);
    }

    for record in records.values() {
        let cands = blocker.candidates(record, &schema, &idx);
        assert!(
            !cands.contains(&record.id),
            "Record {} appears in its own candidate list (custom category)",
            record.id
        );
    }
}

#[test]
fn no_self_candidates_kvk() {
    let schema = kvk_schema();
    let records = load_kvk_records();

    let blocker = BlockerFactory::from_schema(&schema);
    let mut idx = InvertedIndex::new();

    for record in records.values() {
        blocker.index_record(record, &schema, &mut idx);
    }

    for record in records.values() {
        let cands = blocker.candidates(record, &schema, &idx);
        assert!(
            !cands.contains(&record.id),
            "Record {} appears in its own candidate list",
            record.id
        );
    }
}
