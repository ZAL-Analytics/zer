/// Integration test: blocking recall on synthetic SIS II person data.
///
/// Ground truth: `ground_truth_alias_pairs.csv` lists pairs of SIS II entries
/// that represent the same person under different name forms (alias romanization
/// variants, name transpositions, etc.).
///
/// The test builds an index with three complementary blocking keys:
/// - `PhoneticNameDobKey` , surname phonetic + DOB year (primary name)
/// - `AliasPhoneticKey`   , surname phonetic + DOB year (alias field)
/// - `FuzzyYearKey`       , ±1 year window for estimated (Jan-1) DOBs
///
/// Target: recall ≥ 0.85, at least 85% of true alias pairs share a key bucket.
use std::collections::{HashMap, HashSet};

use csv::Reader;
use zer_blocking::{
    CompositeBlocker, InvertedIndex,
    keys::{AliasPhoneticKey, FuzzyYearKey, PhoneticNameDobKey},
};
use zer_core::{
    record::{Record, RecordId},
    schema::{FieldKind, Schema, SchemaBuilder},
    traits::Blocker,
};

const SIS_PERSONS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/tests/sis/sis_persons.csv"
);
const SIS_GROUND_TRUTH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/tests/sis/ground_truth_alias_pairs.csv"
);

fn sis_schema() -> Schema {
    SchemaBuilder::new()
        .field("achternaam",      FieldKind::Name)
        .field("voornamen",       FieldKind::Name)
        .field("alias_namen",     FieldKind::Alias)
        .field("geboortedatum",   FieldKind::Date)
        .field("document_nummer", FieldKind::Id)
        .build()
        .unwrap()
}

fn load_persons() -> (Vec<Record>, HashMap<String, RecordId>) {
    let mut rdr      = Reader::from_path(SIS_PERSONS).expect("SIS persons CSV not found");
    let mut records  = vec![];
    let mut id_map: HashMap<String, RecordId> = HashMap::new();
    let mut next_id: u64 = 1;

    for result in rdr.records() {
        let row = result.expect("CSV read error");
        let sis_id      = row.get(0).unwrap_or("").to_string();
        let achternaam  = row.get(3).unwrap_or("").to_string();
        let voornamen   = row.get(2).unwrap_or("").to_string();
        let alias_namen = row.get(4).unwrap_or("").to_string();
        let geboortedatum = row.get(5).unwrap_or("").to_string();
        let document_nummer = row.get(11).unwrap_or("").to_string();

        if sis_id.is_empty() { continue; }

        let r = Record::new(next_id)
            .insert("achternaam",      achternaam)
            .insert("voornamen",       voornamen)
            .insert("alias_namen",     alias_namen)
            .insert("geboortedatum",   geboortedatum)
            .insert("document_nummer", document_nummer);

        id_map.insert(sis_id, next_id);
        records.push(r);
        next_id += 1;
    }

    (records, id_map)
}

fn load_true_pairs(id_map: &HashMap<String, RecordId>) -> Vec<(RecordId, RecordId)> {
    let mut rdr  = Reader::from_path(SIS_GROUND_TRUTH).expect("SIS ground truth CSV not found");
    let mut pairs = vec![];

    for result in rdr.records() {
        let row = result.expect("CSV read error");
        let sis_id_a = row.get(0).unwrap_or("");
        let sis_id_b = row.get(1).unwrap_or("");
        let is_match = row.get(2).unwrap_or("False");
        if is_match != "True" { continue; }

        if let (Some(&a), Some(&b)) = (id_map.get(sis_id_a), id_map.get(sis_id_b)) {
            pairs.push((a, b));
        }
    }
    pairs
}

#[test]
fn blocking_recall_sis_alias_pairs() {
    let schema             = sis_schema();
    let (records, id_map)  = load_persons();
    let true_pairs         = load_true_pairs(&id_map);

    assert!(!records.is_empty(),    "SIS persons CSV produced no records");
    assert!(!true_pairs.is_empty(), "SIS ground truth has no true pairs");

    let blocker = CompositeBlocker::new()
        .add(PhoneticNameDobKey::new("achternaam", "geboortedatum"))
        .add(AliasPhoneticKey::new("alias_namen", "geboortedatum"))
        .add(FuzzyYearKey::new("achternaam", "geboortedatum", 1));

    let mut idx = InvertedIndex::new();
    let record_map: HashMap<RecordId, &Record> =
        records.iter().map(|r| (r.id, r)).collect();

    for record in &records {
        blocker.index_record(record, &schema, &mut idx);
    }

    let mut found = 0usize;
    for (a_id, b_id) in &true_pairs {
        if let Some(rec_a) = record_map.get(a_id) {
            let candidates: HashSet<RecordId> =
                blocker.candidates(rec_a, &schema, &idx).into_iter().collect();
            if candidates.contains(b_id) {
                found += 1;
            }
        }
    }

    let recall = found as f64 / true_pairs.len() as f64;
    println!(
        "SIS II blocking recall: {:.4} ({}/{} alias pairs found, {} records)",
        recall, found, true_pairs.len(), records.len()
    );

    assert!(
        recall >= 0.85,
        "SIS II recall {:.4} below 0.85, alias + phonetic blocking should catch name variants",
        recall
    );
}

#[test]
fn sis_no_self_candidates() {
    let schema            = sis_schema();
    let (records, _)      = load_persons();

    let blocker = CompositeBlocker::new()
        .add(PhoneticNameDobKey::new("achternaam", "geboortedatum"));

    let mut idx = InvertedIndex::new();
    for record in &records {
        blocker.index_record(record, &schema, &mut idx);
    }

    for record in &records {
        let cands = blocker.candidates(record, &schema, &idx);
        assert!(
            !cands.contains(&record.id),
            "Record {} appears in its own candidate list",
            record.id
        );
    }
}
