/// Integration test: blocking recall on synthetic ANPR vehicle passage data.
///
/// Ground truth: `ground_truth_vehicle_pairs.csv` lists passage records where
/// the OCR system read the license plate with a single-character confusion
/// (0/O, 1/I, 8/B, 5/S, 2/Z).
///
/// The test verifies that `PlateOCRFuzzyKey` causes the OCR-confused passage
/// to be a candidate of at least one passage that carries the true plate.
///
/// Target: recall ≥ 0.97, the OCR confusion pairs in the fuzzy key table
/// cover all injected confusions, so recall should approach 1.0.
use std::collections::{HashMap, HashSet};

use csv::Reader;
use zer_blocking::{
    CompositeBlocker, InvertedIndex,
    keys::{LicensePlateNormKey, PlateOCRFuzzyKey},
};
use zer_core::{
    record::{Record, RecordId},
    schema::{FieldKind, Schema, SchemaBuilder},
    traits::Blocker,
};

fn anpr_passages() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "tests/anpr/anpr_passages.csv")
}
fn anpr_ground_truth() -> std::path::PathBuf {
    zer_test_utils::dataset_path(env!("CARGO_MANIFEST_DIR"), "tests/anpr/ground_truth_vehicle_pairs.csv")
}

fn anpr_schema() -> Schema {
    SchemaBuilder::new()
        .field("kenteken",  FieldKind::LicensePlate)
        .field("camera_id", FieldKind::Categorical)
        .field("tijdstip",  FieldKind::Timestamp)
        .field("lat",       FieldKind::GpsCoordinate)
        .field("lon",       FieldKind::GpsCoordinate)
        .build()
        .unwrap()
}

fn norm_plate(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

fn load_passages() -> (
    Vec<Record>,
    HashMap<String, RecordId>,
    HashMap<String, Vec<RecordId>>,
) {
    let mut rdr       = Reader::from_path(anpr_passages()).expect("ANPR passages CSV not found");
    let mut records   = vec![];
    let mut pid_map:   HashMap<String, RecordId>     = HashMap::new();
    let mut plate_map: HashMap<String, Vec<RecordId>> = HashMap::new();
    let mut next_id: u64 = 1;

    for result in rdr.records() {
        let row        = result.expect("CSV read error");
        let passage_id = row.get(0).unwrap_or("").to_string();
        let tijdstip   = row.get(1).unwrap_or("").to_string();
        let camera_id  = row.get(2).unwrap_or("").to_string();
        let lat        = row.get(4).unwrap_or("").to_string();
        let lon        = row.get(5).unwrap_or("").to_string();
        let kenteken   = row.get(7).unwrap_or("").to_string();

        if passage_id.is_empty() { continue; }

        let norm = norm_plate(&kenteken);
        let r = Record::new(next_id)
            .insert("kenteken",  kenteken)
            .insert("camera_id", camera_id)
            .insert("tijdstip",  tijdstip)
            .insert("lat",       lat)
            .insert("lon",       lon);
        if !norm.is_empty() {
            plate_map.entry(norm).or_default().push(next_id);
        }
        pid_map.insert(passage_id, next_id);
        records.push(r);
        next_id += 1;
    }

    (records, pid_map, plate_map)
}

#[test]
fn blocking_recall_anpr_ocr_confusion() {
    let schema = anpr_schema();
    let (records, pid_map, plate_map) = load_passages();

    assert!(!records.is_empty(), "ANPR passages CSV produced no records");

    let blocker = CompositeBlocker::new()
        .add(LicensePlateNormKey::new("kenteken"))
        .add(PlateOCRFuzzyKey::new("kenteken"));

    let mut idx = InvertedIndex::new();
    let record_map: HashMap<RecordId, &Record> =
        records.iter().map(|r| (r.id, r)).collect();

    for record in &records {
        blocker.index_record(record, &schema, &mut idx);
    }

    let mut rdr   = Reader::from_path(anpr_ground_truth()).expect("ANPR ground truth CSV not found");
    let mut total = 0usize;
    let mut found = 0usize;

    for result in rdr.records() {
        let row           = result.expect("CSV read error");
        let passage_id_a  = row.get(0).unwrap_or("");
        let kenteken_true = row.get(1).unwrap_or("");
        let is_match      = row.get(3).unwrap_or("False");
        if is_match != "True" { continue; }

        let ocr_id = match pid_map.get(passage_id_a) {
            Some(&id) => id,
            None      => continue,
        };
        let true_norm = norm_plate(kenteken_true);
        let true_ids: Vec<RecordId> = plate_map.get(&true_norm).cloned().unwrap_or_default();

        if true_ids.is_empty() { continue; }
        total += 1;

        if let Some(rec_ocr) = record_map.get(&ocr_id) {
            let candidates: HashSet<RecordId> =
                blocker.candidates(rec_ocr, &schema, &idx).into_iter().collect();
            if true_ids.iter().any(|id| candidates.contains(id)) {
                found += 1;
            }
        }
    }

    assert!(total > 0, "No usable OCR confusion pairs in ground truth");

    let recall = found as f64 / total as f64;
    println!(
        "ANPR OCR blocking recall: {:.4} ({}/{} confusion pairs recovered, {} passages)",
        recall, found, total, records.len()
    );

    assert!(
        recall >= 0.97,
        "ANPR recall {:.4} below 0.97, PlateOCRFuzzyKey should cover all injected confusions",
        recall
    );
}

#[test]
fn anpr_no_self_candidates() {
    let schema = anpr_schema();
    let (records, _, _) = load_passages();

    let blocker = CompositeBlocker::new()
        .add(LicensePlateNormKey::new("kenteken"));

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
