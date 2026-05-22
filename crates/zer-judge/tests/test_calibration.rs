/// Integration tests for `CalibrationTable`.
///
/// The module-level unit tests cover the update_* arithmetic.  These tests
/// focus on the public struct contract: field values, constructors, Clone, and
/// serde round-trips.

use zer_judge::CalibrationTable;

// ── Default values ────────────────────────────────────────────────────────────

#[test]
fn default_lr_increase_is_above_one() {
    let table = CalibrationTable::default();
    assert!(table.lr_increase > 1.0, "lr_increase must boost probability; got {}", table.lr_increase);
}

#[test]
fn default_lr_decrease_is_below_one() {
    let table = CalibrationTable::default();
    assert!(table.lr_decrease < 1.0, "lr_decrease must reduce probability; got {}", table.lr_decrease);
}

#[test]
fn default_lr_no_change_is_one() {
    let table = CalibrationTable::default();
    assert_eq!(table.lr_no_change, 1.0);
}

// ── Constructor ───────────────────────────────────────────────────────────────

#[test]
fn new_sets_all_fields() {
    let table = CalibrationTable::new(3.0, 0.25, 1.5);
    assert_eq!(table.lr_increase,  3.0);
    assert_eq!(table.lr_decrease,  0.25);
    assert_eq!(table.lr_no_change, 1.5);
}

// ── Clone ─────────────────────────────────────────────────────────────────────

#[test]
fn clone_is_independent() {
    let original = CalibrationTable::default();
    let mut cloned = original.clone();
    cloned.lr_increase = 99.0;
    assert_ne!(original.lr_increase, cloned.lr_increase, "clone must be a deep copy");
}

// ── Serde round-trip ──────────────────────────────────────────────────────────

#[test]
fn json_roundtrip_preserves_all_fields() {
    let original = CalibrationTable::new(5.0, 0.08, 1.2);
    let json = serde_json::to_string(&original).expect("serialization failed");
    let back: CalibrationTable = serde_json::from_str(&json).expect("deserialization failed");
    assert_eq!(back.lr_increase,  original.lr_increase);
    assert_eq!(back.lr_decrease,  original.lr_decrease);
    assert_eq!(back.lr_no_change, original.lr_no_change);
}

#[test]
fn default_json_roundtrip() {
    let original = CalibrationTable::default();
    let json = serde_json::to_string(&original).unwrap();
    let back: CalibrationTable = serde_json::from_str(&json).unwrap();
    assert_eq!(back.lr_increase,  original.lr_increase);
    assert_eq!(back.lr_decrease,  original.lr_decrease);
    assert_eq!(back.lr_no_change, original.lr_no_change);
}
