/// Integration tests for `zer_judge::serialize::serialize_pair`.
///
/// Exercises the NLI cross-encoder input format from outside the module.
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_judge::serialize::serialize_pair;

fn name_dob_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("naam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .unwrap()
}

// ── Structural invariants ─────────────────────────────────────────────────────

#[test]
fn output_starts_with_cls() {
    let schema = name_dob_schema();
    let r = Record::new(1).insert("naam", FieldValue::Text("alice".into()));
    let s = serialize_pair(&r, &r, &schema);
    assert!(s.starts_with("[CLS]"), "must start with [CLS]: {s}");
}

#[test]
fn output_ends_with_sep() {
    let schema = name_dob_schema();
    let r = Record::new(1).insert("naam", FieldValue::Text("alice".into()));
    let s = serialize_pair(&r, &r, &schema);
    assert!(s.ends_with("[SEP]"), "must end with [SEP]: {s}");
}

#[test]
fn output_has_exactly_two_sep_tokens() {
    let schema = name_dob_schema();
    let r = Record::new(1).insert("naam", FieldValue::Text("test".into()));
    let s = serialize_pair(&r, &r, &schema);
    assert_eq!(
        s.matches("[SEP]").count(),
        2,
        "expected exactly two [SEP]: {s}"
    );
}

// ── Record content ────────────────────────────────────────────────────────────

#[test]
fn both_records_appear_in_output() {
    let schema = name_dob_schema();
    let a = Record::new(1).insert("naam", FieldValue::Text("jan smits".into()));
    let b = Record::new(2).insert("naam", FieldValue::Text("jan smyts".into()));
    let s = serialize_pair(&a, &b, &schema);
    assert!(
        s.contains("COL:naam VAL:jan smits"),
        "left record missing: {s}"
    );
    assert!(
        s.contains("COL:naam VAL:jan smyts"),
        "right record missing: {s}"
    );
}

#[test]
fn absent_field_renders_as_empty_val() {
    let schema = name_dob_schema();
    // geboortedatum intentionally absent
    let a = Record::new(1).insert("naam", FieldValue::Text("jan".into()));
    let b = Record::new(2).insert("naam", FieldValue::Text("jan".into()));
    let s = serialize_pair(&a, &b, &schema);
    assert!(
        s.contains("COL:geboortedatum VAL:"),
        "absent field should produce empty VAL: {s}",
    );
}

// ── Schema ordering ───────────────────────────────────────────────────────────

#[test]
fn fields_appear_in_schema_order_regardless_of_insertion_order() {
    let schema = name_dob_schema(); // naam first, then geboortedatum
    let r = Record::new(1)
        .insert("geboortedatum", FieldValue::Text("1990-01-01".into()))
        .insert("naam", FieldValue::Text("alice".into()));
    let s = serialize_pair(&r, &r, &schema);
    let naam_pos = s.find("COL:naam").expect("COL:naam missing");
    let dob_pos = s
        .find("COL:geboortedatum")
        .expect("COL:geboortedatum missing");
    assert!(
        naam_pos < dob_pos,
        "naam must appear before geboortedatum (schema order): {s}"
    );
}
