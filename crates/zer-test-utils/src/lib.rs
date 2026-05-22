/// Shared test fixtures for the zer workspace.
///
/// Provides canonical schema builders, record constructors, and field-value
/// helpers so integration tests don't duplicate 20-line boilerplate.

use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, Schema, SchemaBuilder},
};

// ── FieldValue helpers ────────────────────────────────────────────────────────

/// Convenience wrapper: `FieldValue::Text(s.into())`.
pub fn text(s: &str) -> FieldValue {
    FieldValue::Text(s.into())
}

// ── Schema builders ───────────────────────────────────────────────────────────

/// Minimal 3-field person schema used in most unit/integration tests.
pub fn person_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .expect("person_schema must build")
}

/// Full 10-field BRP schema matching the canonical benchmark dataset.
pub fn brp_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("record_id",     FieldKind::Id)
        .field("geboorteland",  FieldKind::Categorical)
        .field("nationaliteit", FieldKind::Categorical)
        .field("straatnaam",    FieldKind::Address)
        .field("postcode",      FieldKind::Id)
        .field("woonplaats",    FieldKind::Address)
        .build()
        .expect("brp_schema must build")
}

// ── Record builders ───────────────────────────────────────────────────────────

/// Build a person record with the 3 fields from [`person_schema`].
pub fn make_person_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen",     text(first))
        .insert("achternaam",    text(last))
        .insert("geboortedatum", text(dob))
}

/// Build a person record with an explicit source label.
pub fn make_person_record_with_source(
    id:     u64,
    first:  &str,
    last:   &str,
    dob:    &str,
    source: &str,
) -> Record {
    make_person_record(id, first, last, dob).with_source(source)
}
