//! Shared test fixtures for the zer workspace.
//!
//! Provides canonical schema builders, record constructors, and field-value
//! helpers so integration tests don't duplicate 20-line boilerplate.

// ── Dataset path resolution ───────────────────────────────────────────────────

/// Returns the path to a dataset file, honouring `ZER_DATASET_DIR` at runtime.
///
/// `manifest_dir` should be `env!("CARGO_MANIFEST_DIR")` from the calling crate.
/// `relative` is the path within the data directory, without the `data/` prefix
/// (e.g. `"tests/brp/brp_persons.csv"`).
///
/// Resolution order:
/// 1. `ZER_DATASET_DIR` env var  →  `$ZER_DATASET_DIR/<relative>`
/// 2. Workspace fallback         →  `<manifest_dir>/../../data/<relative>`
pub fn dataset_path(manifest_dir: &str, relative: &str) -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("ZER_DATASET_DIR") {
        return std::path::PathBuf::from(dir).join(relative);
    }
    std::path::PathBuf::from(manifest_dir)
        .join("../..")
        .join("data")
        .join(relative)
}

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
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .build()
        .expect("person_schema must build")
}

/// Full 10-field BRP schema matching the canonical benchmark dataset.
pub fn brp_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("record_id", FieldKind::Id)
        .field("geboorteland", FieldKind::Categorical)
        .field("nationaliteit", FieldKind::Categorical)
        .field("straatnaam", FieldKind::Address)
        .field("postcode", FieldKind::Id)
        .field("woonplaats", FieldKind::Address)
        .build()
        .expect("brp_schema must build")
}

// ── Record builders ───────────────────────────────────────────────────────────

/// Build a person record with the 3 fields from [`person_schema`].
///
/// Uses [`Record::new`] so the `key` defaults to `id.to_string()`.  For
/// records loaded from real data use [`Record::from_key`] together with a
/// [`zer_adapters::DatasetConfig`] instead.
pub fn make_person_record(id: u64, first: &str, last: &str, dob: &str) -> Record {
    Record::new(id)
        .insert("voornamen", text(first))
        .insert("achternaam", text(last))
        .insert("geboortedatum", text(dob))
}

/// Build a person record with an explicit source label.
pub fn make_person_record_with_source(
    id: u64,
    first: &str,
    last: &str,
    dob: &str,
    source: &str,
) -> Record {
    make_person_record(id, first, last, dob).with_source(source)
}
