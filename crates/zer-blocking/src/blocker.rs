use std::collections::HashMap;

use zer_core::{
    record::{Record, RecordId},
    schema::Schema,
    traits::{BlockIndex, Blocker},
};

use crate::keys::BlockingKey;

/// Composite blocker that applies multiple blocking keys.
///
/// For cross-schema linkage, `source_remaps` stores per-source field-name
/// translations (b_field to a_field).  Before key extraction, any record
/// whose `source` label has an entry in `source_remaps` gets its fields
/// renamed to the canonical (A-side) names so the existing `BlockingKey`
/// implementations can extract values without knowing about schema differences.
pub struct CompositeBlocker {
    keys:          Vec<Box<dyn BlockingKey>>,
    source_remaps: HashMap<String, HashMap<String, String>>,
}

impl CompositeBlocker {
    pub fn new() -> Self {
        Self { keys: vec![], source_remaps: HashMap::new() }
    }

    pub fn add(mut self, key: impl BlockingKey + 'static) -> Self {
        self.keys.push(Box::new(key));
        self
    }

    pub fn add_boxed(mut self, key: Box<dyn BlockingKey>) -> Self {
        self.keys.push(key);
        self
    }

    /// Register a field-name remap for records from `source`.
    ///
    /// `remap` maps b_field to a_field so that the source-B fields are
    /// visible under canonical source-A names during blocking key extraction.
    pub fn with_source_remap(
        mut self,
        source: impl Into<String>,
        remap:  HashMap<String, String>,
    ) -> Self {
        self.source_remaps.insert(source.into(), remap);
        self
    }

    fn effective_record<'r>(&self, record: &'r Record) -> Option<Record> {
        let src = record.source.as_deref()?;
        let remap = self.source_remaps.get(src)?;
        let mut new_rec = Record::new(record.id);
        if let Some(s) = &record.source {
            new_rec = new_rec.with_source(s);
        }
        for (field_name, value) in &record.fields {
            let canonical = remap.get(field_name).cloned()
                .unwrap_or_else(|| field_name.clone());
            new_rec.fields.insert(canonical, value.clone());
        }
        Some(new_rec)
    }
}

impl Default for CompositeBlocker {
    fn default() -> Self {
        Self::new()
    }
}

impl Blocker for CompositeBlocker {
    fn blocking_keys(&self, record: &Record, schema: &Schema) -> Vec<String> {
        let remapped = self.effective_record(record);
        let effective = remapped.as_ref().unwrap_or(record);
        self.keys
            .iter()
            .flat_map(|k| {
                k.extract(effective, schema)
                    .into_iter()
                    .map(|val| format!("{}:{}", k.name(), val))
            })
            .collect()
    }

    fn index_record(&self, record: &Record, schema: &Schema, index: &mut dyn BlockIndex) {
        let keys = self.blocking_keys(record, schema);
        index.insert(record.id, keys);
    }

    fn candidates(&self, record: &Record, schema: &Schema, index: &dyn BlockIndex) -> Vec<RecordId> {
        let keys = self.blocking_keys(record, schema);
        index.lookup_union(&keys, record.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{record::FieldValue, schema::{SchemaBuilder, FieldKind}};
    use crate::{index::InvertedIndex, keys::ExactFieldKey};

    fn schema() -> Schema {
        SchemaBuilder::new()
            .field("category", FieldKind::Categorical)
            .build()
            .unwrap()
    }

    #[test]
    fn index_and_candidates_round_trip() {
        let schema  = schema();
        let blocker = CompositeBlocker::new().add(ExactFieldKey::new("category"));
        let mut idx = InvertedIndex::new();

        let r1 = Record::new(1).insert("category", FieldValue::Text("TypeA".into()));
        let r2 = Record::new(2).insert("category", FieldValue::Text("TypeA".into()));
        let r3 = Record::new(3).insert("category", FieldValue::Text("TypeB".into()));

        blocker.index_record(&r1, &schema, &mut idx);
        blocker.index_record(&r2, &schema, &mut idx);
        blocker.index_record(&r3, &schema, &mut idx);

        let cands_r1 = blocker.candidates(&r1, &schema, &idx);
        assert!(cands_r1.contains(&2), "r2 should be a candidate for r1");
        assert!(!cands_r1.contains(&1), "r1 should not be its own candidate");
        assert!(!cands_r1.contains(&3), "r3 should not match r1 (different category)");
    }

    #[test]
    fn no_self_candidates() {
        let schema  = schema();
        let blocker = CompositeBlocker::new().add(ExactFieldKey::new("category"));
        let mut idx = InvertedIndex::new();

        let r = Record::new(1).insert("category", FieldValue::Text("X".into()));
        blocker.index_record(&r, &schema, &mut idx);

        let cands = blocker.candidates(&r, &schema, &idx);
        assert!(!cands.contains(&1));
    }
}
