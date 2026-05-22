use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::{
    comparison::{ComparisonBatch, ComparisonVector},
    entity::{Entity, EntityId},
    error::ZerError,
    record::{Record, RecordId},
    record_pool::RecordPool,
    schema::Schema,
    scoring::{ModelParams, ScoredPair},
};

pub type Result<T> = std::result::Result<T, ZerError>;

// ── RecordStore ───────────────────────────────────────────────────────────────

/// Backing store for records used during ingestion and batch runs.
pub trait RecordStore: Send + Sync {
    /// Persist a record.  Must be callable from the ingester background task.
    fn insert(&self, record: Record);

    /// Retrieve a single record by ID.  Returns `None` if not present.
    fn get(&self, id: RecordId) -> Option<Cow<'_, Record>>;

    /// Retrieve multiple records in one call (allows batch I/O optimisation).
    /// The default impl calls `get` in a loop; override for bulk reads.
    fn get_many(&self, ids: &[RecordId]) -> Vec<Option<Cow<'_, Record>>> {
        ids.iter().map(|id| self.get(*id)).collect()
    }

    /// Total number of records held.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── VecRecordStore (default in-memory impl) ───────────────────────────────────

struct VecRecordStoreInner {
    records:   Vec<Record>,
    id_to_idx: HashMap<RecordId, usize>,
}

/// Default in-memory [`RecordStore`] backed by a `Vec`, zero-config.
pub struct VecRecordStore {
    inner: RwLock<VecRecordStoreInner>,
}

impl VecRecordStore {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(VecRecordStoreInner {
                records:   Vec::new(),
                id_to_idx: HashMap::new(),
            }),
        }
    }
}

impl Default for VecRecordStore {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordStore for VecRecordStore {
    fn insert(&self, record: Record) {
        let mut inner = self.inner.write().unwrap();
        let idx = inner.records.len();
        inner.id_to_idx.insert(record.id, idx);
        inner.records.push(record);
    }

    fn get(&self, id: RecordId) -> Option<Cow<'_, Record>> {
        let inner = self.inner.read().unwrap();
        let idx = *inner.id_to_idx.get(&id)?;
        Some(Cow::Owned(inner.records[idx].clone()))
    }

    fn len(&self) -> usize {
        self.inner.read().unwrap().records.len()
    }
}

// ── BlockIndex ────────────────────────────────────────────────────────────────

/// Opaque blocking index.
///
/// The `as_any` / `as_any_mut` escape hatches allow access to concrete fields
/// not covered by the trait, such as index statistics.
pub trait BlockIndex: Send + Sync {
    /// Index `record_id` under the given set of blocking keys.
    fn insert(&mut self, record_id: RecordId, keys: Vec<String>);

    /// Return all record IDs sharing at least one key with `keys`, excluding
    /// `exclude` (the querying record itself).  Result must be deduplicated.
    fn lookup_union(&self, keys: &[String], exclude: RecordId) -> Vec<RecordId>;

    /// Remove all index entries for `record_id`.
    fn remove(&mut self, record_id: RecordId);

    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Extracts blocking keys from records and looks up candidates in an index.
pub trait Blocker: Send + Sync {
    fn blocking_keys(&self, record: &Record, schema: &Schema) -> Vec<String>;
    fn index_record(&self, record: &Record, schema: &Schema, index: &mut dyn BlockIndex);
    fn candidates(&self, record: &Record, schema: &Schema, index: &dyn BlockIndex) -> Vec<RecordId>;
}

pub trait Comparator: Send + Sync {
    /// Compare a single pair, always CPU, returns an individual vector.
    fn compare(&self, a: &Record, b: &Record, schema: &Schema) -> ComparisonVector;

    /// Pool-native batch comparison, the primary hot path.
    ///
    /// Reads `RecordPool` columns directly: zero HashMap lookups, no
    /// `Record::clone()`.  Implementors SHOULD override this method.
    /// The default falls back to `compare` per pair, which is correct but
    /// slower than a native pool implementation.
    fn compare_batch_from_pool(
        &self,
        pool:    &RecordPool,
        indices: &[(usize, usize)],
        schema:  &Schema,
    ) -> ComparisonBatch {
        let n_pairs  = indices.len();
        let n_fields = schema.fields.len();
        if n_pairs == 0 {
            return ComparisonBatch::new(0, n_fields, vec![]);
        }
        let pair_ids: Vec<(u64, u64)> = indices.iter()
            .map(|&(i, j)| (pool.ids[i], pool.ids[j]))
            .collect();
        let mut batch = ComparisonBatch::new(n_pairs, n_fields, pair_ids);
        for (p, &(i, j)) in indices.iter().enumerate() {
            use crate::record::FieldValue;
            let mut a = Record::new(pool.ids[i]);
            let mut b = Record::new(pool.ids[j]);
            for (f, field) in schema.fields.iter().enumerate() {
                let va = pool.get(f, i);
                let vb = pool.get(f, j);
                if !va.is_empty() {
                    a = a.insert(&field.name, FieldValue::Text(va.to_string()));
                }
                if !vb.is_empty() {
                    b = b.insert(&field.name, FieldValue::Text(vb.to_string()));
                }
            }
            let v = self.compare(&a, &b, schema);
            for (f, &level) in v.levels.iter().enumerate() {
                batch.set_level(f, p, level);
            }
        }
        batch
    }

}

pub trait Scorer: Send + Sync {
    /// Score a single pair, always CPU, cheap dot product.
    fn score(&self, vector: &ComparisonVector, params: &ModelParams) -> ScoredPair;

    /// Score a batch using the field-major `ComparisonBatch`.
    fn score_batch(&self, batch: &ComparisonBatch, params: &ModelParams) -> Vec<ScoredPair> {
        (0..batch.n_pairs)
            .map(|p| self.score(&batch.pair_as_vector(p), params))
            .collect()
    }

    fn estimate_params(
        &self,
        batch:    &ComparisonBatch,
        init:     Option<ModelParams>,
        max_iter: usize,
    ) -> Result<ModelParams>;
}

/// Groups scored pairs into entity clusters.
pub trait Clusterer: Send + Sync {
    fn cluster(&self, pairs: &[ScoredPair], params: &ModelParams) -> Vec<Entity>;
}

/// Persistent store for resolved entities.
pub trait EntityStore: Send + Sync {
    fn upsert_entity(&self, entity: &Entity) -> Result<EntityId>;
    fn get_entity(&self, id: EntityId) -> Result<Entity>;
    fn record_to_entity(&self, record_id: RecordId) -> Result<Option<EntityId>>;
    fn all_entities(&self) -> Result<Vec<Entity>>;
}

/// Convert an external row type into a [`Record`].
///
/// Implement this in an adapter crate (e.g. `zer-adapters`) for foreign
/// row types such as a Polars `LazyFrame` row or an Arrow `RecordBatch` row.
/// The `id` parameter lets callers assign a stable [`RecordId`].
pub trait IntoRecord {
    fn into_record(self, id: RecordId) -> Record;
}

impl IntoRecord for Record {
    fn into_record(self, _id: RecordId) -> Record {
        self
    }
}

/// Neural re-ranker that adjudicates borderline record pairs.
pub trait Judge: Send + Sync {
    fn adjudicate(&self, pairs: &[ScoredPair]) -> Result<Vec<JudgeVerdict>>;
}

impl<J: Judge + ?Sized> Judge for Box<J> {
    fn adjudicate(&self, pairs: &[ScoredPair]) -> Result<Vec<JudgeVerdict>> {
        (**self).adjudicate(pairs)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum JudgeVerdict {
    IncreaseConfidence,
    DecreaseConfidence,
    NoChange,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traits_are_object_safe() {
        let _: Box<dyn super::BlockIndex>;
        let _: Box<dyn super::Blocker>;
        let _: Box<dyn super::Comparator>;
        let _: Box<dyn super::Scorer>;
        let _: Box<dyn super::Clusterer>;
        let _: Box<dyn super::EntityStore>;
        let _: Box<dyn super::Judge>;
        let _: Box<dyn super::RecordStore>;
    }

    #[test]
    fn vec_record_store_insert_and_get() {
        use crate::record::{FieldValue, Record};
        let store = VecRecordStore::new();
        assert!(store.is_empty());

        let r = Record::new(42).insert("name", FieldValue::Text("Alice".into()));
        store.insert(r);

        assert_eq!(store.len(), 1);
        let fetched = store.get(42).expect("record 42 must exist");
        assert_eq!(fetched.id, 42);
    }

    #[test]
    fn vec_record_store_get_missing_returns_none() {
        let store = VecRecordStore::new();
        assert!(store.get(999).is_none());
    }

    #[test]
    fn vec_record_store_get_many() {
        use crate::record::Record;
        let store = VecRecordStore::new();
        store.insert(Record::new(1));
        store.insert(Record::new(2));
        store.insert(Record::new(3));

        let results = store.get_many(&[1, 3, 99]);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none());
    }

    #[test]
    fn vec_record_store_is_sendable() {
        use std::sync::Arc;
        let store: Arc<dyn RecordStore> = Arc::new(VecRecordStore::new());
        let store2 = Arc::clone(&store);
        let handle = std::thread::spawn(move || {
            store2.insert(crate::record::Record::new(7));
        });
        handle.join().unwrap();
        assert_eq!(store.len(), 1);
    }
}
