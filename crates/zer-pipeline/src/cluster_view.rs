use std::sync::Arc;

use zer_core::{
    entity::{Entity, EntityId, ResolutionMethod},
    record::Record,
    traits::{EntityStore, RecordStore},
};

/// A single cross-source link: one record from source A matched to one record
/// from source B within the same resolved entity.
///
/// `record_key_a` and `record_key_b` are the natural keys of the two records
/// (i.e. the values from the column nominated as the identity column when
/// loading the dataset).  For records created with `Record::new(id)` the key
/// is the numeric ID as a string.
#[derive(Debug, Clone)]
pub struct LinkedPair {
    pub entity_id: EntityId,
    pub record_key_a: String,
    pub source_a: Option<String>,
    pub record_key_b: String,
    pub source_b: Option<String>,
    pub score: f32,
    pub method: ResolutionMethod,
}

/// A lazy view over resolved clusters: iterates `(Entity, Vec<Record>)` pairs
/// by joining the entity store with the record store.
///
/// Constructed via [`crate::pipeline::Pipeline::cluster_view`].  Create once, iterate many times;
/// each call to `into_iter` re-reads all entities from the store.
pub struct ClusterView {
    entity_store: Arc<dyn EntityStore>,
    record_store: Arc<dyn RecordStore>,
}

impl ClusterView {
    pub fn new(entity_store: Arc<dyn EntityStore>, record_store: Arc<dyn RecordStore>) -> Self {
        Self {
            entity_store,
            record_store,
        }
    }

    /// Emit cross-source linked pairs from all resolved entities.
    ///
    /// For each entity with members from at least two distinct sources, yields
    /// one [`LinkedPair`] per (source_a record, source_b record) combination.
    /// Entities whose members all share the same source label (or all have no
    /// label) produce no output, they are purely within-source clusters.
    ///
    /// This is the primary output format for [`crate::config::LinkMode::LinkOnly`] and
    /// [`crate::config::LinkMode::LinkAndDedupe`] runs.
    pub fn linked_pairs(&self) -> Vec<LinkedPair> {
        let entities = self.entity_store.all_entities().unwrap_or_default();
        let mut out = Vec::new();

        for entity in entities {
            let n = entity.members.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let ma = &entity.members[i];
                    let mb = &entity.members[j];
                    // Skip pairs from the same source (including both-None).
                    if ma.source == mb.source {
                        continue;
                    }
                    out.push(LinkedPair {
                        entity_id: entity.id,
                        record_key_a: ma.record_key.clone(),
                        source_a: ma.source.clone(),
                        record_key_b: mb.record_key.clone(),
                        source_b: mb.source.clone(),
                        score: ma.score.min(mb.score),
                        method: ma.method,
                    });
                }
            }
        }

        out
    }

    /// Emit all within-entity member pairs regardless of source label.
    ///
    /// Unlike [`Self::linked_pairs`], this does not filter by source, it emits
    /// every (member_i, member_j) combination within each entity, including
    /// same-source and unlabelled pairs.
    ///
    /// Use this for accuracy evaluation in `deduplicate` and `link-and-dedupe`
    /// modes where the ground truth contains both within-source and cross-source
    /// pairs.
    pub fn all_member_pairs(&self) -> Vec<LinkedPair> {
        let entities = self.entity_store.all_entities().unwrap_or_default();
        let mut out = Vec::new();

        for entity in entities {
            let n = entity.members.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let ma = &entity.members[i];
                    let mb = &entity.members[j];
                    out.push(LinkedPair {
                        entity_id: entity.id,
                        record_key_a: ma.record_key.clone(),
                        source_a: ma.source.clone(),
                        record_key_b: mb.record_key.clone(),
                        source_b: mb.source.clone(),
                        score: ma.score.min(mb.score),
                        method: ma.method,
                    });
                }
            }
        }

        out
    }
}

impl<'a> IntoIterator for &'a ClusterView {
    type Item = (Entity, Vec<Record>);
    type IntoIter = ClusterIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let entities = self.entity_store.all_entities().unwrap_or_default();
        ClusterIter {
            view: self,
            entities: entities.into_iter(),
        }
    }
}

pub struct ClusterIter<'a> {
    view: &'a ClusterView,
    entities: std::vec::IntoIter<Entity>,
}

impl<'a> Iterator for ClusterIter<'a> {
    type Item = (Entity, Vec<Record>);

    fn next(&mut self) -> Option<Self::Item> {
        let entity = self.entities.next()?;
        let records: Vec<Record> = entity
            .members
            .iter()
            .filter_map(|m| {
                self.view
                    .record_store
                    .get(m.record_id)
                    .map(|cow| cow.into_owned())
            })
            .collect();
        Some((entity, records))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::sync::RwLock;
    use zer_core::{
        entity::{EntityId, EntityMember, ResolutionMethod},
        error::ZerError,
        record::{FieldValue, RecordId},
        traits::EntityStore,
    };

    // Minimal in-memory entity store for testing
    struct TestEntityStore {
        entities: RwLock<Vec<Entity>>,
    }

    impl TestEntityStore {
        fn new() -> Self {
            Self {
                entities: RwLock::new(Vec::new()),
            }
        }
    }

    impl EntityStore for TestEntityStore {
        fn upsert_entity(&self, entity: &Entity) -> Result<EntityId, ZerError> {
            let mut guard = self.entities.write().unwrap();
            if let Some(pos) = guard.iter().position(|e| e.id == entity.id) {
                guard[pos] = entity.clone();
            } else {
                guard.push(entity.clone());
            }
            Ok(entity.id)
        }

        fn get_entity(&self, id: EntityId) -> Result<Entity, ZerError> {
            let guard = self.entities.read().unwrap();
            guard
                .iter()
                .find(|e| e.id == id)
                .cloned()
                .ok_or_else(|| ZerError::Store(format!("entity {id} not found")))
        }

        fn record_to_entity(&self, record_id: RecordId) -> Result<Option<EntityId>, ZerError> {
            let guard = self.entities.read().unwrap();
            Ok(guard
                .iter()
                .find(|e| e.members.iter().any(|m| m.record_id == record_id))
                .map(|e| e.id))
        }

        fn all_entities(&self) -> Result<Vec<Entity>, ZerError> {
            Ok(self.entities.read().unwrap().clone())
        }
    }

    // Minimal in-memory record store for testing
    struct TestRecordStore {
        inner: RwLock<HashMap<RecordId, Record>>,
    }

    impl TestRecordStore {
        fn new() -> Self {
            Self {
                inner: RwLock::new(HashMap::new()),
            }
        }
    }

    impl zer_core::traits::RecordStore for TestRecordStore {
        fn insert(&self, record: Record) {
            self.inner.write().unwrap().insert(record.id, record);
        }

        fn get(&self, id: RecordId) -> Option<Cow<'_, Record>> {
            self.inner.read().unwrap().get(&id).cloned().map(Cow::Owned)
        }

        fn len(&self) -> usize {
            self.inner.read().unwrap().len()
        }
    }

    fn make_member(record_id: RecordId, record_key: &str, source: Option<&str>) -> EntityMember {
        EntityMember {
            record_id,
            record_key: record_key.to_string(),
            score: 0.95,
            method: ResolutionMethod::AutoMatch,
            source: source.map(str::to_string),
        }
    }

    #[test]
    fn cluster_view_iterates_entity_with_records() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        let rec = Record::new(1).insert("naam", FieldValue::Text("Alice".into()));
        record_store.insert(rec);

        let entity = Entity {
            id: 1,
            members: vec![make_member(1, "1", None)],
        };
        entity_store.upsert_entity(&entity).unwrap();

        let view = ClusterView::new(
            entity_store as Arc<dyn EntityStore>,
            record_store as Arc<dyn RecordStore>,
        );

        let clusters: Vec<_> = view.into_iter().collect();
        assert_eq!(clusters.len(), 1);
        let (e, recs) = &clusters[0];
        assert_eq!(e.id, 1);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].key, "1");
    }

    #[test]
    fn cluster_view_skips_missing_records() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        let entity = Entity {
            id: 1,
            members: vec![make_member(99, "99", None)],
        };
        entity_store.upsert_entity(&entity).unwrap();

        let view = ClusterView::new(
            entity_store as Arc<dyn EntityStore>,
            record_store as Arc<dyn RecordStore>,
        );

        let clusters: Vec<_> = view.into_iter().collect();
        assert_eq!(clusters.len(), 1);
        let (_e, recs) = &clusters[0];
        assert!(recs.is_empty(), "missing record should be skipped");
    }

    #[test]
    fn cluster_view_empty_when_no_entities() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        let view = ClusterView::new(
            entity_store as Arc<dyn EntityStore>,
            record_store as Arc<dyn RecordStore>,
        );

        let clusters: Vec<_> = view.into_iter().collect();
        assert!(clusters.is_empty());
    }

    #[test]
    fn linked_pairs_skips_single_source_entities() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        let entity = Entity {
            id: 1,
            members: vec![
                make_member(1, "key-001", Some("brp")),
                make_member(2, "key-002", Some("brp")),
            ],
        };
        entity_store.upsert_entity(&entity).unwrap();

        let view = ClusterView::new(
            entity_store as Arc<dyn EntityStore>,
            record_store as Arc<dyn RecordStore>,
        );

        let pairs = view.linked_pairs();
        assert!(
            pairs.is_empty(),
            "single-source entity must produce no LinkedPairs"
        );
    }

    #[test]
    fn linked_pairs_emits_cross_source_pairs() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        let entity = Entity {
            id: 1,
            members: vec![
                make_member(10, "brp-001", Some("brp")),
                make_member(20, "kvk-001", Some("kvk")),
            ],
        };
        entity_store.upsert_entity(&entity).unwrap();

        let view = ClusterView::new(
            entity_store as Arc<dyn EntityStore>,
            record_store as Arc<dyn RecordStore>,
        );

        let pairs = view.linked_pairs();
        assert_eq!(pairs.len(), 1, "one cross-source pair expected");
        let lp = &pairs[0];
        assert_eq!(lp.entity_id, 1);
        assert!(
            (lp.record_key_a == "brp-001" && lp.record_key_b == "kvk-001")
                || (lp.record_key_a == "kvk-001" && lp.record_key_b == "brp-001")
        );
        assert_ne!(lp.source_a, lp.source_b);
    }

    #[test]
    fn linked_pairs_no_source_labels_produces_no_pairs() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        let entity = Entity {
            id: 1,
            members: vec![make_member(1, "k1", None), make_member(2, "k2", None)],
        };
        entity_store.upsert_entity(&entity).unwrap();

        let view = ClusterView::new(
            entity_store as Arc<dyn EntityStore>,
            record_store as Arc<dyn RecordStore>,
        );

        let pairs = view.linked_pairs();
        assert!(
            pairs.is_empty(),
            "members without source labels must not produce LinkedPairs"
        );
    }
}
