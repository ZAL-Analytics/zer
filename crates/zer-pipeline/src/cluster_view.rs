use std::sync::Arc;

use zer_core::{
    entity::{Entity, EntityId, ResolutionMethod},
    record::{Record, RecordId},
    traits::{EntityStore, RecordStore},
};

/// A single cross-source link: one record from source A matched to one record
/// from source B within the same resolved entity.
#[derive(Debug, Clone)]
pub struct LinkedPair {
    pub entity_id: EntityId,
    pub record_id_a: RecordId,
    pub source_a: Option<String>,
    pub record_id_b: RecordId,
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
            // Partition members by source label.
            // Members with source A ≠ source B are eligible for cross-source output.
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
                        record_id_a: ma.record_id,
                        source_a: ma.source.clone(),
                        record_id_b: mb.record_id,
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
                        record_id_a: ma.record_id,
                        source_a: ma.source.clone(),
                        record_id_b: mb.record_id,
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

    #[test]
    fn cluster_view_iterates_entity_with_records() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        let rec = Record::new(1).insert("naam", FieldValue::Text("Alice".into()));
        record_store.insert(rec);

        let entity = Entity {
            id: 1,
            members: vec![EntityMember {
                record_id: 1,
                score: 1.0,
                method: ResolutionMethod::Manual,
                source: None,
            }],
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
        assert_eq!(recs[0].id, 1);
    }

    #[test]
    fn cluster_view_skips_missing_records() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        // Entity member points to record 99 which doesn't exist in the store
        let entity = Entity {
            id: 1,
            members: vec![EntityMember {
                record_id: 99,
                score: 1.0,
                method: ResolutionMethod::Manual,
                source: None,
            }],
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

        // Entity with two members, both from "brp"
        let entity = Entity {
            id: 1,
            members: vec![
                EntityMember {
                    record_id: 1,
                    score: 0.95,
                    method: ResolutionMethod::AutoMatch,
                    source: Some("brp".into()),
                },
                EntityMember {
                    record_id: 2,
                    score: 0.90,
                    method: ResolutionMethod::AutoMatch,
                    source: Some("brp".into()),
                },
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

        // Entity with one brp member and one kvk member
        let entity = Entity {
            id: 1,
            members: vec![
                EntityMember {
                    record_id: 10,
                    score: 0.95,
                    method: ResolutionMethod::AutoMatch,
                    source: Some("brp".into()),
                },
                EntityMember {
                    record_id: 20,
                    score: 0.88,
                    method: ResolutionMethod::AutoMatch,
                    source: Some("kvk".into()),
                },
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
            (lp.record_id_a == 10 && lp.record_id_b == 20)
                || (lp.record_id_a == 20 && lp.record_id_b == 10)
        );
        assert_ne!(lp.source_a, lp.source_b);
    }

    #[test]
    fn linked_pairs_no_source_labels_produces_no_pairs() {
        let entity_store = Arc::new(TestEntityStore::new());
        let record_store = Arc::new(TestRecordStore::new());

        // Members with no source labels, treated as same source
        let entity = Entity {
            id: 1,
            members: vec![
                EntityMember {
                    record_id: 1,
                    score: 0.95,
                    method: ResolutionMethod::AutoMatch,
                    source: None,
                },
                EntityMember {
                    record_id: 2,
                    score: 0.90,
                    method: ResolutionMethod::AutoMatch,
                    source: None,
                },
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
            "members without source labels must not produce LinkedPairs"
        );
    }
}
