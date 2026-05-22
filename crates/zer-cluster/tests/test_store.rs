use zer_cluster::ZalEntityStore;
use zer_core::{
    entity::{Entity, EntityMember, ResolutionMethod},
    record::RecordId,
    traits::EntityStore,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_entity(record_ids: &[RecordId]) -> Entity {
    Entity {
        id: 0,
        members: record_ids
            .iter()
            .map(|&rid| EntityMember {
                record_id: rid,
                score:     0.95,
                method:    ResolutionMethod::AutoMatch,
                source:    None,
            })
            .collect(),
    }
}

// ── Unit tests (in-memory) ────────────────────────────────────────────────────

#[test]
fn in_memory_store_opens_successfully() {
    ZalEntityStore::open_in_memory().unwrap();
}

#[test]
fn upsert_returns_increasing_entity_ids() {
    let store = ZalEntityStore::open_in_memory().unwrap();
    let id1 = store.upsert_entity(&make_entity(&[1, 2])).unwrap();
    let id2 = store.upsert_entity(&make_entity(&[3, 4])).unwrap();
    assert_ne!(id1, id2);
    assert!(id1 >= 1);
    assert!(id2 >= 1);
}

#[test]
fn get_entity_round_trips_all_members() {
    let store = ZalEntityStore::open_in_memory().unwrap();
    let eid = store.upsert_entity(&make_entity(&[10, 20, 30])).unwrap();
    let loaded = store.get_entity(eid).unwrap();
    let mut rids: Vec<RecordId> = loaded.members.iter().map(|m| m.record_id).collect();
    rids.sort();
    assert_eq!(rids, vec![10, 20, 30]);
}

#[test]
fn upsert_merges_overlapping_members() {
    let store = ZalEntityStore::open_in_memory().unwrap();
    let eid = store.upsert_entity(&make_entity(&[1, 2])).unwrap();
    // Share record 2 → should merge into the same entity.
    let eid2 = store.upsert_entity(&make_entity(&[2, 3])).unwrap();
    assert_eq!(eid, eid2, "overlapping upsert must resolve to same entity");

    let loaded = store.get_entity(eid).unwrap();
    let mut rids: Vec<RecordId> = loaded.members.iter().map(|m| m.record_id).collect();
    rids.sort();
    assert_eq!(rids, vec![1, 2, 3]);
}

#[test]
fn record_to_entity_returns_correct_mapping() {
    let store = ZalEntityStore::open_in_memory().unwrap();
    let eid = store.upsert_entity(&make_entity(&[100, 200, 300])).unwrap();
    assert_eq!(store.record_to_entity(100).unwrap(), Some(eid));
    assert_eq!(store.record_to_entity(200).unwrap(), Some(eid));
    assert_eq!(store.record_to_entity(300).unwrap(), Some(eid));
}

#[test]
fn record_to_entity_returns_none_for_unknown_record() {
    let store = ZalEntityStore::open_in_memory().unwrap();
    assert!(store.record_to_entity(999_999).unwrap().is_none());
}

#[test]
fn all_entities_returns_every_entity() {
    let store = ZalEntityStore::open_in_memory().unwrap();
    store.upsert_entity(&make_entity(&[1, 2])).unwrap();
    store.upsert_entity(&make_entity(&[3, 4])).unwrap();
    store.upsert_entity(&make_entity(&[5, 6])).unwrap();
    let all = store.all_entities().unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
fn all_entities_empty_store_returns_empty() {
    let store = ZalEntityStore::open_in_memory().unwrap();
    assert!(store.all_entities().unwrap().is_empty());
}

#[test]
fn resolution_method_round_trips_correctly() {
    let store = ZalEntityStore::open_in_memory().unwrap();
    let entity = Entity {
        id: 0,
        members: vec![
            EntityMember { record_id: 1, score: 0.9, method: ResolutionMethod::JudgePromoted, source: None },
            EntityMember { record_id: 2, score: 0.8, method: ResolutionMethod::JudgeDemoted,  source: None },
            EntityMember { record_id: 3, score: 0.7, method: ResolutionMethod::Manual,        source: None },
        ],
    };
    let eid = store.upsert_entity(&entity).unwrap();
    let loaded = store.get_entity(eid).unwrap();

    let find = |rid: RecordId| -> ResolutionMethod {
        loaded.members.iter().find(|m| m.record_id == rid).unwrap().method
    };
    assert_eq!(find(1), ResolutionMethod::JudgePromoted);
    assert_eq!(find(2), ResolutionMethod::JudgeDemoted);
    assert_eq!(find(3), ResolutionMethod::Manual);
}

// ── Integration test: .zes file persistence ───────────────────────────────────

#[test]
fn zes_file_persists_across_reopen() {
    let tmp = tempfile::Builder::new()
        .suffix(".zes")
        .tempfile()
        .unwrap();
    let path = tmp.path().to_path_buf();

    let expected_count = 10_usize;

    // Write 10 entities.
    {
        let store = ZalEntityStore::open(&path).unwrap();
        for i in 0..expected_count as u64 {
            store
                .upsert_entity(&make_entity(&[i * 2, i * 2 + 1]))
                .unwrap();
        }
    } // store dropped, connection closed

    // Re-open and verify.
    {
        let store = ZalEntityStore::open(&path).unwrap();
        let all = store.all_entities().unwrap();
        assert_eq!(
            all.len(),
            expected_count,
            "all {expected_count} entities must survive a store close/reopen"
        );

        // Spot-check a few record_to_entity lookups.
        let eid0 = store.record_to_entity(0).unwrap();
        let eid1 = store.record_to_entity(1).unwrap();
        assert!(eid0.is_some());
        assert_eq!(eid0, eid1, "records 0 and 1 must belong to the same entity");
    }
}
