use rusqlite::Connection;
use zer_core::{
    entity::{EntityId, ResolutionMethod},
    error::ZerError,
    record::RecordId,
};

/// Events written to the `resolution_events` table to provide an audit trail
/// for every structural change to the entity store.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ResolutionEvent {
    EntityCreated { entity_id: EntityId, record_ids: Vec<RecordId> },
    RecordsAdded  { entity_id: EntityId, record_ids: Vec<RecordId>, method: ResolutionMethod },
    EntityMerged  { source_a: EntityId, source_b: EntityId, into: EntityId },
    EntitySplit   { source: EntityId, into: Vec<EntityId> },
    JudgeApplied  { entity_id: EntityId, pair: (RecordId, RecordId), verdict: String },
}

/// Append a provenance event to `resolution_events`.
///
/// Called from `ZalEntityStore` with the locked connection, no additional
/// locking is needed here.
pub fn append_event(conn: &Connection, event: &ResolutionEvent) -> Result<(), ZerError> {
    let (event_type, entity_id, record_ids, score, judge_verdict) = match event {
        ResolutionEvent::EntityCreated { entity_id, record_ids } => (
            "EntityCreated",
            *entity_id,
            record_ids.clone(),
            None::<f32>,
            None::<String>,
        ),
        ResolutionEvent::RecordsAdded { entity_id, record_ids, .. } => (
            "RecordsAdded",
            *entity_id,
            record_ids.clone(),
            None,
            None,
        ),
        ResolutionEvent::EntityMerged { into, source_a, source_b } => (
            "EntityMerged",
            *into,
            vec![*source_a, *source_b],
            None,
            None,
        ),
        ResolutionEvent::EntitySplit { source, into } => (
            "EntitySplit",
            *source,
            into.clone(),
            None,
            None,
        ),
        ResolutionEvent::JudgeApplied { entity_id, pair, verdict } => (
            "JudgeApplied",
            *entity_id,
            vec![pair.0, pair.1],
            None,
            Some(verdict.clone()),
        ),
    };

    let ids_json = serde_json::to_string(&record_ids)
        .map_err(|e| ZerError::Serialization(e.to_string()))?;
    let now = unix_now();

    conn.execute(
        "INSERT INTO resolution_events
             (event_type, entity_id, record_ids, score, judge_verdict, occurred_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![event_type, entity_id as i64, ids_json, score, judge_verdict, now],
    )
    .map_err(|e| ZerError::Store(e.to_string()))?;

    Ok(())
}

pub(crate) fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
