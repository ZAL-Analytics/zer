use std::{path::Path, sync::Mutex};

use rusqlite::{Connection, OptionalExtension};
use zer_core::{
    entity::{Entity, EntityId, EntityMember, ResolutionMethod},
    error::ZerError,
    record::RecordId,
    traits::{EntityStore, Result},
};

use crate::provenance::{append_event, unix_now, ResolutionEvent};

/// SQLite-backed entity store persisted as a single `.zes` file.
///
/// Uses `rusqlite/bundled` so no system SQLite installation is required.
/// All mutations hold the connection `Mutex` for the duration of the
/// transaction, suitable for single-threaded or lightly-concurrent use.
pub struct ZalEntityStore {
    conn: Mutex<Connection>,
}

impl ZalEntityStore {
    /// Open (or create) a `.zes` store at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| ZerError::Store(e.to_string()))?;
        init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory store. No file is created; data is lost on drop.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| ZerError::Store(e.to_string()))?;
        init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS entities (
            entity_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS entity_members (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entities(entity_id),
            record_id INTEGER NOT NULL,
            score     REAL    NOT NULL,
            method    TEXT    NOT NULL,
            source    TEXT,
            added_at  INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_record_entity ON entity_members(record_id);

        CREATE TABLE IF NOT EXISTS resolution_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type    TEXT    NOT NULL,
            entity_id     INTEGER NOT NULL,
            record_ids    TEXT    NOT NULL,
            score         REAL,
            judge_verdict TEXT,
            occurred_at   INTEGER NOT NULL
        );",
    )
    .map_err(|e| ZerError::Store(e.to_string()))
}

impl EntityStore for ZalEntityStore {
    fn upsert_entity(&self, entity: &Entity) -> Result<EntityId> {
        let conn = self.conn.lock().unwrap();
        let now = unix_now();

        // Find if any member already belongs to an existing entity.
        let mut existing_id: Option<EntityId> = None;
        for member in &entity.members {
            let id: Option<i64> = conn
                .query_row(
                    "SELECT entity_id FROM entity_members WHERE record_id = ?1",
                    [member.record_id as i64],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| ZerError::Store(e.to_string()))?;

            if let Some(eid) = id {
                existing_id = Some(eid as EntityId);
                break;
            }
        }

        if let Some(eid) = existing_id {
            // Entity already exists, merge new members in.
            conn.execute(
                "UPDATE entities SET updated_at = ?1 WHERE entity_id = ?2",
                rusqlite::params![now, eid as i64],
            )
            .map_err(|e| ZerError::Store(e.to_string()))?;

            let new_record_ids: Vec<RecordId> =
                entity.members.iter().map(|m| m.record_id).collect();
            for member in &entity.members {
                conn.execute(
                    "INSERT OR IGNORE INTO entity_members
                         (entity_id, record_id, score, method, source, added_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![
                        eid as i64,
                        member.record_id as i64,
                        member.score,
                        method_to_str(member.method),
                        member.source.as_deref(),
                        now,
                    ],
                )
                .map_err(|e| ZerError::Store(e.to_string()))?;
            }

            append_event(
                &conn,
                &ResolutionEvent::RecordsAdded {
                    entity_id: eid,
                    record_ids: new_record_ids,
                    method: entity
                        .members
                        .first()
                        .map(|m| m.method)
                        .unwrap_or(ResolutionMethod::AutoMatch),
                },
            )?;

            Ok(eid)
        } else {
            // Brand-new entity.
            conn.execute(
                "INSERT INTO entities (created_at, updated_at) VALUES (?1, ?2)",
                rusqlite::params![now, now],
            )
            .map_err(|e| ZerError::Store(e.to_string()))?;

            let eid = conn.last_insert_rowid() as EntityId;

            let record_ids: Vec<RecordId> = entity.members.iter().map(|m| m.record_id).collect();
            for member in &entity.members {
                conn.execute(
                    "INSERT INTO entity_members
                         (entity_id, record_id, score, method, source, added_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![
                        eid as i64,
                        member.record_id as i64,
                        member.score,
                        method_to_str(member.method),
                        member.source.as_deref(),
                        now,
                    ],
                )
                .map_err(|e| ZerError::Store(e.to_string()))?;
            }

            append_event(
                &conn,
                &ResolutionEvent::EntityCreated {
                    entity_id: eid,
                    record_ids,
                },
            )?;

            Ok(eid)
        }
    }

    fn get_entity(&self, id: EntityId) -> Result<Entity> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT record_id, score, method, source
                 FROM entity_members WHERE entity_id = ?1",
            )
            .map_err(|e| ZerError::Store(e.to_string()))?;

        let members: Vec<EntityMember> = stmt
            .query_map([id as i64], |row| {
                Ok((
                    row.get::<_, i64>(0)? as RecordId,
                    row.get::<_, f32>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            })
            .map_err(|e| ZerError::Store(e.to_string()))?
            .map(|r| {
                r.map_err(|e| ZerError::Store(e.to_string()))
                    .map(|(rid, score, method, source)| EntityMember {
                        record_id: rid,
                        score,
                        method: method_from_str(&method),
                        source,
                    })
            })
            .collect::<Result<_>>()?;

        Ok(Entity { id, members })
    }

    fn record_to_entity(&self, record_id: RecordId) -> Result<Option<EntityId>> {
        let conn = self.conn.lock().unwrap();
        let id: Option<i64> = conn
            .query_row(
                "SELECT entity_id FROM entity_members WHERE record_id = ?1",
                [record_id as i64],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| ZerError::Store(e.to_string()))?;
        Ok(id.map(|i| i as EntityId))
    }

    fn all_entities(&self) -> Result<Vec<Entity>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT em.entity_id, em.record_id, em.score, em.method, em.source
                 FROM entity_members em
                 ORDER BY em.entity_id",
            )
            .map_err(|e| ZerError::Store(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)? as EntityId,
                    row.get::<_, i64>(1)? as RecordId,
                    row.get::<_, f32>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            })
            .map_err(|e| ZerError::Store(e.to_string()))?;

        let mut entities: Vec<Entity> = Vec::new();

        for row in rows {
            let (eid, rid, score, method, source) =
                row.map_err(|e| ZerError::Store(e.to_string()))?;
            let member = EntityMember {
                record_id: rid,
                score,
                method: method_from_str(&method),
                source,
            };
            match entities.last_mut() {
                Some(e) if e.id == eid => e.members.push(member),
                _ => entities.push(Entity {
                    id: eid,
                    members: vec![member],
                }),
            }
        }

        Ok(entities)
    }
}

// ── Method round-trip helpers ─────────────────────────────────────────────────

fn method_to_str(method: ResolutionMethod) -> &'static str {
    match method {
        ResolutionMethod::AutoMatch => "AutoMatch",
        ResolutionMethod::JudgePromoted => "JudgePromoted",
        ResolutionMethod::JudgeDemoted => "JudgeDemoted",
        ResolutionMethod::Manual => "Manual",
    }
}

fn method_from_str(s: &str) -> ResolutionMethod {
    match s {
        "JudgePromoted" => ResolutionMethod::JudgePromoted,
        "JudgeDemoted" => ResolutionMethod::JudgeDemoted,
        "Manual" => ResolutionMethod::Manual,
        _ => ResolutionMethod::AutoMatch,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{entity::ResolutionMethod, traits::EntityStore};

    fn make_entity(id: EntityId, record_ids: &[RecordId]) -> Entity {
        Entity {
            id,
            members: record_ids
                .iter()
                .map(|&rid| EntityMember {
                    record_id: rid,
                    score: 0.95,
                    method: ResolutionMethod::AutoMatch,
                    source: None,
                })
                .collect(),
        }
    }

    #[test]
    fn open_in_memory_creates_schema() {
        ZalEntityStore::open_in_memory().unwrap();
    }

    #[test]
    fn upsert_new_entity_returns_id() {
        let store = ZalEntityStore::open_in_memory().unwrap();
        let entity = make_entity(0, &[1, 2, 3]);
        let eid = store.upsert_entity(&entity).unwrap();
        assert!(eid >= 1, "autoincrement id must be ≥ 1");
    }

    #[test]
    fn upsert_same_entity_merges_members() {
        let store = ZalEntityStore::open_in_memory().unwrap();

        let e1 = make_entity(0, &[1, 2]);
        let eid = store.upsert_entity(&e1).unwrap();

        // Second upsert shares record 2, should merge into the same entity.
        let e2 = make_entity(0, &[2, 3]);
        let eid2 = store.upsert_entity(&e2).unwrap();

        assert_eq!(eid, eid2, "same entity_id must be returned on merge");

        let loaded = store.get_entity(eid).unwrap();
        let rids: Vec<RecordId> = loaded.members.iter().map(|m| m.record_id).collect();
        assert!(rids.contains(&1));
        assert!(rids.contains(&2));
        assert!(rids.contains(&3));
    }

    #[test]
    fn record_to_entity_returns_correct_id() {
        let store = ZalEntityStore::open_in_memory().unwrap();
        let entity = make_entity(0, &[10, 20]);
        let eid = store.upsert_entity(&entity).unwrap();

        assert_eq!(store.record_to_entity(10).unwrap(), Some(eid));
        assert_eq!(store.record_to_entity(20).unwrap(), Some(eid));
    }

    #[test]
    fn record_to_entity_missing_returns_none() {
        let store = ZalEntityStore::open_in_memory().unwrap();
        assert!(store.record_to_entity(999).unwrap().is_none());
    }

    #[test]
    fn all_entities_returns_all() {
        let store = ZalEntityStore::open_in_memory().unwrap();
        store.upsert_entity(&make_entity(0, &[1, 2])).unwrap();
        store.upsert_entity(&make_entity(0, &[3, 4])).unwrap();
        store.upsert_entity(&make_entity(0, &[5, 6])).unwrap();

        let all = store.all_entities().unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn provenance_event_written_on_create() {
        let store = ZalEntityStore::open_in_memory().unwrap();
        store.upsert_entity(&make_entity(0, &[1, 2])).unwrap();

        let conn = store.conn.lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM resolution_events WHERE event_type = 'EntityCreated'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}
