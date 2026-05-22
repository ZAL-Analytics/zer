use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};
use zer_blocking::InvertedIndex;
use zer_core::{
    entity::{Entity, EntityId, EntityMember, ResolutionMethod},
    error::ZerError,
    record::{Record, RecordId},
    record_pool::RecordPool,
    scoring::{MatchBand, ModelParams, ScoredPair},
    traits::{IntoRecord, RecordStore},
};
use zer_schema::SchemaFingerprint;

use crate::{batch::default_params, pipeline::Pipeline, rate::RateAdapter};

/// Result returned for each ingested record.
#[derive(Debug, Clone)]
pub struct IngestResult {
    pub record_id: RecordId,
    /// Assigned entity, if the record was auto-matched or auto-rejected into one.
    pub entity_id: Option<EntityId>,
    /// Scoring band for the best candidate pair (or AutoReject for singletons).
    pub band:      MatchBand,
    /// The highest-scoring candidate pair, if any candidates existed.
    pub top_match: Option<ScoredPair>,
}

// ── Internal message type ─────────────────────────────────────────────────────

enum IngesterMsg {
    Ingest(Record, oneshot::Sender<Result<IngestResult, ZerError>>),
    FlushBorderlines(oneshot::Sender<Result<(), ZerError>>),
}

// ── Public handle ─────────────────────────────────────────────────────────────

/// Streaming record intake handle produced by [`Pipeline::ingester`].
///
/// Internally drives a single background tokio task that owns the blocking
/// index and per-record state.  Send records one at a time with [`Ingester::send`].
pub struct Ingester {
    tx: mpsc::Sender<IngesterMsg>,
}

impl Ingester {
    /// Spawn the background task.  Called by [`Pipeline::ingester`].
    pub(crate) fn new(pipeline: Arc<Pipeline>) -> Self {
        let (tx, rx) = mpsc::channel(1_024);
        tokio::spawn(run_ingester(pipeline, rx));
        Self { tx }
    }

    /// Ingest one record and await its resolution result.
    pub async fn send(&self, record: Record) -> Result<IngestResult, ZerError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(IngesterMsg::Ingest(record, resp_tx))
            .await
            .map_err(|_| ZerError::Store("ingester task has shut down".into()))?;
        resp_rx
            .await
            .map_err(|_| ZerError::Store("ingester task dropped response".into()))?
    }

    /// Ingest an iterator of rows that implement [`IntoRecord`], assigning IDs
    /// sequentially starting from `id_start`.
    ///
    /// Returns results in the same order as the input iterator.  The first
    /// error short-circuits the remaining rows.
    pub async fn send_all<I>(&self, rows: I, id_start: RecordId) -> Result<Vec<IngestResult>, ZerError>
    where
        I: IntoIterator,
        I::Item: IntoRecord,
    {
        let mut results = Vec::new();
        for (i, row) in rows.into_iter().enumerate() {
            let record = row.into_record(id_start + i as RecordId);
            results.push(self.send(record).await?);
        }
        Ok(results)
    }

    /// Acknowledge any queued borderline records (they remain unresolved for human review).
    pub async fn flush_borderlines(&self) -> Result<(), ZerError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(IngesterMsg::FlushBorderlines(resp_tx))
            .await
            .map_err(|_| ZerError::Store("ingester task has shut down".into()))?;
        resp_rx
            .await
            .map_err(|_| ZerError::Store("ingester task dropped response".into()))?
    }
}

// ── Background task ───────────────────────────────────────────────────────────

struct IngesterState {
    pipeline:     Arc<Pipeline>,
    index:        InvertedIndex,
    record_store: Arc<dyn RecordStore>,
    params:       ModelParams,
    rate_adapter: RateAdapter,
}

impl IngesterState {
    fn new(pipeline: Arc<Pipeline>) -> Self {
        let rate_adapter  = RateAdapter::new(pipeline.config.rate_config.clone());
        let params        = load_initial_params(&pipeline);
        let record_store  = Arc::clone(&pipeline.record_store);
        Self {
            pipeline,
            index: InvertedIndex::new(),
            record_store,
            params,
            rate_adapter,
        }
    }
}

fn load_initial_params(pipeline: &Pipeline) -> ModelParams {
    let fp = SchemaFingerprint::from_schema(&pipeline.schema);
    match pipeline.registry.lookup_startup_mode(&fp) {
        Ok(zer_schema::StartupMode::WarmLoad(art))              => art.params,
        Ok(zer_schema::StartupMode::WarmStart { artifact, .. }) => artifact.params,
        _                                                        => default_params(pipeline.schema.fields.len()),
    }
}

async fn run_ingester(pipeline: Arc<Pipeline>, mut rx: mpsc::Receiver<IngesterMsg>) {
    let mut state = IngesterState::new(Arc::clone(&pipeline));

    while let Some(msg) = rx.recv().await {
        match msg {
            IngesterMsg::Ingest(record, resp) => {
                let result = process_record(&mut state, record);
                let _ = resp.send(result);
            }
            IngesterMsg::FlushBorderlines(resp) => {
                let _ = resp.send(Ok(()));
            }
        }
    }
}

fn process_record(state: &mut IngesterState, record: Record) -> Result<IngestResult, ZerError> {
    let record_id = record.id;
    state.rate_adapter.tick();

    // Candidates BEFORE indexing so the record isn't its own candidate.
    let cand_ids = state
        .pipeline
        .blocker
        .candidates(&record, &state.pipeline.schema, &state.index);

    // Persist and index the new record.
    state.record_store.insert(record.clone());
    state.pipeline.blocker.index_record(&record, &state.pipeline.schema, &mut state.index);

    if cand_ids.is_empty() {
        return singleton_result(&*state.pipeline.store, record_id);
    }

    // Build a mini-pool: new record at position 0, candidates at 1..N.
    let mut ids_for_pool: Vec<RecordId> = vec![record_id];
    ids_for_pool.extend_from_slice(&cand_ids);
    let pool = RecordPool::from_store(&*state.record_store, &ids_for_pool, &state.pipeline.schema);

    // All pairs involve the new record (position 0), already canonical (0 < i).
    let pair_indices: Vec<(usize, usize)> = (1..pool.len()).map(|i| (0, i)).collect();

    if pair_indices.is_empty() {
        return singleton_result(&*state.pipeline.store, record_id);
    }

    let batch = state.pipeline.comparator.compare_batch_from_pool(
        &pool,
        &pair_indices,
        &state.pipeline.schema,
    );
    let effective_params = state.rate_adapter.adjusted_params(&state.params);
    let scored = state.pipeline.scorer.score_batch(&batch, &effective_params);

    // Best pair involving this record.
    let top_match = scored
        .iter()
        .filter(|sp| sp.record_a == record_id || sp.record_b == record_id)
        .max_by(|a, b| {
            a.match_weight
                .partial_cmp(&b.match_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    let band = top_match.as_ref().map_or(MatchBand::AutoReject, |sp| sp.band);

    let entity_id = match band {
        MatchBand::AutoMatch => {
            if let Some(ref sp) = top_match {
                let partner_id = if sp.record_a == record_id { sp.record_b } else { sp.record_a };
                merge_into_entity(
                    &*state.pipeline.store,
                    record_id,
                    partner_id,
                    sp.match_probability,
                )?
            } else {
                singleton_entity_id(&*state.pipeline.store, record_id)?
            }
        }
        MatchBand::AutoReject => singleton_entity_id(&*state.pipeline.store, record_id)?,
        MatchBand::Borderline => {
            // Leave unresolved, caller can call flush_borderlines or handle externally.
            return Ok(IngestResult { record_id, entity_id: None, band, top_match });
        }
    };

    Ok(IngestResult { record_id, entity_id: Some(entity_id), band, top_match })
}

// ── Entity persistence helpers ────────────────────────────────────────────────

fn singleton_result(
    store:     &dyn zer_core::traits::EntityStore,
    record_id: RecordId,
) -> Result<IngestResult, ZerError> {
    let entity_id = singleton_entity_id(store, record_id)?;
    Ok(IngestResult {
        record_id,
        entity_id: Some(entity_id),
        band:      MatchBand::AutoReject,
        top_match: None,
    })
}

fn singleton_entity_id(
    store:     &dyn zer_core::traits::EntityStore,
    record_id: RecordId,
) -> Result<EntityId, ZerError> {
    // If this record already belongs to an entity, return that entity.
    if let Some(eid) = store.record_to_entity(record_id)? {
        return Ok(eid);
    }
    let entity = Entity {
        id:      record_id,
        members: vec![EntityMember {
            record_id,
            score:  1.0,
            method: ResolutionMethod::Manual,
            source: None,
        }],
    };
    store.upsert_entity(&entity)
}

fn merge_into_entity(
    store:       &dyn zer_core::traits::EntityStore,
    record_id:   RecordId,
    partner_id:  RecordId,
    score:       f32,
) -> Result<EntityId, ZerError> {
    let existing_eid = store.record_to_entity(partner_id)?;
    let mut entity = if let Some(eid) = existing_eid {
        store.get_entity(eid)?
    } else {
        Entity {
            id:      partner_id,
            members: vec![EntityMember {
                record_id: partner_id,
                score,
                method: ResolutionMethod::AutoMatch,
                source: None,
            }],
        }
    };
    // Only add if not already a member.
    if !entity.members.iter().any(|m| m.record_id == record_id) {
        entity.members.push(EntityMember {
            record_id,
            score,
            method: ResolutionMethod::AutoMatch,
            source: None,
        });
    }
    store.upsert_entity(&entity)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use zer_cluster::ZalEntityStore;
    use zer_core::{
        record::FieldValue,
        schema::{FieldKind, SchemaBuilder},
    };

    use crate::{config::PipelineConfig, pipeline::Pipeline};

    fn person_schema() -> zer_core::schema::Schema {
        SchemaBuilder::new()
            .field("voornamen",     FieldKind::Name)
            .field("achternaam",    FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap()
    }

    fn make_pipeline(dir: &TempDir) -> Arc<Pipeline> {
        Pipeline::builder()
            .schema(person_schema())
            .store(ZalEntityStore::open_in_memory().unwrap())
            .config(PipelineConfig {
                registry_path: dir.path().join("test.zsm"),
                ..PipelineConfig::default()
            })
            .build()
            .unwrap()
    }

    fn make_record(id: u64, name: &str, last: &str, dob: &str) -> Record {
        Record::new(id)
            .insert("voornamen",     FieldValue::Text(name.into()))
            .insert("achternaam",    FieldValue::Text(last.into()))
            .insert("geboortedatum", FieldValue::Text(dob.into()))
    }

    #[tokio::test]
    async fn singleton_gets_entity() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let ingester = Arc::clone(&pipeline).ingester();
        let result   = ingester.send(make_record(1, "Alice", "Smith", "1990-01-01")).await.unwrap();
        assert_eq!(result.record_id, 1);
        assert!(result.entity_id.is_some(), "singleton must be assigned an entity");
    }

    #[tokio::test]
    async fn second_record_has_correct_id() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let ingester = Arc::clone(&pipeline).ingester();
        let _r1 = ingester.send(make_record(1, "Jan", "de Vries", "1985-03-15")).await.unwrap();
        let r2  = ingester.send(make_record(2, "Jan", "de Vries", "1985-03-15")).await.unwrap();
        assert_eq!(r2.record_id, 2);
    }

    #[tokio::test]
    async fn flush_borderlines_succeeds() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let ingester = Arc::clone(&pipeline).ingester();
        ingester.send(make_record(1, "Test", "User", "2000-01-01")).await.unwrap();
        ingester.flush_borderlines().await.unwrap();
    }

    #[tokio::test]
    async fn multiple_records_returned_in_order() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let ingester = Arc::clone(&pipeline).ingester();
        for i in 1u64..=5 {
            let result = ingester
                .send(make_record(i, "Anna", "Jansen", "1992-07-04"))
                .await
                .unwrap();
            assert_eq!(result.record_id, i);
        }
    }

    #[tokio::test]
    async fn distinct_records_each_get_entity() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let ingester = Arc::clone(&pipeline).ingester();
        let r1 = ingester.send(make_record(1, "Alice",  "Smith",   "1990-01-01")).await.unwrap();
        let r2 = ingester.send(make_record(2, "Bob",    "Jones",   "1975-06-20")).await.unwrap();
        let r3 = ingester.send(make_record(3, "Carlos", "Ramirez", "1988-11-03")).await.unwrap();
        assert!(r1.entity_id.is_some());
        assert!(r2.entity_id.is_some());
        assert!(r3.entity_id.is_some());
    }
}
