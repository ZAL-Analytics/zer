use std::sync::Arc;

use zer_blocking::BlockerFactory;
use zer_cluster::ConnectedComponentsClusterer;
use zer_compare::{FellegiSunterScorer, FieldComparator};
use zer_core::{
    error::ZerError,
    schema::Schema,
    traits::{Blocker, Clusterer, Comparator, EntityStore, Judge, RecordStore, Scorer},
    VecRecordStore,
};
use zer_schema::SchemaRegistry;

use crate::{
    cluster_view::ClusterView, config::PipelineConfig, ingester::Ingester, progress::PipelineEvent,
};

/// Fully wired entity-resolution pipeline.
///
/// Create via [`PipelineBuilder`] (`Pipeline::builder()`). All component fields
/// are `pub(crate)` so they can be accessed from `batch.rs` and `ingester.rs`
/// without exposing them in the public API.
pub struct Pipeline {
    pub(crate) schema: Schema,
    pub(crate) blocker: Arc<dyn Blocker>,
    pub(crate) comparator: Arc<dyn Comparator>,
    /// Non-None when `config.field_mappings` is non-empty.  Used by `batch.rs`
    /// to compare cross-schema pairs via `compare_batch_mapped` instead of the
    /// union-schema pool path.
    pub(crate) mapped_comparator: Option<Arc<FieldComparator>>,
    pub(crate) scorer: Arc<dyn Scorer>,
    pub(crate) clusterer: Arc<dyn Clusterer>,
    pub(crate) store: Arc<dyn EntityStore>,
    pub(crate) record_store: Arc<dyn RecordStore>,
    pub(crate) registry: Arc<SchemaRegistry>,
    pub(crate) judge: Option<Arc<dyn Judge>>,
    pub(crate) config: PipelineConfig,
    pub(crate) progress: Option<tokio::sync::mpsc::UnboundedSender<PipelineEvent>>,
}

impl Pipeline {
    /// Create a new builder.
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::default()
    }

    /// Consume this `Arc<Pipeline>` and return an [`Ingester`] for streaming
    /// record intake.  The ingester spawns a background tokio task that owns
    /// the blocking index and per-record state.
    pub fn ingester(self: Arc<Self>) -> Ingester {
        Ingester::new(self)
    }

    /// Borrow the entity store for queries.
    pub fn store(&self) -> &Arc<dyn EntityStore> {
        &self.store
    }

    /// Borrow the record store for direct record access.
    pub fn record_store(&self) -> &Arc<dyn RecordStore> {
        &self.record_store
    }

    /// Return a [`ClusterView`] that joins the entity store with the record store.
    pub fn cluster_view(&self) -> ClusterView {
        ClusterView::new(Arc::clone(&self.store), Arc::clone(&self.record_store))
    }

    /// Borrow the schema registry.
    pub fn registry(&self) -> &Arc<SchemaRegistry> {
        &self.registry
    }

    /// Borrow the schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

// ── PipelineBuilder ───────────────────────────────────────────────────────────

/// Fluent builder for [`Pipeline`].
///
/// Required: [`PipelineBuilder::schema`] and [`PipelineBuilder::store`].
/// Everything else defaults to CPU implementations.
#[derive(Default)]
pub struct PipelineBuilder {
    schema: Option<Schema>,
    blocker: Option<Arc<dyn Blocker>>,
    comparator: Option<Arc<dyn Comparator>>,
    scorer: Option<Arc<dyn Scorer>>,
    clusterer: Option<Arc<dyn Clusterer>>,
    store: Option<Arc<dyn EntityStore>>,
    record_store: Option<Arc<dyn RecordStore>>,
    judge: Option<Arc<dyn Judge>>,
    config: PipelineConfig,
    progress: Option<tokio::sync::mpsc::UnboundedSender<PipelineEvent>>,
}

/// Attach a source label to every record in `records` in one call.
///
/// Equivalent to `records.into_iter().map(|r| r.with_source(source)).collect()`.
/// The returned `Vec` is ready to pass directly to [`Pipeline::run_batch`].
///
/// ```rust
/// use zer_pipeline::label_source;
/// use zer_core::record::Record;
///
/// let brp_records = vec![Record::new(1)];
/// let labelled = label_source(brp_records, "brp");
/// assert_eq!(labelled[0].source.as_deref(), Some("brp"));
/// ```
pub fn label_source(
    records: Vec<zer_core::record::Record>,
    source: &str,
) -> Vec<zer_core::record::Record> {
    records.into_iter().map(|r| r.with_source(source)).collect()
}

impl PipelineBuilder {
    pub fn schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn blocker(mut self, b: impl Blocker + 'static) -> Self {
        self.blocker = Some(Arc::new(b));
        self
    }

    pub fn comparator(mut self, c: impl Comparator + 'static) -> Self {
        self.comparator = Some(Arc::new(c));
        self
    }

    pub fn scorer(mut self, s: impl Scorer + 'static) -> Self {
        self.scorer = Some(Arc::new(s));
        self
    }

    pub fn clusterer(mut self, c: impl Clusterer + 'static) -> Self {
        self.clusterer = Some(Arc::new(c));
        self
    }

    pub fn store(mut self, s: impl EntityStore + 'static) -> Self {
        self.store = Some(Arc::new(s));
        self
    }

    pub fn record_store(mut self, s: impl RecordStore + 'static) -> Self {
        self.record_store = Some(Arc::new(s));
        self
    }

    /// Supply a pre-existing `Arc<dyn RecordStore>` directly.
    ///
    /// Use this when you need to share the same record store with a
    /// `DebertaJudge` (which also holds an `Arc<dyn RecordStore>`).  Sharing
    /// the Arc ensures the judge can look up records that the pipeline inserts.
    pub fn record_store_arc(mut self, s: Arc<dyn RecordStore>) -> Self {
        self.record_store = Some(s);
        self
    }

    pub fn judge(mut self, j: impl Judge + 'static) -> Self {
        self.judge = Some(Arc::new(j));
        self
    }

    pub fn config(mut self, c: PipelineConfig) -> Self {
        self.config = c;
        self
    }

    /// Attach a progress-event channel.
    ///
    /// The pipeline will send a [`PipelineEvent`] at each stage boundary inside
    /// `run_batch`.  Sends are unbounded and fire-and-forget, a slow receiver
    /// never stalls the pipeline.
    pub fn progress(mut self, tx: tokio::sync::mpsc::UnboundedSender<PipelineEvent>) -> Self {
        self.progress = Some(tx);
        self
    }

    /// Build the pipeline.  Returns an error if `schema` or `store` is missing.
    ///
    /// Defaults applied when not explicitly provided:
    /// - `blocker`: `BlockerFactory::from_schema`
    /// - `comparator`: `FieldComparator::from_schema` (CPU)
    /// - `scorer`: `FellegiSunterScorer` (CPU)
    /// - `clusterer`: `ConnectedComponentsClusterer::default()`
    pub fn build(self) -> Result<Arc<Pipeline>, ZerError> {
        let schema = self.schema.ok_or(ZerError::EmptySchema)?;
        let store = self
            .store
            .ok_or_else(|| ZerError::Store("no entity store configured".into()))?;

        let blocker = self
            .blocker
            .unwrap_or_else(|| Arc::new(BlockerFactory::from_schema(&schema)));
        let mapped_comparator: Option<Arc<FieldComparator>> =
            if self.config.field_mappings.is_empty() {
                None
            } else {
                Some(Arc::new(FieldComparator::from_mapping(
                    &self.config.field_mappings,
                    &schema,
                )))
            };
        let comparator = self
            .comparator
            .unwrap_or_else(|| Arc::new(FieldComparator::from_schema(&schema)));
        let scorer = self.scorer.unwrap_or_else(|| Arc::new(FellegiSunterScorer));
        let clusterer = self
            .clusterer
            .unwrap_or_else(|| Arc::new(ConnectedComponentsClusterer::default()));

        let record_store = self
            .record_store
            .unwrap_or_else(|| Arc::new(VecRecordStore::new()));
        let registry = Arc::new(SchemaRegistry::open(&self.config.registry_path)?);

        Ok(Arc::new(Pipeline {
            schema,
            blocker,
            comparator,
            mapped_comparator,
            scorer,
            clusterer,
            store,
            record_store,
            registry,
            judge: self.judge,
            config: self.config,
            progress: self.progress,
        }))
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use zer_cluster::ZalEntityStore;
    use zer_core::schema::{FieldKind, SchemaBuilder};

    fn person_schema() -> Schema {
        SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap()
    }

    fn temp_pipeline(dir: &TempDir) -> Arc<Pipeline> {
        let registry_path = dir.path().join("test.zsm");
        let store = ZalEntityStore::open_in_memory().unwrap();
        Pipeline::builder()
            .schema(person_schema())
            .store(store)
            .config(PipelineConfig {
                registry_path,
                ..PipelineConfig::default()
            })
            .build()
            .unwrap()
    }

    #[test]
    fn builder_with_schema_and_store_succeeds() {
        let dir = TempDir::new().unwrap();
        let pipeline = temp_pipeline(&dir);
        assert_eq!(pipeline.schema().fields.len(), 3);
    }

    #[test]
    fn builder_missing_schema_returns_error() {
        let dir = TempDir::new().unwrap();
        let store = ZalEntityStore::open_in_memory().unwrap();
        let result = Pipeline::builder()
            .store(store)
            .config(PipelineConfig {
                registry_path: dir.path().join("test.zsm"),
                ..PipelineConfig::default()
            })
            .build();
        assert!(result.is_err(), "missing schema must return an error");
    }

    #[test]
    fn builder_missing_store_returns_error() {
        let dir = TempDir::new().unwrap();
        let result = Pipeline::builder()
            .schema(person_schema())
            .config(PipelineConfig {
                registry_path: dir.path().join("test.zsm"),
                ..PipelineConfig::default()
            })
            .build();
        assert!(result.is_err(), "missing store must return an error");
    }

    #[test]
    fn store_and_registry_accessors_work() {
        let dir = TempDir::new().unwrap();
        let pipeline = temp_pipeline(&dir);
        let _store = pipeline.store();
        let _registry = pipeline.registry();
    }
}
