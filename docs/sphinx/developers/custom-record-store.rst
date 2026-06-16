Custom Record Store
====================

The pipeline stores every ingested ``Record`` in a ``RecordStore`` before
processing. By default it uses ``VecRecordStore``, an in-memory ``Vec`` wrapped
in ``Arc``. That is fine for datasets that fit in RAM, but becomes a problem when:

* The total record count exceeds available memory.
* Records are produced by a streaming source and you cannot hold them all in
  memory before the pipeline starts.
* You want to share a record store across multiple pipeline workers or between
  the pipeline and the neural judge without duplicating data.

This guide shows how to implement the ``RecordStore`` trait and wire it into the
pipeline, with or without a neural judge.

The ``RecordStore`` trait
--------------------------

The trait lives in ``zer_core``:

.. code-block:: rust

   use std::borrow::Cow;
   use zer_core::{RecordStore, Record, RecordId};

   pub trait RecordStore: Send + Sync {
       /// Store a record. Called during ingestion, before run_batch.
       fn insert(&self, record: Record);

       /// Retrieve a record by ID. Returns None if not found.
       fn get(&self, id: RecordId) -> Option<Cow<'_, Record>>;

       /// Retrieve multiple records at once. Default implementation calls get() per ID.
       fn get_many(&self, ids: &[RecordId]) -> Vec<Option<Cow<'_, Record>>> {
           ids.iter().map(|&id| self.get(id)).collect()
       }

       /// Total number of records in the store.
       fn len(&self) -> usize;

       fn is_empty(&self) -> bool { self.len() == 0 }
   }

``insert`` is called once per record as the pipeline ingests each batch.
``get`` is called when building the internal ``RecordPool`` for comparison, and
by ``DebertaJudge`` for every pair it evaluates (so its latency directly affects
judge throughput when a judge is configured).

The trait does not propagate errors through return values. Implementations should
panic (via ``expect`` or ``unwrap``) on I/O failures, or handle them internally.

RocksDB example
----------------

RocksDB stores records serialized as MessagePack bytes. This keeps the hot path
for ``get`` at roughly one disk read per call, which is fast enough even on
rotating storage.

Add the dependencies:

.. code-block:: toml

   [dependencies]
   zer          = { version = "1.1", features = ["pipeline"] }
   rocksdb      = { version = "0.22" }
   rmp-serde    = { version = "1" }

Implement the store:

.. code-block:: rust

   use std::borrow::Cow;
   use std::sync::Arc;
   use rocksdb::{DB, Options};
   use zer_core::{RecordStore, Record, RecordId};

   pub struct RocksRecordStore {
       db: Arc<DB>,
   }

   impl RocksRecordStore {
       pub fn open(path: &str) -> anyhow::Result<Self> {
           let mut opts = Options::default();
           opts.create_if_missing(true);
           Ok(Self { db: Arc::new(DB::open(&opts, path)?) })
       }
   }

   impl RecordStore for RocksRecordStore {
       fn insert(&self, record: Record) {
           let key   = record.id.to_le_bytes();
           let value = rmp_serde::to_vec(&record).expect("serialise record");
           self.db.put(key, value).expect("RocksDB write");
       }

       fn get(&self, id: RecordId) -> Option<Cow<'_, Record>> {
           let bytes = self.db.get(id.to_le_bytes()).ok()??;
           let record: Record = rmp_serde::from_slice(&bytes).ok()?;
           Some(Cow::Owned(record))
       }

       fn len(&self) -> usize {
           self.db.iterator(rocksdb::IteratorMode::Start).count()
       }
   }

Using a custom store with the pipeline (no judge)
--------------------------------------------------

Wire the store into the pipeline via ``PipelineBuilder::record_store_arc``:

.. code-block:: rust

   use std::sync::Arc;
   use zer_pipeline::{PipelineConfig, Pipeline};
   use zer_cluster::ZalEntityStore;

   let record_store = Arc::new(RocksRecordStore::open("/tmp/zer_records")?);

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
       .record_store_arc(Arc::clone(&record_store) as Arc<dyn zer_core::RecordStore>)
       .config(PipelineConfig { registry_path: "model.zsm".into(), ..Default::default() })
       .build()?;

   let report = pipeline.run_batch(records).await?;

The pipeline calls ``record_store.insert`` for each record it ingests and uses
``record_store.get`` when building the internal comparison pool. The same store
is available between ``run_batch`` calls, so records from earlier batches can
still be retrieved.

Sharing a store with the judge
-------------------------------

When both the pipeline and the neural judge need access to the same records,
pass the same ``Arc`` to both. Using ``record_store_arc`` ensures insertions from
the pipeline are immediately visible to judge lookups:

.. code-block:: rust

   use std::sync::Arc;
   use zer_judge::{JudgeBackend, DebertaJudge, DebertaJudgeConfig, MiniLmSpec};
   use zer_pipeline::{PipelineConfig, Pipeline};
   use zer_cluster::ZalEntityStore;

   let record_store = Arc::new(RocksRecordStore::open("/tmp/zer_records")?);
   let store_ref: Arc<dyn zer_core::RecordStore> =
       Arc::clone(&record_store) as Arc<dyn zer_core::RecordStore>;

   let backend = JudgeBackend::auto_detect();
   let spec    = MiniLmSpec::from_dir("models/nli-minilm");

   let judge = DebertaJudge::new(
       &spec,
       &backend,
       Arc::clone(&store_ref),
       schema.clone(),
       DebertaJudgeConfig::default(),
   )?;

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
       .record_store_arc(store_ref)
       .judge(judge)
       .build()?;

   let report = pipeline.run_batch(records).await?;

Sizing the RocksDB block cache
---------------------------------

For large datasets the default RocksDB block cache is small. Set it
explicitly to keep frequently accessed records in memory:

.. code-block:: rust

   use rocksdb::{Options, BlockBasedOptions, Cache};

   let cache = Cache::new_lru_cache(512 * 1024 * 1024); // 512 MB
   let mut block_opts = BlockBasedOptions::default();
   block_opts.set_block_cache(&cache);

   let mut opts = Options::default();
   opts.create_if_missing(true);
   opts.set_block_based_table_factory(&block_opts);

   let db = DB::open(&opts, "/tmp/zer_records")?;

A 512 MB LRU cache is usually sufficient: records are read a small number of
times per batch, so cache hit rates are high after the first pass.

What to explore next
---------------------

* :doc:`custom-entity-store`, replace the cluster persistence layer.
* :doc:`streaming-pipeline`, coordinate a live-stream record store with a continuously running pipeline.
* :doc:`/how-to/neural-judge`, full configuration options for ``DebertaJudge``.
