Custom Record Store
====================

The neural judge retrieves the raw ``Record`` values for each candidate pair
it evaluates. By default those records live in ``VecRecordStore``, an in-memory
``Vec`` wrapped in ``Arc``. That is fine for datasets that fit in RAM, but
becomes a problem when:

* The total record count exceeds available memory.
* Records are produced by a streaming source and you cannot hold them all in
  memory before the pipeline starts.
* You want to share a record store across multiple pipeline workers without
  duplicating data.

This guide shows how to implement the ``RecordStore`` trait and use it with the
neural judge.

The ``RecordStore`` trait
--------------------------

The trait lives in ``zer_core``:

.. code-block:: rust

   use zer_core::{RecordStore, Record, RecordId};

   pub trait RecordStore: Send + Sync + 'static {
       type Error: std::error::Error + Send + Sync + 'static;

       /// Store a record. Called during ingestion, before run_batch.
       fn insert(&self, id: RecordId, record: Record) -> Result<(), Self::Error>;

       /// Retrieve a record by ID. Called by the judge for each candidate pair.
       fn get(&self, id: RecordId) -> Result<Option<Record>, Self::Error>;
   }

``insert`` is called once per record as you build up the input set.
``get`` is called by ``DebertaJudge`` for every pair it evaluates, so its
latency directly affects judge throughput.

RocksDB example
----------------

RocksDB stores records serialized as MessagePack bytes. This keeps the hot path
for ``get`` at roughly one disk read per call, which is fast enough for judge
throughput even on rotating storage.

Add the dependencies:

.. code-block:: toml

   [dependencies]
   zer          = { version = "1.0", features = ["pipeline", "judge_cpu"] }
   rocksdb      = { version = "0.22" }
   rmp-serde    = { version = "1" }

Implement the store:

.. code-block:: rust

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
       type Error = anyhow::Error;

       fn insert(&self, id: RecordId, record: Record) -> Result<(), Self::Error> {
           let key   = id.to_le_bytes();
           let value = rmp_serde::to_vec(&record)?;
           self.db.put(key, value)?;
           Ok(())
       }

       fn get(&self, id: RecordId) -> Result<Option<Record>, Self::Error> {
           match self.db.get(id.to_le_bytes())? {
               None        => Ok(None),
               Some(bytes) => Ok(Some(rmp_serde::from_slice(&bytes)?)),
           }
       }
   }

Wire it into the judge and pipeline:

.. code-block:: rust

   use std::sync::Arc;
   use zer_judge::{
       backend::JudgeBackend,
       judge::{DebertaJudge, DebertaJudgeConfig},
       spec::MiniLmSpec,
   };
   use zer_pipeline::{config::PipelineConfig, pipeline::Pipeline};
   use zer_cluster::ZalEntityStore;

   let record_store = Arc::new(RocksRecordStore::open("/tmp/zer_records")?);

   // Populate the store before running the pipeline
   for (id, record) in records.iter().enumerate() {
       record_store.insert(id as u64 + 1, record.clone())?;
   }

   let backend = JudgeBackend::auto_detect();
   let spec    = MiniLmSpec::from_dir("models/nli-minilm");

   let judge = DebertaJudge::new(
       &spec,
       &backend,
       Arc::clone(&record_store) as Arc<dyn zer_core::RecordStore<Error = _>>,
       schema.clone(),
       DebertaJudgeConfig::default(),
   )?;

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
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

A 512 MB LRU cache is usually sufficient for judge workloads: the judge only
reads each record a small number of times, so cache hit rates are high after
the first pass.

Streaming inserts alongside run_batch
---------------------------------------

Because ``RecordStore::insert`` and ``run_batch`` are independent, you can
populate the store in a background task while the pipeline is running earlier
chunks. This is most useful when records arrive over a network stream:

.. code-block:: rust

   use tokio::sync::mpsc;
   use std::sync::Arc;

   let store = Arc::new(RocksRecordStore::open("/tmp/zer_records")?);
   let (tx, mut rx) = mpsc::channel::<(RecordId, Record)>(1_024);

   // Background task: insert records as they arrive
   let store_ref = Arc::clone(&store);
   tokio::spawn(async move {
       while let Some((id, rec)) = rx.recv().await {
           store_ref.insert(id, rec).expect("insert failed");
       }
   });

   // Main task: send records and run pipeline in overlapping chunks
   for (id, record) in source.stream() {
       tx.send((id, record.clone())).await?;
       batch.push(record);
       if batch.len() >= CHUNK_SIZE {
           pipeline.run_batch(std::mem::take(&mut batch)).await?;
       }
   }

What to explore next
---------------------

* :doc:`custom-entity-store`, replace the cluster persistence layer.
* :doc:`streaming-pipeline`, coordinate a live-stream record store with a continuously running pipeline.
* :doc:`/how-to/neural-judge`, full configuration options for ``DebertaJudge``.
