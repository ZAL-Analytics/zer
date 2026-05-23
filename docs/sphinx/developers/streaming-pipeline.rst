Streaming Pipeline
===================

A batch pipeline loads all records up front, resolves entities, and exits.
A streaming pipeline keeps running: new records arrive continuously (from
Kafka, a message queue, a database CDC feed, or an HTTP ingest endpoint) and
the entity graph is updated incrementally without re-processing old records.

This guide shows the key patterns for building a long-running zer process
that handles a continuous record stream.

Core idea: reuse a single Pipeline instance
--------------------------------------------

``Pipeline`` is designed to be called multiple times. The EM model parameters
are persisted in a ``.zsm`` registry file between calls, so each ``run_batch``
starts from where the last one left off. The entity store accumulates clusters
across all calls.

.. code-block:: rust

   use zer_pipeline::{config::PipelineConfig, pipeline::Pipeline};
   use zer_cluster::ZalEntityStore;

   // Open a persistent entity store so clusters survive restarts
   let store = ZalEntityStore::open("/data/entities.zes")?;

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(store)
       .config(PipelineConfig {
           // Persist EM parameters between process restarts
           registry_path: "/data/pipeline.zsm".into(),
           ..PipelineConfig::default()
       })
       .build()?;

   // From here, call pipeline.run_batch(chunk).await? in a loop

Reading from a Kafka topic
---------------------------

Use ``rdkafka`` to consume a topic and feed chunks into the pipeline. Commit
the Kafka offset only after ``run_batch`` succeeds so that a crash does not
lose records:

Add the dependencies:

.. code-block:: toml

   [dependencies]
   zer          = { version = "1.0", features = ["pipeline"] }
   rdkafka      = { version = "0.36", features = ["tokio"] }
   serde_json   = { version = "1" }
   tokio        = { version = "1", features = ["full"] }

.. code-block:: rust

   use rdkafka::consumer::{Consumer, StreamConsumer};
   use rdkafka::message::Message;
   use rdkafka::ClientConfig;
   use zer_core::Record;

   const BATCH_SIZE: usize = 1_000;

   let consumer: StreamConsumer = ClientConfig::new()
       .set("group.id",          "zer-pipeline")
       .set("bootstrap.servers", "kafka:9092")
       .set("enable.auto.commit","false")
       .create()?;

   consumer.subscribe(&["persons"])?;

   let mut batch: Vec<Record>  = Vec::with_capacity(BATCH_SIZE);
   let mut id_cursor: u64      = 1;

   loop {
       // Collect up to BATCH_SIZE messages (or flush after a timeout)
       while batch.len() < BATCH_SIZE {
           match tokio::time::timeout(
               std::time::Duration::from_secs(5),
               consumer.recv(),
           ).await {
               Ok(Ok(msg))  => {
                   if let Some(payload) = msg.payload() {
                       let record: Record = serde_json::from_slice(payload)?;
                       batch.push(record);
                       id_cursor += 1;
                   }
               }
               Ok(Err(e))   => return Err(e.into()),
               Err(_timeout) => break, // flush a partial batch after 5 s idle
           }
       }

       if batch.is_empty() { continue; }

       let report = pipeline.run_batch(std::mem::take(&mut batch)).await?;
       println!(
           "batch done: +{} entities, {} ms",
           report.entities_created, report.elapsed_ms
       );

       // Commit only after a successful run_batch
       consumer.commit_consumer_state(rdkafka::consumer::CommitMode::Sync)?;
   }

Triggering EM re-estimation
------------------------------

The EM parameters are estimated from the data seen so far. In the early life
of a streaming pipeline the parameters are imprecise because the model has
seen few records. As more batches arrive the estimates stabilise.

You can trigger a full re-estimation at any time by calling
``pipeline.reset_em_model()`` and then running a larger consolidation batch.
A good strategy is to do this after the first few thousand records, and then
again at regular intervals (e.g. daily) to adapt to schema drift:

.. code-block:: rust

   use std::time::{Duration, Instant};

   let mut last_reestimate = Instant::now();
   let reestimate_interval = Duration::from_secs(24 * 3600);

   loop {
       // ... consume and run_batch as above ...

       if last_reestimate.elapsed() > reestimate_interval {
           pipeline.reset_em_model();
           last_reestimate = Instant::now();
           println!("EM model reset; parameters will re-converge over next batches");
       }
   }

Combining a custom entity store with streaming
------------------------------------------------

For a truly live entity graph, swap ``ZalEntityStore`` for a custom store that
writes to an external database on every ``upsert_cluster`` call. See
:doc:`custom-entity-store` for the full implementation pattern.

With SurrealDB as the backing store, every ``run_batch`` call automatically
propagates new and merged clusters to SurrealDB so downstream consumers always
see the current entity graph without polling a file:

.. code-block:: rust

   let store = SurrealEntityStore::connect("ws://localhost:8000").await?;

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(store)
       .config(PipelineConfig {
           registry_path: "/data/pipeline.zsm".into(),
           ..PipelineConfig::default()
       })
       .build()?;

   // Kafka consumer loop above, unchanged

Graceful shutdown
------------------

Flush the entity store and persist the EM model before exiting so the next
process start resumes cleanly:

.. code-block:: rust

   use tokio::signal;

   let pipeline = std::sync::Arc::new(pipeline);
   let pipeline_ref = std::sync::Arc::clone(&pipeline);

   tokio::spawn(async move {
       signal::ctrl_c().await.expect("signal handler");
       println!("shutting down,flushing pipeline state");
       pipeline_ref.flush().await.expect("flush failed");
       std::process::exit(0);
   });

``Pipeline::flush`` writes the current EM parameters to the ``.zsm`` file and
calls ``EntityStore::flush`` on the backing store.

Backpressure and memory management
------------------------------------

The comparison step is CPU-bound and proportional to the square of the batch
size (blocking reduces this, but does not eliminate it). If records arrive
faster than the pipeline can process them, the batch buffer grows without
bound.

Apply backpressure by capping the ingest channel:

.. code-block:: rust

   use tokio::sync::mpsc;

   // Buffer at most 10 000 records; producers block when full
   let (tx, rx) = mpsc::channel::<Record>(10_000);

When the channel is full, the Kafka consumer stops polling, which causes the
broker to pause delivery to this consumer group. Kafka retains undelivered
messages until the consumer catches up.

What to explore next
---------------------

* :doc:`custom-entity-store`, write entity clusters directly to a graph database for live querying.
* :doc:`custom-record-store`, back the neural judge's record store with RocksDB so memory stays bounded across long-running streams.
* :doc:`/how-to/async-ingestion`, async patterns for batch workloads that do not need a full streaming setup.
