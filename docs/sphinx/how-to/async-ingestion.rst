How to Use the Async Ingester
==============================

zer exposes two async entry points for ingesting records:

* **Streaming mode** via ``Pipeline::ingester()``: resolves records one at
  a time and returns a per-record result immediately. This is the primary async
  API and the focus of this guide.
* **Batch mode** via ``pipeline.run_batch(records).await``: processes a whole
  ``Vec<Record>`` at once, re-estimates EM parameters, and returns a summary
  report. Covered briefly at the end.

The key difference is that the ``Ingester`` does **not** run EM. It scores
each incoming record against the current model parameters loaded from the
``.zsm`` registry. This makes it fast enough for real-time intake but means
you should run at least one ``run_batch`` first to train the model before
switching to streaming mode.

Setting up the Tokio runtime
-----------------------------

Both APIs require a Tokio runtime. Add the dependency:

.. code-block:: toml

   [dependencies]
   zer   = { version = "1.1", features = ["pipeline"] }
   tokio = { version = "1", features = ["full"] }

Mark your entry point:

.. code-block:: rust

   #[tokio::main]
   async fn main() -> anyhow::Result<()> {
       // ...
       Ok(())
   }

Creating an Ingester
---------------------

``Pipeline::ingester`` consumes the ``Arc<Pipeline>`` and spawns a background
Tokio task that owns the blocking index and per-record state. The returned
``Ingester`` handle is the only way to send records to that task:

.. code-block:: rust

   use std::path::Path;
   use std::sync::Arc;
   use zer_pipeline::{PipelineConfig, Pipeline};
   use zer_cluster::ZalEntityStore;

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open(Path::new("/data/entities.zes"))?)
       .config(PipelineConfig {
           // The ingester loads initial EM parameters from this file.
           // Run run_batch at least once beforehand to populate it.
           registry_path: "/data/pipeline.zsm".into(),
           ..PipelineConfig::default()
       })
       .build()?;

   let ingester = Arc::clone(&pipeline).ingester();

The internal channel buffer holds up to 1 024 in-flight records. Senders
block automatically if the background task falls behind.

Sending a single record
------------------------

``Ingester::send`` takes one ``Record`` and returns an ``IngestResult`` once
the background task has processed it:

.. code-block:: rust

   use zer_core::record::{FieldValue, Record};

   let record = Record::from_key("brp", "893479421")
       .insert("voornamen",     FieldValue::Text("Jan".into()))
       .insert("achternaam",    FieldValue::Text("de Vries".into()))
       .insert("geboortedatum", FieldValue::Text("1985-03-15".into()));

   let result = ingester.send(record).await?;

   println!("record {}  band={:?}  entity={:?}",
       result.record_id,
       result.band,
       result.entity_id,
   );

``IngestResult`` contains:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``record_id``
     - The ``RecordId`` of the ingested record.
   * - ``band``
     - ``AutoMatch``, ``Borderline``, or ``AutoReject``: the Fellegi-Sunter classification.
   * - ``entity_id``
     - The assigned ``EntityId`` for ``AutoMatch`` and ``AutoReject`` records. ``None`` for ``Borderline``.
   * - ``top_match``
     - The highest-scoring candidate pair involving this record, if any candidates existed.

Understanding match bands
--------------------------

Every record is scored against candidates already in the blocking index and
classified into one of three bands:

* **AutoMatch**: the record is above the match threshold. It is merged into
  the entity of its top-scoring candidate. ``entity_id`` is set.
* **AutoReject**: the record is below the reject threshold, or no candidates
  were found. It becomes a singleton entity. ``entity_id`` is set.
* **Borderline**: the score fell between the two thresholds. The record is
  left unresolved; ``entity_id`` is ``None``. Call ``flush_borderlines`` to
  acknowledge these or handle them with the neural judge.

.. code-block:: rust

   use zer_core::scoring::MatchBand;

   let result = ingester.send(record).await?;

   match result.band {
       MatchBand::AutoMatch  => println!("merged into entity {:?}", result.entity_id),
       MatchBand::AutoReject => println!("new entity {:?}", result.entity_id),
       MatchBand::Borderline => println!("record {} needs review", result.record_id),
   }

Sending multiple records
-------------------------

``Ingester::send_all`` accepts any iterator of ``Record`` objects and processes
them in order. Each record's ID should already be derived via ``Record::from_key``
or ``into_records(&config)``. Results are returned in the same order as the input:

.. code-block:: rust

   use zer_pipeline::Ingester;

   // rows: any iterator of IntoRecord: Vec<Record>, CSV rows, etc.
   let results = ingester.send_all(rows, 1).await?;

   for r in &results {
       println!("record {}  band={:?}", r.record_id, r.band);
   }

The first error short-circuits the remaining rows and returns ``Err``.

Flushing borderline records
-----------------------------

``flush_borderlines`` tells the background task that you have finished the
current intake session and any queued borderline records should be acknowledged
(they remain unresolved for external review or neural-judge adjudication):

.. code-block:: rust

   ingester.flush_borderlines().await?;

Call this at the end of each logical batch or before querying the entity store
to ensure consistent state.

Full streaming example
-----------------------

.. code-block:: rust

   use std::path::Path;
   use std::sync::Arc;
   use zer_core::{record::{FieldValue, Record}, scoring::MatchBand};
   use zer_pipeline::{PipelineConfig, Pipeline};
   use zer_cluster::ZalEntityStore;

   #[tokio::main]
   async fn main() -> anyhow::Result<()> {
       let pipeline = Pipeline::builder()
           .schema(schema)
           .store(ZalEntityStore::open(Path::new("/data/entities.zes"))?)
           .config(PipelineConfig {
               registry_path: "/data/pipeline.zsm".into(),
               ..PipelineConfig::default()
           })
           .build()?;

       let ingester = Arc::clone(&pipeline).ingester();

       let mut auto_matched  = 0usize;
       let mut borderline    = 0usize;
       let mut auto_rejected = 0usize;

       // incoming_records() yields Record objects built with Record::from_key
       for record in incoming_records() {
           let result = ingester.send(record).await?;
           match result.band {
               MatchBand::AutoMatch  => auto_matched  += 1,
               MatchBand::Borderline => borderline    += 1,
               MatchBand::AutoReject => auto_rejected += 1,
           }
       }

       ingester.flush_borderlines().await?;

       println!("matched={auto_matched}  borderline={borderline}  rejected={auto_rejected}");
       Ok(())
   }

Batch mode: ``run_batch``
--------------------------

Use ``run_batch`` when you have a full dataset up front and want EM parameter
re-estimation. Each call processes the whole ``Vec<Record>``, updates the EM
model, clusters results, and returns a ``BatchReport``:

.. code-block:: rust

   // pipeline is Arc<Pipeline>: build() returns Arc
   let report = pipeline.run_batch(records).await?;
   println!("entities: {}  elapsed: {} ms", report.entities_created, report.elapsed_ms);

For large datasets that do not fit in memory, call ``run_batch`` in a loop
over pages. EM parameters accumulate across calls:

.. code-block:: rust

   let config = DatasetConfig::new("brp", "bsn");
   for page in data_source.pages() {
       let records = page.into_records(&config);
       pipeline.run_batch(records).await?;
   }

When to use each mode
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - ``Ingester``
     - ``run_batch``
   * - Records available
     - One at a time / live stream
     - All at once / paged file
   * - EM parameter estimation
     - No (uses loaded registry)
     - Yes (re-estimates each call)
   * - Per-record result
     - Yes (``IngestResult`` per send)
     - No (summary ``BatchReport`` only)
   * - Typical use
     - Live intake, Kafka consumer, HTTP ingest
     - Initial training, nightly batch, large file load

What to explore next
---------------------

* :doc:`/developers/streaming-pipeline`, wire the ``Ingester`` to a Kafka topic with at-least-once delivery guarantees.
* :doc:`/how-to/neural-judge`, route ``Borderline`` records from the ingester to the neural judge for adjudication.
* :doc:`/how-to/polars-arrow`, convert DataFrames to ``Record`` iterators for use with ``send_all``.
