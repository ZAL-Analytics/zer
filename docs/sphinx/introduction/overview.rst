Five-Minute Overview
====================

The full zer pipeline has six stages. Here is a brief tour of each one, with
the minimal code needed to wire them together.

.. code-block:: text

   Records
      │
      ▼
   ┌─────────────────────────────────────────────────────┐
   │  Schema          FieldKind annotations              │
   └─────────────────────────────────────────────────────┘
      │
      ▼
   ┌─────────────────────────────────────────────────────┐
   │  Blocker         candidate pair generation          │
   └─────────────────────────────────────────────────────┘
      │
      ▼
   ┌─────────────────────────────────────────────────────┐
   │  Comparator      field-by-field similarity          │
   └─────────────────────────────────────────────────────┘
      │
      ▼
   ┌─────────────────────────────────────────────────────┐
   │  Scorer          Fellegi-Sunter match probability   │
   └─────────────────────────────────────────────────────┘
      │
      ▼
   ┌─────────────────────────────────────────────────────┐
   │  Clusterer       group matches into entities        │
   └─────────────────────────────────────────────────────┘
      │
      ▼
   ┌─────────────────────────────────────────────────────┐
   │  Entity Store    persist + audit trail              │
   └─────────────────────────────────────────────────────┘

Step 1: Define a Schema
------------------------

A ``Schema`` tells zer what each field *means*, not just its name. The
``FieldKind`` annotation drives blocking key selection, similarity function
choice, and EM parameter weighting.

.. code-block:: rust

   use zer_core::schema::{FieldKind, SchemaBuilder};

   let schema = SchemaBuilder::new()
       .field("voornamen",     FieldKind::Name)
       .field("achternaam",    FieldKind::Name)
       .field("geboortedatum", FieldKind::Date)
       .field("postcode",      FieldKind::Id)
       .build()?;

See :doc:`/reference/field-kind` for the full ``FieldKind`` table.

Step 2: Create Records
-----------------------

A ``Record`` is a bag of named ``FieldValue`` entries. IDs must be unique across all
sources you plan to link.

.. code-block:: rust

   use zer_core::record::{FieldValue, Record};

   let record = Record::new(1)
       .with_source("brp")
       .insert("voornamen",     FieldValue::Text("Jan".into()))
       .insert("achternaam",    FieldValue::Text("de Vries".into()))
       .insert("geboortedatum", FieldValue::Text("1985-03-15".into()))
       .insert("postcode",      FieldValue::Text("1011AB".into()));

Step 3: Build and Run a Pipeline
----------------------------------

``Pipeline`` wires all six stages together. ``PipelineConfig`` controls the
link mode, the EM scorer settings, and the path of the model registry file
(``.zsm``).

.. code-block:: rust

   use zer_cluster::ZalEntityStore;
   use zer_pipeline::{
       config::{LinkMode, PipelineConfig},
       label_source,
       pipeline::Pipeline,
   };

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
       .config(PipelineConfig {
           registry_path: "model.zsm".into(),
           link_mode: LinkMode::LinkOnly,
           ..PipelineConfig::default()
       })
       .build()?;

   // Label each source so the pipeline can filter cross-source vs within-source pairs
   let brp = label_source(brp_records, "brp");
   let kvk = label_source(kvk_records, "kvk");
   let all: Vec<Record> = [brp, kvk].concat();

   let report = pipeline.run_batch(all).await?;

Step 4: Read Results
---------------------

``BatchReport`` summarises what happened. ``ClusterView`` lets you iterate
over resolved entities and their member records.

.. code-block:: rust

   println!("auto-matched:    {}", report.auto_matched);
   println!("borderline:      {}", report.borderline);
   println!("entities created:{}", report.entities_created);
   println!("elapsed:         {} ms", report.elapsed_ms);

   let view = pipeline.cluster_view();

   // Iterate every resolved entity
   for (entity, records) in &view {
       println!("entity {} has {} member records", entity.id, records.len());
   }

   // Or just the cross-source linked pairs
   for pair in view.linked_pairs() {
       println!(
           "{} ({}) ↔ {} ({})  score={:.3}",
           pair.record_id_a, pair.source_a.as_deref().unwrap_or("?"),
           pair.record_id_b, pair.source_b.as_deref().unwrap_or("?"),
           pair.score,
       );
   }

Link modes
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Behaviour
   * - ``LinkMode::Deduplicate``
     - Within-source deduplication only. Cross-source pairs are skipped.
   * - ``LinkMode::LinkOnly``
     - Cross-source pairs only. Within-source pairs are skipped.
   * - ``LinkMode::LinkAndDedupe``
     - Both within-source and cross-source pairs are generated.

Model persistence
------------------

The EM-estimated parameters are stored in a ``.zsm`` (zer schema model) file.
On subsequent runs, zer loads the existing model and updates it incrementally, only new records need to be compared.

.. code-block:: rust

   PipelineConfig {
       registry_path: PathBuf::from("data/models/brp_kvk.zsm"),
       ..PipelineConfig::default()
   }

If the file does not exist, zer runs a cold-start EM estimation on the first
batch and then saves the result.

Next steps
----------

* Follow the :doc:`/tutorials/deduplication` tutorial for a full working example.
* Read :doc:`/explanation/entity-resolution` to understand the theory behind each stage.
* Consult :doc:`/reference/field-kind` for the complete ``FieldKind`` reference.
