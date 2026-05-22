Tutorial: ANPR Vehicle Passage Matching
========================================

This tutorial links vehicle passages from ANPR (Automatic Number Plate
Recognition) cameras. ANPR datasets contain OCR errors where similar-looking
characters are confused: ``1`` ↔ ``I``, ``0`` ↔ ``O``, ``8`` ↔ ``B``, ``5`` ↔ ``S``,
``2`` ↔ ``Z``. zer handles these with ``PlateOCRFuzzyKey``, which generates all
single-character variant keys for every plate.

This scenario is different from person linkage: the blocking strategy is
entirely domain-specific and cannot be inferred from the raw field names alone.
You tell zer which domain you are in via ``SchemaCategory::ANPRPassages``.

The full blocking example lives in ``crates/zer-blocking/examples/basic_blocking.rs``.

The problem
-----------

The same car passes a camera twice, but the second read has an OCR error:

.. list-table::
   :header-rows: 1
   :widths: 25 20 22 18

   * - passage_id
     - camera_id
     - tijdstip
     - kenteken
   * - 8DDE6D8D-81A
     - CAM-A12-001
     - 2025-06-01T10:04:00
     - CX-180-W *(true plate)*
   * - F3A2B891-C04
     - CAM-A12-001
     - 2025-06-01T10:04:03
     - **CX-I80-W** *(1→I confusion)*

A naive exact-match on the plate string misses this pair entirely. zer finds it.

Define the ANPR schema
-----------------------

Annotate ``kenteken`` as ``FieldKind::LicensePlate`` and the camera ID as
``FieldKind::Categorical``. This is enough for ``BlockerFactory`` to select the
right keys automatically.

.. code-block:: rust

   use zer_core::schema::{FieldKind, SchemaBuilder};

   let schema = SchemaBuilder::new()
       .field("kenteken",  FieldKind::LicensePlate)
       .field("camera_id", FieldKind::Categorical)
       .field("tijdstip",  FieldKind::Timestamp)
       .field("lat",       FieldKind::GpsCoordinate)
       .field("lon",       FieldKind::GpsCoordinate)
       .build()?;

Create the records
-------------------

.. code-block:: rust

   use zer_core::record::Record;

   let true_passage = Record::new(1)
       .with_source("anpr")
       .insert("kenteken",  "CX-180-W")
       .insert("camera_id", "CAM-A12-001")
       .insert("tijdstip",  "2025-06-01T10:04:00")
       .insert("lat",       "52.345")
       .insert("lon",       "4.901");

   let ocr_passage = Record::new(2)
       .with_source("anpr")
       .insert("kenteken",  "CX-I80-W")      // 1 → I confusion
       .insert("camera_id", "CAM-A12-001")
       .insert("tijdstip",  "2025-06-01T10:07:00")
       .insert("lat",       "52.346")
       .insert("lon",       "4.902");

   let unrelated = Record::new(3)
       .with_source("anpr")
       .insert("kenteken",  "25-XKL-9")      // different vehicle
       .insert("camera_id", "CAM-A20-003")
       .insert("tijdstip",  "2025-06-01T14:00:00")
       .insert("lat",       "51.922")
       .insert("lon",       "4.479");

Use the ANPR domain category
-----------------------------

``SchemaCategory::ANPRPassages`` selects four blocking keys automatically:
``LicensePlateNormKey``, ``PlateOCRFuzzyKey``, ``CameraTimeWindowKey``, and
``GeoGridKey``.

.. code-block:: rust

   use zer_blocking::{BlockerFactory, InvertedIndex, SchemaCategory};
   use zer_core::traits::Blocker;

   let blocker = BlockerFactory::from_schema_category(&schema, SchemaCategory::ANPRPassages);

   let mut index = InvertedIndex::new();
   for record in [&true_passage, &ocr_passage, &unrelated] {
       blocker.index_record(record, &schema, &mut index);
   }

   // Inspect the keys generated for the true plate
   println!("Keys for CX-180-W:");
   for key in blocker.blocking_keys(&true_passage, &schema) {
       println!("  {}", key);
   }

   // Look up candidates
   let candidates = blocker.candidates(&true_passage, &schema, &index);
   assert!(candidates.contains(&2));   // OCR passage found ✓
   assert!(!candidates.contains(&3));  // unrelated excluded ✓

The printed keys show how OCR fuzzy blocking works::

   Keys for CX-180-W:
     plate_norm:CX180W
     plate_ocr:CX180W
     plate_ocr:CX18OW    ← 0→O at position 4
     plate_ocr:CX1B0W    ← 8→B at position 4
     plate_ocr:CXI80W    ← 1→I at position 2   ← shared with CX-I80-W ✓
     cam_time_window:CAM-A12-001:2025-06-01:60
     geo_grid:52:4

``CXI80W`` is a key that both records emit. That is the bridge.

How OCR fuzzy blocking works
-----------------------------

``PlateOCRFuzzyKey`` normalises the plate (uppercase, strip hyphens/spaces) and
then emits one additional key for each single-character OCR confusion in the
table:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Confusion pair
     - Characters that look alike in ANPR camera images
   * - ``0`` ↔ ``O``
     - Zero vs letter O
   * - ``1`` ↔ ``I``
     - One vs letter I
   * - ``8`` ↔ ``B``
     - Eight vs letter B
   * - ``5`` ↔ ``S``
     - Five vs letter S
   * - ``2`` ↔ ``Z``
     - Two vs letter Z

The substitutions are **bidirectional**: the true plate emits an ``I``-variant
key, and the OCR-confused plate emits a ``1``-variant key. Either way, they
share at least one key.

.. note::

   Only single-character substitutions are generated. This keeps the candidate
   set small while covering the vast majority of real ANPR OCR errors.

Run inside the full pipeline
-----------------------------

To run ANPR matching as a full pipeline rather than just blocking exploration,
pass the same schema and use ``LinkMode::Dedupe`` (all passages are from the
same source) or ``LinkMode::LinkAndDedupe`` if you are cross-linking multiple
camera feeds:

.. code-block:: rust

   use zer_pipeline::{Pipeline, PipelineConfig};
   use zer_cluster::ZalEntityStore;

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
       .config(PipelineConfig {
           registry_path: "/tmp/anpr.zsm".into(),
           ..PipelineConfig::default()
       })
       .build()?;

   let report = pipeline.run_batch(passages).await?;
   println!("Linked {} vehicle entities in {} ms",
       report.entities_created, report.elapsed_ms);

What to explore next
---------------------

* :doc:`/explanation/anpr-ocr`, the full OCR confusion table and why
  bidirectional keys are necessary.
* :doc:`/how-to/blocking-strategy`, how to use ``CustomSchemaCategory`` when
  the built-in ``ANPRPassages`` preset does not quite fit.
* :doc:`/reference/blocking-keys`, every blocking key, its parameters, and
  the domains it is designed for.
