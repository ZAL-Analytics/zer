Tutorial: Multi-Source Linkage (BRP + KvK)
===========================================

This tutorial runs the full zer pipeline against two real Dutch administrative
data domains simultaneously:

* **BRP**, Basisregistratie Personen (municipal person register)
* **KvK**, Kamer van Koophandel director extract

The tutorial runs the pipeline **twice** on the same data to demonstrate both
link modes:

1. ``LinkOnly``, find which BRP persons are also registered as KvK directors.
2. ``LinkAndDedupe``, same cross-source linkage, plus deduplication within each
   register.

The full runnable demo lives in ``demos/multi_source_linkage/``.

Prepare the data
-----------------

The demo reads from ``data/demos/multi_source/``. Download the full dataset
bundle (all tutorials share the same download) as described in
:doc:`/introduction/installation`, or regenerate locally:

.. code-block:: bash

   $ python data_generator/generate_demo_multi_source.py
   # Writes:   data/demos/multi_source/source_brp.csv
   #           data/demos/multi_source/source_kvk.csv
   #           data/demos/multi_source/ground_truth.csv

Load both sources
------------------

KvK record IDs are offset by ``10_000_000`` to avoid namespace collisions with
BRP IDs.

.. code-block:: rust

   use zer_pipeline::label_source;

   const KVK_ID_OFFSET: u64 = 10_000_000;

   let brp_records: Vec<Record> = brp_rows.into_iter().map(|row| {
       Record::new(row.record_id)
           .insert("voornamen",     row.voornamen)
           .insert("tussenvoegsel", row.tussenvoegsel)
           .insert("achternaam",    row.achternaam)
           .insert("geboortedatum", row.geboortedatum)
           .insert("geslacht",      row.geslacht)
           .insert("postcode",      row.postcode)
   }).collect();

   let kvk_records: Vec<Record> = kvk_rows.into_iter().map(|row| {
       Record::new(row.record_id + KVK_ID_OFFSET)
           .insert("voornamen",     row.voornamen)
           .insert("tussenvoegsel", row.tussenvoegsel)
           .insert("achternaam",    row.achternaam)
           .insert("geboortedatum", row.geboortedatum)
           .insert("postcode",      row.postcode)
   }).collect();

   // Apply source labels, then merge into one batch
   let all_records: Vec<Record> = {
       let brp = label_source(brp_records, "brp");
       let kvk = label_source(kvk_records, "kvk");
       [brp, kvk].concat()
   };

Mode 1: LinkOnly (cross-source only)
--------------------------------------

.. code-block:: rust

   use std::sync::Arc;
   use zer_compute::{DeviceBackend, DeviceScorer};
   use zer_pipeline::{LinkMode, Pipeline, PipelineConfig};
   use zer_cluster::ZalEntityStore;

   // Auto-detect the best available compute backend (CUDA → AVX2 → CPU)
   let backend = Arc::new(DeviceBackend::auto_detect());
   println!("compute backend: {}", backend.name());

   let pipeline = Pipeline::builder()
       .schema(build_schema())
       .store(ZalEntityStore::open_in_memory()?)
       .scorer(DeviceScorer::new(Arc::clone(&backend)))
       .config(PipelineConfig {
           registry_path: "/tmp/multi_link_only.zsm".into(),
           link_mode:     LinkMode::LinkOnly,
           ..PipelineConfig::default()
       })
       .build()?;

   let report = pipeline.run_batch(all_records.clone()).await?;

   println!("total records     : {}", report.total_records);
   println!("cross-source pairs: {}", report.cross_source_pairs);
   println!("within-source pairs:{}", report.within_source_pairs);  // 0 in LinkOnly
   println!("auto-matched      : {}", report.auto_matched);
   println!("entities created  : {}", report.entities_created);

The ``scorer(DeviceScorer::new(...))`` call is optional, the pipeline defaults
to the CPU scorer if you omit it. Passing an explicit ``DeviceScorer`` activates
GPU-accelerated batch comparison when CUDA is available.

Mode 2: LinkAndDedupe
----------------------

.. code-block:: rust

   let pipeline_lad = Pipeline::builder()
       .schema(build_schema())
       .store(ZalEntityStore::open_in_memory()?)
       .scorer(DeviceScorer::new(Arc::clone(&backend)))
       .config(PipelineConfig {
           registry_path: "/tmp/multi_link_and_dedupe.zsm".into(),
           link_mode:     LinkMode::LinkAndDedupe,
           ..PipelineConfig::default()
       })
       .build()?;

   let report_lad = pipeline_lad.run_batch(all_records).await?;

   println!("cross-source pairs : {}", report_lad.cross_source_pairs);
   println!("within-source pairs: {}", report_lad.within_source_pairs);

With ``LinkAndDedupe``, both counts are non-zero:
``within_source_pairs`` captures internal duplicates within BRP and KvK
independently, while ``cross_source_pairs`` captures BRP ↔ KvK linkage.

Evaluate cross-source linkage
-------------------------------

.. code-block:: rust

   let view       = pipeline.cluster_view();
   let linked     = view.linked_pairs();

   let predicted: HashSet<(u64, u64)> = linked.iter()
       .map(|p| {
           let (brp_id, kvk_id) = if p.source_a.as_deref() == Some("brp") {
               (p.record_id_a, p.record_id_b)
           } else {
               (p.record_id_b, p.record_id_a)
           };
           (brp_id.min(kvk_id), brp_id.max(kvk_id))
       })
       .collect();

   let tp  = predicted.intersection(&gt_cross).count();
   let fp  = predicted.difference(&gt_cross).count();
   let fn_ = gt_cross.difference(&predicted).count();

   println!("precision: {:.3}", tp as f64 / (tp + fp) as f64);
   println!("recall:    {:.3}", tp as f64 / (tp + fn_) as f64);

Evaluate within-source deduplication
--------------------------------------

For the ``LinkAndDedupe`` run, within-source duplicates are found by scanning
clusters for members from the same source:

.. code-block:: rust

   let view_lad = pipeline_lad.cluster_view();
   let mut predicted_within: HashSet<(u64, u64)> = HashSet::new();

   for (_, members) in &view_lad {
       let brp_ids: Vec<u64> = members.iter()
           .filter(|r| r.source.as_deref() == Some("brp"))
           .map(|r| r.id)
           .collect();
       // Enumerate all within-source pairs in this cluster
       for i in 0..brp_ids.len() {
           for j in (i + 1)..brp_ids.len() {
               let a = brp_ids[i].min(brp_ids[j]);
               let b = brp_ids[i].max(brp_ids[j]);
               predicted_within.insert((a, b));
           }
       }
       // Do the same for kvk_ids...
   }

Mode comparison
----------------

Typical output on the synthetic multi-source dataset:

.. list-table::
   :header-rows: 1
   :widths: 35 20 20

   * - Metric
     - LinkOnly
     - LinkAndDedupe
   * - candidate pairs
     - 4,812
     - 7,091
   * - cross-source pairs
     - 4,812
     - 4,831
   * - within-source pairs
     - 0
     - 2,260
   * - auto-matched
     - 423
     - 611
   * - entities created
     - 423
     - 599

Run the full demo
------------------

.. code-block:: bash

   $ cargo run -p multi_source_linkage

What to explore next
---------------------

* :doc:`/how-to/gpu-backend`, configure CUDA or Vulkan for large datasets.
* :doc:`/explanation/blocking-recall`, why recall is the critical blocking metric.
* :doc:`/how-to/tune-scorer`, adjust EM thresholds for your precision/recall trade-off.
