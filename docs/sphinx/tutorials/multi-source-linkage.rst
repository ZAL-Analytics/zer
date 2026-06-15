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

The demo reads from ``data/v1.1/demos/multi_source/``. Download the full dataset
bundle (all tutorials share the same download) as described in
:doc:`/introduction/installation`, or regenerate locally:

.. code-block:: bash

   $ python data_generator/generate_demo_multi_source.py
   # Writes:   data/v1.1/demos/multi_source/source_brp.csv
   #           data/v1.1/demos/multi_source/source_kvk.csv
   #           data/v1.1/demos/multi_source/ground_truth.csv

Load both sources
------------------

Each source gets its own ``DatasetConfig`` naming its source label and natural-key
column. IDs are derived from ``FNV-1a(source:key)``, so BRP and KvK records
never collide even if the raw key values overlap. no manual offset needed:

.. code-block:: rust

   use zer_adapters::{DatasetConfig, PolarsIngest};

   let brp_records = load_csv("source_brp.csv")?
       .into_records(&DatasetConfig::new("brp", "bsn"));

   let kvk_records = load_csv("source_kvk.csv")?
       .into_records(&DatasetConfig::new("kvk", "kvk_nummer"));

   let all_records: Vec<Record> = [brp_records, kvk_records].concat();

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

   let predicted: HashSet<(String, String)> = linked.iter()
       .map(|p| {
           let (brp_key, kvk_key) = if p.source_a.as_deref() == Some("brp") {
               (p.record_key_a.clone(), p.record_key_b.clone())
           } else {
               (p.record_key_b.clone(), p.record_key_a.clone())
           };
           (brp_key, kvk_key)
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
   let mut predicted_within: HashSet<(String, String)> = HashSet::new();

   for (_, members) in &view_lad {
       let brp_keys: Vec<&str> = members.iter()
           .filter(|m| m.source.as_deref() == Some("brp"))
           .map(|m| m.record_key.as_str())
           .collect();
       // Enumerate all within-source pairs in this cluster
       for i in 0..brp_keys.len() {
           for j in (i + 1)..brp_keys.len() {
               let a = brp_keys[i].min(brp_keys[j]).to_string();
               let b = brp_keys[i].max(brp_keys[j]).to_string();
               predicted_within.insert((a, b));
           }
       }
       // Do the same for kvk_keys...
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
