Tutorial: Cross-Source Linkage
===============================

This tutorial links two independently-perturbed views of the same population:
a source A (authoritative municipal register) and a source B (downstream benefits
system). The two datasets share no common key, they are linked purely by name,
date of birth, and address similarity.

The full runnable demo lives in ``demos/cross_source_linkage/``.

What you will build
--------------------

A pipeline that:

1. Assigns a ``"source"`` label to records from each dataset.
2. Runs ``LinkMode::LinkOnly`` to generate only cross-source candidate pairs.
3. Evaluates the linked pairs against a ground-truth file.
4. Prints a side-by-side table of the top linked pairs with their scores.

Prepare the data
-----------------

The demo reads from ``data/v1.1/demos/linkage/``. Download the full dataset bundle
(all tutorials share the same download) as described in
:doc:`/introduction/installation`, or regenerate locally:

.. code-block:: bash

   $ python data_generator/generate_demo_linkage.py
   # Writes:   data/v1.1/demos/linkage/source_a.csv
   #           data/v1.1/demos/linkage/source_b.csv
   #           data/v1.1/demos/linkage/ground_truth.csv

The two datasets share the same synthetic population. Source A has minimal
perturbation (the authoritative view). Source B has name variants, address lag,
and missing fields that simulate a downstream system updated less frequently.

Label and load the sources
---------------------------

Source labels tell zer which records belong to which dataset.
``LinkOnly`` mode then skips any pair where both records share the same label.

Each source gets its own ``DatasetConfig`` naming the source label and the
natural-key column. IDs are derived from ``FNV-1a(source:key)``, so records
from different sources never collide even if the raw key values overlap:

.. code-block:: rust

   use zer_adapters::{DatasetConfig, PolarsIngest};

   // Source A (authoritative register) is keyed by BSN.
   // Source B (downstream system) has no authoritative ID, so use record_id.
   let records_a = load_csv("source_a.csv")?
       .into_records(&DatasetConfig::new("A", "bsn"));

   let records_b = load_csv("source_b.csv")?
       .into_records(&DatasetConfig::new("B", "record_id"));

   let all: Vec<Record> = [records_a, records_b].concat();

Build the pipeline with ``LinkOnly``
--------------------------------------

.. code-block:: rust

   use zer_pipeline::{
       config::{LinkMode, PipelineConfig},
       pipeline::Pipeline,
   };
   use zer_cluster::ZalEntityStore;

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
       .config(PipelineConfig {
           registry_path: "/tmp/demo_linkage.zsm".into(),
           link_mode:     LinkMode::LinkOnly,
           ..PipelineConfig::default()
       })
       .build()?;

   let report = pipeline.run_batch(all).await?;

   println!("candidate pairs   : {}", report.candidate_pairs);
   println!("cross-source pairs: {}", report.cross_source_pairs);
   println!("within-source pairs:{}", report.within_source_pairs);  // always 0 in LinkOnly
   println!("auto-matched      : {}", report.auto_matched);

With ``LinkOnly``, ``report.within_source_pairs`` is always zero, records from
the same source are never compared.

Read the linked pairs
----------------------

``LinkedPair`` now exposes ``record_key_a`` and ``record_key_b``. the natural
key values from the source datasets. instead of raw numeric IDs:

.. code-block:: rust

   let view   = pipeline.cluster_view();
   let linked = view.linked_pairs();

   println!("Linked pairs ({}):", linked.len());
   for pair in linked.iter().take(20) {
       let (key_a, key_b) = if pair.source_a.as_deref() == Some("A") {
           (&pair.record_key_a, &pair.record_key_b)
       } else {
           (&pair.record_key_b, &pair.record_key_a)
       };
       println!(
           "  A:{:<12} ↔ B:{:<12}  score={:.3}",
           key_a, key_b, pair.score
       );
   }

Evaluate
---------

.. code-block:: rust

   // Build set of predicted pairs as (key_a, key_b) string tuples
   let predicted: HashSet<(String, String)> = linked.iter()
       .map(|p| {
           if p.source_a.as_deref() == Some("A") {
               (p.record_key_a.clone(), p.record_key_b.clone())
           } else {
               (p.record_key_b.clone(), p.record_key_a.clone())
           }
       })
       .collect();

   let tp  = predicted.intersection(&ground_truth).count();
   let fp  = predicted.difference(&ground_truth).count();
   let fn_ = ground_truth.difference(&predicted).count();

   let precision = tp as f64 / (tp + fp) as f64;
   let recall    = tp as f64 / (tp + fn_) as f64;
   let f1        = 2.0 * precision * recall / (precision + recall);

   println!("precision: {:.3}", precision);   // typically ≥ 0.95
   println!("recall:    {:.3}", recall);       // typically ≥ 0.92
   println!("F1:        {:.3}", f1);

Run the full demo
------------------

.. code-block:: bash

   $ cargo run -p cross_source_linkage

What to explore next
---------------------

* :doc:`multi-source-linkage`, link three sources simultaneously and also
  deduplicate within each source.
* :doc:`/how-to/define-schema`, add or change fields for your own datasets.
* :doc:`/explanation/fellegi-sunter`, understand how the match probability
  is computed from comparison vectors.
