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

The demo reads from ``data/demos/linkage/``. Download the full dataset bundle
(all tutorials share the same download) as described in
:doc:`/introduction/installation`, or regenerate locally:

.. code-block:: bash

   $ python data_generator/generate_demo_linkage.py
   # Writes:   data/demos/linkage/source_a.csv
   #           data/demos/linkage/source_b.csv
   #           data/demos/linkage/ground_truth.csv

The two datasets share the same synthetic population. Source A has minimal
perturbation (the authoritative view). Source B has name variants, address lag,
and missing fields that simulate a downstream system updated less frequently.

Label and load the sources
---------------------------

Source labels tell zer which records belong to which dataset.
``LinkOnly`` mode then skips any pair where both records share the same label.

.. code-block:: rust

   use zer_pipeline::label_source;

   // Source B IDs are offset to avoid collisions with Source A in the
   // same record store.
   let id_offset: u64 = source_a_rows.len() as u64 + 1;

   let records_a: Vec<Record> = source_a_rows
       .into_iter()
       .map(|row| {
           Record::new(row.record_id)
               .with_source("A")
               .insert("voornamen",     row.voornamen)
               .insert("achternaam",    row.achternaam)
               .insert("geboortedatum", row.geboortedatum)
               .insert("postcode",      row.postcode)
               // ... other fields
       })
       .collect();

   let records_b: Vec<Record> = source_b_rows
       .into_iter()
       .map(|row| {
           Record::new(row.record_id + id_offset)
               .with_source("B")
               .insert("voornamen",     row.voornamen)
               .insert("achternaam",    row.achternaam)
               .insert("geboortedatum", row.geboortedatum)
               .insert("postcode",      row.postcode)
       })
       .collect();

   // Alternatively, use label_source() to apply the same label to a whole Vec
   let all: Vec<Record> = [records_a, records_b].concat();

.. note::

   ID namespaces must not overlap. If Source A uses IDs 1–500, start Source B
   IDs at 501 or higher. An ID collision will silently merge two unrelated
   records into the same slot in the entity store.

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

.. code-block:: rust

   let view   = pipeline.cluster_view();
   let linked = view.linked_pairs();

   println!("Linked pairs ({}):", linked.len());
   for pair in linked.iter().take(20) {
       // Identify which side is A and which is B
       let (a_id, b_id) = if pair.source_a.as_deref() == Some("A") {
           (pair.record_id_a, pair.record_id_b - id_offset)
       } else {
           (pair.record_id_b, pair.record_id_a - id_offset)
       };
       println!(
           "  A:{:<6} ↔ B:{:<6}  score={:.3}",
           a_id, b_id, pair.score
       );
   }

Evaluate
---------

.. code-block:: rust

   // Build set of predicted pairs in (original_a_id, original_b_id) form
   let predicted: HashSet<(u64, u64)> = linked.iter()
       .map(|p| {
           if p.source_a.as_deref() == Some("A") {
               (p.record_id_a, p.record_id_b - id_offset)
           } else {
               (p.record_id_b, p.record_id_a - id_offset)
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
