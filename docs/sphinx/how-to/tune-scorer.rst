How to Tune the Scorer
=======================

The zer scorer uses a Fellegi-Sunter probabilistic model whose parameters are
estimated by EM (Expectation-Maximization) directly from the comparison vectors, 
no labelled training data is required. This guide explains how to inspect the
estimated parameters and how to override the match/reject thresholds when the
EM defaults do not match your precision/recall requirements.

How the thresholds work
------------------------

After EM converges, the scorer produces two probability thresholds:

* **upper_threshold**, pairs with ``match_probability ≥ upper`` are
  ``AutoMatch``. Raise this to accept only higher-confidence matches.
* **lower_threshold**, pairs with ``match_probability ≤ lower`` are
  ``AutoReject``. Lower this to exclude weaker non-matches earlier.

Pairs between the two thresholds are ``Borderline``, they are either
discarded or passed to the neural judge for further adjudication.

The default thresholds are estimated from the EM log-likelihood; the pipeline
stores them in the model registry (``.zsm``) so that subsequent runs warm-start
from the previous batch's parameters.

Running with default thresholds
---------------------------------

The simplest case: omit ``upper_threshold`` and ``lower_threshold`` from
``PipelineConfig`` and let EM decide.

.. code-block:: rust

   use zer_pipeline::{config::PipelineConfig, pipeline::Pipeline};
   use zer_cluster::ZalEntityStore;

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
       .config(PipelineConfig::default())
       .build()?;

   let report = pipeline.run_batch(records).await?;
   println!("auto-matched : {}", report.auto_matched);
   println!("borderline   : {}", report.borderline);
   println!("auto-rejected: {}", report.auto_rejected);

Overriding the thresholds
--------------------------

Set ``upper_threshold`` and/or ``lower_threshold`` in ``PipelineConfig`` to
pin them. Only the ones you set are overridden; the others remain EM-estimated.

.. code-block:: rust

   use zer_pipeline::config::PipelineConfig;

   // High-precision mode: only match pairs with ≥ 95% probability
   let tight = PipelineConfig {
       upper_threshold: Some(0.95),
       lower_threshold: Some(0.05),
       ..PipelineConfig::default()
   };

   // Aggressive mode: match anything with ≥ 70% probability
   let wide = PipelineConfig {
       upper_threshold: Some(0.70),
       lower_threshold: Some(0.30),
       ..PipelineConfig::default()
   };

Tightening the thresholds pushes more pairs into the ``Borderline`` band. If
you have a neural judge attached, those pairs are re-evaluated. Without a
judge, borderline pairs are treated as non-matches.

Comparing threshold effects
----------------------------

Run the same batch under multiple configs to see the trade-off:

.. code-block:: rust

   use std::sync::Arc;
   use tempfile::TempDir;
   use zer_cluster::ZalEntityStore;
   use zer_pipeline::{config::PipelineConfig, pipeline::Pipeline};

   async fn run_with(records: Vec<Record>, schema: Schema, config: PipelineConfig)
       -> BatchReport
   {
       let dir = TempDir::new().unwrap();
       Pipeline::builder()
           .schema(schema)
           .store(ZalEntityStore::open_in_memory().unwrap())
           .config(PipelineConfig {
               registry_path: dir.path().join("demo.zsm"),
               ..config
           })
           .build()
           .unwrap()
           .run_batch(records)
           .await
           .unwrap()
   }

Typical output on a synthetic BRP dataset of 50 records:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Config
     - matched
     - borderline
     - rejected
     - elapsed_ms
   * - default (EM)
     - 12
     - 3
     - 235
     - 42
   * - tightened (0.95/0.05)
     - 9
     - 6
     - 235
     - 43
   * - wide (0.70/0.30)
     - 15
     - 0
     - 235
     - 41

Reading EM parameters directly
--------------------------------

After a batch run, the pipeline writes a ``.zsm`` registry file. You can
also estimate and inspect parameters without running the full pipeline, using
the scorer directly:

.. code-block:: rust

   use zer_compare::{FieldComparator, FellegiSunterScorer};
   use zer_core::{record_pool::RecordPool, traits::{Comparator, Scorer}};

   let cmp  = FieldComparator::from_schema(&schema);
   let pool = RecordPool::from_pairs(&training_pairs, &schema);
   let idx  = (0..training_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect::<Vec<_>>();
   let batch = cmp.compare_batch_from_pool(&pool, &idx, &schema);

   let scorer = FellegiSunterScorer;
   let params = scorer.estimate_params(&batch, None, 100)?;

   for (i, field) in schema.fields.iter().enumerate() {
       println!("{:20}  m={:.3?}  u={:.3?}",
           field.name, &params.m[i], &params.u[i]);
   }
   println!("upper={:.3}  lower={:.3}",
       params.upper_threshold, params.lower_threshold);

The ``m`` vector is the probability that each comparison level occurs **given
a true match**. The ``u`` vector is the same probability given a non-match.
A field is informative when its ``m`` and ``u`` distributions diverge.

Adjusting minimum training pairs
----------------------------------

EM needs enough comparison vectors to estimate reliable parameters. If your
dataset is small or a field is mostly null, EM may converge poorly. Two options:

* **Increase batch size**, run a larger initial ingestion to give EM more
  vectors.
* **Warm-start**, if a prior ``.zsm`` exists from a related dataset, zer
  loads it automatically and uses the previous parameters as the starting
  point for EM, reducing the iterations needed.

.. code-block:: rust

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open(std::path::Path::new("entities.zes"))?)
       .config(PipelineConfig {
           // Reuse parameters from a previous run on similar data
           registry_path: "/data/models/brp_prod.zsm".into(),
           ..PipelineConfig::default()
       })
       .build()?;

What to explore next
---------------------

* :doc:`/explanation/fellegi-sunter`, how EM estimates m/u parameters and
  why no labels are needed.
* :doc:`/how-to/neural-judge`, handle borderline pairs with a neural
  cross-encoder instead of dropping them.
* :doc:`/reference/benchmarks`, precision/recall figures for standard
  Dutch administrative datasets.
