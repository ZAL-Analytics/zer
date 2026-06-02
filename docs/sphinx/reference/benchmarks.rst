Benchmarks
===========

Performance figures for zer on standard Dutch administrative datasets.
All benchmarks were run on the synthetic Dutch law-enforcement datasets
available on
`Hugging Face <https://huggingface.co/datasets/arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset>`_
(or regenerated locally via the ``data_generator`` scripts). Production
figures will vary with data quality, field cardinality, and hardware.

Splink comparisons use identical datasets and field schemas.  zer runs
unsupervised Fellegi-Sunter EM; Splink runs its built-in DuckDB EM on a
single CPU thread (Python 3.12, Splink 4.x).

.. contents:: On this page
   :local:
   :depth: 2

----

Accuracy
---------

The table below covers all eight benchmark scenarios.  All runs use
default thresholds; no per-dataset tuning was applied.

.. list-table::
   :header-rows: 2
   :widths: 30 10 10 10 10 10 10 10 10

   * - Scenario
     - \
     - zer
     - \
     - \
     - \
     - Splink
     - \
     - \
   * - \
     - Records
     - Precision
     - Recall
     - F1
     - PR-AUC
     - Precision
     - Recall
     - F1
   * - BRP deduplication
     - 22 200
     - 0.984
     - 0.982
     - **0.983**
     - 0.991
     - 0.996
     - 0.922
     - 0.958
   * - KvK deduplication
     - 22 200
     - 0.910
     - 1.000
     - **0.953**
     - 0.916
     - 0.998
     - 0.925
     - 0.960
   * - BRP link (LinkOnly)
     - 28 400
     - 0.976
     - 0.991
     - **0.983**
     - 0.997
     - 0.964
     - 0.714
     - 0.820
   * - BRP + KvK link (LinkOnly)
     - 25 200
     - 0.788
     - 0.975
     - 0.872
     - 0.938
     - 0.912
     - 0.877
     - **0.895**
   * - BRP + SIS II link (LinkOnly)
     - 21 200
     - 1.000
     - 0.984
     - **0.992**
     - 0.984
     - 0.926
     - 0.823
     - 0.871
   * - BRP + HKS link (LinkOnly)
     - 23 200
     - 1.000
     - 0.992
     - **0.996**
     - 0.993
     - 0.924
     - 0.819
     - 0.868
   * - BRP + KvK link + dedupe (LinkAndDedupe)
     - 31 200
     - 0.843
     - 0.976
     - **0.905**
     - 0.923
     - 0.904
     - 0.771
     - 0.832
   * - BRP + KvK + HKS link + dedupe (LinkAndDedupe)
     - 30 200
     - 0.850
     - 0.985
     - **0.913**
     - 0.916
     - 0.702
     - 0.889
     - 0.784

.. raw:: html

   <div style="margin: 1.5rem 0;">
     <img src="../res/accuracy_comparison.svg"
          alt="Accuracy comparison across all scenarios"
          style="max-width:100%; border-radius:6px;" />
   </div>

The chart above plots precision, recall, and F1 for every scenario
side-by-side.  zer consistently achieves higher recall; Splink can
yield higher precision on clean-field dedupe tasks (BRP/KvK) at the
cost of substantially lower recall on cross-source linkage.

PR curves
~~~~~~~~~

.. raw:: html

   <div style="margin: 1.5rem 0;">
     <img src="../res/pr_curves.svg"
          alt="Precision-recall curves across all scenarios"
          style="max-width:100%; border-radius:6px;" />
   </div>

PR-AUC is threshold-independent and captures overall discriminative
power.  zer's phonetic + suffix blocking surfaces more true matches in
the candidate set, which directly raises the ceiling for recall at any
given precision threshold.

----

Throughput
-----------

Throughput is measured as end-to-end pairs scored per second, covering
all pipeline stages (blocking, comparison, EM scoring).  Benchmarks use
the same ~22 200-record BRP and KvK dedupe datasets.

.. note::

   Splink runs on the host CPU in all cases; the backend column refers
   to the zer compute backend selected for that run.  zer always uses
   the same CPU for blocking and comparison,only the EM scoring stage
   is accelerated by AVX2/CUDA/Vulkan.

BRP deduplication (22 200 records, ~2.68 M candidate pairs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 20

   * - Backend
     - Throughput (pairs/s)
     - Elapsed (ms)
     - Peak memory (MB)
     - Speedup vs Splink
   * - Splink (CPU)
     - ~432 k
     - 6 200
     - 3 112
     - 1 times 
   * - zer CPU
     - ~735 k
     - 3 653
     - 163
     - **1.7 times**
   * - zer AVX2
     - ~768 k
     - 3 494
     - 147
     - **1.8 times**
   * - zer CUDA
     - ~4.1 M
     - 661
     - 246
     - **9.4 times**
   * - zer Vulkan
     - ~3.9 M
     - 680
     - 280
     - **9.1 times**

KvK deduplication (22 200 records, ~2.64 M candidate pairs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 20

   * - Backend
     - Throughput (pairs/s)
     - Elapsed (ms)
     - Peak memory (MB)
     - Speedup vs Splink
   * - Splink (CPU)
     - ~688 k
     - 3 830
     - 2 737
     - 1 times 
   * - zer CPU
     - ~748 k
     - 3 523
     - 407
     - **1.1 times**
   * - zer AVX2
     - ~787 k
     - 3 348
     - 407
     - **1.1 times**
   * - zer CUDA
     - ~4.6 M
     - 573
     - 516
     - **6.7 times**
   * - zer Vulkan
     - ~4.5 M
     - 584
     - 533
     - **6.6 times**

.. raw:: html

   <div style="margin: 1.5rem 0;">
     <img src="../res/throughput_comparison_cuda.svg"
          alt="Throughput comparison, CUDA backend"
          style="max-width:100%; border-radius:6px;" />
   </div>

The largest memory gap is the most practical: Splink loads the full
scored-pair matrix into a Polars/DuckDB DataFrame (~3 GB peak for 2.6 M
pairs), whereas zer processes pairs in streaming batches (~150–530 MB
depending on backend).

----

Pipeline stage breakdown
-------------------------

For reference, the zer pipeline cost breakdown on BRP dedupe (AVX2):

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Stage
     - zer AVX2 (ms)
     - zer CUDA (ms)
     - Notes
   * - Setup
     - 6
     - 7
     - Schema compilation, index init
   * - Blocking
     - 106
     - 105
     - Always on CPU; phonetic + suffix keys
   * - Comparison
     - 357
     - 349
     - Field-level similarity vectors; CPU SIMD
   * - EM scoring
     - 2 897
     - 34
     - Fellegi-Sunter iteration; GPU-accelerated on CUDA/Vulkan
   * - Score / classify
     - 128
     - 166
     - Threshold application, cluster update
   * - **Total**
     - **3 494**
     - **661**
     - \

The EM stage dominates on CPU.  CUDA reduces it from ~2.9 s to ~34 ms
(85 times  speedup on that stage alone), yielding a ~5 times  end-to-end speedup
after accounting for the fixed blocking and comparison costs.

----

Run the benchmarks
-------------------

.. code-block:: bash

   # Accuracy benchmarks (all scenarios, all libraries)
   cargo run -p zer-bench -- accuracy

   # Throughput,CPU backend
   cargo run -p zer-bench -- throughput

   # Throughput,AVX2 backend
   cargo run -p zer-bench --features avx2 -- throughput

   # Throughput,CUDA backend
   cargo run -p zer-bench --features cuda -- throughput

   # Throughput,Vulkan backend
   cargo run -p zer-bench --features vulkan -- throughput

The Python Splink comparison benchmarks are in ``benchmarks/splink/``:

.. code-block:: bash

   cd benchmarks/splink
   pip install -r ../../docs/sphinx/requirements.txt
   python strategies/brp_link.py
