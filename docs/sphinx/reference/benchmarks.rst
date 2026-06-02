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

All eight benchmark scenarios are shown below select a tab to compare
zer against Splink side by side.  The higher value in each metric column is
shown in **bold**.  All runs use default thresholds; no per-dataset tuning
was applied.  Splink PR-AUC is computed from each scenario's
``*_scored_pairs.csv`` in ``bench_results/data/accuracy_all/``.

.. tab-set::

   .. tab-item:: BRP dedupe

      22 200 records

      .. list-table::
         :header-rows: 1
         :widths: 20 20 20 20 20

         * - Library
           - Precision
           - Recall
           - F1
           - PR-AUC
         * - zer
           - 0.984
           - **0.982**
           - **0.983**
           - **0.990**
         * - Splink
           - **0.996**
           - 0.922
           - 0.958
           - 0.951

   .. tab-item:: KvK dedupe

      22 200 records

      .. list-table::
         :header-rows: 1
         :widths: 20 20 20 20 20

         * - Library
           - Precision
           - Recall
           - F1
           - PR-AUC
         * - zer
           - 0.910
           - **1.000**
           - 0.953
           - 0.915
         * - Splink
           - **0.998**
           - 0.925
           - **0.960**
           - **0.974**

   .. tab-item:: BRP link

      28 400 records

      .. list-table::
         :header-rows: 1
         :widths: 20 20 20 20 20

         * - Library
           - Precision
           - Recall
           - F1
           - PR-AUC
         * - zer
           - **0.976**
           - **0.991**
           - **0.983**
           - **0.997**
         * - Splink
           - 0.964
           - 0.714
           - 0.820
           - 0.858

   .. tab-item:: BRP + KvK link

      25 200 records

      .. list-table::
         :header-rows: 1
         :widths: 20 20 20 20 20

         * - Library
           - Precision
           - Recall
           - F1
           - PR-AUC
         * - zer
           - 0.788
           - **0.975**
           - 0.872
           - 0.938
         * - Splink
           - **0.912**
           - 0.877
           - **0.895**
           - **0.951**

   .. tab-item:: BRP + SIS link

      21 200 records

      .. list-table::
         :header-rows: 1
         :widths: 20 20 20 20 20

         * - Library
           - Precision
           - Recall
           - F1
           - PR-AUC
         * - zer
           - **1.000**
           - **0.984**
           - **0.992**
           - **0.999**
         * - Splink
           - 0.926
           - 0.823
           - 0.871
           - 0.942

   .. tab-item:: BRP + HKS link

      23 200 records

      .. list-table::
         :header-rows: 1
         :widths: 20 20 20 20 20

         * - Library
           - Precision
           - Recall
           - F1
           - PR-AUC
         * - zer
           - **1.000**
           - **0.992**
           - **0.996**
           - **0.999**
         * - Splink
           - 0.924
           - 0.819
           - 0.868
           - 0.942

   .. tab-item:: BRP + KvK L+D

      31 200 records

      .. list-table::
         :header-rows: 1
         :widths: 20 20 20 20 20

         * - Library
           - Precision
           - Recall
           - F1
           - PR-AUC
         * - zer
           - 0.843
           - **0.976**
           - **0.905**
           - **0.923**
         * - Splink
           - **0.904**
           - 0.771
           - 0.832
           - 0.874

   .. tab-item:: BRP + KvK + HKS L+D

      30 200 records

      .. list-table::
         :header-rows: 1
         :widths: 20 20 20 20 20

         * - Library
           - Precision
           - Recall
           - F1
           - PR-AUC
         * - zer
           - **0.850**
           - **0.985**
           - **0.913**
           - **0.920**
         * - Splink
           - 0.702
           - 0.889
           - 0.784
           - 0.831

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
   the same CPU for blocking and comparison; only the EM scoring stage
   is accelerated by AVX2/CUDA/Vulkan.

BRP deduplication
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 20

   * - Backend
     - Throughput (pairs/s)
     - Elapsed (ms)
     - Peak memory (MB)
     - Speedup vs Splink
   * - Splink (CPU)
     - ~488 k
     - 5 498
     - 3 828
     - 1×
   * - zer CPU
     - ~628 k
     - 4 275
     - 413
     - **1.3×**
   * - zer AVX2
     - ~671 k
     - 3 998
     - 414
     - **1.4×**
   * - zer CUDA
     - ~3.6 M
     - 739
     - 523
     - **7.4×**
   * - zer Vulkan
     - ~3.5 M
     - 757
     - 553
     - **7.3×**

KvK deduplication
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 20

   * - Backend
     - Throughput (pairs/s)
     - Elapsed (ms)
     - Peak memory (MB)
     - Speedup vs Splink
   * - Splink (CPU)
     - ~813 k
     - 3 240
     - 3 370
     - 1×
   * - zer CPU
     - ~713 k
     - 3 695
     - 410
     - 0.9×
   * - zer AVX2
     - ~741 k
     - 3 560
     - 410
     - 0.9×
   * - zer CUDA
     - ~4.0 M
     - 654
     - 521
     - **5.0×**
   * - zer Vulkan
     - ~4.0 M
     - 657
     - 551
     - **4.9×**

.. raw:: html

   <div style="margin: 1.5rem 0;">
     <img src="../res/throughput_comparison_cuda.svg"
          alt="Throughput comparison, CUDA backend"
          style="max-width:100%; border-radius:6px;" />
   </div>

   <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1.5rem 0;">
     <div>
       <p style="text-align:center; font-weight:bold; margin-bottom:0.4rem;">BRP dedupe zer (CUDA)</p>
       <img src="../res/throughput_stage_pie_brp_zer_cuda.svg"
            alt="BRP dedupe pipeline stage breakdown, zer CUDA"
            style="width:100%; border-radius:6px;" />
     </div>
     <div>
       <p style="text-align:center; font-weight:bold; margin-bottom:0.4rem;">BRP dedupe Splink</p>
       <img src="../res/throughput_stage_pie_brp_splink.svg"
            alt="BRP dedupe pipeline stage breakdown, Splink"
            style="width:100%; border-radius:6px;" />
     </div>
     <div>
       <p style="text-align:center; font-weight:bold; margin-bottom:0.4rem;">KvK dedupe zer (CUDA)</p>
       <img src="../res/throughput_stage_pie_kvk_zer_cuda.svg"
            alt="KvK dedupe pipeline stage breakdown, zer CUDA"
            style="width:100%; border-radius:6px;" />
     </div>
     <div>
       <p style="text-align:center; font-weight:bold; margin-bottom:0.4rem;">KvK dedupe Splink</p>
       <img src="../res/throughput_stage_pie_kvk_splink.svg"
            alt="KvK dedupe pipeline stage breakdown, Splink"
            style="width:100%; border-radius:6px;" />
     </div>
   </div>

   <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1.5rem 0;">
     <div>
       <p style="text-align:center; font-weight:bold; margin-bottom:0.4rem;">BRP dedupe zer (CUDA)</p>
       <img src="../res/throughput_mem_timeline_brp_zer_cuda.svg"
            alt="BRP dedupe memory timeline, zer CUDA"
            style="width:100%; border-radius:6px;" />
     </div>
     <div>
       <p style="text-align:center; font-weight:bold; margin-bottom:0.4rem;">BRP dedupe Splink</p>
       <img src="../res/throughput_mem_timeline_brp_splink.svg"
            alt="BRP dedupe memory timeline, Splink"
            style="width:100%; border-radius:6px;" />
     </div>
     <div>
       <p style="text-align:center; font-weight:bold; margin-bottom:0.4rem;">KvK dedupe zer (CUDA)</p>
       <img src="../res/throughput_mem_timeline_kvk_zer_cuda.svg"
            alt="KvK dedupe memory timeline, zer CUDA"
            style="width:100%; border-radius:6px;" />
     </div>
     <div>
       <p style="text-align:center; font-weight:bold; margin-bottom:0.4rem;">KvK dedupe Splink</p>
       <img src="../res/throughput_mem_timeline_kvk_splink.svg"
            alt="KvK dedupe memory timeline, Splink"
            style="width:100%; border-radius:6px;" />
     </div>
   </div>

The largest memory gap is the most practical: Splink loads the full
scored-pair matrix into a Polars/DuckDB DataFrame (~3.4–3.8 GB peak for 2.6–2.7 M
pairs), whereas zer processes pairs in streaming batches (~410–555 MB
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
     - 6
     - Schema compilation, index init
   * - Blocking
     - 131
     - 131
     - Always on CPU; phonetic + suffix keys
   * - Comparison
     - 405
     - 393
     - Field-level similarity vectors; CPU SIMD
   * - EM scoring
     - 3 318
     - 34
     - Fellegi-Sunter iteration; GPU-accelerated on CUDA/Vulkan
   * - Score / classify
     - 138
     - 175
     - Threshold application, cluster update
   * - **Total**
     - **3 998**
     - **739**
     - \

The EM stage dominates on CPU.  CUDA reduces it from ~3.3 s to ~34 ms
(~97× speedup on that stage alone), yielding a ~5.4× end-to-end speedup
after accounting for the fixed blocking and comparison costs.

----

Run the benchmarks
-------------------

``zer-bench`` is the unified benchmark harness.  It can be installed as a
standalone CLI tool or run directly from a repository clone.

Install
~~~~~~~

.. code-block:: bash

   # Install from crates.io (CPU backend, no extra toolchain required)
   cargo install zer-bench

   # With a specific compute backend
   cargo install zer-bench --features avx2
   cargo install zer-bench --features cuda     # requires CUDA Toolkit 13.1+
   cargo install zer-bench --features vulkan   # requires Vulkan 1.3 driver

   # With a neural judge execution provider
   cargo install zer-bench --features judge_cuda      # NVIDIA CUDA ORT provider
   cargo install zer-bench --features judge_tensorrt  # TensorRT FP16 (requires TensorRT 8.0+)
   cargo install zer-bench --features judge_rocm      # AMD ROCm ORT provider
   cargo install zer-bench --features judge_directml  # Windows DirectML ORT provider
   cargo install zer-bench --features judge_openvino  # Intel OpenVINO ORT provider

   # Combine compute backend and judge provider
   cargo install zer-bench --features "cuda,judge_tensorrt"

Neural judge targets
~~~~~~~~~~~~~~~~~~~~

Pass ``--judge-target`` to enable the neural judge and select its ONNX Runtime
execution provider.  The chosen target must match a ``judge_*`` feature compiled
into the binary.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - ``--judge-target``
     - Required feature
     - Notes
   * - ``cpu``
     - *(none)*
     - Default when ``--judge-target`` is omitted; always available
   * - ``cuda``
     - ``judge_cuda``
     - NVIDIA GPU via CUDA ORT provider
   * - ``tensorrt``
     - ``judge_tensorrt``
     - NVIDIA TensorRT FP16; caches engine on first run; requires TensorRT 8.0+
   * - ``rocm``
     - ``judge_rocm``
     - AMD GPU via ROCm ORT provider
   * - ``directml``
     - ``judge_directml``
     - Windows DirectML (any DirectX 12 GPU)
   * - ``openvino``
     - ``judge_openvino``
     - Intel hardware via OpenVINO ORT provider

Datasets
~~~~~~~~

Download the benchmark datasets from HuggingFace and set the
``ZER_DATASET_DIR`` environment variable so ``zer-bench`` can find them:

.. code-block:: bash

   hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
       --repo-type dataset --local-dir ~/datasets/zer
   export ZER_DATASET_DIR=~/datasets/zer

For runs that use the neural judge (``--judge``), also download model files:

.. code-block:: bash

   hf download arsalan-anwari/zjudge --local-dir ~/.cache/zer/models
   # ZER_MODEL_DIR defaults to ~/.cache/zer/models

Environment variables
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Variable
     - Default
     - Description
   * - ``ZER_DATASET_DIR``
     - ``<workspace>/data``
     - Root directory of benchmark datasets downloaded from HuggingFace.
       Dataset paths are resolved as ``$ZER_DATASET_DIR/benchmarks/<scenario>/...``.
       When unset, falls back to ``<workspace>/data`` (repository clone layout).
   * - ``ZER_MODEL_DIR``
     - ``~/.cache/zer/models``
     - Directory containing neural judge ONNX model files.
       Mirrors the layout from ``arsalan-anwari/zjudge`` on HuggingFace.
   * - ``ZER_EXTERNAL_BENCHMARKS_DIR``
     - ``<workspace>/benchmarks``
     - Root directory containing external library benchmark scripts.
       Scripts are resolved as ``$ZER_EXTERNAL_BENCHMARKS_DIR/<library>/<mode>/run.py``
       (or ``run.R``).  Set this when running ``zer-bench library`` outside of a
       zer repository clone.  Can also be passed as ``--external-benchmarks-dir``.

Subcommands
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Subcommand
     - Description
   * - ``throughput``
     - Raw compare/EM/score throughput on a single dataset.  Pass ``--compare-libs`` to also run competitor libraries
   * - ``accuracy``
     - Precision, recall, F1, and PR-AUC against labelled ground truth.  Pass ``--compare-libs`` to also run competitor libraries
   * - ``compare``
     - Read multiple ``_summary.csv`` files and print a side-by-side comparison table
   * - ``plot``
     - Generate plots from benchmark summary CSVs via ``plot_results.py``

Quick examples
~~~~~~~~~~~~~~

Direct ``zer-bench`` invocations (use after ``cargo install zer-bench``):

.. code-block:: bash

   # List available scenarios
   zer-bench accuracy --list-scenarios

   # Accuracy on a scenario
   zer-bench accuracy --scenario brp/dedupe --out bench_results/

   # Accuracy with neural judge (replace cuda with tensorrt / rocm / directml / openvino)
   zer-bench accuracy --scenario brp/dedupe --judge-target cuda --out bench_results/

   # Throughput (note: only dedupe scenarios are supported for throughput)
   zer-bench throughput --scenario brp/dedupe --out bench_results/
   zer-bench throughput --scenario brp/dedupe --target cuda --out bench_results/

   # Run all dedupe throughput scenarios back-to-back
   zer-bench throughput --scenario all --out bench_results/

   # zer vs Splink: run both and print an inline comparison table
   zer-bench accuracy  --scenario brp/dedupe --compare-libs splink --out bench_results/
   zer-bench throughput --scenario brp/dedupe --compare-libs splink --out bench_results/

   # Compare previously written summary CSVs side by side
   zer-bench compare --results bench_results/

   # Generate plots from summary CSVs
   zer-bench plot --input bench_results/ --output bench_results/plots/

   # Library scripts outside a zer repo clone
   zer-bench accuracy --scenario brp/dedupe --compare-libs splink \
       --external-benchmarks-dir /path/to/my/benchmarks --out bench_results/

