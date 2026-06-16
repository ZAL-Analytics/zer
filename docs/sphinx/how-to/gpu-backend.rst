How to Use the GPU Backend
===========================

zer has two independent acceleration layers with separate feature flags and
runtime selection:

1. **Comparison / EM backend** (``zer-compute``), accelerates batch
   pairwise comparison and the EM scoring step. Uses CUDA or AVX2.
2. **Neural judge backend** (``zer-judge``), accelerates DeBERTa/MiniLM
   inference via ONNX Runtime. Controlled by ``--judge-target=``.

This guide covers the comparison/EM backend. For the neural judge see
:doc:`neural-judge`.

Selecting a backend at compile time
-------------------------------------

Feature flags in ``Cargo.toml`` control which backends are compiled in:

.. code-block:: toml

   [dependencies]
   zer = { version = "1.1", features = ["pipeline"] }          # CPU only
   zer = { version = "1.1", features = ["pipeline", "avx2"] }  # + AVX2 SIMD
   zer = { version = "1.1", features = ["pipeline", "cuda"] }  # + NVIDIA CUDA

Multiple flags can be combined. ``auto_detect`` selects the best compiled-in
backend at runtime.

Auto-detecting the best backend
---------------------------------

``GpuBackend::auto_detect()`` probes available backends in priority order:
CUDA, then AVX2, then CPU. It returns the first one that is both compiled in and
available at runtime.

.. code-block:: rust

   use std::sync::Arc;
   use zer_compute::{GpuBackend, DeviceComparator, DeviceScorer};
   use zer_core::schema::{FieldKind, SchemaBuilder};

   let backend = Arc::new(GpuBackend::auto_detect());
   println!("Selected backend : {}", backend.name());

   // VRAM info (None for CPU backend)
   if let Some(total) = backend.total_vram_bytes() {
       println!("Total VRAM       : {:.1} GiB", total as f64 / (1 << 30) as f64);
   }
   if let Some(avail) = backend.available_vram_bytes() {
       println!("Available VRAM   : {:.1} GiB", avail as f64 / (1 << 30) as f64);
   }

Querying the auto-tuned batch size
------------------------------------

``BatchSizer`` reads available VRAM and returns the maximum number of record
pairs that fit in a single GPU batch without OOM:

.. code-block:: rust

   use zer_compute::BatchSizer;

   let available = backend.available_vram_bytes().unwrap_or(4 * 1024 * 1024 * 1024);
   let sizer     = BatchSizer::new();
   let max_batch = sizer.max_batch_size(available, schema.fields.len());
   println!("Max GPU batch    : {} pairs", max_batch);

The pipeline uses this automatically when ``DeviceScorer`` is attached.

Passing the backend to the pipeline
--------------------------------------

Pass a ``DeviceScorer`` built from the backend into ``Pipeline::builder()``.
If you omit it the pipeline defaults to the CPU scorer.

.. code-block:: rust

   use std::sync::Arc;
   use zer_compute::{GpuBackend, DeviceScorer};
   use zer_pipeline::{PipelineConfig, Pipeline};
   use zer_cluster::ZalEntityStore;

   let backend = Arc::new(GpuBackend::auto_detect());

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
       .scorer(DeviceScorer::new(Arc::clone(&backend)))
       .config(PipelineConfig {
           registry_path: "/tmp/pipeline.zsm".into(),
           ..PipelineConfig::default()
       })
       .build()?;

   let report = pipeline.run_batch(records).await?;
   println!("elapsed: {} ms", report.elapsed_ms);

Forcing a specific backend
----------------------------

Override ``auto_detect`` with the ``--target=`` CLI flag. zer reads this
flag at startup:

.. code-block:: bash

   # Force CPU even when CUDA is compiled in
   my_app --target=cpu

   # Force AVX2 SIMD
   my_app --target=avx2

   # Force CUDA (fails with exit(1) if cuda feature is not compiled in)
   my_app --target=cuda

You can also construct the backend directly for tests or scripted pipelines:

.. code-block:: rust

   use zer_compute::GpuBackend;

   // Always CPU, regardless of compiled features
   let cpu_backend = GpuBackend::cpu();

Building and verifying the CUDA backend
------------------------------------------

CUDA builds require:

* NVIDIA CUDA Toolkit **13.1 or later** (enforced at build time by ``build.rs``)
* Minimum GPU compute capability: **SM 8.6** (NVIDIA Ampere, e.g. RTX 3000 series or A-series)
* ``libcuda.so`` on the runtime library path (driver version 575+ recommended)
* Cargo feature: ``cuda``

.. code-block:: bash

   # Build with CUDA
   cargo build --features cuda -p zer-compute

   # Verify CUDA is detected at runtime
   cargo run --features cuda -p zer-compute --example auto_detect

Expected output on a CUDA-capable machine:

.. code-block:: text

   Selected backend : cuda
   Total VRAM       : 24.0 GiB
   Available VRAM   : 22.4 GiB
   Max GPU batch    : 131072 pairs
   DeviceComparator and DeviceScorer constructed successfully.

On a machine without CUDA:

.. code-block:: text

   Selected backend : avx2
   Total VRAM       : N/A (CPU backend)

Building and verifying the Vulkan backend
-------------------------------------------

Vulkan builds require:

* **Vulkan 1.3** runtime driver (any vendor, NVIDIA, AMD, Intel)
* **Slang shader compiler** (``slangc``) on ``PATH`` at build time.
  Download from `shader-slang/slang releases <https://github.com/shader-slang/slang/releases>`_.
  Shaders compile to SPIR-V 1.5.
* Cargo feature: ``vulkan``

.. code-block:: bash

   # Build with Vulkan
   cargo build --features vulkan -p zer-compute

   # Verify Vulkan is detected at runtime
   cargo run --features vulkan -p zer-compute --example auto_detect

.. note::

   ``slangc`` is only needed at **build time** to compile the ``.slang`` shader
   sources to SPIR-V. End users of a pre-built binary do not need it installed.

Backend comparison
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Backend
     - Feature flag
     - Throughput (BRP, 10 fields)
     - When to use
   * - CPU
     - (none)
     - ~50 k pairs/s
     - Development, small datasets (< 100 k records)
   * - AVX2
     - ``avx2``
     - ~200 k pairs/s
     - Production without a GPU; x86-64 servers
   * - CUDA
     - ``cuda``
     - ~2 M pairs/s
     - Large datasets (> 500 k records); batch processing
   * - Vulkan
     - ``vulkan``
     - ~1.5 M pairs/s
     - Cross-vendor GPU support (AMD, Intel, NVIDIA); requires Vulkan 1.3

.. note::

   The neural judge backend (DeBERTa/MiniLM) uses a separate ONNX Runtime
   CUDA execution provider controlled by ``judge_cuda``. The two CUDA
   instances are independent, you can run the comparator on AVX2 and the
   judge on GPU, or vice versa.

What to explore next
---------------------

* :doc:`neural-judge`, enable GPU acceleration for borderline pair adjudication.
* :doc:`/reference/benchmarks`, throughput figures for each backend on
  standard Dutch administrative datasets.
* :doc:`/tutorials/multi-source-linkage`, example using ``DeviceBackend::auto_detect()``.
