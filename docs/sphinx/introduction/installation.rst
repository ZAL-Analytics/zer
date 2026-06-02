Installation
============

zer is published on `crates.io <https://crates.io/crates/zer>`_. Add the crate
to your ``Cargo.toml`` and cargo handles the rest.

.. contents:: On this page
   :local:
   :depth: 1

----

System requirements
--------------------

The following table lists what must be present on the **build** and **runtime**
machine depending on which features you enable. CPU-only builds have no
external dependencies beyond Rust itself.

.. list-table::
   :header-rows: 1
   :widths: 20 22 22 36

   * - Component
     - Required version
     - Feature flag(s)
     - Notes
   * - **Rust** (stable)
     - 1.75 or later
     - all
     - ``rustup update stable``
   * - **CUDA Toolkit**
     - 13.1 or later
     - ``cuda``, ``judge_cuda``, ``judge_tensorrt``
     - Provides ``nvcc`` and ``libcuda.so``. Minimum GPU compute capability:
       SM 8.6 (NVIDIA Ampere or later). Driver version 575+ recommended.
   * - **Vulkan SDK**
     - 1.3 or later
     - ``vulkan``
     - Runtime: a Vulkan 1.3-capable driver. Build: ``slangc`` (the Slang
       shader compiler) on ``PATH``. Download from
       `shader-slang/slang releases <https://github.com/shader-slang/slang/releases>`_.
   * - **ONNX Runtime**
     - 2.0 (bundled)
     - ``judge_cpu``, ``judge_cuda``, ``judge_tensorrt``
     - Downloaded automatically at build time via the ``ort`` crate
       (``download-binaries`` feature). No manual install needed unless you
       want to supply your own ORT build via ``ORT_LIB_LOCATION``.
   * - **TensorRT**
     - 8.0 or later
     - ``judge_tensorrt``
     - Must be installed separately; not bundled. TensorRT engines are cached
       under ``~/.cache/zer-judge/trt-engines/``. Implies ``judge_cuda``.
   * - **Python 3.10+**
     - 3.10 or later
     - (demos only)
     - Only needed to run the synthetic demo data generators in
       ``scripts/``. Not required to build or use the library.

Linux system packages
~~~~~~~~~~~~~~~~~~~~~

.. note::

   Windows and macOS installation instructions will be added in a future update.

.. tab-set::

   .. tab-item:: Ubuntu / Debian

      .. code-block:: bash

         sudo apt-get install \
             build-essential pkg-config \
             libssl-dev libonig-dev \
             libvulkan-dev vulkan-tools

   .. tab-item:: Fedora / RHEL

      .. code-block:: bash

         sudo dnf install \
             gcc gcc-c++ make pkgconfig \
             openssl-devel oniguruma-devel \
             vulkan-devel

For a complete post-install setup on RHEL 10 (Rust toolchain, CUDA, Vulkan, developer tools),
see `ZAL-Analytics/rhel10-post-install <https://github.com/ZAL-Analytics/rhel10-post-install>`_.

----

Add to Cargo.toml
------------------

Minimal (CPU only, full pipeline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [dependencies]
   zer = { version = "1.0", features = ["pipeline"] }

This brings in ``zer-core``, ``zer-blocking``, ``zer-compare``,
``zer-cluster``, ``zer-schema``, and ``zer-pipeline``.

With CUDA GPU acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [dependencies]
   zer = { version = "1.0", features = ["pipeline", "cuda"] }

Requires CUDA Toolkit 13.1+ and an Ampere-class GPU (SM 8.6+). The pipeline
falls back to the CPU scorer automatically at runtime when no CUDA device is
found.

With AVX2 SIMD (CPU servers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [dependencies]
   zer = { version = "1.0", features = ["pipeline", "avx2"] }

Good for production x86-64 servers without a GPU. Provides roughly 4 times  the
throughput of the generic CPU backend.

With the neural judge (ONNX Runtime, CPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [dependencies]
   zer = { version = "1.0", features = ["pipeline", "judge_cpu"] }

Enables the DeBERTa/MiniLM NLI cross-encoder for borderline pair adjudication.
ONNX Runtime is downloaded at build time; no manual install needed.

With the neural judge (CUDA inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [dependencies]
   zer = { version = "1.0", features = ["pipeline", "judge_cuda"] }

Runs ORT on the CUDA execution provider. Requires CUDA Toolkit 13.1+.

With the neural judge (TensorRT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [dependencies]
   zer = { version = "1.0", features = ["pipeline", "judge_tensorrt"] }

Uses the TensorRT ORT execution provider for FP16 inference with engine
caching. Requires TensorRT 8.0+ and CUDA Toolkit 13.1+.

With Polars / Arrow adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [dependencies]
   zer      = { version = "1.0", features = ["pipeline"] }
   zer-adapters = { version = "1.0", features = ["polars"] }
   polars       = "0.44"

Individual crates
~~~~~~~~~~~~~~~~~~

You can depend on individual crates if you only need part of the pipeline:

.. code-block:: toml

   [dependencies]
   zer-core     = "1.0"   # Record, Schema, FieldKind, traits
   zer-blocking = "1.0"   # Blocker, BlockerFactory, blocking keys
   zer-compare  = "1.0"   # FieldComparator, FellegiSunterScorer
   zer-cluster  = "1.0"   # Clusterer, ZalEntityStore
   zer-pipeline = "1.0"   # Pipeline, PipelineConfig, BatchReport

----

Downloading models
-------------------

The neural judge requires ONNX model files that are **not** bundled with the
crate. Download them from Hugging Face:

**Model repository:** `arsalan-anwari/zjudge <https://huggingface.co/arsalan-anwari/zjudge>`_

.. code-block:: bash

   # Using the huggingface_hub CLI (pip install huggingface_hub[cli])
   $ hf download arsalan-anwari/zjudge --local-dir ~/.cache/zer/models

   # Or clone with git-lfs
   $ git lfs install
   $ git clone https://huggingface.co/arsalan-anwari/zjudge ~/.cache/zer/models

Expected directory layout after download:

.. code-block:: text

   ~/.cache/zer/models/
     nli-base/
       base/             # FP32 weights (CPU / CUDA)
         model.onnx
         tokenizer.json
       fp16/             # FP16 weights (GPU / TensorRT)
         model.onnx
         tokenizer.json

zer resolves the model directory in this order:

1. ``$ZER_MODEL_DIR`` environment variable (explicit override).
2. ``~/.cache/zer/models`` (populated by the download above).
3. ``./models`` relative to the working directory (workspace fallback).

Override the default path:

.. code-block:: bash

   export ZER_MODEL_DIR=/data/zer/models

----

Downloading datasets
---------------------

The benchmark suite and demo generators use Dutch law-enforcement entity
resolution datasets published on Hugging Face:

**Dataset repository:**
`arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset <https://huggingface.co/datasets/arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset>`_

.. code-block:: bash

   $ hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
       --repo-type dataset \
       --local-dir data/

The demos and benchmarks expect the data under ``data/`` at the repository
root, or wherever you set ``ZER_DATASET_DIR``.

.. note::

   The datasets contain **synthetic records only**. They are generated from
   statistical distributions derived from Dutch administrative data and do not
   contain real personal information.

----

Environment variables
----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Variable
     - Purpose
   * - ``ZER_MODEL_DIR``
     - Override the directory searched for ONNX model files and tokenizer
       configs. See `Downloading models`_ above.
   * - ``ZER_NAME_HEURISTICS``
     - Path to a TOML file overriding the embedded name-field heuristics used
       by ``SchemaInferrer``. Falls back to the built-in
       ``heuristics_name.toml`` when unset or unreadable.
   * - ``ZER_VALUE_PATTERNS``
     - Path to a TOML file overriding the embedded value-pattern rules used
       by ``SchemaInferrer`` to detect postcodes, BSNs, IBANs, and other
       structured fields. Falls back to the built-in ``value_patterns.toml``.
   * - ``ORT_LIB_LOCATION``
     - Path to a custom ONNX Runtime installation. When unset, the ``ort``
       crate downloads a compatible ORT binary at build time.

----

Building the demos
-------------------

The demo programs live in the `GitHub repository
<https://github.com/ZAL-Analytics/zer>`_ and are not published to crates.io.
Clone the repo to run them:

.. code-block:: bash

   $ git clone https://github.com/ZAL-Analytics/zer
   $ cd zer

   # Download datasets first (see above)
   $ hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
       --repo-type dataset --local-dir data/

   # Run a demo
   $ cargo run -p person_deduplication
   $ cargo run -p cross_source_linkage
   $ cargo run -p multi_source_linkage

   # GPU demos require the cuda feature
   $ cargo run -p person_deduplication --features cuda
