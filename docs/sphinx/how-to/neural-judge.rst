How to Use the Neural Judge
============================

The neural judge is an optional second-pass component that re-evaluates
``Borderline`` pairs, pairs whose Fellegi-Sunter probability fell between
the auto-match and auto-reject thresholds. It loads a DeBERTa or MiniLM NLI
cross-encoder via ONNX Runtime and classifies each pair as a match or
non-match using the serialized record text as input.

When to use it
---------------

* Your dataset has fields that are frequently abbreviated, transliterated, or
  missing, making EM parameters less reliable for borderline cases.
* You need higher precision than the EM thresholds can deliver alone.
* You can tolerate the extra latency for borderline pairs (~5-50 ms/pair on
  CPU, <1 ms on GPU).

Enable the feature flag
------------------------

The neural judge is gated behind a feature flag in ``Cargo.toml``:

.. code-block:: toml

   [dependencies]
   zer = { version = "1.1", features = ["pipeline", "judge_cpu"] }

   # GPU variants (mutually exclusive; pick one):
   # zer = { version = "1.1", features = ["pipeline", "judge_cuda"] }
   # zer = { version = "1.1", features = ["pipeline", "judge_tensorrt"] }
   # zer = { version = "1.1", features = ["pipeline", "judge_rocm"] }

These flags are **separate** from ``zer-compute``'s ``cuda``/``avx2`` flags.
``judge_*`` selects the ONNX Runtime execution provider; ``cuda`` selects the
comparison/EM compute backend.

Download a model
-----------------

Models are published on Hugging Face and must be downloaded separately before
the neural judge can run:

`arsalan-anwari/zjudge <https://huggingface.co/arsalan-anwari/zjudge>`_

.. code-block:: bash

   # Using the huggingface_hub CLI
   $ pip install huggingface_hub[cli]
   $ hf download arsalan-anwari/zjudge --local-dir ~/.cache/zer/models

   # Or with git-lfs
   $ git lfs install
   $ git clone https://huggingface.co/arsalan-anwari/zjudge ~/.cache/zer/models

Two built-in model specs are provided:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Spec
     - ONNX size
     - VRAM
     - When to use
   * - ``MiniLmSpec``
     - ~23 MB
     - ~256 MB
     - Fast CPU inference; good for most cases
   * - ``DebertaBaseSpec``
     - ~180 MB
     - ~1.5 GB
     - Higher accuracy; use with GPU

Both variants expect the ONNX weights and tokenizer config under ``nli-base/``
inside the model directory:

.. code-block:: text

   ~/.cache/zer/models/         ← default; override with $ZER_MODEL_DIR
     nli-base/
       base/                    ← FP32, used by MiniLmSpec and DebertaBaseSpec
         model.onnx
         tokenizer.json
       fp16/                    ← FP16, used when precision = ModelPrecision::Fp16
         model.onnx
         tokenizer.json

zer resolves the model directory in this order:

1. ``$ZER_MODEL_DIR`` environment variable.
2. ``~/.cache/zer/models``.
3. ``./models`` relative to the working directory.

To use a custom location:

.. code-block:: bash

   export ZER_MODEL_DIR=/data/models/zer

Load and run the judge
-----------------------

.. code-block:: rust

   use std::sync::Arc;
   use zer_judge::{JudgeBackend, DebertaJudge, DebertaJudgeConfig, MiniLmSpec};
   use zer_core::VecRecordStore;

   // Auto-detects: reads --judge-target= from process args (separate from --target=)
   let backend      = JudgeBackend::auto_detect();
   let spec         = MiniLmSpec::from_dir("models/nli-minilm");
   let record_store = Arc::new(VecRecordStore::new());

   let judge = DebertaJudge::new(
       &spec,
       &backend,
       record_store,
       schema.clone(),
       DebertaJudgeConfig::default(),
   )?;

Selecting the execution provider at runtime
--------------------------------------------

The judge target is controlled by a CLI flag, not by the feature flag. The
feature flag enables compilation; the CLI flag selects which compiled provider
to use:

.. code-block:: bash

   # Use the CPU ORT provider
   my_app --judge-target=cpu

   # Use CUDA ORT provider (requires judge_cuda feature)
   my_app --judge-target=cuda

   # Use TensorRT (requires judge_tensorrt feature, implies CUDA)
   my_app --judge-target=tensorrt

If ``--judge-target`` is absent, ``JudgeBackend::auto_detect()`` defaults to
CPU. All compiled targets are listed at startup:

.. code-block:: text

   All known judge targets:
     Cpu           compiled-in  (--judge-target=cpu)    ◀ selected
     Cuda          not compiled (--judge-target=cuda)
     TensorRt      not compiled (--judge-target=tensorrt)
     Rocm          not compiled (--judge-target=rocm)
     DirectMl      not compiled (--judge-target=directml)
     OpenVino      not compiled (--judge-target=openvino)

Tuning promote and demote thresholds
--------------------------------------

``DebertaJudgeConfig`` controls when the judge promotes a pair to a match or
demotes it to a non-match:

.. code-block:: rust

   use zer_judge::DebertaJudgeConfig;

   let config = DebertaJudgeConfig {
       // Promote to match when entailment probability ≥ 0.75
       promote_threshold: 0.75,
       // Demote to non-match when entailment probability < 0.40
       demote_threshold:  0.40,
       // Send pairs to the ORT worker in chunks of 32
       batch_size:        32,
       ..DebertaJudgeConfig::default()
   };

Pairs where the entailment probability falls between ``demote_threshold`` and
``promote_threshold`` remain borderline after the judge and are treated as
non-matches in the final cluster step.

Enabling the audit log
-----------------------

The judge can write a JSONL audit trail of every adjudicated pair, including
the serialized record text, the entailment score, and the verdict:

.. code-block:: rust

   use std::sync::Arc;
   use zer_judge::{audit::AuditLog, DebertaJudgeConfig};

   let audit = Arc::new(AuditLog::open("/data/audit/judge_2025.jsonl")?);

   let config = DebertaJudgeConfig {
       audit_log: Some(audit),
       ..DebertaJudgeConfig::default()
   };

Each audit entry records the record pair, the comparison vector, the
entailment probability, and the final verdict (``Promote``/``Demote``/
``Abstain``). Use this to review difficult cases and tune thresholds.

Choosing MiniLM vs DeBERTa
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Criterion
     - MiniLM-L6
     - DeBERTa-v3-base
   * - Model size
     - ~23 MB ONNX
     - ~180 MB ONNX
   * - CPU latency
     - ~5 ms/pair
     - ~50 ms/pair
   * - GPU latency
     - <1 ms/pair
     - ~2 ms/pair
   * - Accuracy (NLI)
     - Good for clean text
     - Better for abbreviations and transliterations
   * - VRAM required
     - ~256 MB
     - ~1.5 GB

For most Dutch administrative data (BRP, KvK), MiniLM is sufficient. Use
DeBERTa when you have many borderline pairs with Arabic or Slavic name
variants that the phonetic keys did not group.

What to explore next
---------------------

* :doc:`/explanation/judge-internals`, how the NLI cross-encoder classifies
  record pairs and what the serialized text format looks like.
* :doc:`/how-to/tune-scorer`, set EM thresholds to control how many pairs
  reach the judge.
* :doc:`/how-to/gpu-backend`, configure CUDA for both the comparator and the
  judge independently.
