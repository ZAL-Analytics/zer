Neural Judge Internals
=======================

The neural judge is the third and final decision stage in the zer pipeline.
It handles ``Borderline`` pairs, those whose Fellegi-Sunter probability fell
between the auto-match and auto-reject thresholds, using a DeBERTa or MiniLM
NLI cross-encoder loaded via ONNX Runtime.

The NLI framing
----------------

The judge uses Natural Language Inference (NLI): given a *premise* (record A)
and a *hypothesis* (record B), classify the relationship as:

* **Entailment**, A and B refer to the same entity (promote to match).
* **Contradiction**, A and B are definitely different entities (demote to non-match).
* **Neutral**, not enough evidence to decide (abstain, pair remains borderline).

The judge maps ``entailment_probability >= promote_threshold`` to Promote and
``entailment_probability < demote_threshold`` to Demote. Pairs in between
receive Abstain.

Record serialization
---------------------

Before inference, each record is serialized to a flat text string. The
serialization follows a structured key-value format:

.. code-block:: text

   voornamen: Alice | achternaam: van den Berg | geboortedatum: 1990-03-15 | postcode: 1011AB

The pair is fed to the cross-encoder as a two-segment input, premise and
hypothesis, separated by a ``[SEP]`` token in the tokenizer. The model sees
the full pair at once, rather than scoring each record independently, which is
why cross-encoders are more accurate than bi-encoders for this task.

Why cross-encoder over bi-encoder
-----------------------------------

A bi-encoder (like SBERT) would encode each record separately and compare
embeddings. This is fast but misses interactions: the model cannot attend
to "Alice" in record A while deciding how to interpret "A." in record B.

A cross-encoder attends across both records simultaneously, so it correctly
handles:

* Abbreviation: "Johannes" vs. "J.", the cross-encoder learns that an
  initial is a plausible abbreviation of the full name.
* Nickname: "Maria" vs. "Ria", learned from the NLI training distribution.
* Translit: "Mohammed" vs. "Mohamed", variant spellings that would not match
  on exact comparison but do share a meaning.

The DeBERTa-v3-base model achieves ~88% F1 on standard NLI benchmarks.
MiniLM-L6 achieves ~82% F1 at roughly one-tenth the inference latency.

Thread architecture
--------------------

ORT inference is synchronous and blocking. Running it directly inside a tokio
task would block all worker threads. zer routes inference through a dedicated
``std::thread`` that owns the ORT session:

.. code-block:: text

   tokio task (async)
       │  adjudicate(pair)
       ▼
   std::sync::mpsc channel
       │
   Worker thread (ORT session)
       │  tokenize + run session
       ▼
   Result<f32> (entailment probability)
       │
   tokio task (async)
       │  apply threshold → Promote / Demote / Abstain

The channel carries ``Vec<String>`` (serialized pairs) and receives
``Vec<f32>`` (entailment probabilities). This design keeps the tokio runtime
responsive even during GPU inference.

Batching
---------

The judge collects borderline pairs and sends them to the worker in chunks of
``batch_size`` (default 64). Chunking keeps GPU memory usage bounded and
allows the pipeline to log progress between chunks.

On a consumer GPU (RTX 3080, 10 GiB VRAM):

* MiniLM: ~2,000 pairs/second at batch size 64
* DeBERTa-base: ~500 pairs/second at batch size 32

On CPU:

* MiniLM: ~200 pairs/second
* DeBERTa-base: ~20 pairs/second

Calibration
------------

After inference, the judge optionally applies Bayesian calibration to adjust
the raw entailment probability for the base rate of true matches in the
borderline band. The calibration table is estimated from a hold-out set and
stored alongside the model. When no calibration table is provided,
raw probabilities are used directly.

The audit log
--------------

When an audit log is attached, the judge writes one JSONL entry per pair:

.. code-block:: json

   {
     "record_a": 12345,
     "record_b": 67890,
     "comparison_vector": [3, 2, 3, 1, 3],
     "fs_probability": 0.71,
     "entailment_probability": 0.83,
     "verdict": "Promote"
   }

This log is invaluable for post-hoc analysis of difficult cases and for
calibrating promote/demote thresholds on your specific dataset.

What to explore next
---------------------

* :doc:`/how-to/neural-judge`, load and configure the judge.
* :doc:`/how-to/tune-scorer`, control how many pairs reach the judge by
  adjusting FS thresholds.
* :doc:`fellegi-sunter`, what happens before the judge gets the pair.
