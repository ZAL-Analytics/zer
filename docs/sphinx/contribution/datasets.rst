Test Datasets
==============

zer ships with a Python-based data generator that produces fully synthetic
Dutch administrative records. No real personal data is used anywhere in the
test suite, benchmarks, or tutorials. All datasets are generated from
configurable statistical distributions using ``faker`` and a set of
Dutch-specific noise models (OCR confusion, tussenvoegsel variants, name
abbreviations, date transpositions).

Published dataset on Hugging Face
-----------------------------------

The full benchmark dataset is published openly on Hugging Face:

`arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset <https://huggingface.co/datasets/arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset>`_

It contains synthetic records for all eight benchmark scenarios across four
simulated Dutch administrative sources: BRP (person registry), KvK (chamber
of commerce), SIS II (Schengen alert), and HKS (crime intelligence). Every
record has a deterministic ground-truth entity assignment so precision and
recall can be computed exactly.

Download it with the Hugging Face CLI:

.. code-block:: bash

   pip install huggingface_hub[cli]
   hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
       --repo-type dataset \
       --local-dir data/benchmarks

Generating datasets locally
-----------------------------

The ``data_generator/`` scripts re-generate all datasets from scratch. This
is useful when you want to change the noise parameters, increase record counts,
or generate data for a new benchmark scenario.

One-time setup:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install faker

Generate everything at once:

.. code-block:: bash

   ./scripts/generate_data.sh

Or generate only what you need:

.. code-block:: bash

   # Tutorial and demo datasets
   ./scripts/generate_data.sh --demos

   # Benchmark datasets (all eight scenarios)
   ./scripts/generate_data.sh --benchmarks

   # Datasets for crate examples and integration tests
   ./scripts/generate_data.sh --examples --tests

Individual generator scripts
------------------------------

Each script accepts parameters to control record count and random seed:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Script
     - Output
     - Used by
   * - ``generate_demo_persons.py``
     - ``data/demos/persons/``
     - Person deduplication tutorial
   * - ``generate_demo_linkage.py``
     - ``data/demos/linkage/``
     - Cross-source linkage tutorial
   * - ``generate_demo_multi_source.py``
     - ``data/demos/multi_source/``
     - Multi-source linkage tutorial
   * - ``generate_bench.py``
     - ``data/benchmarks/``
     - Accuracy and throughput benchmarks
   * - ``generate_examples_tests.py``
     - ``data/examples/``, ``data/tests/``
     - Crate examples and integration tests
   * - ``generate_raw.py``
     - ``data/raw/``
     - Raw provider export format datasets

Example: generate a larger deduplcation dataset with a fixed seed:

.. code-block:: bash

   python data_generator/generate_demo_persons.py --records 5000 --seed 99

The seed parameter makes output fully reproducible, which is important when
comparing results across different pipeline configurations or library versions.

Ground-truth format
---------------------

Every dataset directory contains a ``ground_truth.csv`` with two columns:

.. code-block:: text

   record_id_a,record_id_b
   1,42
   1,87
   42,87

Each row is a pair of ``RecordId`` values that refer to the same underlying
entity. The benchmark runner loads this file and computes precision and recall
against the clusters produced by the pipeline.

Contributing new datasets
--------------------------

If you have designed a synthetic generator for a domain not covered by the
existing scripts (financial transactions, medical records, geospatial
locations, non-Dutch administrative data), contributions are welcome. Please
open a GitHub issue with the ``datasets`` label before writing the generator,
so we can discuss the schema and noise model.

For datasets that cannot be generated synthetically (e.g. real-world
benchmarks under a permissive licence), reach out directly before sharing
anything,see :doc:`contact`.

What to explore next
---------------------

* :doc:`running-benchmarks`, use these datasets to run the accuracy and throughput suite.
* :doc:`/tutorials/deduplication`, the person deduplication tutorial uses the ``data/demos/persons/`` dataset.
