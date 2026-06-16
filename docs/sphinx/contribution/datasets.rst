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

   pip install huggingface_hub
   hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
       --repo-type dataset \
       --local-dir data/benchmarks

Generating datasets locally
-----------------------------

The ``data_generator/`` scripts re-generate all datasets from scratch. This
is useful when you want to change the noise parameters, increase record counts,
or generate data for a new benchmark scenario.

**Step 1: Clone the repository:**

.. code-block:: bash

   git clone https://github.com/ZAL-Analytics/zer
   cd zer

**Step 2: Install Python dependencies:**

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r data_generator/requirements.txt

**Step 3: Download** ``data/base/`` **from Hugging Face:**

All generator scripts depend on ``data/base/``, which holds the name, address,
and CDR base datasets. **The scripts will not work without this directory
present.** Download it together with all other published data using the
included helper script:

.. code-block:: bash

   pip install huggingface_hub
   ./scripts/download_datasets.sh --base

This populates ``data/base/``

**Step 4: Run the generator(s):**

.. code-block:: bash

   # All categories at once
   ./scripts/generate_data.sh

   # Or only what you need:
   ./scripts/generate_data.sh --demos          # tutorial and demo datasets
   ./scripts/generate_data.sh --benchmarks     # all benchmark scenarios
   ./scripts/generate_data.sh --examples --tests  # crate examples + tests

.. note::

   Individual generator scripts (shown in the table below) can also be invoked
   directly, but they require ``data/base/`` to already be present (step 3
   above).

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
     - ``data/v1.1/demos/persons/``
     - Person deduplication tutorial
   * - ``generate_demo_linkage.py``
     - ``data/v1.1/demos/linkage/``
     - Cross-source linkage tutorial
   * - ``generate_demo_multi_source.py``
     - ``data/v1.1/demos/multi_source/``
     - Multi-source linkage tutorial
   * - ``generate_bench.py``
     - ``data/benchmarks/``
     - Accuracy and throughput benchmarks
   * - ``generate_examples_tests.py``
     - ``data/v1.1/examples/``, ``data/v1.1/tests/``
     - Crate examples and integration tests

Example: generate a larger deduplication dataset with a fixed seed:

.. code-block:: bash

   python data_generator/generate_demo_persons.py --records 5000 --seed 99

The seed parameter makes output fully reproducible, which is important when
comparing results across different pipeline configurations or library versions.

Ground-truth format
---------------------

Every dataset directory contains a ``ground_truth.csv`` with two columns
holding the natural keys of records that refer to the same underlying entity:

.. code-block:: text

   key_a,key_b
   893479421,891234567
   893479421,899876543
   891234567,899876543

Each row is a pair of natural key values (e.g. BSN, UUID, or primary-key
column) as produced by the dataset generator. The benchmark runner loads this
file and computes precision and recall against the ``record_key`` values in the
clusters produced by the pipeline.

Contributing new datasets
--------------------------

If you have designed a synthetic generator for a domain not covered by the
existing scripts (financial transactions, medical records, geospatial
locations, non-Dutch administrative data), contributions are welcome. Please
open a GitHub issue with the ``datasets`` label before writing the generator,
so we can discuss the schema and noise model.

For datasets that cannot be generated synthetically (e.g. real-world
benchmarks under a permissive licence), reach out directly before sharing
anything; see :doc:`contact`.

What to explore next
---------------------

* :doc:`running-benchmarks`, use these datasets to run the accuracy and throughput suite.
* :doc:`/tutorials/deduplication`, the person deduplication tutorial uses the ``data/v1.1/demos/persons/`` dataset.
