Running Benchmarks
===================

zer includes a benchmark suite under ``benchmarks/`` that measures both
accuracy (precision, recall, F1) and throughput (pairs per second) across
eight standard scenarios. The published figures in :doc:`/reference/benchmarks`
were produced by this suite. Running it yourself lets you verify results on
your own hardware and, if you find something different, share those numbers
with the project.

Prerequisites
--------------

Generate the benchmark datasets before running (they are not checked in to
the repository):

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install faker
   ./scripts/generate_data.sh --benchmarks

This produces the synthetic Dutch administrative datasets under
``data/benchmarks/`` that the benchmark runner expects.

Running the Rust benchmarks
-----------------------------

The ``zer-bench`` crate drives all accuracy and throughput scenarios:

.. code-block:: bash

   # Accuracy: all eight scenarios, CPU backend
   cargo run -p zer-bench -- accuracy

   # Throughput: CPU backend
   cargo run -p zer-bench -- throughput

   # Throughput: AVX2 backend
   cargo run -p zer-bench --features avx2 -- throughput

   # Throughput: CUDA backend (requires CUDA Toolkit 13.1+)
   cargo run -p zer-bench --features cuda -- throughput

   # Throughput: Vulkan backend (requires slangc on PATH)
   cargo run -p zer-bench --features vulkan -- throughput

Benchmark output is written to ``bench_results/`` as JSON and CSV files.

Running zer alongside competitor libraries
------------------------------------------

Pass ``--compare-libs`` to any ``accuracy`` or ``throughput`` invocation to
run competitor libraries on the same dataset and print an inline comparison
table.  The library scripts live in ``benchmarks/<library>/``.

.. code-block:: bash

   # Accuracy: zer vs Splink on BRP dedupe
   cargo run -p zer-bench -- accuracy --scenario brp/dedupe --compare-libs splink

   # Throughput: zer (CUDA) vs Splink on all dedupe scenarios
   cargo run -p zer-bench --features cuda -- throughput --scenario all --compare-libs splink

   # Compare previously written CSV summaries side by side
   cargo run -p zer-bench -- compare --results bench_results/

Sharing your results
---------------------

If you run the benchmarks on hardware not represented in the published table
(a different GPU vendor, an ARM server, a very large dataset), please share
the output. Open a GitHub issue with the ``benchmarks`` label and attach or
paste the contents of ``bench_results/``, along with:

* CPU model and core count.
* GPU model, VRAM, and driver version (if applicable).
* OS and kernel version.
* Rust toolchain version (``rustc --version``).
* Any non-default feature flags or configuration changes.

Results from diverse hardware help build a more complete picture of where zer
performs well and where there is room to improve.

Adding a new benchmark scenario
---------------------------------

The benchmark scenarios are defined in ``benchmarks/`` alongside the data
generator scripts in ``data_generator/``. To add a new scenario:

1. Add a generator script to ``data_generator/`` that produces a
   ``data/benchmarks/<scenario>/`` directory with ``source_*.csv`` and
   ``ground_truth.csv`` files following the same format as the existing
   datasets.
2. Add the scenario to ``zer-bench`` following the pattern of the existing
   eight scenarios.
3. If the scenario represents a domain not currently covered (financial
   records, medical data, geospatial, etc.), open an issue describing it so
   we can consider including it in the published benchmark table.

What to explore next
---------------------

* :doc:`/reference/benchmarks`, the full published benchmark results and pipeline stage breakdown.
* :doc:`datasets`, how to generate and work with the synthetic datasets used by the benchmarks.
