# zer-bench

Benchmark harness for zer, measures throughput, accuracy, and head-to-head comparisons against competitor libraries on Dutch administrative datasets.

- **Documentation**: [docs.zal-analytics.ch](https://docs.zal-analytics.ch)
- **Website**: [www.zal-analytics.ch](https://www.zal-analytics.ch)
- **Support & feedback**: [info@zal-analytics.ch](mailto:info@zal-analytics.ch)

## Install

```bash
cargo install zer-bench
```

For GPU-accelerated benchmarks pass the matching feature flag(s) at install time:

```bash
cargo install zer-bench --features avx2            # x86-64 AVX2 SIMD
cargo install zer-bench --features cuda            # NVIDIA CUDA (requires CUDA Toolkit 13.1+)
cargo install zer-bench --features vulkan          # Vulkan 1.3 compute
cargo install zer-bench --features judge_tensorrt  # TensorRT judge
```

To install with every compute backend and judge provider enabled:

```bash
cargo install zer-bench --features \
    "cuda,avx2,vulkan,judge_cuda,judge_tensorrt,judge_rocm,judge_directml,judge_openvino"
```

Once installed, `--target` and `--judge-target` switch backends at runtime no rebuild needed.

## Datasets and models

zer-bench resolves dataset and model paths from environment variables. Download the benchmark datasets from HuggingFace and point `ZER_DATASET_DIR` at the local copy:

```bash
hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
    --repo-type dataset --local-dir ~/datasets/zer
export ZER_DATASET_DIR=~/datasets/zer
```

Neural judge benchmarks (`--judge-target`) also need model files:

```bash
hf download arsalan-anwari/zjudge --local-dir ~/.cache/zer/models
# ZER_MODEL_DIR defaults to ~/.cache/zer/models; override if needed
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ZER_DATASET_DIR` | `<workspace>/data` | Root directory for benchmark datasets. Paths resolve as `$ZER_DATASET_DIR/benchmarks/<scenario>/...`. Falls back to `<workspace>/data` when unset. |
| `ZER_MODEL_DIR` | `~/.cache/zer/models` | Directory containing neural judge ONNX model files. Mirrors the layout from `arsalan-anwari/zjudge` on HuggingFace. |
| `ZER_EXTERNAL_BENCHMARKS_DIR` | `<workspace>/benchmarks` | Root directory for external library benchmark scripts. Scripts resolve as `$ZER_EXTERNAL_BENCHMARKS_DIR/<library>/<mode>/run.py`. Can also be passed as `--external-benchmarks-dir`. |

## Subcommands

| Subcommand | Description |
|---|---|
| `accuracy` | Precision, recall, F1, and PR-AUC against labelled ground truth |
| `throughput` | Raw compare/EM/score throughput on a single dataset |
| `compare` | Read multiple CSV summaries and print a side-by-side comparison table |
| `plot` | Generate plots from benchmark JSON files via `plot_results.py` |

Competitor libraries are run inline via `--compare-libs` on both `accuracy` and `throughput` no separate subcommand needed.

## Quick start

### List available scenarios

```bash
zer-bench accuracy  --list-scenarios
zer-bench throughput --list-scenarios
```

### Accuracy

```bash
# Single scenario datasets, mode, and ground truth wired up automatically
zer-bench accuracy --scenario brp/dedupe --out bench_results/

# All 8 full-size scenarios back-to-back
zer-bench accuracy --scenario all --out bench_results/

# zer vs splink (runs both, prints inline comparison table)
zer-bench accuracy --scenario brp/dedupe --compare-libs splink --out bench_results/

# zer vs splink across all scenarios
zer-bench accuracy --scenario all --compare-libs splink --out bench_results/
```

### Judge dual-pass

When `--judge-target` is supplied, zer-bench automatically runs **zer without judge** then **zer with judge**, then prints a side-by-side comparison table. No extra flags needed.

```bash
# CPU judge dual-pass (no extra feature flag needed)
zer-bench accuracy --scenario brp/dedupe --judge-target cpu

# TensorRT judge (requires --features judge_tensorrt at build time)
zer-bench accuracy --scenario brp/dedupe --judge-target tensorrt

# TensorRT judge vs splink baseline 3 results per scenario: zer, zer+judge, splink
zer-bench accuracy --scenario brp/dedupe --judge-target tensorrt --compare-libs splink

# All 8 scenarios × (zer + zer+judge + splink) 24 runs total, one table per scenario
zer-bench accuracy --scenario all --judge-target tensorrt --compare-libs splink
```

### Throughput

```bash
# CPU throughput (dedupe scenarios only)
zer-bench throughput --scenario brp/dedupe --out bench_results/

# CUDA throughput (requires --features cuda)
zer-bench throughput --scenario brp/dedupe --target cuda --out bench_results/

# All dedupe scenarios back-to-back (brp/dedupe and kvk/dedupe)
zer-bench throughput --scenario all --target cuda --out bench_results/

# zer vs splink throughput
zer-bench throughput --scenario brp/dedupe --compare-libs splink --out bench_results/

# CUDA throughput + TensorRT judge dual-pass
zer-bench throughput --scenario brp/dedupe --target cuda --judge-target tensorrt
```

### Comparing existing results

```bash
# Print a table from all summary CSVs in a directory
zer-bench compare --results bench_results/

# Filter by mode and dataset
zer-bench compare --results bench_results/ --mode dedupe --dataset brp_persons
```

### Plotting

```bash
zer-bench plot --input bench_results/data/<run>/
```

## Helper script

`scripts/run_benchmark.sh` is a thin driver that selects the correct Cargo features based on `--target` and `--judge-target` (backends must be compiled in), generates a timestamped output directory, then forwards everything to `zer-bench`:

```bash
# Equivalent to the direct zer-bench calls above
./scripts/run_benchmark.sh --scenario brp/dedupe
./scripts/run_benchmark.sh --scenario all --compare-libs splink
./scripts/run_benchmark.sh --type throughput --scenario brp/dedupe --target cuda
./scripts/run_benchmark.sh --scenario brp/dedupe --judge-target tensorrt --compare-libs splink
./scripts/run_benchmark.sh --list
```

Use the script during development. For a pre-built all-features binary, call `zer-bench` directly.

## Feature flags

### Compute backends

| Flag | `--target` value | Description |
|---|---|---|
| *(none)* | `cpu` or `auto` | Scalar CPU fallback, always available |
| `avx2` | `avx2` | x86-64 AVX2 SIMD (~4× vs scalar CPU) |
| `cuda` | `cuda` | NVIDIA CUDA, requires CUDA Toolkit 13.1+ |
| `vulkan` | `vulkan` | Vulkan 1.3 compute |

### Neural judge execution providers

| Feature flag | `--judge-target` value | Description |
|---|---|---|
| *(none)* | `cpu` | CPU ONNX Runtime, always available |
| `judge_cuda` | `cuda` | NVIDIA CUDA ORT execution provider |
| `judge_tensorrt` | `tensorrt` | TensorRT FP16 with engine caching, requires TensorRT 8.0+ |
| `judge_rocm` | `rocm` | AMD ROCm ORT execution provider |
| `judge_directml` | `directml` | Windows DirectML ORT execution provider |
| `judge_openvino` | `openvino` | Intel OpenVINO ORT execution provider |

### Diagnostics

| Flag | Description |
|---|---|
| `progress` | Print pipeline stage progress during accuracy runs |
| `perf-metrics` | Print per-phase timing metrics (blocking_ms, compare_ms, etc.) |
| `collect-pairs` | Collect all scored pairs after judging for unbiased PR-AUC (on by default) |
| `nvtx` | Map tracing spans to Nsight Systems ranges (profiling only) |

## Available scenarios

### Accuracy (`zer-bench accuracy --list-scenarios`)

| Scenario | Mode | Description |
|---|---|---|
| `brp/dedupe` | deduplicate | BRP person deduplication |
| `brp/link` | link-only | BRP → external source linkage |
| `brp/link_and_dedupe` | link-and-dedupe | BRP simultaneous dedup + link |
| `brp_kvk/link` | link-only | BRP × KVK cross-schema linkage |
| `brp_sis/link` | link-only | BRP × SIS cross-schema linkage |
| `brp_hks/link` | link-only | BRP × HKS cross-schema linkage |
| `brp_kvk_hks/link_and_dedupe` | link-and-dedupe | BRP × KVK × HKS multi-source |
| `kvk/dedupe` | deduplicate | KVK business-register deduplication |

`--scenario all` runs all 8. Micro/smoke-test variants are also listed by `--list-scenarios`.

### Throughput (`zer-bench throughput --list-scenarios`)

Throughput only supports dedupe scenarios (`brp/dedupe`, `kvk/dedupe`).
`--scenario all` runs both back-to-back.

## Output format

Every `accuracy` and `throughput` run writes to `--out`:

| File | Description |
|---|---|
| `<run_id>_summary.csv` | Single-row CSV consumed by `zer-bench compare` |
| `<run_id>_benchmark.json` | Full metadata: metrics, timings, memory snapshots |
| `<run_id>_scored_pairs.csv` | `(score, is_match)` pairs for PR curve plotting (accuracy only) |

Use `zer-bench compare --results <dir>` to aggregate rows from multiple runs into a formatted table.

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
