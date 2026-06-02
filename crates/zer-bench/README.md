# zer-bench

Benchmark harness for zer, measures throughput, accuracy, and head-to-head comparisons against competitor libraries on Dutch administrative datasets.

- **Documentation**: [docs.zal-analytics.ch](https://docs.zal-analytics.ch)
- **Website**: [www.zal-analytics.ch](https://www.zal-analytics.ch)
- **Support & feedback**: [info@zal-analytics.ch](mailto:info@zal-analytics.ch)

## Install

```bash
cargo install zer-bench
```

For GPU-accelerated benchmarks pass a feature flag at install time:

```bash
cargo install zer-bench --features avx2     # x86-64 AVX2 SIMD
cargo install zer-bench --features cuda     # NVIDIA CUDA (requires CUDA Toolkit 13.1+)
cargo install zer-bench --features vulkan   # Vulkan 1.3 compute
```

## Datasets and models

zer-bench resolves dataset and model paths from environment variables. Download the benchmark datasets from HuggingFace and point `ZER_DATASET_DIR` at the local copy:

```bash
hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
    --repo-type dataset --local-dir ~/datasets/zer
export ZER_DATASET_DIR=~/datasets/zer
```

Neural judge benchmarks (`--judge`) also need model files:

```bash
hf download arsalan-anwari/zjudge --local-dir ~/.cache/zer/models
# ZER_MODEL_DIR defaults to ~/.cache/zer/models; override if needed
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ZER_DATASET_DIR` | `<workspace>/data` | Root directory for benchmark datasets downloaded from HuggingFace. Dataset paths are resolved as `$ZER_DATASET_DIR/benchmarks/<scenario>/...`. When unset, falls back to `<workspace>/data` (repo clone layout). |
| `ZER_MODEL_DIR` | `~/.cache/zer/models` | Directory containing neural judge ONNX model files. Mirrors the layout from `arsalan-anwari/zjudge` on HuggingFace. |
| `ZER_EXTERNAL_BENCHMARKS_DIR` | `<workspace>/benchmarks` | Root directory containing external library benchmark scripts. Scripts are resolved as `$ZER_EXTERNAL_BENCHMARKS_DIR/<library>/<mode>/run.py` (or `run.R`). Set this when running `zer-bench library` outside of a zer repository clone. Can also be passed as `--external-benchmarks-dir`. |

## Subcommands

| Subcommand | Description |
|---|---|
| `throughput` | Raw compare/EM/score throughput on a single dataset |
| `accuracy` | Precision, recall, F1, and PR-AUC against labelled ground truth |
| `library` | Run a competitor library script and capture its summary CSV |
| `library-all` | Run all configured competitor libraries for a given mode and dataset |
| `compare` | Read multiple CSV summaries and print a side-by-side comparison table |

## Quick start

```bash
# List available scenarios
zer-bench accuracy --list-scenarios

# Accuracy; results written to bench_results/ by default
zer-bench accuracy --scenario brp/dedupe --out bench_results/

# Accuracy with neural judge (replace cuda with tensorrt / rocm / directml / openvino)
zer-bench accuracy --scenario brp/dedupe --judge-target cuda --out bench_results/

# Throughput, CPU; switch --target to avx2 / cuda / vulkan for GPU backends
zer-bench throughput --scenario brp/dedupe --out bench_results/
zer-bench throughput --scenario brp/dedupe --target cuda --out bench_results/

# Run an external library (Splink) on the same scenario
zer-bench library --library splink --scenario brp/dedupe --out bench_results/

# Same, but scripts live outside a zer repo clone
zer-bench library --library splink --scenario brp/dedupe \
    --external-benchmarks-dir /path/to/my/benchmarks --out bench_results/

# Side-by-side comparison table
zer-bench compare --results bench_results/
```

## Feature flags

### Compute backends

| Flag | Description |
|---|---|
| `cpu` | Scalar CPU fallback (always available without this flag too) |
| `avx2` | x86_64 AVX2 SIMD (~4× throughput vs scalar CPU) |
| `cuda` | NVIDIA CUDA (SM 8.6+), requires CUDA Toolkit 13.1+ |
| `vulkan` | Vulkan 1.3 compute, requires Vulkan 1.3 driver |

### Neural judge execution providers

Install the feature flag that matches the hardware you want to use, then pass
`--judge-target <value>` at runtime to select the execution provider:

| Feature flag | `--judge-target` value | Description |
|---|---|---|
| *(none)* | `cpu` | CPU ONNX Runtime (default when `--judge-target` is omitted) |
| `judge_cuda` | `cuda` | NVIDIA CUDA ORT execution provider |
| `judge_tensorrt` | `tensorrt` | TensorRT FP16 with engine caching, requires TensorRT 8.0+ |
| `judge_rocm` | `rocm` | AMD ROCm ORT execution provider |
| `judge_directml` | `directml` | Windows DirectML ORT execution provider |
| `judge_openvino` | `openvino` | Intel OpenVINO ORT execution provider |

Example:

```bash
# Install with TensorRT support
cargo install zer-bench --features judge_tensorrt

# Run accuracy with TensorRT judge (--judge-target enables the judge automatically)
zer-bench accuracy --scenario brp/dedupe --judge-target tensorrt
```

### Diagnostics

| Flag | Description |
|---|---|
| `progress` | Print pipeline stage progress during accuracy runs |
| `perf-metrics` | Print per-phase timing metrics (blocking_ms, compare_ms, etc.) |
| `collect-pairs` | Collect all scored pairs after judging for unbiased PR-AUC |
| `nvtx` | Map tracing spans to Nsight Systems ranges (profiling only) |

## Output format

Every `accuracy`, `throughput`, and `library` run appends a CSV row to the `--out` directory:

```
library,dataset,mode,precision,recall,f1,pr_auc,elapsed_ms,peak_memory_mb
zer,brp_persons,deduplicate,0.984,0.982,0.983,0.991,3653,163
```

Use `zer-bench compare` to aggregate rows from multiple runs into a formatted side-by-side table.

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
