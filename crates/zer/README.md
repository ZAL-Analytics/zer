# zer

Zero-shot probabilistic entity resolution, with GPU acceleration and neural NLI judging.

`zer` is the top-level facade crate. It re-exports all sub-crates under a single dependency and exposes feature flags to opt into GPU backends and the neural judge. Built-in optimizations for Dutch administrative data (BRP, KvK, SIS II, ANPR) are included; all components are pluggable for other domains.

- **Documentation**: [docs.zal-analytics.ch](https://docs.zal-analytics.ch)
- **Website**: [www.zal-analytics.ch](https://www.zal-analytics.ch)
- **Support & feedback**: [info@zal-analytics.ch](mailto:info@zal-analytics.ch)

## Feature flags

### Compute backends

| Flag      | Description |
|-----------|-------------|
| `cuda`    | NVIDIA CUDA (SM 8.6+), requires CUDA Toolkit 13.1+ and `nvcc` |
| `vulkan`  | Vulkan 1.3 compute, requires `slangc` at build time |
| `avx2`    | x86_64 AVX2 SIMD, no extra toolchain |
| `cpu`     | Scalar CPU fallback (always available without this flag too) |

### Pipeline

| Flag       | Description |
|------------|-------------|
| `pipeline` | Enables `Pipeline`, `Ingester`, and async progress events |

### Neural judge (ORT execution providers)

| Flag              | Description |
|-------------------|-------------|
| `judge_cpu`       | CPU ORT execution provider |
| `judge_cuda`      | NVIDIA CUDA ORT execution provider |
| `judge_tensorrt`  | NVIDIA TensorRT EP (FP16, engine caching) |
| `judge_rocm`      | AMD ROCm ORT execution provider |
| `judge_directml`  | Windows DirectML ORT execution provider |
| `judge_openvino`  | Intel OpenVINO ORT execution provider |

The judge flags are independent from the compute backend flags.

## Models and datasets

- **Judge models**: [arsalan-anwari/zjudge](https://huggingface.co/arsalan-anwari/zjudge) on HuggingFace
- **Test / example datasets**: [arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset](https://huggingface.co/datasets/arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset) on HuggingFace

Download models before using the judge:

```bash
bash scripts/download_models.sh
# or set the env var:
export ZER_MODEL_DIR=/path/to/your/models
```

Datasets for tests and examples must be generated before use. See the [dataset generation guide](https://docs.zal-analytics.ch/zer/contribution/datasets.html) for prerequisites and full instructions.

## Sub-crates

| Crate | Description |
|-------|-------------|
| [`zer-core`](https://crates.io/crates/zer-core) | Core traits and types |
| [`zer-blocking`](https://crates.io/crates/zer-blocking) | Blocking strategies and inverted index |
| [`zer-compare`](https://crates.io/crates/zer-compare) | Fellegi-Sunter comparison and EM scoring |
| [`zer-schema`](https://crates.io/crates/zer-schema) | Schema inference and model registry |
| [`zer-cluster`](https://crates.io/crates/zer-cluster) | Connected-components clustering |
| [`zer-compute`](https://crates.io/crates/zer-compute) | GPU-accelerated compute backend |
| [`zer-pipeline`](https://crates.io/crates/zer-pipeline) | End-to-end pipeline orchestration |
| [`zer-judge`](https://crates.io/crates/zer-judge) | ONNX neural NLI judge |
| [`zer-adapters`](https://crates.io/crates/zer-adapters) | Polars / Arrow data-frame adapters |
| [`zer-prof`](https://crates.io/crates/zer-prof) | NVTX profiling annotations |

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
