# zer-judge

ONNX-based neural judge for the zer entity-resolution library.

Loads a DeBERTa-v3 or MiniLM NLI cross-encoder model via ONNX Runtime and uses it to adjudicate borderline record pairs that the Fellegi-Sunter scorer could not classify with high confidence. Models are hosted on HuggingFace at [arsalan-anwari/zjudge](https://huggingface.co/arsalan-anwari/zjudge).

- **Documentation**: [docs.zal-analytics.ch](https://docs.zal-analytics.ch)
- **Website**: [www.zal-analytics.ch](https://www.zal-analytics.ch)
- **Support & feedback**: [info@zal-analytics.ch](mailto:info@zal-analytics.ch)

## Getting the models

```bash
# Download all model variants to ~/.cache/zer/models/
bash scripts/download_models.sh

# Or point at a custom location:
export ZER_MODEL_DIR=/path/to/your/models
```

The `ZER_MODEL_DIR` environment variable controls where zer looks for models. If unset, zer checks `~/.cache/zer/models` and falls back to `./models`.

## Feature flags

| Flag              | Description |
|-------------------|-------------|
| `judge_cpu`       | CPU execution provider for ORT |
| `judge_cuda`      | NVIDIA CUDA execution provider (requires CUDA toolkit + cuDNN) |
| `judge_tensorrt`  | NVIDIA TensorRT EP, FP16 + engine caching (requires TensorRT 8+) |
| `judge_rocm`      | AMD ROCm execution provider |
| `judge_directml`  | Windows DirectML execution provider |
| `judge_openvino`  | Intel OpenVINO execution provider |

These are independent from `zer-compute`'s `cuda`/`avx2`/`vulkan` flags.

## Usage

```rust
use zer_judge::{spec_from_env, ModelPrecision, backend::JudgeBackend};

// Reads ZER_MODEL_DIR → ~/.cache/zer/models → ./models
let backend = JudgeBackend::auto_detect();
let spec    = spec_from_env(ModelPrecision::Fp16Fused, backend.available_vram_bytes());

// Or pick a specific model explicitly:
use zer_judge::spec::DebertaBaseSpec;
let spec = DebertaBaseSpec::from_env(ModelPrecision::Base);
```

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
