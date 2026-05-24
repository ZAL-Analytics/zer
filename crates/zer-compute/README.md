# zer-compute

Hardware-accelerated backend for the zer entity-resolution library.

Provides `DeviceComparator` and `DeviceScorer` as drop-in replacements for the CPU-only counterparts in `zer-compare`. Both implement the same `zer_core` traits, so the rest of the pipeline is fully backend-agnostic.

- **Documentation**: [docs.zal-analytics.ch](https://docs.zal-analytics.ch)
- **Website**: [www.zal-analytics.ch](https://www.zal-analytics.ch)
- **Support & feedback**: [info@zal-analytics.ch](mailto:info@zal-analytics.ch)

## Feature flags

| Flag     | Backend | Requirements |
|----------|---------|--------------|
| `cuda`   | NVIDIA CUDA (SM 8.6+, Ampere / RTX 30-series) | CUDA Toolkit 13.1+, `nvcc` at build time |
| `vulkan` | Vulkan compute (NVIDIA Maxwell+, AMD, Intel) | Vulkan 1.3 GPU, `slangc` on PATH at build time |
| `avx2`   | x86_64 AVX2 SIMD | No external toolchain |
| `cpu`    | Scalar CPU fallback | Always available |

When no flag is set the crate compiles normally using the scalar fallback from `zer-compare`.

## Usage

```rust
use std::sync::Arc;
use zer_compute::{GpuBackend, DeviceComparator, DeviceScorer};

// Auto-detect: tries CUDA → AVX2 → CPU in order.
let backend    = Arc::new(GpuBackend::auto_detect());
let comparator = DeviceComparator::new(Arc::clone(&backend), &schema)?;
let scorer     = DeviceScorer::new(Arc::clone(&backend));
```

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
