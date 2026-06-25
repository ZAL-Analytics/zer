# zer-prof

Host-side NVTX profiling annotations for the zer entity-resolution library.

Provides `trace!`, `trace_cuda!`, and `trace_vulkan!` macros that wrap code blocks with RAII NVTX ranges visible in **Nsight Systems** (`nsys`). All macros are zero-cost no-ops when no feature flag is enabled.

- **Documentation**: [docs.zal-analytics.nl](https://docs.zal-analytics.nl)
- **Website**: [www.zal-analytics.nl](https://www.zal-analytics.nl)
- **Support & feedback**: [info@zal-analytics.nl](mailto:info@zal-analytics.nl)

## Feature flags

| Flag      | Effect |
|-----------|--------|
| `cuda`    | Activates NVTX; `trace_cuda!` emits named ranges |
| `vulkan`  | Activates NVTX; `trace_vulkan!` emits named ranges |
| `avx2`    | Activates NVTX for AVX2 SIMD profiling |
| `cpu`     | Activates NVTX for CPU-path profiling |
| `nvtx`    | Standalone NVTX activation, no compute backend |
| *(none)*  | All macros expand to bare blocks, zero overhead |

## Usage

```rust
zer_prof::init(); // call once in main()

let result = zer_prof::trace!("compare_batch", {
    comparator.compare_batch(&pairs, &schema)
});
```

Filter by CUDA regions in Nsight Compute:
```bash
ncu --nvtx --nvtx-include "regex:^CUDA:.*" ./your_binary
```

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
