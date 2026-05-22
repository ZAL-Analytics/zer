//! NVTX profiling guard, used by all backends (CUDA, Vulkan, AVX2, CPU).
//!
//! NVTX ranges appear in Nsight Systems timeline as labelled bands correlated
//! with code regions. The `nvtx` crate links against `libnvToolsExt`, which
//! ships with the CUDA toolkit.

/// RAII guard that starts an NVTX range on construction and ends it on drop.
///
/// Wraps [`nvtx::RangeGuard`], which calls `ffi_range_end` automatically when
/// it goes out of scope. Construct via [`zer_prof::trace!`](crate::trace) or
/// [`zer_prof::trace_cuda!`](crate::trace_cuda), do not construct directly.
pub struct NvtxGuard {
    _range: nvtx::RangeGuard,
}

impl NvtxGuard {
    /// Start a named NVTX range (no prefix).
    pub fn new(name: &str) -> Self {
        Self { _range: nvtx::range!("{}", name) }
    }

    /// Start an NVTX range prefixed with `"CUDA: "`.
    ///
    /// Used by [`trace_cuda!`](crate::trace_cuda) so that Nsight Compute's
    /// `--nvtx-include "regex:^CUDA:.*"` filter selects only CUDA kernel regions.
    pub fn new_cuda(name: &str) -> Self {
        Self { _range: nvtx::range!("CUDA: {}", name) }
    }

    /// Start an NVTX range prefixed with `"VULKAN: "` for a Vulkan shader region.
    ///
    /// `shader` is the SPIR-V shader entry-point name (e.g. `"compare_fields"`).
    /// Used by [`trace_vulkan!`](crate::trace_vulkan) so that Nsight Compute's
    /// `--nvtx-include "regex:^GPU:.*"` filter selects only Vulkan shader regions.
    pub fn new_vulkan(shader: &str) -> Self {
        Self { _range: nvtx::range!("VULKAN: {}", shader) }
    }
}
// nvtx::RangeGuard::drop() calls ffi_range_end automatically.
