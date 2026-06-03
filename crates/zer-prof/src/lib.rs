//! Host-side NVTX profiling annotations for `zer`, consumed by `nsys`.
//!
//! Provides macros that wrap a block with RAII NVTX ranges visible in the
//! **Nsight Systems** (`nsys`) timeline:
//!
//! | Macro            | NVTX name            | Active when           | Use for                        |
//! |------------------|----------------------|-----------------------|--------------------------------|
//! | `trace!`         | `{name}`             | any feature           | CPU and GPU host regions       |
//! | `trace_cuda!`    | `"CUDA: {name}"`     | `cuda` feature only   | CUDA kernel dispatch sites     |
//! | `trace_vulkan!`  | `"VULKAN: {shader}"`    | `vulkan` feature only | Vulkan shader dispatch sites   |
//!
//! `trace_cuda!` lets `ncu` filter to CUDA-specific regions:
//!   - `ncu --nvtx --nvtx-include "regex:^CUDA:.*" ./your_binary`
//!
//! `trace_vulkan!` lets `ncu` filter to Vulkan shader regions:
//!   - `ncu --nvtx --nvtx-include "regex:^GPU:.*" ./your_binary`
//!
//! Both macros are **zero-cost no-ops** when no feature is compiled in.
//!
//! # Feature flags
//!
//! | Feature  | Effect                                                                      |
//! |----------|-----------------------------------------------------------------------------|
//! | `nvtx`   | Activates NVTX standalone, without any compute backend                      |
//! | `cuda`   | Activates NVTX; `trace_cuda!` active; `trace_vulkan!` is a no-op           |
//! | `vulkan` | Activates NVTX; `trace_vulkan!` active; `trace_cuda!` is a no-op           |
//! | `avx2`   | Activates NVTX; `trace_cuda!` and `trace_vulkan!` are no-ops               |
//! | `cpu`    | Activates NVTX; `trace_cuda!` and `trace_vulkan!` are no-ops               |
//! | *(none)* | All macros expand to bare blocks, zero overhead, no link dep               |
//!
//! # Usage
//!
//! ```rust,ignore
//! zer_prof::init();  // call once at the start of main()
//!
//! // Host-side region, visible in nsys timeline for all backends.
//! let vectors = zer_prof::trace!("compare_batch", {
//!     comparator.compare_batch(&pairs, &schema)
//! });
//!
//! // CUDA kernel dispatch, filtered by ncu --nvtx-include "regex:^CUDA:.*".
//! let out = zer_prof::trace_cuda!("em_reduce_mstep", {
//!     backend.run::<EmReduce>(input)
//! })?;
//!
//! // Vulkan shader dispatch, filtered by ncu --nvtx-include "regex:^GPU:.*".
//! let out = zer_prof::trace_vulkan!("compare_fields", {
//!     backend.run::<CompareFields>(input)
//! })?;
//! ```

// ── NVTX guard module ─────────────────────────────────────────────────────────

#[cfg(any(
    feature = "nvtx",
    feature = "cuda",
    feature = "avx2",
    feature = "cpu",
    feature = "vulkan"
))]
pub mod nvtx;
#[cfg(any(
    feature = "nvtx",
    feature = "cuda",
    feature = "avx2",
    feature = "cpu",
    feature = "vulkan"
))]
pub use nvtx::NvtxGuard;

// ── init ─────────────────────────────────────────────────────────────────────

/// Initialise profiling state.
///
/// Currently a no-op for all backends; call once at the start of `main()`
/// before any [`trace!`] or [`trace_cuda!`] invocations.
pub fn init() {}

// ── trace! ────────────────────────────────────────────────────────────────────

/// Wrap a block with a named NVTX range.
///
/// Evaluates to the block's value.  The range is visible in Nsight Systems as
/// a labelled band and in Nsight Compute as a host-side context annotation.
/// Expands to a bare block when no feature is compiled in.
#[cfg(any(
    feature = "nvtx",
    feature = "cuda",
    feature = "avx2",
    feature = "cpu",
    feature = "vulkan"
))]
#[macro_export]
macro_rules! trace {
    ($name:expr, $body:block) => {{
        let _guard = $crate::NvtxGuard::new($name);
        $body
    }};
}

#[cfg(not(any(
    feature = "nvtx",
    feature = "cuda",
    feature = "avx2",
    feature = "cpu",
    feature = "vulkan"
)))]
#[macro_export]
macro_rules! trace {
    ($name:expr, $body:block) => {
        $body
    };
}

// ── trace_cuda! ───────────────────────────────────────────────────────────────

/// Wrap a CUDA kernel dispatch with an NVTX range prefixed `"CUDA: {name}"`.
///
/// The prefix is the anchor for Nsight Compute's NVTX filter:
/// ```text
/// ncu --nvtx --nvtx-include "regex:^CUDA:.*" ./your_binary
/// ```
/// This limits the `.ncu-rep` file to only the kernels launched inside a
/// `trace_cuda!` region.
///
/// Only active when the `cuda` feature is compiled in; expands to a bare block
/// otherwise.
#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! trace_cuda {
    ($name:expr, $body:block) => {{
        let _guard = $crate::NvtxGuard::new_cuda($name);
        $body
    }};
}

#[cfg(not(feature = "cuda"))]
#[macro_export]
macro_rules! trace_cuda {
    ($name:expr, $body:block) => {
        $body
    };
}

// ── trace_vulkan! ─────────────────────────────────────────────────────────────

/// Wrap a Vulkan shader dispatch with an NVTX range prefixed `"VULKAN: {shader}"`.
///
/// `shader` should be the SPIR-V entry-point name (e.g. `"compare_fields"`),
/// making the range easy to locate in both Nsight Systems and Nsight Compute.
///
/// The prefix is the anchor for Nsight Compute's NVTX filter:
/// ```text
/// ncu --nvtx --nvtx-include "regex:^GPU:.*" ./your_binary
/// ```
/// This limits the `.ncu-rep` file to only the shader dispatches inside a
/// `trace_vulkan!` region.
///
/// Only active when the `vulkan` feature is compiled in; expands to a bare block
/// otherwise.
#[cfg(feature = "vulkan")]
#[macro_export]
macro_rules! trace_vulkan {
    ($shader:expr, $body:block) => {{
        let _guard = $crate::NvtxGuard::new_vulkan($shader);
        $body
    }};
}

#[cfg(not(feature = "vulkan"))]
#[macro_export]
macro_rules! trace_vulkan {
    ($shader:expr, $body:block) => {
        $body
    };
}
