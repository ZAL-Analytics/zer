//! CPU backend, pure-CPU implementations of all kernels via `zer-compare`.
//!
//! Always compiled, regardless of feature flags.  When neither `cuda` nor
//! `vulkan` is enabled, or when the GPU is unavailable / the batch is below
//! the minimum threshold, every call routes through here.

pub mod device;
pub mod launch;

pub use device::{CpuDevice, CpuFallbackComparator, CpuFallbackScorer, cpu_estimate_params};
