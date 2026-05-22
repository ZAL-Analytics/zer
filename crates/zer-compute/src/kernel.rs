//! Core trait definitions for the kernel dispatch system.
//!
//! [`Kernel`] is a marker trait that associates typed input/output with a
//! compute operation.  [`KernelDispatch<K>`] is implemented by each concrete
//! backend device (`CudaDevice`, `VulkanDevice`, `CpuDevice`) **and** by
//! [`DeviceBackend`] itself, so callers never need to name the backend:
//!
//! ```rust,ignore
//! let output = backend.run::<CompareScore>(input)?;
//! ```
//!
//! # Adding a new kernel
//!
//! 1. Create `src/kernels/my_kernel.rs`, define a marker struct + typed
//!    `Input`/`Output` and `impl Kernel for MyKernel`.
//! 2. For CUDA add `src/backend/cuda/launch/my_kernel.rs`
//!    with `impl KernelDispatch<MyKernel> for CudaDevice`.
//! 3. Add the CPU fallback `impl KernelDispatch<MyKernel> for CpuDevice` in
//!    `src/backend/cpu/launch/my_kernel.rs`.
//! 4. Add `impl KernelDispatch<MyKernel> for DeviceBackend` in
//!    `src/backend/mod.rs` (a match that delegates to the above).
//! 5. Register the new kernel in `build.rs` so it gets compiled to PTX.
//!
//! [`DeviceBackend`]: crate::backend::DeviceBackend

use crate::error::GpuError;

/// Marker trait that binds typed `Input` and `Output` to a compute operation.
///
/// Implement this on a zero-sized marker struct, the struct itself carries no
/// data; it just names the operation so Rust can resolve the right dispatch.
pub trait Kernel: Sized + 'static {
    /// Input type for this kernel.  The lifetime parameter allows borrowing
    /// host data (records, schema) without copying.
    type Input<'a>;
    /// Output type produced after the kernel completes and results are
    /// downloaded back to host memory.
    type Output;
}

/// Execute kernel `K` on `self`.
///
/// Implemented by:
/// - Backend devices (`CudaDevice`, `CpuDevice`), the actual
///   upload / launch / download logic lives here.
/// - [`DeviceBackend`], a thin match that delegates to the active variant.
///
/// Callers should go through [`DeviceBackend::run`] rather than calling
/// `dispatch` directly.
///
/// [`DeviceBackend`]: crate::backend::DeviceBackend
/// [`DeviceBackend::run`]: crate::backend::DeviceBackend::run
pub trait KernelDispatch<K: Kernel> {
    fn dispatch(&self, input: K::Input<'_>) -> Result<K::Output, GpuError>;
}
