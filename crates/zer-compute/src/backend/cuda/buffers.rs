//! CUDA buffer helpers, upload, download, and zero-allocation wrappers.
//!
//! In cudarc 0.19, memory ops use `self: &Arc<CudaStream>`.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits};

use crate::error::GpuError;

/// Upload a host slice to a newly allocated device buffer.
pub fn upload<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    data: &[T],
) -> Result<CudaSlice<T>, GpuError> {
    stream
        .clone_htod(data)
        .map_err(|e| GpuError::TransferFailed(format!("clone_htod: {e}")))
}

/// Download a device slice or view to a host `Vec`.
pub fn download<T: DeviceRepr, Src: DevicePtr<T>>(
    stream: &Arc<CudaStream>,
    d: &Src,
) -> Result<Vec<T>, GpuError> {
    stream
        .clone_dtoh(d)
        .map_err(|e| GpuError::TransferFailed(format!("clone_dtoh: {e}")))
}

/// Allocate a zero-initialised device buffer of `n` elements.
pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    n: usize,
) -> Result<CudaSlice<T>, GpuError> {
    stream
        .alloc_zeros::<T>(n)
        .map_err(|e| GpuError::AllocationFailed {
            requested_bytes: (n * std::mem::size_of::<T>()) as u64,
            detail: e.to_string(),
        })
}
