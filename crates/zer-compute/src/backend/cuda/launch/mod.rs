//! CUDA kernel launch implementations.
//!
//! Each submodule contains `impl KernelDispatch<K> for CudaDevice` for one
//! kernel `K`.  Add a new file here when introducing a new CUDA kernel.

pub mod em_reduce;
pub mod hello_backend;
