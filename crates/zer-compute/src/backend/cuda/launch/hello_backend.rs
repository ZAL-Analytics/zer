//! CUDA dispatch for [`HelloBackend`].
//!
//! Kernel spec constants (`PTX_SRC`, `LAUNCH_FN`) are read by
//! `CudaDevice::init()` so all kernel-specific details live here.

// ── Kernel spec (consumed by CudaDevice::init) ───────────────────────────────

pub(crate) static PTX_SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/hello_backend.ptx"));

pub(crate) const LAUNCH_FN: &str = "hello_backend_kernel";

// ── Dispatch ─────────────────────────────────────────────────────────────────

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{
    backend::cuda::{buffers::{alloc_zeros, download}, device::CudaDevice},
    error::GpuError,
    kernel::KernelDispatch,
    kernels::hello_backend::{HelloBackend, HelloBackendInput, HelloBackendOutput},
};

impl KernelDispatch<HelloBackend> for CudaDevice {
    fn dispatch(&self, _input: HelloBackendInput) -> Result<HelloBackendOutput, GpuError> {
        let mut d_out = alloc_zeros::<u32>(&self.stream, 1)?;

        let cfg = LaunchConfig {
            grid_dim:         (1, 1, 1),
            block_dim:        (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.hello.launch_fn)
                .arg(&mut d_out)
                .launch(cfg)
        }
        .map_err(|e| GpuError::LaunchFailed(format!("hello_backend launch: {e}")))?;

        self.stream.synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("hello_backend sync: {e}")))?;

        let h_out = download(&self.stream, &d_out)?;
        Ok(HelloBackendOutput { token: h_out[0] })
    }
}
