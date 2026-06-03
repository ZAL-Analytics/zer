//! CUDA device initialisation.
//!
//! `CudaDevice` owns a `CudaContext` + `CudaStream` and pre-loads all kernel
//! PTX modules at construction time.  Kernel-specific resources are grouped
//! into per-kernel structs (`CompareKernels`, `EmKernels`, `HelloKernels`) so
//! that `device.rs` stays free of per-kernel dispatch logic.
//!
//! The actual per-kernel dispatch logic lives in `launch/` submodules as
//! `impl KernelDispatch<K> for CudaDevice`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};

use crate::error::GpuError;

use super::launch::{em_reduce, hello_backend};

/// Block size shared across all GPU kernels.  256 threads is a safe starting
/// point for register-heavy kernels across Pascal (SM 6.x) and newer.
pub(crate) const BLOCK_DIM: u32 = 256;

// ── Per-kernel resource bundles ───────────────────────────────────────────────

pub(crate) struct EmKernels {
    pub _module: Arc<CudaModule>,
    pub estep_fn: CudaFunction,
    pub partial_fn: CudaFunction,
    pub final_fn: CudaFunction,
}

pub(crate) struct HelloKernels {
    pub _module: Arc<CudaModule>,
    pub launch_fn: CudaFunction,
}

// ── CudaDevice ────────────────────────────────────────────────────────────────

/// Wrapper around a cudarc 0.19 `CudaContext` with pre-loaded kernels.
///
/// Fields are `pub(crate)` so that the `launch/` submodules can access them
/// directly.
pub struct CudaDevice {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) em: EmKernels,
    pub(crate) hello: HelloKernels,
    device_name: String,
    total_vram: u64,
}

impl CudaDevice {
    pub fn init() -> Result<Self, GpuError> {
        let ctx = CudaContext::new(0)
            .map_err(|e| GpuError::Cuda(format!("CudaContext::new failed: {e}")))?;

        let (sm_major, sm_minor) = ctx
            .compute_capability()
            .map_err(|e| GpuError::Cuda(format!("compute_capability query failed: {e}")))?;
        if (sm_major, sm_minor) < (8, 6) {
            return Err(GpuError::Cuda(format!(
                "GPU compute capability {sm_major}.{sm_minor} is below the required 8.6 \
                 (Ampere, RTX 30-series or newer). For older GPUs, use the Vulkan backend instead."
            )));
        }
        tracing::debug!(sm_major, sm_minor, "CUDA compute capability check passed");

        let device_name = ctx.name().map_err(|e| GpuError::Cuda(e.to_string()))?;
        let total_vram = ctx.total_mem().map_err(|e| GpuError::Cuda(e.to_string()))? as u64;

        let (free_mb, total_mb) = {
            let (f, t) = ctx
                .mem_get_info()
                .map_err(|e| GpuError::Cuda(e.to_string()))?;
            (f / (1024 * 1024), t / (1024 * 1024))
        };
        tracing::debug!(device = %device_name, free_mb, total_mb, "CUDA device info");

        let stream = ctx.default_stream();

        let em = {
            let ptx = cudarc::nvrtc::Ptx::from_src(em_reduce::PTX_SRC.to_string());
            let module = ctx
                .load_module(ptx)
                .map_err(|e| GpuError::Cuda(format!("load_module em_reduce: {e}")))?;
            let estep_fn = module
                .load_function(em_reduce::ESTEP_FN)
                .map_err(|_| GpuError::ShaderNotFound(em_reduce::ESTEP_FN.into()))?;
            let partial_fn = module
                .load_function(em_reduce::PARTIAL_FN)
                .map_err(|_| GpuError::ShaderNotFound(em_reduce::PARTIAL_FN.into()))?;
            let final_fn = module
                .load_function(em_reduce::FINAL_FN)
                .map_err(|_| GpuError::ShaderNotFound(em_reduce::FINAL_FN.into()))?;
            EmKernels {
                _module: module,
                estep_fn,
                partial_fn,
                final_fn,
            }
        };

        let hello = {
            let ptx = cudarc::nvrtc::Ptx::from_src(hello_backend::PTX_SRC.to_string());
            let module = ctx
                .load_module(ptx)
                .map_err(|e| GpuError::Cuda(format!("load_module hello_backend: {e}")))?;
            let launch_fn = module
                .load_function(hello_backend::LAUNCH_FN)
                .map_err(|_| GpuError::ShaderNotFound(hello_backend::LAUNCH_FN.into()))?;
            HelloKernels {
                _module: module,
                launch_fn,
            }
        };

        Ok(Self {
            ctx,
            stream,
            em,
            hello,
            device_name,
            total_vram,
        })
    }

    pub fn name(&self) -> &str {
        &self.device_name
    }
    pub fn total_vram_bytes(&self) -> u64 {
        self.total_vram
    }

    pub fn available_vram_bytes(&self) -> Result<u64, GpuError> {
        let (free, _) = self
            .ctx
            .mem_get_info()
            .map_err(|e| GpuError::Cuda(e.to_string()))?;
        Ok(free as u64)
    }
}
