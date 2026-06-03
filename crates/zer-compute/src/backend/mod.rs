//! Backend abstraction, [`DeviceBackend`] enum, [`BackendPreference`], and
//! auto-detection logic.
//!
//! # Selecting a backend
//!
//! ```rust,ignore
//! // Auto: CUDA → AVX2 → CPU scalar
//! let backend = DeviceBackend::auto_detect();
//!
//! // Explicit with error if not compiled in or hardware missing
//! let backend = DeviceBackend::from_preference(BackendPreference::Cuda)?;
//! ```
//!
//! # Dispatching kernels
//!
//! ```rust,ignore
//! use zer_compute::kernels::hello_backend::{HelloBackend, HelloBackendInput};
//!
//! let out = backend.run::<HelloBackend>(HelloBackendInput)?;
//! ```
//!
//! `run<K>` is generic over any `K: Kernel` for which `DeviceBackend:
//! KernelDispatch<K>` is implemented.  The `impl KernelDispatch<K> for
//! DeviceBackend` blocks at the bottom of this file do the N-arm match that
//! delegates to the active variant.

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "avx2")]
pub mod avx2;

#[cfg(feature = "vulkan")]
pub mod vulkan;

use crate::{
    backend::cpu::CpuDevice,
    error::GpuError,
    kernel::{Kernel, KernelDispatch},
    kernels::{
        em_reduce::{EmReduce, EmReduceInput, EmReduceOutput},
        hello_backend::{HelloBackend, HelloBackendInput, HelloBackendOutput},
    },
};

// ── DeviceBackend ─────────────────────────────────────────────────────────────

/// Active backend selected at runtime.
///
/// Obtain via [`DeviceBackend::auto_detect`] for automatic selection, or
/// [`DeviceBackend::from_preference`] to request a specific backend.
///
/// # Feature gating
///
/// - `Cuda` variant requires `--features cuda`.
/// - `Avx2` variant requires `--features avx2` on an x86_64 host.
/// - `Cpu` is always present and is the fallback of last resort.
pub enum DeviceBackend {
    /// Scalar CPU path, delegates to `zer-compare` (Rayon parallel).
    Cpu,

    /// NVIDIA CUDA path via `cudarc`, preferred when available.
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaDevice),

    /// Vulkan 1.3 compute path, works on NVIDIA Maxwell+ and other Vulkan-capable GPUs.
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanDevice),

    /// x86_64 AVX2 SIMD path, no external toolchain required.
    #[cfg(feature = "avx2")]
    Avx2,
}

/// Backward-compatibility alias.  New code should use [`DeviceBackend`].
pub type GpuBackend = DeviceBackend;

// ── BackendPreference ─────────────────────────────────────────────────────────

/// Explicit backend preference passed to [`DeviceBackend::from_preference`].
#[non_exhaustive]
pub enum BackendPreference {
    /// Try CUDA → Vulkan → AVX2 → CPU in order (same as `auto_detect`).
    Auto,
    /// Require CUDA; error if not compiled in or no CUDA GPU available.
    Cuda,
    /// Require Vulkan; error if not compiled in or no Vulkan GPU available.
    Vulkan,
    /// Require AVX2; error if not compiled in or CPU lacks AVX2 support.
    Avx2,
    /// Always use the scalar CPU path.
    Cpu,
}

// ── DeviceBackend impl ────────────────────────────────────────────────────────

impl DeviceBackend {
    /// Auto-detect the best available backend: CUDA → AVX2 → CPU scalar.
    ///
    /// Never panics; always returns a usable backend.  Tracing output explains
    /// which path was selected and why alternatives were skipped.
    pub fn auto_detect() -> Self {
        #[cfg(feature = "cuda")]
        match cuda::CudaDevice::init() {
            Ok(dev) => {
                tracing::info!(
                    device_name = %dev.name(),
                    vram_bytes  = dev.total_vram_bytes(),
                    "compute backend: CUDA selected"
                );
                return Self::Cuda(dev);
            }
            Err(e) => tracing::warn!(%e, "CUDA init failed, trying Vulkan"),
        }

        #[cfg(feature = "vulkan")]
        match vulkan::VulkanDevice::init() {
            Ok(dev) => {
                tracing::info!(
                    device_name = %dev.name(),
                    vram_bytes  = dev.total_vram_bytes(),
                    "compute backend: Vulkan selected"
                );
                return Self::Vulkan(dev);
            }
            Err(e) => tracing::warn!(%e, "Vulkan init failed, trying AVX2"),
        }

        #[cfg(feature = "avx2")]
        if is_x86_feature_detected!("avx2") {
            tracing::info!("compute backend: AVX2 selected");
            return Self::Avx2;
        }

        tracing::warn!("compute backend: scalar CPU fallback");
        Self::Cpu
    }

    /// Force the scalar CPU backend regardless of available hardware.
    ///
    /// Useful in tests where deterministic, non-SIMD behaviour is required.
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// Initialise the CUDA backend explicitly.
    ///
    /// Requires `--features cuda`; the method does not exist without it.
    /// Returns `Err` when no CUDA-capable GPU is present or driver init fails.
    #[cfg(feature = "cuda")]
    pub fn cuda() -> Result<Self, GpuError> {
        Ok(Self::Cuda(cuda::CudaDevice::init()?))
    }

    /// Initialise the Vulkan compute backend explicitly.
    ///
    /// Requires `--features vulkan`; the method does not exist without it.
    /// Returns `Err` when no Vulkan-capable GPU is present or init fails.
    #[cfg(feature = "vulkan")]
    pub fn vulkan() -> Result<Self, GpuError> {
        Ok(Self::Vulkan(vulkan::VulkanDevice::init()?))
    }

    /// Initialise the AVX2 SIMD backend explicitly.
    ///
    /// Requires `--features avx2`; the method does not exist without it.
    /// Returns `Err` when the running CPU does not support AVX2.
    #[cfg(feature = "avx2")]
    pub fn avx2() -> Result<Self, GpuError> {
        if is_x86_feature_detected!("avx2") {
            Ok(Self::Avx2)
        } else {
            Err(GpuError::BackendUnavailable(
                "AVX2 not supported by this CPU".into(),
            ))
        }
    }

    /// Request a specific backend.
    ///
    /// Returns `Err(BackendUnavailable)` when:
    /// - The requested feature flag was not compiled in.
    /// - The hardware initialisation fails (e.g. no CUDA GPU present).
    /// - The requested ISA extension is absent at runtime (AVX2).
    pub fn from_preference(pref: BackendPreference) -> Result<Self, GpuError> {
        match pref {
            BackendPreference::Auto => Ok(Self::auto_detect()),
            BackendPreference::Cpu => Ok(Self::Cpu),

            BackendPreference::Cuda => {
                #[cfg(feature = "cuda")]
                return Ok(Self::Cuda(cuda::CudaDevice::init()?));
                #[allow(unreachable_code)]
                Err(GpuError::BackendUnavailable(
                    "CUDA backend not compiled in; rebuild with --features cuda".into(),
                ))
            }

            BackendPreference::Vulkan => {
                #[cfg(feature = "vulkan")]
                return Ok(Self::Vulkan(vulkan::VulkanDevice::init()?));
                #[allow(unreachable_code)]
                Err(GpuError::BackendUnavailable(
                    "Vulkan backend not compiled in; rebuild with --features vulkan".into(),
                ))
            }

            BackendPreference::Avx2 => {
                #[cfg(feature = "avx2")]
                {
                    if is_x86_feature_detected!("avx2") {
                        return Ok(Self::Avx2);
                    }
                    return Err(GpuError::BackendUnavailable(
                        "AVX2 not supported by this CPU".into(),
                    ));
                }
                #[allow(unreachable_code)]
                Err(GpuError::BackendUnavailable(
                    "AVX2 backend not compiled in; rebuild with --features avx2".into(),
                ))
            }
        }
    }

    /// Dispatch kernel `K` on this backend.
    pub fn run<K: Kernel>(&self, input: K::Input<'_>) -> Result<K::Output, GpuError>
    where
        Self: KernelDispatch<K>,
    {
        self.dispatch(input)
    }

    /// Human-readable name of the active backend.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda",
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => "vulkan",
            #[cfg(feature = "avx2")]
            Self::Avx2 => "avx2",
        }
    }

    /// `true` when a GPU backend (CUDA or Vulkan) is active.
    pub fn is_gpu(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => true,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => true,
            _ => false,
        }
    }

    /// `true` when this backend provides any hardware acceleration
    /// (GPU or SIMD), i.e. it is not the scalar CPU fallback.
    pub fn is_accelerated(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Available VRAM in bytes, or `None` for CPU/AVX2 paths.
    pub fn available_vram_bytes(&self) -> Option<u64> {
        match self {
            Self::Cpu => None,
            #[cfg(feature = "cuda")]
            Self::Cuda(dev) => dev.available_vram_bytes().ok(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(dev) => dev.available_vram_bytes(),
            #[cfg(feature = "avx2")]
            Self::Avx2 => None,
        }
    }

    /// Total (installed) VRAM in bytes, or `None` for CPU/AVX2 paths.
    pub fn total_vram_bytes(&self) -> Option<u64> {
        match self {
            Self::Cpu => None,
            #[cfg(feature = "cuda")]
            Self::Cuda(dev) => Some(dev.total_vram_bytes()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(dev) => Some(dev.total_vram_bytes()),
            #[cfg(feature = "avx2")]
            Self::Avx2 => None,
        }
    }
}

// ── Session-based full-GPU EM ─────────────────────────────────────────────────

/// Unified EM session handle, wraps CUDA, Vulkan, or AVX2 session state.
///
/// Callers treat this as an opaque token.  The owning `DeviceBackend` is
/// responsible for cleanup via [`DeviceBackend::em_drop_session`].
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "avx2"))]
pub(crate) enum EmSession {
    #[cfg(feature = "cuda")]
    Cuda(cuda::launch::em_reduce::CudaEmSession),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::launch::em_reduce::VulkanEmSession),
    #[cfg(feature = "avx2")]
    Avx2(avx2::launch::em_reduce::Avx2EmSession),
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "avx2"))]
impl DeviceBackend {
    /// Allocate an EM session: upload `comparison_levels` once, pre-allocate
    /// all backend-specific buffers.  Call this once before the EM loop.
    pub(crate) fn em_init_session(
        &self,
        comparison_levels: &[u32],
        n_pairs: usize,
        n_fields: usize,
    ) -> Result<EmSession, GpuError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(dev) => dev
                .em_init_session(comparison_levels, n_pairs, n_fields)
                .map(EmSession::Cuda),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(dev) => dev
                .em_init_session(comparison_levels, n_pairs, n_fields)
                .map(EmSession::Vulkan),
            #[cfg(feature = "avx2")]
            Self::Avx2 => Ok(EmSession::Avx2(avx2::device::Avx2Device::em_init_session(
                comparison_levels,
                n_pairs,
                n_fields,
            ))),
            _ => Err(GpuError::BackendUnavailable(
                "em_init_session requires an accelerated backend".into(),
            )),
        }
    }

    /// Run one full EM iteration (E-step + M-step) on the active backend.
    ///
    /// `weights` must be `ln(m[f][l] / u[f][l])`, `n_fields * 4` floats.
    /// Returns raw M-step counts; the caller normalises them into `ModelParams`.
    pub(crate) fn em_run_iteration(
        &self,
        session: &mut EmSession,
        weights: &[f32],
        log_prior_odds: f32,
    ) -> Result<EmReduceOutput, GpuError> {
        match (self, session) {
            #[cfg(feature = "cuda")]
            (Self::Cuda(dev), EmSession::Cuda(s)) => {
                dev.em_run_iteration(s, weights, log_prior_odds)
            }
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(dev), EmSession::Vulkan(s)) => {
                dev.em_run_iteration(s, weights, log_prior_odds)
            }
            #[cfg(feature = "avx2")]
            (Self::Avx2, EmSession::Avx2(s)) => {
                avx2::device::Avx2Device::em_run_iteration(s, weights, log_prior_odds)
            }
            _ => Err(GpuError::BackendUnavailable(
                "em_run_iteration requires an accelerated backend".into(),
            )),
        }
    }

    /// Release all backend-side resources held by `session`.
    ///
    /// For CUDA/AVX2: fields auto-drop.
    /// For Vulkan: explicit `VulkanEmSession::destroy` is required because
    /// `VulkanBuffer` has no `Drop` impl.
    pub(crate) fn em_drop_session(&self, session: EmSession) {
        match (self, session) {
            #[cfg(feature = "cuda")]
            (Self::Cuda(_), EmSession::Cuda(_s)) => { /* CudaSlice fields auto-drop */ }
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(dev), EmSession::Vulkan(s)) => {
                let mut alloc = dev.allocator.lock().unwrap();
                s.destroy(&dev.device, &mut alloc);
            }
            #[cfg(feature = "avx2")]
            (Self::Avx2, EmSession::Avx2(_s)) => { /* Vec fields auto-drop */ }
            _ => {}
        }
    }
}

// ── KernelDispatch impls for DeviceBackend ────────────────────────────────────
//
// One impl block per kernel.  Each block delegates to the active variant's
// per-backend KernelDispatch impl (defined in the respective launch/ modules).

impl KernelDispatch<HelloBackend> for DeviceBackend {
    fn dispatch(&self, input: HelloBackendInput) -> Result<HelloBackendOutput, GpuError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(dev) => {
                <cuda::CudaDevice as KernelDispatch<HelloBackend>>::dispatch(dev, input)
            }
            #[cfg(feature = "vulkan")]
            Self::Vulkan(dev) => {
                <vulkan::VulkanDevice as KernelDispatch<HelloBackend>>::dispatch(dev, input)
            }
            #[cfg(feature = "avx2")]
            Self::Avx2 => <avx2::Avx2Device as KernelDispatch<HelloBackend>>::dispatch(
                &avx2::Avx2Device,
                input,
            ),
            Self::Cpu => <CpuDevice as KernelDispatch<HelloBackend>>::dispatch(&CpuDevice, input),
        }
    }
}

impl KernelDispatch<EmReduce> for DeviceBackend {
    fn dispatch(&self, input: EmReduceInput<'_>) -> Result<EmReduceOutput, GpuError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(dev) => <cuda::CudaDevice as KernelDispatch<EmReduce>>::dispatch(dev, input),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(dev) => {
                <vulkan::VulkanDevice as KernelDispatch<EmReduce>>::dispatch(dev, input)
            }
            #[cfg(feature = "avx2")]
            Self::Avx2 => {
                <avx2::Avx2Device as KernelDispatch<EmReduce>>::dispatch(&avx2::Avx2Device, input)
            }
            Self::Cpu => <CpuDevice as KernelDispatch<EmReduce>>::dispatch(&CpuDevice, input),
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detect_does_not_panic() {
        let backend = DeviceBackend::auto_detect();
        let name = backend.name();
        assert!(
            matches!(name, "cpu" | "cuda" | "vulkan" | "avx2"),
            "unexpected backend name: {name}"
        );
    }

    #[test]
    fn cpu_backend_has_no_vram() {
        let b = DeviceBackend::cpu();
        assert_eq!(b.available_vram_bytes(), None);
        assert_eq!(b.total_vram_bytes(), None);
        assert!(!b.is_gpu());
        assert!(!b.is_accelerated());
    }

    #[test]
    fn cpu_backend_name() {
        assert_eq!(DeviceBackend::cpu().name(), "cpu");
    }

    #[test]
    fn cpu_preference_always_succeeds() {
        assert!(DeviceBackend::from_preference(BackendPreference::Cpu).is_ok());
    }

    #[cfg(feature = "avx2")]
    #[test]
    fn avx2_backend_is_accelerated_not_gpu() {
        let b = DeviceBackend::Avx2;
        assert!(b.is_accelerated());
        assert!(!b.is_gpu());
        assert_eq!(b.name(), "avx2");
        assert_eq!(b.available_vram_bytes(), None);
    }
}
