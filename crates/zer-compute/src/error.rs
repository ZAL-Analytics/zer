use zer_core::error::ZerError;

/// GPU-specific error type. Variants are forwarded to `ZerError::Gpu` at the
/// trait boundary so callers that only depend on `zer-core` don't need to
/// know about the GPU backend.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Vulkan error: {0}")]
    Vulkan(String),

    #[error("shader not compiled: {0}. Rebuild with the appropriate feature flag.")]
    ShaderNotFound(String),

    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),

    #[error("device memory allocation failed: requested {requested_bytes} bytes, {detail}")]
    AllocationFailed {
        requested_bytes: u64,
        detail: String,
    },

    #[error("host↔device transfer failed: {0}")]
    TransferFailed(String),

    #[error("schema mismatch in GPU kernel: expected {expected} fields, got {got}")]
    SchemaMismatch { expected: usize, got: usize },

    #[error("backend not available: {0}")]
    BackendUnavailable(String),
}

impl From<GpuError> for ZerError {
    fn from(e: GpuError) -> Self {
        ZerError::Gpu(e.to_string())
    }
}
