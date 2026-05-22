/// Execution-provider selection for ORT-based judge models.
///
/// Reads `--judge-target=<name>` from process args.  This flag is entirely
/// separate from `--target=<name>` used by `zer::Backend` (which controls the
/// pairwise-comparison and EM compute backend).
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use zer_judge::JudgeBackend;
/// let jb = JudgeBackend::auto_detect();   // reads --judge-target=
/// let session_builder = jb.configure_session(ort::session::Session::builder()?)?;
/// # Ok(()) }
/// ```
use ort::ep::ExecutionProviderDispatch;

// ── TrtProfile ────────────────────────────────────────────────────────────────

/// TensorRT dynamic-shape profile: min / opt / max for batch and sequence length.
///
/// TRT compiles a kernel specialised for `opt`; `min`/`max` bound the range it
/// will accept without falling back to the CUDA or CPU EP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrtProfile {
    pub min_batch: usize,
    pub min_seq:   usize,
    pub opt_batch: usize,
    pub opt_seq:   usize,
    pub max_batch: usize,
    pub max_seq:   usize,
}

impl TrtProfile {
    /// Shapes tuned for the BRP workload: ~20 borderline pairs padded to ~35–64 tokens.
    pub const DEFAULT: Self = Self {
        min_batch: 1,  min_seq: 1,
        opt_batch: 32, opt_seq: 64,
        max_batch: 64, max_seq: 512,
    };

    fn to_shape_string(self, batch: usize, seq: usize) -> String {
        format!(
            "input_ids:{batch}x{seq},attention_mask:{batch}x{seq},token_type_ids:{batch}x{seq}",
        )
    }

    pub fn min_shapes(self) -> String { self.to_shape_string(self.min_batch, self.min_seq) }
    pub fn opt_shapes(self) -> String { self.to_shape_string(self.opt_batch, self.opt_seq) }
    pub fn max_shapes(self) -> String { self.to_shape_string(self.max_batch, self.max_seq) }
}

/// Which ORT execution provider to use for neural judge inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JudgeTarget {
    /// Run inference on CPU (always available, no features required).
    #[default]
    Cpu,
    /// NVIDIA CUDA via ORT, requires the `judge_cuda` feature.
    Cuda,
    /// NVIDIA TensorRT via ORT, requires the `judge_tensorrt` feature.
    ///
    /// Enables FP16 kernel compilation and engine caching under
    /// `~/.cache/zer-judge/trt-engines`.  Faster than raw CUDA EP for fixed
    /// sequence-length workloads after a one-time engine build.
    TensorRt,
    /// AMD ROCm via ORT, requires the `judge_rocm` feature.
    Rocm,
    /// Windows DirectML via ORT, requires the `judge_directml` feature.
    DirectMl,
    /// Intel OpenVINO via ORT, requires the `judge_openvino` feature.
    OpenVino,
}

impl JudgeTarget {
    /// Parse a target name as supplied to `--judge-target=`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cpu"       => Some(Self::Cpu),
            "cuda"      => Some(Self::Cuda),
            "tensorrt"  => Some(Self::TensorRt),
            "rocm"      => Some(Self::Rocm),
            "directml"  => Some(Self::DirectMl),
            "openvino"  => Some(Self::OpenVino),
            _           => None,
        }
    }

    /// Canonical lowercase name shown in diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu      => "cpu",
            Self::Cuda     => "cuda",
            Self::TensorRt => "tensorrt",
            Self::Rocm     => "rocm",
            Self::DirectMl => "directml",
            Self::OpenVino => "openvino",
        }
    }
}

/// Opaque handle that configures an ORT `SessionBuilder` with the right
/// execution provider.
///
/// Create once per process; pass a reference wherever a session is built so
/// that all judge models share the same EP selection.
pub struct JudgeBackend {
    target:      JudgeTarget,
    trt_profile: TrtProfile,
}

impl JudgeBackend {
    /// Read `--judge-target=<name>` from process args and return the matching backend.
    ///
    /// Falls back to CPU when the flag is absent, no hardware probing.
    pub fn auto_detect() -> Self {
        let target = std::env::args()
            .find_map(|a| a.strip_prefix("--judge-target=").map(str::to_owned));

        match target.as_deref() {
            Some(t) => Self::from_target(t),
            None    => Self::cpu(),
        }
    }

    /// Force the CPU execution provider regardless of available hardware.
    pub fn cpu() -> Self {
        Self { target: JudgeTarget::Cpu, trt_profile: TrtProfile::DEFAULT }
    }

    /// Use CUDA if compiled in, otherwise fall back to CPU.
    ///
    /// Used when TRT is requested but the model contains ORT-fused ops that
    /// TRT cannot parse, CUDA still accelerates inference without the noise.
    pub fn cuda_or_cpu() -> Self {
        if cfg!(feature = "judge_cuda") {
            Self { target: JudgeTarget::Cuda, trt_profile: TrtProfile::DEFAULT }
        } else {
            Self::cpu()
        }
    }

    /// Override the TensorRT dynamic-shape profile (only takes effect when `target` is `TensorRt`).
    ///
    /// Any prior engine cache built with different shapes will be ignored; TRT rebuilds the engine.
    pub fn with_trt_profile(mut self, profile: TrtProfile) -> Self {
        self.trt_profile = profile;
        self
    }

    /// Select a backend by name, called by `auto_detect()` to resolve `--judge-target=<name>`.
    ///
    /// Accepted values: `"cpu"`, `"cuda"`, `"rocm"`, `"directml"`, `"openvino"`.
    ///
    /// Exits with a diagnostic if the target is unknown or not compiled in.
    pub fn from_target(target: &str) -> Self {
        match JudgeTarget::from_name(target) {
            Some(t) => {
                if !Self::target_compiled_in(t) {
                    tracing::error!(target, "judge target not compiled in; rebuild with 'judge_{{target}}' feature flag");
                    std::process::exit(1);
                }
                Self { target: t, trt_profile: TrtProfile::DEFAULT }
            }
            None => {
                tracing::error!(target, "unknown --judge-target; valid: cpu, cuda, tensorrt, rocm, directml, openvino");
                std::process::exit(1);
            }
        }
    }

    fn target_compiled_in(target: JudgeTarget) -> bool {
        match target {
            JudgeTarget::Cpu      => true,
            JudgeTarget::Cuda     => cfg!(feature = "judge_cuda"),
            JudgeTarget::TensorRt => cfg!(feature = "judge_tensorrt"),
            JudgeTarget::Rocm     => cfg!(feature = "judge_rocm"),
            JudgeTarget::DirectMl => cfg!(feature = "judge_directml"),
            JudgeTarget::OpenVino => cfg!(feature = "judge_openvino"),
        }
    }

    /// Human-readable name of the selected execution provider.
    pub fn name(&self) -> &'static str {
        self.target.as_str()
    }

    /// Preferred models subdirectory name for this backend (no filesystem probing).
    ///
    /// For filesystem-aware resolution with fallback use [`resolve_models_dir`].
    ///
    /// - `TensorRt` / `Cpu`: `"base"`, plain FP32 ONNX, no ORT fusions.
    ///   TRT compiles its own FP16 kernels internally; CPU runs F32 directly.
    /// - All others: `"fp16_fused"`, FP16 weights + ORT graph fusions, fastest
    ///   for CUDA / ROCm / DirectML / OpenVINO execution providers.
    ///
    /// [`resolve_models_dir`]: JudgeBackend::resolve_models_dir
    pub fn models_subdir(&self) -> &'static str {
        match self.target {
            JudgeTarget::TensorRt | JudgeTarget::Cpu => "base",
            _                                        => "fp16_fused",
        }
    }

    /// Resolve the models directory under `base`, applying a fallback chain.
    ///
    /// - `TensorRt` and `Cpu`: always use `base/base` (plain FP32, no fusions).
    /// - All others (`Cuda`, `Rocm`, `DirectMl`, `OpenVino`): prefer
    ///   `base/fp16_fused`, then `base/fp16`, then fall back to `base/base`.
    pub fn resolve_models_dir(&self, base: &std::path::Path) -> std::path::PathBuf {
        match self.target {
            JudgeTarget::TensorRt | JudgeTarget::Cpu => base.join("base"),
            _ => {
                let fp16_fused = base.join("fp16_fused");
                if fp16_fused.exists() {
                    return fp16_fused;
                }
                let fp16 = base.join("fp16");
                if fp16.exists() {
                    return fp16;
                }
                base.join("base")
            }
        }
    }

    /// The selected [`JudgeTarget`].
    pub fn target(&self) -> JudgeTarget {
        self.target
    }

    /// Build the ORT `ExecutionProviderDispatch` list for this backend.
    ///
    /// The returned vec is passed to `SessionBuilder::with_execution_providers`.
    /// CPU is always appended as the final fallback.
    pub fn execution_providers(&self) -> Vec<ExecutionProviderDispatch> {
        let mut eps: Vec<ExecutionProviderDispatch> = vec![];

        match self.target {
            JudgeTarget::Cpu => {}

            JudgeTarget::Cuda => {
                #[cfg(feature = "judge_cuda")]
                eps.push(ort::ep::CUDA::default().build());
                #[cfg(not(feature = "judge_cuda"))]
                unreachable!("judge_cuda feature not compiled in, guarded by from_target()");
            }

            JudgeTarget::TensorRt => {
                #[cfg(feature = "judge_tensorrt")]
                {
                    // Engine cache lives at ~/.cache/zer-judge/trt-engines so that
                    // re-runs skip the expensive JIT compilation step.
                    let cache_dir = std::env::var("HOME")
                        .unwrap_or_else(|_| ".".to_string())
                        + "/.cache/zer-judge/trt-engines";
                    let _ = std::fs::create_dir_all(&cache_dir);
                    let p = self.trt_profile;
                    eps.push(
                        ort::ep::TensorRT::default()
                            .with_fp16(true)
                            .with_engine_cache(true)
                            .with_engine_cache_path(&cache_dir)
                            .with_profile_min_shapes(&p.min_shapes())
                            .with_profile_opt_shapes(&p.opt_shapes())
                            .with_profile_max_shapes(&p.max_shapes())
                            .build(),
                    );
                    // CUDA EP sits between TRT and CPU so that shapes TRT rejects
                    // (e.g. during engine warm-up) still run on GPU, not on CPU.
                    #[cfg(feature = "judge_cuda")]
                    eps.push(ort::ep::CUDA::default().build());
                }
                #[cfg(not(feature = "judge_tensorrt"))]
                unreachable!("judge_tensorrt feature not compiled in, guarded by from_target()");
            }

            JudgeTarget::Rocm => {
                #[cfg(feature = "judge_rocm")]
                eps.push(ort::ep::ROCm::default().build());
                #[cfg(not(feature = "judge_rocm"))]
                unreachable!("judge_rocm feature not compiled in");
            }

            JudgeTarget::DirectMl => {
                #[cfg(feature = "judge_directml")]
                eps.push(ort::ep::DirectML::default().build());
                #[cfg(not(feature = "judge_directml"))]
                unreachable!("judge_directml feature not compiled in");
            }

            JudgeTarget::OpenVino => {
                #[cfg(feature = "judge_openvino")]
                eps.push(ort::ep::OpenVINO::default().build());
                #[cfg(not(feature = "judge_openvino"))]
                unreachable!("judge_openvino feature not compiled in");
            }
        }

        // CPU is always the final fallback.
        eps.push(ort::ep::CPU::default().build());
        eps
    }

    /// Configure an ORT `SessionBuilder` with this backend's execution providers.
    pub fn configure_session(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> ort::Result<ort::session::builder::SessionBuilder> {
        Ok(builder.with_execution_providers(self.execution_providers())?)
    }
}

impl std::fmt::Display for JudgeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JudgeBackend({})", self.name())
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_name_cpu() {
        assert_eq!(JudgeTarget::from_name("cpu"), Some(JudgeTarget::Cpu));
    }

    #[test]
    fn from_name_cuda() {
        assert_eq!(JudgeTarget::from_name("cuda"), Some(JudgeTarget::Cuda));
    }

    #[test]
    fn from_name_tensorrt() {
        assert_eq!(JudgeTarget::from_name("tensorrt"), Some(JudgeTarget::TensorRt));
    }

    #[test]
    fn from_name_rocm() {
        assert_eq!(JudgeTarget::from_name("rocm"), Some(JudgeTarget::Rocm));
    }

    #[test]
    fn from_name_directml() {
        assert_eq!(JudgeTarget::from_name("directml"), Some(JudgeTarget::DirectMl));
    }

    #[test]
    fn from_name_openvino() {
        assert_eq!(JudgeTarget::from_name("openvino"), Some(JudgeTarget::OpenVino));
    }

    #[test]
    fn from_name_unknown_returns_none() {
        assert_eq!(JudgeTarget::from_name("vulkan"), None);
        assert_eq!(JudgeTarget::from_name(""), None);
        assert_eq!(JudgeTarget::from_name("CUDA"), None);
    }

    #[test]
    fn as_str_round_trips_for_all_variants() {
        let targets = [
            JudgeTarget::Cpu,
            JudgeTarget::Cuda,
            JudgeTarget::TensorRt,
            JudgeTarget::Rocm,
            JudgeTarget::DirectMl,
            JudgeTarget::OpenVino,
        ];
        for target in targets {
            let name = target.as_str();
            assert_eq!(
                JudgeTarget::from_name(name),
                Some(target),
                "round-trip failed for {name}"
            );
        }
    }

    #[test]
    fn judge_backend_cpu_has_cpu_name() {
        let backend = JudgeBackend::cpu();
        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.target(), JudgeTarget::Cpu);
    }

    #[test]
    fn judge_backend_display() {
        let backend = JudgeBackend::cpu();
        assert_eq!(format!("{backend}"), "JudgeBackend(cpu)");
    }

    #[test]
    fn cpu_execution_providers_has_cpu_fallback() {
        let backend = JudgeBackend::cpu();
        let eps     = backend.execution_providers();
        assert!(!eps.is_empty(), "execution_providers must never return an empty vec");
    }

    #[test]
    fn cpu_target_is_always_compiled_in() {
        assert!(JudgeBackend::target_compiled_in(JudgeTarget::Cpu));
    }

    #[test]
    fn models_subdir_trt_returns_base() {
        let mut backend = JudgeBackend::cpu();
        backend.target = JudgeTarget::TensorRt;
        assert_eq!(backend.models_subdir(), "base");
    }

    #[test]
    fn models_subdir_cpu_returns_base() {
        let backend = JudgeBackend::cpu();
        assert_eq!(backend.models_subdir(), "base");
    }

    #[test]
    fn models_subdir_gpu_providers_return_fp16_fused() {
        for target in [JudgeTarget::Cuda, JudgeTarget::Rocm, JudgeTarget::DirectMl, JudgeTarget::OpenVino] {
            let mut backend = JudgeBackend::cpu();
            backend.target = target;
            assert_eq!(backend.models_subdir(), "fp16_fused", "expected fp16_fused for {}", target.as_str());
        }
    }

    #[test]
    fn resolve_models_dir_trt_always_returns_base() {
        let tmp = std::env::temp_dir();
        let mut backend = JudgeBackend::cpu();
        backend.target = JudgeTarget::TensorRt;
        assert_eq!(backend.resolve_models_dir(&tmp), tmp.join("base"));
    }

    #[test]
    fn resolve_models_dir_cpu_always_returns_base() {
        let tmp = std::env::temp_dir();
        let backend = JudgeBackend::cpu();
        assert_eq!(backend.resolve_models_dir(&tmp), tmp.join("base"));
    }

    #[test]
    fn resolve_models_dir_cuda_falls_back_to_base_when_no_dirs_exist() {
        // Use a non-existent base so no subdir exists → should fall back to base/base
        let base = std::path::Path::new("/nonexistent/models/nli-base");
        let mut backend = JudgeBackend::cpu();
        backend.target = JudgeTarget::Cuda;
        assert_eq!(backend.resolve_models_dir(base), base.join("base"));
    }

    #[test]
    fn trt_profile_default_shape_strings() {
        let p = TrtProfile::DEFAULT;
        assert_eq!(p.min_shapes(), "input_ids:1x1,attention_mask:1x1,token_type_ids:1x1");
        assert_eq!(p.opt_shapes(), "input_ids:32x64,attention_mask:32x64,token_type_ids:32x64");
        assert_eq!(p.max_shapes(), "input_ids:64x512,attention_mask:64x512,token_type_ids:64x512");
    }

    #[test]
    fn trt_profile_custom_values() {
        let p = TrtProfile { min_batch: 1, min_seq: 1, opt_batch: 16, opt_seq: 128, max_batch: 32, max_seq: 256 };
        assert_eq!(p.opt_shapes(), "input_ids:16x128,attention_mask:16x128,token_type_ids:16x128");
        assert_eq!(p.max_shapes(), "input_ids:32x256,attention_mask:32x256,token_type_ids:32x256");
    }

    #[test]
    fn with_trt_profile_overrides_default() {
        let custom = TrtProfile { min_batch: 1, min_seq: 1, opt_batch: 8, opt_seq: 32, max_batch: 16, max_seq: 128 };
        let backend = JudgeBackend::cpu().with_trt_profile(custom);
        assert_eq!(backend.trt_profile, custom);
    }
}
