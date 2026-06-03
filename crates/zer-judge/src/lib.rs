//! `zer-judge`, ONNX-based neural judge for the zer entity-resolution pipeline.
//!
//! Loads a DeBERTa or MiniLM NLI cross-encoder model via ORT and uses it to
//! adjudicate borderline record pairs that the Fellegi-Sunter scorer could not
//! classify with high confidence.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use zer_judge::{
//!     backend::JudgeBackend,
//!     judge::{DebertaJudge, DebertaJudgeConfig},
//!     spec::DebertaBaseSpec,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Backend reads --judge-target= from process args (separate from --target=).
//! let backend      = JudgeBackend::auto_detect();
//! let spec         = DebertaBaseSpec::from_dir("models/nli-base/fp16_fused/nli-deberta-v3-base-onnx");
//! let record_store = Arc::new(zer_core::VecRecordStore::new());
//! # let schema = zer_core::schema::Schema { fields: vec![] };
//!
//! let judge = DebertaJudge::new(
//!     &spec,
//!     &backend,
//!     record_store,
//!     schema,
//!     DebertaJudgeConfig::default(),
//! )?;
//! # let _ = judge;
//! # Ok(()) }
//! ```
//!
//! # Feature flags
//!
//! | Flag               | Description                                                          |
//! |--------------------|----------------------------------------------------------------------|
//! | `judge_cpu`        | Scalar CPU execution provider for ORT (always available, no extras) |
//! | `judge_cuda`       | NVIDIA CUDA execution provider for ORT                               |
//! | `judge_tensorrt`   | NVIDIA TensorRT EP, FP16 + engine caching (requires `judge_cuda`)   |
//! | `judge_rocm`       | AMD ROCm execution provider for ORT                                  |
//! | `judge_directml`   | Windows DirectML execution provider for ORT                          |
//! | `judge_openvino`   | Intel OpenVINO execution provider for ORT                            |
//!
//! These are **completely independent** from `zer-compute`'s `cuda`/`avx2`/`vulkan`
//! feature flags.  Use `--judge-target=<name>` to select the ORT execution provider
//! at runtime; use `--target=<name>` to select the comparison/EM compute backend.

pub mod audit;
pub mod backend;
pub mod calibration;
pub mod dummy;
pub mod error;
pub mod judge;
pub mod serialize;
pub mod session;
pub mod spec;
pub mod test_utils;
pub mod tokenize;

pub use backend::{JudgeBackend, TrtProfile};
pub use calibration::CalibrationTable;
pub use dummy::DummyJudge;
pub use error::JudgeError;
pub use judge::{DebertaJudge, DebertaJudgeConfig};
pub use spec::{
    default_models_dir, spec_from_env, spec_from_vram, DebertaBaseSpec, JudgeModelSpec, MiniLmSpec,
    ModelPrecision, TokenizerSource,
};
pub use test_utils::NearDuplicateGenerator;
