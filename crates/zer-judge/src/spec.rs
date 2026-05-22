/// Model specifications, describes where to find the ONNX model and tokenizer,
/// and how to interpret its output.
///
/// # Model resolution
///
/// Models are loaded from a local directory.  The recommended layout (produced
/// by `scripts/download_models.sh`) is:
///
/// ```text
/// $ZER_MODEL_DIR/
///   nli-base/
///     base/             # FP32, CPU baseline
///     fp16/             # FP16 weights
///     fp16_fused/       # FP16 + graph fusions (CUDA / TensorRT preferred)
/// ```
///
/// Resolution order for the `from_env` constructors:
///
/// 1. `ZER_MODEL_DIR` environment variable (explicit override)
/// 2. `~/.cache/zer/models` (user cache, populated by `scripts/download_models.sh`)
/// 3. `./models` relative to the current working directory (workspace default)
use std::path::{Path, PathBuf};

// ── Model resolution helpers ──────────────────────────────────────────────────

/// Returns the directory where zer looks for judge models.
///
/// Resolution order:
/// 1. `ZER_MODEL_DIR` environment variable
/// 2. `~/.cache/zer/models`
/// 3. `./models` (workspace fallback)
pub fn default_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ZER_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let cache = PathBuf::from(home).join(".cache").join("zer").join("models");
        if cache.exists() {
            return cache;
        }
    }
    PathBuf::from("models")
}

// ── TokenizerSource ───────────────────────────────────────────────────────────

/// Specifies how the tokenizer is loaded.
#[derive(Debug, Clone)]
pub enum TokenizerSource {
    /// Load from a `tokenizer.json` file on disk.
    File(PathBuf),
    /// Use a Hugging Face model identifier (downloads on first use via the
    /// `tokenizers` crate's built-in hub).
    HuggingFace(String),
}

impl TokenizerSource {
    /// Convenience: file path from any `AsRef<Path>`.
    pub fn file(path: impl AsRef<Path>) -> Self {
        Self::File(path.as_ref().to_owned())
    }

    /// Convenience: HuggingFace model id as a string.
    pub fn hub(model_id: impl Into<String>) -> Self {
        Self::HuggingFace(model_id.into())
    }
}

// ── JudgeModelSpec ────────────────────────────────────────────────────────────

/// Everything needed to load and run a judge model.
///
/// Implement this trait to add a new model variant; the built-in specs are
/// [`MiniLmSpec`] (small default) and [`DebertaBaseSpec`] (large default).
pub trait JudgeModelSpec: Send + Sync {
    /// Human-readable model name for diagnostics.
    fn name(&self) -> &str;

    /// Path to the ONNX model file on disk.
    fn model_path(&self) -> &Path;

    /// Where to load the tokenizer from.
    fn tokenizer_source(&self) -> &TokenizerSource;

    /// Maximum token sequence length the model accepts.
    fn max_length(&self) -> usize;

    /// Index of the "entailment" class in the model output logits.
    ///
    /// For NLI cross-encoders: `entailment` → match, `contradiction` → no-match.
    fn entailment_idx(&self) -> usize;

    /// Approximate VRAM requirement in bytes for this model.
    fn vram_bytes(&self) -> u64;
}

// ── ModelPrecision ────────────────────────────────────────────────────────────

/// Precision variant of the ONNX model to load.
///
/// Matches the subdirectory layout produced by `scripts/download_models.sh`:
///
/// | Variant      | Subfolder      | Notes                                         |
/// |--------------|----------------|-----------------------------------------------|
/// | `Base`       | `base/`        | FP32, no optimisation; CPU baseline           |
/// | `Fp16`       | `fp16/`        | FP16 weights, no graph fusions                |
/// | `Fp16Fused`  | `fp16_fused/`  | FP16 + level-2 fusions; CUDA / TensorRT best  |
#[derive(Debug, Clone, Copy, Default)]
pub enum ModelPrecision {
    Base,
    Fp16,
    #[default]
    Fp16Fused,
}

impl ModelPrecision {
    pub fn subfolder(self) -> &'static str {
        match self {
            Self::Base     => "base",
            Self::Fp16     => "fp16",
            Self::Fp16Fused => "fp16_fused",
        }
    }
}

// ── MiniLmSpec ────────────────────────────────────────────────────────────────

/// MiniLM-L6-v2 NLI cross-encoder (~23 MB ONNX, fits in 256 MB VRAM).
pub struct MiniLmSpec {
    model_path:       PathBuf,
    tokenizer_source: TokenizerSource,
}

impl MiniLmSpec {
    pub fn new(model_path: impl AsRef<Path>, tokenizer_source: TokenizerSource) -> Self {
        Self {
            model_path:       model_path.as_ref().to_owned(),
            tokenizer_source,
        }
    }

    /// Convenience: load both model and tokenizer from the same directory.
    /// Expects `<dir>/model.onnx` and `<dir>/tokenizer.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            model_path:       dir.join("model.onnx"),
            tokenizer_source: TokenizerSource::file(dir.join("tokenizer.json")),
        }
    }

    /// Load from the resolved models directory (see [`default_models_dir`]).
    ///
    /// Looks for the FP16-fused variant first, falling back to the FP32 base.
    /// Download models first with `scripts/download_models.sh` or set
    /// `ZER_MODEL_DIR` to point at a local directory.
    pub fn from_env(precision: ModelPrecision) -> Self {
        let base = default_models_dir()
            .join("nli-base")
            .join(precision.subfolder())
            .join("nli-minilm-onnx");
        Self::from_dir(base)
    }
}

impl JudgeModelSpec for MiniLmSpec {
    fn name(&self)            -> &str  { "cross-encoder/nli-MiniLM2-L6-H768" }
    fn model_path(&self)      -> &Path { &self.model_path }
    fn tokenizer_source(&self) -> &TokenizerSource { &self.tokenizer_source }
    fn max_length(&self)      -> usize { 512 }
    fn entailment_idx(&self)  -> usize { 1 }
    fn vram_bytes(&self)      -> u64   { 256 * 1024 * 1024 } // 256 MB
}

// ── DebertaBaseSpec ───────────────────────────────────────────────────────────

/// DeBERTa-v3-base NLI (~185 MB ONNX, fits in 2 GB VRAM).
pub struct DebertaBaseSpec {
    model_path:       PathBuf,
    tokenizer_source: TokenizerSource,
}

impl DebertaBaseSpec {
    pub fn new(model_path: impl AsRef<Path>, tokenizer_source: TokenizerSource) -> Self {
        Self {
            model_path:       model_path.as_ref().to_owned(),
            tokenizer_source,
        }
    }

    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            model_path:       dir.join("model.onnx"),
            tokenizer_source: TokenizerSource::file(dir.join("tokenizer.json")),
        }
    }

    /// Load from the resolved models directory (see [`default_models_dir`]).
    ///
    /// Download models first with `scripts/download_models.sh` or set
    /// `ZER_MODEL_DIR` to point at a local directory.
    pub fn from_env(precision: ModelPrecision) -> Self {
        let base = default_models_dir()
            .join("nli-base")
            .join(precision.subfolder())
            .join("nli-deberta-v3-base-onnx");
        Self::from_dir(base)
    }
}

impl JudgeModelSpec for DebertaBaseSpec {
    fn name(&self)            -> &str  { "cross-encoder/nli-deberta-v3-base" }
    fn model_path(&self)      -> &Path { &self.model_path }
    fn tokenizer_source(&self) -> &TokenizerSource { &self.tokenizer_source }
    fn max_length(&self)      -> usize { 512 }
    fn entailment_idx(&self)  -> usize { 1 }
    fn vram_bytes(&self)      -> u64   { 2 * 1024 * 1024 * 1024 } // 2 GB
}

// ── spec_from_env / spec_from_vram ───────────────────────────────────────────

/// Select the most capable spec that fits within `available_vram_bytes`, loading
/// from the resolved models directory (see [`default_models_dir`]).
///
/// This is the easiest entry-point for end users: run `scripts/download_models.sh`
/// (or set `ZER_MODEL_DIR`), then call `spec_from_env` and let zer pick the best
/// model for the available hardware.
pub fn spec_from_env(precision: ModelPrecision, available_vram_bytes: u64) -> Box<dyn JudgeModelSpec> {
    let models_dir = default_models_dir().join("nli-base").join(precision.subfolder());
    spec_from_vram(&models_dir, available_vram_bytes)
}

/// Select the most capable built-in spec that fits within `available_vram_bytes`.
///
/// Two defaults: small → MiniLM-L6, large → DeBERTa-v3-base.
///
/// Requires a directory layout produced by `models/generate_onnx_model.py`:
/// ```text
/// models/nli-base/base/         ← TensorRT and CPU (plain FP32, no fusions)
///   nli-deberta-v3-base-onnx/model.onnx
///   nli-minilm-onnx/model.onnx
/// models/nli-base/fp16_fused/   ← CUDA / ROCm / DirectML / OpenVINO (preferred)
///   nli-deberta-v3-base-onnx/model.onnx
///   nli-minilm-onnx/model.onnx
/// models/nli-base/fp16/         ← CUDA fallback (FP16 weights, no ORT fusions)
///   nli-deberta-v3-base-onnx/model.onnx
///   nli-minilm-onnx/model.onnx
/// ```
pub fn spec_from_vram(models_dir: &Path, available_vram_bytes: u64) -> Box<dyn JudgeModelSpec> {
    let base  = models_dir.join("nli-deberta-v3-base-onnx");
    let mini  = models_dir.join("nli-minilm-onnx");

    if available_vram_bytes >= 2 * 1024 * 1024 * 1024 && base.exists() {
        tracing::info!("judge: selecting DeBERTa-v3-base ({:.1} GB VRAM available)",
            available_vram_bytes as f64 / 1e9);
        return Box::new(DebertaBaseSpec::from_dir(&base));
    }

    tracing::info!("judge: selecting MiniLM-L6 (CPU or low VRAM)");
    Box::new(MiniLmSpec::from_dir(&mini))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dummy_path(name: &str) -> PathBuf {
        PathBuf::from(format!("/nonexistent/{name}"))
    }

    // ── MiniLmSpec ────────────────────────────────────────────────────────────

    #[test]
    fn minilm_from_dir_sets_expected_paths() {
        let spec = MiniLmSpec::from_dir("/some/dir");
        assert_eq!(spec.model_path(), Path::new("/some/dir/model.onnx"));
        assert!(matches!(spec.tokenizer_source(), TokenizerSource::File(p) if p == Path::new("/some/dir/tokenizer.json")));
    }

    #[test]
    fn minilm_metadata() {
        let spec = MiniLmSpec::from_dir("/d");
        assert_eq!(spec.name(), "cross-encoder/nli-MiniLM2-L6-H768");
        assert_eq!(spec.max_length(), 512);
        assert_eq!(spec.entailment_idx(), 1);
        assert_eq!(spec.vram_bytes(), 256 * 1024 * 1024);
    }

    // ── DebertaBaseSpec ───────────────────────────────────────────────────────

    #[test]
    fn deberta_base_from_dir_sets_expected_paths() {
        let spec = DebertaBaseSpec::from_dir("/fp16_fused/dir");
        assert_eq!(spec.model_path(), Path::new("/fp16_fused/dir/model.onnx"));
        assert!(matches!(spec.tokenizer_source(), TokenizerSource::File(p) if p == Path::new("/fp16_fused/dir/tokenizer.json")));
    }

    #[test]
    fn deberta_base_metadata() {
        let spec = DebertaBaseSpec::from_dir("/d");
        assert_eq!(spec.name(), "cross-encoder/nli-deberta-v3-base");
        assert_eq!(spec.max_length(), 512);
        assert_eq!(spec.entailment_idx(), 1);
        assert_eq!(spec.vram_bytes(), 2 * 1024 * 1024 * 1024);
    }

    // ── spec_from_vram ────────────────────────────────────────────────────────

    #[test]
    fn spec_from_vram_no_dirs_returns_minilm() {
        // Non-existent dirs → all exist() == false → always falls through to MiniLm
        let spec = spec_from_vram(Path::new("/nonexistent"), 16 * 1024 * 1024 * 1024);
        assert_eq!(spec.name(), "cross-encoder/nli-MiniLM2-L6-H768");
    }

    #[test]
    fn spec_from_vram_selects_minilm_when_low_vram() {
        // Even if dirs existed, 512 MB isn't enough for base (needs 2 GB)
        let spec = spec_from_vram(Path::new("/nonexistent"), 512 * 1024 * 1024);
        assert_eq!(spec.name(), "cross-encoder/nli-MiniLM2-L6-H768");
    }

    #[test]
    fn spec_from_vram_with_real_models_dir_selects_best_available() {
        // Test with the actual models directory if it exists
        let models_dir = Path::new("../../models/nli-base/fp16_fused");
        if !models_dir.exists() {
            return; // Skip if models directory not available
        }
        // With 0 VRAM, must fall back to MiniLM
        let spec = spec_from_vram(models_dir, 0);
        assert_eq!(spec.name(), "cross-encoder/nli-MiniLM2-L6-H768");
    }

    #[test]
    fn token_source_file_convenience() {
        let ts = TokenizerSource::file("/tmp/tok.json");
        assert!(matches!(ts, TokenizerSource::File(p) if p == Path::new("/tmp/tok.json")));
    }

    #[test]
    fn token_source_hub_convenience() {
        let ts = TokenizerSource::hub("cross-encoder/nli-deberta-v3-base");
        assert!(matches!(ts, TokenizerSource::HuggingFace(s) if s == "cross-encoder/nli-deberta-v3-base"));
    }

    #[test]
    fn minilm_new_constructor() {
        let spec = MiniLmSpec::new(
            dummy_path("model.onnx"),
            TokenizerSource::file(dummy_path("tok.json")),
        );
        assert_eq!(spec.model_path(), Path::new("/nonexistent/model.onnx"));
    }
}
