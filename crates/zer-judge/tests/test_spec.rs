/// Integration tests for `zer_judge::spec`, model spec constructors and
/// `spec_from_vram` selection logic.

use std::path::Path;
use zer_judge::spec::{
    DebertaBaseSpec, JudgeModelSpec, MiniLmSpec,
    TokenizerSource, spec_from_vram,
};

const MODELS_BASE: &str = "../../models/fp16_fused";

fn models_base() -> &'static Path {
    Path::new(MODELS_BASE)
}

// ── Built-in spec constructors ────────────────────────────────────────────────

#[test]
fn minilm_from_dir_model_path_is_correct() {
    let spec = MiniLmSpec::from_dir("/some/dir");
    assert_eq!(spec.model_path(), Path::new("/some/dir/model.onnx"));
}

#[test]
fn minilm_from_dir_tokenizer_is_file() {
    let spec = MiniLmSpec::from_dir("/some/dir");
    assert!(
        matches!(spec.tokenizer_source(), TokenizerSource::File(p) if p == Path::new("/some/dir/tokenizer.json")),
        "MiniLmSpec tokenizer source should be a file path"
    );
}

#[test]
fn deberta_base_from_dir_model_path_is_correct() {
    let spec = DebertaBaseSpec::from_dir("/fp16_fused");
    assert_eq!(spec.model_path(), Path::new("/fp16_fused/model.onnx"));
}

#[test]
fn spec_names_are_distinct() {
    let mini = MiniLmSpec::from_dir("/d");
    let base = DebertaBaseSpec::from_dir("/d");
    assert_ne!(mini.name(), base.name(), "each spec must have a unique name");
}

#[test]
fn vram_requirements_ordered() {
    let mini = MiniLmSpec::from_dir("/d");
    let base = DebertaBaseSpec::from_dir("/d");
    assert!(mini.vram_bytes() < base.vram_bytes(), "MiniLM should need less VRAM than Base");
}

#[test]
fn all_specs_have_max_length_512() {
    assert_eq!(MiniLmSpec::from_dir("/d").max_length(), 512);
    assert_eq!(DebertaBaseSpec::from_dir("/d").max_length(), 512);
}

#[test]
fn all_specs_have_entailment_idx_one() {
    // Both models use label order [contradiction=0, entailment=1, neutral=2].
    assert_eq!(MiniLmSpec::from_dir("/d").entailment_idx(), 1);
    assert_eq!(DebertaBaseSpec::from_dir("/d").entailment_idx(), 1);
}

// ── spec_from_vram selection ──────────────────────────────────────────────────

#[test]
fn spec_from_vram_zero_vram_returns_minilm() {
    let spec = spec_from_vram(Path::new("/nonexistent"), 0);
    assert_eq!(spec.name(), "cross-encoder/nli-MiniLM2-L6-H768");
}

#[test]
fn spec_from_vram_nonexistent_dirs_always_returns_minilm() {
    // Even with abundant VRAM, no dir → MiniLM (the default)
    let spec = spec_from_vram(Path::new("/nonexistent"), u64::MAX);
    assert_eq!(spec.name(), "cross-encoder/nli-MiniLM2-L6-H768");
}

#[test]
fn spec_from_vram_low_vram_returns_minilm_even_with_real_dirs() {
    if !models_base().exists() { return; }
    let spec = spec_from_vram(models_base(), 256 * 1024 * 1024);
    assert_eq!(spec.name(), "cross-encoder/nli-MiniLM2-L6-H768",
        "256 MB VRAM is not enough for DeBERTa-base");
}

#[test]
fn spec_from_vram_2gb_returns_base_when_dir_exists() {
    if !models_base().join("nli-deberta-v3-base-onnx").exists() { return; }
    let two_gb = 2 * 1024 * 1024 * 1024_u64;
    let spec = spec_from_vram(models_base(), two_gb);
    assert_eq!(spec.name(), "cross-encoder/nli-deberta-v3-base");
}

// ── TokenizerSource ───────────────────────────────────────────────────────────

#[test]
fn tokenizer_source_file_roundtrip() {
    let ts = TokenizerSource::file("/path/to/tokenizer.json");
    assert!(matches!(ts, TokenizerSource::File(p) if p == Path::new("/path/to/tokenizer.json")));
}

#[test]
fn tokenizer_source_hub_roundtrip() {
    let ts = TokenizerSource::hub("cross-encoder/nli-deberta-v3-base");
    assert!(matches!(ts, TokenizerSource::HuggingFace(s) if s == "cross-encoder/nli-deberta-v3-base"));
}
