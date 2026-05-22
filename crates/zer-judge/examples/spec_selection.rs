//! Show how `spec_from_vram` selects the best built-in model spec, and how
//! individual specs are constructed from a directory.
//!
//! Run with:
//!   cargo run -p zer-judge --example spec_selection
//!
//! To simulate different VRAM amounts:
//!   cargo run -p zer-judge --example spec_selection -- --vram-gb=8

use std::path::Path;
use zer_judge::spec::{
    DebertaBaseSpec, JudgeModelSpec, MiniLmSpec, spec_from_vram,
};

fn main() {
    // ── 1. Parse optional --vram-gb= flag ────────────────────────────────────
    let vram_gb: f64 = std::env::args()
        .find_map(|a| a.strip_prefix("--vram-gb=").and_then(|v| v.parse().ok()))
        .unwrap_or(0.5);

    let vram_bytes = (vram_gb * 1024.0 * 1024.0 * 1024.0) as u64;

    println!("Simulated VRAM: {vram_gb:.1} GB ({vram_bytes} bytes)");
    println!();

    // ── 2. spec_from_vram with the real models directory ─────────────────────
    let models_dir = Path::new("models/fp16_fused");
    let spec = spec_from_vram(models_dir, vram_bytes);

    println!("spec_from_vram selected:");
    print_spec(spec.as_ref());
    println!();

    // ── 3. Individual constructors ────────────────────────────────────────────
    println!("All built-in specs (from_dir constructors):");
    println!();

    let specs: Vec<(&str, Box<dyn JudgeModelSpec>)> = vec![
        ("nli-minilm-onnx",          Box::new(MiniLmSpec::from_dir("models/fp16_fused/nli-minilm-onnx"))),
        ("nli-deberta-v3-base-onnx", Box::new(DebertaBaseSpec::from_dir("models/fp16_fused/nli-deberta-v3-base-onnx"))),
    ];

    for (dir, spec) in &specs {
        let exists = spec.model_path().exists();
        let status = if exists { "found   " } else { "missing " };
        println!(
            "  [{status}]  {:<46}  VRAM: {:>7.0} MB  max_len: {}",
            spec.name(),
            spec.vram_bytes() as f64 / (1024.0 * 1024.0),
            spec.max_length(),
        );
        println!("             dir: {dir}  model: {}", spec.model_path().display());
        println!();
    }
}

fn print_spec(spec: &dyn JudgeModelSpec) {
    println!("  name        : {}", spec.name());
    println!("  model_path  : {}", spec.model_path().display());
    println!("  max_length  : {}", spec.max_length());
    println!("  vram_bytes  : {} MB", spec.vram_bytes() / (1024 * 1024));
    println!("  entail_idx  : {}", spec.entailment_idx());
}
