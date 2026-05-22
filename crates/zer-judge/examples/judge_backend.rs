//! Show which ORT execution provider would be selected by `JudgeBackend`.
//!
//! Parses `--judge-target=` the same way `auto_detect()` does, but without
//! calling `std::process::exit()` for uncompiled targets, so this example
//! runs safely under `run_all_examples.sh` on any build.
//!
//! Run with:
//!   cargo run -p zer-judge --example judge_backend
//!
//! To see what a specific target resolves to (no model is loaded):
//!   cargo run -p zer-judge --example judge_backend -- --judge-target=cuda

use zer_judge::{JudgeBackend, backend::JudgeTarget};

fn main() {
    // ── 1. Parse --judge-target= without the production exit-on-error path ────
    // JudgeBackend::auto_detect() calls std::process::exit(1) when the target
    // is not compiled in.  For an informational example we use from_name()
    // (returns None, no exit) and fall back to Cpu for unknown values.
    let target_arg = std::env::args()
        .find_map(|a| a.strip_prefix("--judge-target=").map(str::to_owned));

    let selected = target_arg
        .as_deref()
        .and_then(JudgeTarget::from_name)
        .unwrap_or_default(); // Cpu when flag absent or value unrecognised

    println!("--judge-target arg : {}", target_arg.as_deref().unwrap_or("(not set)"));
    println!("Resolved target    : {:?}  ({})", selected, selected.as_str());
    println!();

    // ── 2. All known targets with compile-time availability ───────────────────
    println!("All known judge targets:");
    for t in [
        JudgeTarget::Cpu,
        JudgeTarget::Cuda,
        JudgeTarget::TensorRt,
        JudgeTarget::Rocm,
        JudgeTarget::DirectMl,
        JudgeTarget::OpenVino,
    ] {
        let compiled = match t {
            JudgeTarget::Cpu      => true,
            JudgeTarget::Cuda     => cfg!(feature = "judge_cuda"),
            JudgeTarget::TensorRt => cfg!(feature = "judge_tensorrt"),
            JudgeTarget::Rocm     => cfg!(feature = "judge_rocm"),
            JudgeTarget::DirectMl => cfg!(feature = "judge_directml"),
            JudgeTarget::OpenVino => cfg!(feature = "judge_openvino"),
        };
        let status = if compiled { "compiled-in " } else { "not compiled" };
        let marker = if t == selected { "  ◀ selected" } else { "" };
        println!(
            "  {:12}  {}  (--judge-target={}){}",
            format!("{t:?}"), status, t.as_str(), marker,
        );
    }
    println!();

    // ── 3. Execution-provider chain for the always-safe CPU backend ───────────
    let cpu = JudgeBackend::cpu();
    println!(
        "CPU backend: {} execution provider(s) in the ORT chain",
        cpu.execution_providers().len(),
    );

    println!("\nDone.");
}
