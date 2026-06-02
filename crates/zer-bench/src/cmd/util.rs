use std::path::PathBuf;

/// Error if `target` names a compute backend that was not compiled in.
///
/// "auto" and "cpu" are always accepted.  Any other value requires the
/// matching feature flag (e.g. `--features=cuda`); without it the backend
/// would silently fall back to CPU, producing a misleading benchmark label.
pub fn validate_compute_target(target: &str) -> anyhow::Result<()> {
    match target {
        "auto" | "cpu" => {}
        "cuda" => {
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!(
                "--target cuda requires compiling with --features=cuda\n  \
                 Rebuild: cargo build --release -p zer-bench --features=cuda"
            );
        }
        "avx2" => {
            #[cfg(not(feature = "avx2"))]
            anyhow::bail!(
                "--target avx2 requires compiling with --features=avx2\n  \
                 Rebuild: cargo build --release -p zer-bench --features=avx2"
            );
        }
        "vulkan" => {
            #[cfg(not(feature = "vulkan"))]
            anyhow::bail!(
                "--target vulkan requires compiling with --features=vulkan\n  \
                 Rebuild: cargo build --release -p zer-bench --features=vulkan"
            );
        }
        other => anyhow::bail!(
            "unknown compute target: {other:?}; valid: auto, cpu, cuda, avx2, vulkan"
        ),
    }
    Ok(())
}

/// Error if `target` names a judge execution provider that was not compiled in.
///
/// "cpu" is always accepted.  Any GPU/accelerator provider requires its
/// matching feature flag; without it ORT would silently fall back to CPU.
pub fn validate_judge_target(target: &str) -> anyhow::Result<()> {
    match target {
        "cpu" => {}
        "cuda" => {
            #[cfg(not(feature = "judge_cuda"))]
            anyhow::bail!(
                "--judge-target cuda requires compiling with --features=judge_cuda\n  \
                 Rebuild: cargo build --release -p zer-bench --features=judge_cuda"
            );
        }
        "tensorrt" => {
            #[cfg(not(feature = "judge_tensorrt"))]
            anyhow::bail!(
                "--judge-target tensorrt requires compiling with --features=judge_tensorrt\n  \
                 Rebuild: cargo build --release -p zer-bench --features=judge_tensorrt"
            );
        }
        "rocm" => {
            #[cfg(not(feature = "judge_rocm"))]
            anyhow::bail!(
                "--judge-target rocm requires compiling with --features=judge_rocm\n  \
                 Rebuild: cargo build --release -p zer-bench --features=judge_rocm"
            );
        }
        "directml" => {
            #[cfg(not(feature = "judge_directml"))]
            anyhow::bail!(
                "--judge-target directml requires compiling with --features=judge_directml\n  \
                 Rebuild: cargo build --release -p zer-bench --features=judge_directml"
            );
        }
        "openvino" => {
            #[cfg(not(feature = "judge_openvino"))]
            anyhow::bail!(
                "--judge-target openvino requires compiling with --features=judge_openvino\n  \
                 Rebuild: cargo build --release -p zer-bench --features=judge_openvino"
            );
        }
        other => anyhow::bail!(
            "unknown judge target: {other:?}; valid: cpu, cuda, tensorrt, rocm, directml, openvino"
        ),
    }
    Ok(())
}

/// Print a visual section header that separates benchmark runs in the output.
///
/// `parts` are joined with ` │ ` and surrounded by a full-width `═` bar.
///
/// ```text
/// ════════════════════════════════════════════════════
///   [1/2] zer  │  accuracy  │  brp/dedupe  │  cpu
/// ════════════════════════════════════════════════════
/// ```
pub fn print_bench_header(parts: &[&str]) {
    let content = parts.join("  │  ");
    let width = content.chars().count() + 4; // 2 spaces padding each side
    let bar = "═".repeat(width.max(52));
    println!("\n{bar}");
    println!("  {content}");
    println!("{bar}");
}

/// Log whether TensorRT engine cache is warm (engines present) or cold (first-run compile).
pub fn log_trt_cache_status() {
    let cache_dir = format!(
        "{}/.cache/zer-judge/trt-engines",
        std::env::var("HOME").unwrap_or_else(|_| ".".to_owned())
    );
    let engine_count = std::fs::read_dir(&cache_dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().map_or(false, |x| x == "engine"))
                .count()
        })
        .unwrap_or(0);
    if engine_count > 0 {
        println!("TRT warm: cached engines found  engine_count={engine_count}  cache_dir={cache_dir}");
    } else {
        println!("TRT cold: no cached engines, TRT will compile now (takes 2-5 min, cached after)  cache_dir={cache_dir}");
    }
}

/// Walk up from the current directory to find the workspace root (the
/// `Cargo.toml` that contains `[workspace]`).  Falls back to `.` if not found.
pub fn workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_default();
    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(content) = std::fs::read_to_string(&candidate) {
                if content.contains("[workspace]") {
                    return dir;
                }
            }
        }
        if !dir.pop() {
            return PathBuf::from(".");
        }
    }
}

/// Returns the root directory used to resolve benchmark dataset paths.
///
/// Resolution order:
/// 1. `ZER_DATASET_DIR` environment variable, set this to the local directory
///    where the HuggingFace benchmark dataset was downloaded (the repo root maps
///    directly to `benchmarks/...`).
/// 2. `<workspace_root>/data`, used automatically when running from inside the repo.
/// 3. `./data` (current directory fallback).
pub fn bench_data_root() -> PathBuf {
    if let Ok(dir) = std::env::var("ZER_DATASET_DIR") {
        return PathBuf::from(dir);
    }
    workspace_root().join("data")
}

/// Resolve an `--out` / `--results` path relative to the workspace root when
/// the supplied path is relative.  Absolute paths are returned unchanged.
pub fn resolve_out_dir(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        workspace_root().join(p)
    }
}
