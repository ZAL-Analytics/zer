//! Internal library runner used by `accuracy` and `throughput` via `--compare-libs`.
//!
//! Discovers the script at `<external-benchmarks-dir>/<library>/<mode>/run.<ext>`,
//! optionally runs `setup.sh` once (using a sentinel file `.setup_done`), then
//! executes the script with the standardised `--dataset`, `--ground-truth`,
//! `--out`, `--scenario` arguments via `std::process::Command`.

use std::path::PathBuf;
use std::process::Command;

use super::util::workspace_root;

// ── Core library runner ───────────────────────────────────────────────────────

pub fn run_library(
    root: &PathBuf,
    library: &str,
    mode: &str,
    scenario: Option<&str>,
    datasets: &[&str],
    ground_truth: Option<&str>,
    out: &str,
    max_records: Option<usize>,
    force_setup: bool,
) -> anyhow::Result<()> {
    let lib_dir = root.join(library);
    let mode_dir = lib_dir.join(mode_dir_name(mode));

    // Ensure setup has been run (idempotent via sentinel file).
    // --force-setup ignores the sentinel; useful after switching Python environments.
    let setup_sentinel = lib_dir.join(".setup_done");
    let setup_sh = lib_dir.join("setup.sh");
    if (!setup_sentinel.exists() || force_setup) && setup_sh.exists() {
        println!("running setup  library={library}");
        let status = Command::new("bash")
            .arg(&setup_sh)
            .current_dir(&lib_dir)
            .status()
            .map_err(|e| anyhow::anyhow!("failed to run setup.sh for {library}: {e}"))?;
        if !status.success() {
            anyhow::bail!("{library}/setup.sh failed with {status}");
        }
        std::fs::write(&setup_sentinel, b"")
            .map_err(|e| anyhow::anyhow!("cannot write sentinel: {e}"))?;
    }

    // Resolve script path (Python or R)
    let (interpreter, script) = resolve_script(&mode_dir)?;

    // Build command, pass each dataset with its own --dataset flag so that
    // link-mode scripts (which use action="append") receive both source files.
    let mut cmd = Command::new(interpreter);
    cmd.arg(&script);
    for ds in datasets {
        cmd.arg("--dataset").arg(ds);
    }
    cmd.arg("--out").arg(out);

    if let Some(gt) = ground_truth {
        cmd.arg("--ground-truth").arg(gt);
    }
    if let Some(slug) = scenario {
        cmd.arg("--scenario").arg(slug);
    }
    if mode == "throughput" {
        if let Some(n) = max_records {
            cmd.arg("--max-records").arg(n.to_string());
        }
    }

    println!("running benchmark script  library={library}  mode={mode}  script={script:?}");
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to execute {library} script: {e}"))?;

    if !status.success() {
        anyhow::bail!(
            "{library}/{mode} script exited with {status}\n\
             If Python dependencies are missing, re-run with --force-setup to reinstall them.",
        );
    }

    // Print the resulting summary file path
    let out_path = super::util::resolve_out_dir(out);
    if let Ok(entries) = std::fs::read_dir(&out_path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.contains(library)
                && (name.ends_with("_benchmark.json") || name.ends_with("_summary.csv"))
            {
                println!("result file written  path={}", entry.path().display());
            }
        }
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn mode_dir_name(mode: &str) -> &str {
    match mode.to_lowercase().replace('-', "_").as_str() {
        "dedupe" | "deduplicate" => "dedupe",
        "link_only" => "link_only",
        "link_and_dedupe" | "link_dedupe" => "link_and_dedupe",
        "throughput" => "throughput",
        _ => mode,
    }
}

fn resolve_script(dir: &PathBuf) -> anyhow::Result<(&'static str, PathBuf)> {
    let py = dir.join("run.py");
    let r = dir.join("run.R");

    if py.exists() {
        return Ok(("python3", py));
    }
    if r.exists() {
        return Ok(("Rscript", r));
    }
    anyhow::bail!(
        "no run.py or run.R found in {dir}\n\
         Ensure the library benchmark scripts are present under \
         <external-benchmarks-dir>/<library>/<mode>/.\n\
         Set --external-benchmarks-dir or ZER_EXTERNAL_BENCHMARKS_DIR to point \
         to your scripts directory.",
        dir = dir.display()
    )
}

pub fn resolve_benchmarks_root(external_benchmarks_dir: Option<&str>) -> PathBuf {
    if let Some(p) = external_benchmarks_dir {
        return PathBuf::from(p);
    }
    workspace_root().join("benchmarks")
}
