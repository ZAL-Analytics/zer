//! `zer-bench plot`, delegate to `benchmarks/utils/plot_results.py`.

use std::path::PathBuf;

use clap::Args;

use super::util::workspace_root;

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct PlotArgs {
    /// Directory or file containing `_summary.csv` files to plot.
    #[arg(long)]
    pub input: String,

    /// Output path for the generated plot (e.g. `results.png`).
    /// Defaults to the plot script's own default when omitted.
    #[arg(long)]
    pub output: Option<String>,

    /// Root directory containing external library benchmark scripts.
    /// Used to locate `utils/plot_results.py` relative to this directory.
    /// Falls back to the `ZER_EXTERNAL_BENCHMARKS_DIR` env var, then
    /// `benchmarks/utils/plot_results.py` inside the workspace root.
    #[arg(long, env = "ZER_EXTERNAL_BENCHMARKS_DIR")]
    pub external_benchmarks_dir: Option<String>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run(args: PlotArgs) -> anyhow::Result<()> {
    let script = resolve_plot_script(args.external_benchmarks_dir.as_deref())?;
    let mut cmd = std::process::Command::new("python3");
    cmd.arg(&script).arg("--input").arg(&args.input);
    if let Some(out) = &args.output {
        cmd.arg("--output").arg(out);
    }
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run plot script: {e}"))?;
    if !status.success() {
        anyhow::bail!("plot script exited with {status}");
    }
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn resolve_plot_script(external_benchmarks_dir: Option<&str>) -> anyhow::Result<PathBuf> {
    if let Some(ext_dir) = external_benchmarks_dir {
        let p = PathBuf::from(ext_dir).join("utils/plot_results.py");
        let p = p.canonicalize().unwrap_or(p);
        if p.exists() {
            return Ok(p);
        }
    }
    let p = workspace_root().join("benchmarks/utils/plot_results.py");
    if p.exists() {
        return Ok(p);
    }
    anyhow::bail!(
        "plot_results.py not found; expected at benchmarks/utils/plot_results.py\n\
         Set --external-benchmarks-dir or ZER_EXTERNAL_BENCHMARKS_DIR to point \
         to your benchmarks directory."
    )
}
