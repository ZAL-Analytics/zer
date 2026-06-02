//! `zer-bench library`, run a competitor library benchmark script.
//!
//! Discovers the script at `<external-benchmarks-dir>/<library>/<mode>/run.<ext>`,
//! optionally runs `setup.sh` once (using a sentinel file `.setup_done`), then
//! executes the script with the standardised `--dataset`, `--ground-truth`,
//! `--out` arguments via `std::process::Command`.
//!
//! When `--dataset` is omitted the canonical dataset(s) for the given scenario
//! (or mode, for backward compat) are used automatically.
//!
//! `zer-bench library-all` is a convenience wrapper that iterates all configured
//! library/mode combinations.

use std::path::PathBuf;
use std::process::Command;

use clap::Args;

use super::scenarios::{ALL_SCENARIOS, find_scenario, datasets_for_scenario};
use super::util::{bench_data_root, resolve_out_dir, workspace_root};

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct LibraryArgs {
    /// Library name (e.g. `splink`).  When omitted, all configured libraries
    /// are run (equivalent to `library-all`).
    #[arg(long)]
    pub library: Option<String>,

    /// Benchmark scenario slug (e.g. `brp/dedupe`, `brp_sis/link`).
    /// Use --list-scenarios to see all options.
    /// When provided, datasets and ground-truth are resolved automatically.
    #[arg(long)]
    pub scenario: Option<String>,

    /// List all available scenarios and exit.
    #[arg(long)]
    pub list_scenarios: bool,

    /// Benchmark mode: dedupe, link-only, link-and-dedupe.
    /// Used only when --scenario is not provided.
    #[arg(long, default_value = "dedupe")]
    pub mode: String,

    /// Path to an input CSV dataset.  May be specified multiple times for
    /// multi-source runs.  When omitted, resolved from --scenario or --mode.
    #[arg(long = "dataset")]
    pub datasets: Vec<String>,

    /// Path to the ground-truth labels CSV.
    /// When omitted, resolved from --scenario or --mode automatically.
    #[arg(long)]
    pub ground_truth: Option<String>,

    /// Output directory for the summary CSV.
    #[arg(long, default_value = "bench_results")]
    pub out: String,

    /// Root directory that contains external library benchmark scripts.
    /// Scripts are resolved as `<dir>/<library>/<mode>/run.py` (or `run.R`).
    /// Can also be set via the `ZER_EXTERNAL_BENCHMARKS_DIR` environment
    /// variable.  Defaults to `benchmarks/` inside the workspace root when
    /// running from a repository clone.
    #[arg(long, env = "ZER_EXTERNAL_BENCHMARKS_DIR")]
    pub external_benchmarks_dir: Option<String>,

    /// Maximum number of records to process (throughput mode only).
    #[arg(long)]
    pub max_records: Option<usize>,

    /// Re-run setup.sh even if the `.setup_done` sentinel already exists.
    /// Use this when Python dependencies are missing after switching environments.
    #[arg(long)]
    pub force_setup: bool,
}

// ── All configured libraries ──────────────────────────────────────────────────

const KNOWN_LIBRARIES: &[(&str, &[&str])] = &[
    ("splink", &["dedupe", "link-only", "link-and-dedupe", "throughput"]),
];

// ── Entry points ──────────────────────────────────────────────────────────────

pub fn run(args: LibraryArgs) -> anyhow::Result<()> {
    if args.list_scenarios {
        print_scenarios();
        return Ok(());
    }

    let root = resolve_benchmarks_root(args.external_benchmarks_dir.as_deref());
    let (dataset_strs, auto_gt, effective_mode) =
        resolve_datasets_and_mode(&args.datasets, args.scenario.as_deref(), &args.mode)?;
    let datasets: Vec<&str> = dataset_strs.iter().map(String::as_str).collect();
    let ground_truth = args.ground_truth.as_deref().or(auto_gt.as_deref());

    match args.library.as_deref() {
        Some(library) => {
            run_library(&root, library, &effective_mode, args.scenario.as_deref(), &datasets, ground_truth, &args.out, args.max_records, args.force_setup)
        }
        None => {
            let mut errors: Vec<String> = Vec::new();
            for (library, modes) in KNOWN_LIBRARIES {
                if modes.contains(&effective_mode.as_str()) {
                    println!("running library  library={library}  mode={}", effective_mode.as_str());
                    if let Err(e) = run_library(
                        &root, library, &effective_mode, args.scenario.as_deref(),
                        &datasets, ground_truth, &args.out, args.max_records, args.force_setup,
                    ) {
                        eprintln!("warning: library failed  library={library}  error={e}");
                        errors.push(format!("{library}: {e}"));
                    }
                }
            }
            if !errors.is_empty() {
                anyhow::bail!("some libraries failed:\n{}", errors.join("\n"));
            }
            Ok(())
        }
    }
}

pub fn run_all(args: LibraryArgs) -> anyhow::Result<()> { run(args) }

// ── Core library runner ───────────────────────────────────────────────────────

fn run_library(
    root:         &PathBuf,
    library:      &str,
    mode:         &str,
    scenario:     Option<&str>,
    datasets:     &[&str],
    ground_truth: Option<&str>,
    out:          &str,
    max_records:  Option<usize>,
    force_setup:  bool,
) -> anyhow::Result<()> {
    let lib_dir  = root.join(library);
    let mode_dir = lib_dir.join(mode_dir_name(mode));

    // Ensure setup has been run (idempotent via sentinel file).
    // --force-setup ignores the sentinel; useful after switching Python environments.
    let setup_sentinel = lib_dir.join(".setup_done");
    let setup_sh       = lib_dir.join("setup.sh");
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
    let status = cmd.status()
        .map_err(|e| anyhow::anyhow!("failed to execute {library} script: {e}"))?;

    if !status.success() {
        anyhow::bail!(
            "{library}/{mode} script exited with {status}\n\
             If Python dependencies are missing, re-run with --force-setup to reinstall them.",
        );
    }

    // Print the resulting summary file path
    let out_path = resolve_out_dir(out);
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

// ── Dataset resolution ────────────────────────────────────────────────────────

/// Returns `(dataset_paths, Option<ground_truth_path>, effective_mode_str)`.
///
/// Resolution order:
/// 1. Explicit `--dataset` args (no auto ground-truth, mode passed through).
/// 2. `--scenario` slug: paths and ground-truth from the scenario registry.
/// 3. Fallback: `brp/dedupe` scenario for backward compat with old `--mode` flag.
fn resolve_datasets_and_mode(
    explicit: &[String],
    scenario: Option<&str>,
    mode:     &str,
) -> anyhow::Result<(Vec<String>, Option<String>, String)> {
    if !explicit.is_empty() {
        return Ok((explicit.to_vec(), None, mode.to_owned()));
    }

    let root = bench_data_root();

    if let Some(slug) = scenario {
        let spec = find_scenario(slug).ok_or_else(|| {
            let names: Vec<&str> = ALL_SCENARIOS.iter().map(|s| s.name).collect();
            anyhow::anyhow!("unknown scenario {slug:?}; available: {}", names.join(", "))
        })?;
        let (datasets, _sources, gt) = datasets_for_scenario(spec, &root);
        println!("scenario resolved  scenario={slug}  mode={}  ground_truth={}", spec.mode.as_str(), gt.as_str());
        for d in &datasets { println!("dataset={}", d.as_str()); }
        return Ok((datasets, Some(gt), spec.mode.as_str().to_owned()));
    }

    // Legacy fallback: map --mode to the canonical BRP same-schema scenario.
    let fallback_scenario = match mode_dir_name(mode) {
        "dedupe"          => "brp/dedupe",
        "link_only"       => "brp/link",
        "link_and_dedupe" => "brp/link_and_dedupe",
        _                 => "brp/dedupe",
    };
    if let Some(spec) = find_scenario(fallback_scenario) {
        let (datasets, _sources, gt) = datasets_for_scenario(spec, &root);
        println!("using fallback scenario  scenario={fallback_scenario}  ground_truth={}", gt.as_str());
        for d in &datasets { println!("dataset={}", d.as_str()); }
        Ok((datasets, Some(gt), spec.mode.as_str().to_owned()))
    } else {
        anyhow::bail!(
            "no --dataset given and fallback scenario {fallback_scenario:?} not found; \
             use --scenario to specify a scenario or --dataset to pass files directly"
        )
    }
}

fn print_scenarios() {
    println!("{:<35}  {}", "SCENARIO", "DESCRIPTION");
    println!("{}", "-".repeat(80));
    for s in ALL_SCENARIOS {
        println!("{:<35}  {}", s.name, s.description);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn mode_dir_name(mode: &str) -> &str {
    match mode.to_lowercase().replace('-', "_").as_str() {
        "dedupe" | "deduplicate" => "dedupe",
        "link_only"                        => "link_only",
        "link_and_dedupe" | "link_dedupe"  => "link_and_dedupe",
        "throughput"                       => "throughput",
        _ => mode,
    }
}

fn resolve_script(dir: &PathBuf) -> anyhow::Result<(&'static str, PathBuf)> {
    let py = dir.join("run.py");
    let r  = dir.join("run.R");

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

fn resolve_benchmarks_root(external_benchmarks_dir: Option<&str>) -> PathBuf {
    if let Some(p) = external_benchmarks_dir {
        return PathBuf::from(p);
    }
    workspace_root().join("benchmarks")
}
