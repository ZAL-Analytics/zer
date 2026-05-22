//! `zer-bench compare`, read multiple `_summary.csv` files and print a
//! side-by-side comparison table.
//!
//! Reads all `_summary.csv` files in `--results`, optionally filtered by
//! `--mode` and `--dataset`, and prints a formatted table.  Also writes a
//! combined `comparison_<mode>_<dataset>.csv` for downstream analysis.
//!
//! # Example
//!
//! ```bash
//! zer-bench compare --results bench_results/ --mode dedupe --dataset brp_persons
//! ```
//!
//! Output:
//!
//! ```text
//! library        mode    dataset       precision  recall   f1      elapsed_ms
//! zer            dedupe   brp_persons   0.979      0.982    0.980   3120
//! splink         dedupe   brp_persons   0.976      0.978    0.977   8500
//! ```

use std::path::PathBuf;

use clap::Args;

use super::util::resolve_out_dir;

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct CompareArgs {
    /// Directory containing `_summary.csv` files.
    #[arg(long)]
    pub results: String,

    /// Filter by mode (e.g. `dedupe`, `link-only`).  Empty = all modes.
    #[arg(long, default_value = "")]
    pub mode: String,

    /// Filter by dataset name.  Empty = all datasets.
    #[arg(long, default_value = "")]
    pub dataset: String,

    /// Output directory for the merged comparison CSV (defaults to --results).
    #[arg(long)]
    pub out: Option<String>,
}

// ── Summary row (shared CSV schema) ──────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct SummaryRow {
    library:         String,
    mode:            String,
    dataset:         String,
    run_id:          String,
    timestamp:       String,
    total_records:   String,
    candidate_pairs: String,
    auto_matched:    String,
    borderline:      String,
    auto_rejected:   String,
    elapsed_ms:      String,
    true_pos:        String,
    false_pos:       String,
    false_neg:       String,
    precision:       String,
    recall:          String,
    f1:              String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run(args: CompareArgs) -> anyhow::Result<()> {
    let results_dir = resolve_out_dir(&args.results);
    if !results_dir.exists() {
        anyhow::bail!("results directory not found: {}", results_dir.display());
    }

    let rows = load_summary_rows(&results_dir, &args.mode, &args.dataset)?;

    if rows.is_empty() {
        eprintln!("warning: no matching summary files found  dir={}  mode={}  dataset={}", results_dir.display(), args.mode, args.dataset);
        return Ok(());
    }

    print_comparison_table(&rows);

    let out_dir = resolve_out_dir(
        args.out.as_deref().unwrap_or(&args.results)
    );
    write_combined_csv(&rows, &out_dir, &args.mode, &args.dataset)?;

    Ok(())
}

// ── Loader ────────────────────────────────────────────────────────────────────

fn load_summary_rows(
    dir:     &PathBuf,
    mode:    &str,
    dataset: &str,
) -> anyhow::Result<Vec<SummaryRow>> {
    let mut rows = Vec::new();

    let entries = std::fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", dir.display()))?;

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with("_summary.csv") {
            continue;
        }

        let path = entry.path();
        let mut rdr = csv::Reader::from_path(&path)
            .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.display()))?;

        for result in rdr.deserialize() {
            let row: SummaryRow = result
                .map_err(|e| anyhow::anyhow!("parse error in {}: {e}", path.display()))?;

            if !mode.is_empty() && normalise_mode(&row.mode) != normalise_mode(mode) {
                continue;
            }
            if !dataset.is_empty() && !row.dataset.contains(dataset) {
                continue;
            }
            rows.push(row);
        }
    }

    // Sort: mode → dataset → library for a stable table
    rows.sort_by(|a, b| {
        (&a.mode, &a.dataset, &a.library).cmp(&(&b.mode, &b.dataset, &b.library))
    });

    Ok(rows)
}

// ── Table printer ─────────────────────────────────────────────────────────────

fn print_comparison_table(rows: &[SummaryRow]) {
    // Determine column widths
    let lib_w     = rows.iter().map(|r| r.library.len()).max().unwrap_or(7).max(7);
    let mode_w    = rows.iter().map(|r| r.mode.len()).max().unwrap_or(4).max(4);
    let dataset_w = rows.iter().map(|r| r.dataset.len()).max().unwrap_or(7).max(7);

    println!(
        "{:<lib_w$}  {:<mode_w$}  {:<dataset_w$}  {:>9}  {:>6}  {:>6}  {:>10}",
        "library", "mode", "dataset", "precision", "recall", "f1", "elapsed_ms",
        lib_w = lib_w, mode_w = mode_w, dataset_w = dataset_w
    );
    println!("{}", "-".repeat(lib_w + mode_w + dataset_w + 45));

    for row in rows {
        let prec  = fmt_float3(&row.precision);
        let rec   = fmt_float3(&row.recall);
        let f1    = fmt_float3(&row.f1);
        let ms    = fmt_opt(&row.elapsed_ms);
        println!(
            "{:<lib_w$}  {:<mode_w$}  {:<dataset_w$}  {:>9}  {:>6}  {:>6}  {:>10}",
            row.library, row.mode, row.dataset, prec, rec, f1, ms,
            lib_w = lib_w, mode_w = mode_w, dataset_w = dataset_w
        );
    }
}

fn normalise_mode(m: &str) -> String {
    let s = m.to_lowercase().replace('-', "_");
    match s.as_str() {
        "deduplicate" | "dedupe" => "deduplicate".into(),
        "link_only"              => "link_only".into(),
        "link_and_dedupe"        => "link_and_dedupe".into(),
        "throughput"             => "throughput".into(),
        _                        => s,
    }
}

fn fmt_opt(s: &str) -> String {
    if s.is_empty() { "-".into() } else { s.to_owned() }
}

fn fmt_float3(s: &str) -> String {
    if s.is_empty() {
        return "-".into();
    }
    match s.parse::<f64>() {
        Ok(v) => format!("{:.3}", v),
        Err(_) => s.to_owned(),
    }
}

// ── Combined CSV writer ───────────────────────────────────────────────────────

fn write_combined_csv(
    rows:    &[SummaryRow],
    out_dir: &PathBuf,
    mode:    &str,
    dataset: &str,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", out_dir.display()))?;

    let mode_slug    = if mode.is_empty()    { "all".into() } else { mode.replace('-', "_")    };
    let dataset_slug = if dataset.is_empty() { "all".into() } else { dataset.replace('-', "_") };
    let filename     = format!("comparison_{mode_slug}_{dataset_slug}.csv");
    let path         = out_dir.join(&filename);

    let file = std::fs::File::create(&path)
        .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", path.display()))?;

    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(&[
        "library","mode","dataset","run_id","timestamp",
        "total_records","candidate_pairs","auto_matched","borderline","auto_rejected",
        "elapsed_ms","true_pos","false_pos","false_neg","precision","recall","f1",
    ])?;

    for row in rows {
        wtr.write_record(&[
            &row.library, &row.mode, &row.dataset, &row.run_id, &row.timestamp,
            &row.total_records, &row.candidate_pairs, &row.auto_matched,
            &row.borderline, &row.auto_rejected, &row.elapsed_ms,
            &row.true_pos, &row.false_pos, &row.false_neg,
            &row.precision, &row.recall, &row.f1,
        ])?;
    }
    wtr.flush()?;

    println!("combined CSV written  path={}", path.display());
    Ok(())
}
