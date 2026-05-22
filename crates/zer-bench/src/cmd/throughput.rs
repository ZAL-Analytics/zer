//! `zer-bench throughput`, raw compare/EM/score throughput measurement.
//!
//! Measures compare/EM/score throughput against the canonical BRP benchmark
//! dataset (`data/benchmarks/bench_dedup/source.csv`).  Supply `--dataset`
//! to use a custom CSV instead.
//!
//! ## Profiling
//!
//! Pass `--profile` to print the nsys/ncu command line instead of running the
//! benchmark:
//!
//! ```bash
//! cargo run --release -p zer-bench --features=cuda -- \
//!     throughput --dataset data/benchmarks/bench_dedup/source.csv \
//!                --target cuda --profile
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Args;
use sysinfo::{ProcessRefreshKind, RefreshKind, System};

use zer_lib::prelude::*;
use zer_adapters::time::{fmt_unix_secs, unix_secs_now};
use zer_judge::{DebertaJudge, DebertaJudgeConfig, JudgeBackend, MiniLmSpec};

use super::util::{log_trt_cache_status, resolve_out_dir, workspace_root};

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct ThroughputArgs {
    /// Path to the input CSV file.  Defaults to the canonical BRP benchmark
    /// dataset (`data/benchmarks/bench_dedup/source.csv`).
    #[arg(long)]
    pub dataset: Option<String>,

    /// EM iterations.
    #[arg(long, default_value = "30")]
    pub em_iter: usize,

    /// Compute target: auto, cpu, cuda, avx2, vulkan.
    #[arg(long, default_value = "auto")]
    pub target: String,

    /// ORT execution provider for the neural judge.
    /// When specified, the judge is enabled automatically.
    /// Valid values: cpu, cuda, tensorrt, rocm, directml, openvino.
    #[arg(long)]
    pub judge_target: Option<String>,

    /// Directory containing judge model directories.
    /// Defaults to `models/nli-base/base` for TensorRT/CPU; CUDA and others try
    /// `models/nli-base/fp16_fused` → `models/nli-base/fp16` → `models/nli-base/base`.
    #[arg(long)]
    pub judge_models_dir: Option<String>,

    /// Run device-vs-CPU scaling comparison (like old device_scale).
    #[arg(long)]
    pub scale: bool,

    /// Print the profiler command instead of running the benchmark.
    #[arg(long)]
    pub profile: bool,

    /// Scenario slug (e.g. `brp/dedupe`).  When provided, the slugified name
    /// (e.g. `brp_dedupe`) is used as the dataset label in output files so it
    /// matches the label written by competitor library scripts.
    #[arg(long)]
    pub scenario: Option<String>,

    /// Output directory for the summary CSV (same schema as accuracy/library).
    /// When omitted, only stdout metrics are printed.
    #[arg(long)]
    pub out: Option<String>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run(args: ThroughputArgs) -> anyhow::Result<()> {
    if args.profile {
        print_profile_cmd(&args);
        return Ok(());
    }

    let path = args.dataset.clone()
        .unwrap_or_else(|| workspace_root()
            .join("data/benchmarks/bench_dedup/source.csv")
            .to_string_lossy()
            .into_owned());
    let label = args.scenario.as_deref()
        .map(|s| s.replace('/', "_"))
        .unwrap_or_else(|| "brp".to_string());
    if label.starts_with("kvk") {
        run_kvk(&path, &label, &args)
    } else {
        run_brp(&path, &label, &args)
    }
}

// ── BRP scenario (replaces brp_throughput) ────────────────────────────────────

fn brp_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("straatnaam",    FieldKind::Address)
        .build()
        .expect("BRP schema must not be empty")
}

fn kvk_schema() -> Schema {
    SchemaBuilder::new()
        .field("handelsnaam",   FieldKind::Name)
        .field("voornamen",     FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("straatnaam",    FieldKind::Address)
        .build()
        .expect("KVK schema must not be empty")
}

fn load_brp(path: &str) -> anyhow::Result<(Vec<Record>, Schema)> {
    let schema = brp_schema();
    let mut rdr = csv::Reader::from_path(path)?;
    let mut records = Vec::new();

    for (i, result) in rdr.records().enumerate() {
        let row = result?;
        let id: u64 = row.get(0).and_then(|s| s.parse().ok()).unwrap_or(i as u64);
        records.push(Record::new(id)
            .insert("voornamen",     text_col(&row, 1))
            .insert("achternaam",    text_col(&row, 3))
            .insert("geboortedatum", text_col(&row, 4))
            .insert("straatnaam",    text_col(&row, 9)));
    }
    Ok((records, schema))
}

/// Load a CSV by matching field names from the schema to column headers.
/// Used for datasets (like KVK) whose column order differs from BRP.
fn load_by_headers(path: &str, schema: Schema, fields: &[&str]) -> anyhow::Result<(Vec<Record>, Schema)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let col_idx: std::collections::HashMap<&str, usize> = headers.iter()
        .enumerate()
        .map(|(i, h)| (h, i))
        .collect();

    let id_col = col_idx.get("record_id").copied();
    let field_cols: Vec<(&str, Option<usize>)> = fields.iter()
        .map(|&f| (f, col_idx.get(f).copied()))
        .collect();

    let mut records = Vec::new();
    for (i, result) in rdr.records().enumerate() {
        let row = result?;
        let id: u64 = id_col
            .and_then(|c| row.get(c))
            .and_then(|s| s.parse().ok())
            .unwrap_or(i as u64);
        let mut rec = Record::new(id);
        for &(name, col) in &field_cols {
            rec = rec.insert(name, FieldValue::Text(
                col.and_then(|c| row.get(c)).unwrap_or("").to_string()
            ));
        }
        records.push(rec);
    }
    Ok((records, schema))
}

fn load_kvk(path: &str) -> anyhow::Result<(Vec<Record>, Schema)> {
    load_by_headers(path, kvk_schema(), &[
        "handelsnaam", "voornamen", "geboortedatum", "straatnaam", "achternaam",
    ])
}

fn run_brp(path: &str, label: &str, args: &ThroughputArgs) -> anyhow::Result<()> {
    println!("loading records  path={path}");
    let (records, schema) = load_brp(path)?;
    let blocker = CompositeBlocker::new()
        .add(ExactFieldKey::new("achternaam"))
        .add(ExactFieldKey::new("geboortedatum"));
    run_scenario(records, schema, blocker, label, args)
}

fn run_kvk(path: &str, label: &str, args: &ThroughputArgs) -> anyhow::Result<()> {
    println!("loading records  path={path}");
    let (records, schema) = load_kvk(path)?;
    let blocker = CompositeBlocker::new()
        .add(ExactFieldKey::new("achternaam"))
        .add(ExactFieldKey::new("geboortedatum"));
    run_scenario(records, schema, blocker, label, args)
}

fn run_scenario(records: Vec<Record>, schema: Schema, blocker: CompositeBlocker, label: &str, args: &ThroughputArgs) -> anyhow::Result<()> {
    println!("records loaded  count={}", records.len());

    let t = Instant::now();
    let mut index     = InvertedIndex::new();
    let mut id_to_idx = HashMap::with_capacity(records.len());
    for (pos, record) in records.iter().enumerate() {
        id_to_idx.insert(record.id, pos);
        blocker.index_record(record, &schema, &mut index);
    }
    let pairs    = index.all_pairs(&id_to_idx, 0);
    let block_ms = t.elapsed().as_millis();
    println!("blocking complete  pairs={}  block_ms={block_ms}", pairs.len());

    let backend = resolve_backend(&args.target);
    println!("backend={}", backend.name());

    // Build judge outside the pipeline timer, init is not counted.
    let judge = build_judge(args, &records, &schema)?;

    if args.scale {
        run_scale_comparison(&records, &pairs, &schema, &backend, args.em_iter);
    } else {
        let metrics = run_pipeline_timed(&backend, &records, &pairs, &schema, args.em_iter, block_ms, judge.as_ref());
        print_metrics(label, &records, &pairs, &backend, &metrics, args.out.as_deref(), args.judge_target.as_deref());
    }
    Ok(())
}

fn build_judge(args: &ThroughputArgs, records: &[Record], schema: &Schema) -> anyhow::Result<Option<DebertaJudge>> {
    let Some(ref jt) = args.judge_target else { return Ok(None) };

    if jt == "tensorrt" {
        log_trt_cache_status();
    }

    let record_store: Arc<dyn RecordStore> = Arc::new(VecRecordStore::new());
    for rec in records {
        record_store.insert(rec.clone());
    }

    let judge_backend = JudgeBackend::from_target(jt);
    let models_base = args.judge_models_dir.as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| judge_backend.resolve_models_dir(std::path::Path::new("models/nli-base")));
    let minilm_dir = models_base.join("nli-minilm-onnx");
    let spec = MiniLmSpec::from_dir(&minilm_dir);

    println!("loading judge  target={}  path={}", jt.as_str(), minilm_dir.display());
    let t_load = Instant::now();
    let judge = DebertaJudge::new(&spec, &judge_backend, record_store, schema.clone(), DebertaJudgeConfig::default())
        .map_err(|e| anyhow::anyhow!("failed to load judge: {e}"))?;
    println!("judge ready  load_ms={}", t_load.elapsed().as_millis());

    Ok(Some(judge))
}

// ── Shared pipeline runner ────────────────────────────────────────────────────

struct PipelineMetrics {
    block_ms:   u128,
    setup_ms:   u128,
    compare_ms: u128,
    em_ms:      u128,
    score_ms:   u128,
    judge_ms:   Option<u128>,
    auto_match:  usize,
    borderline:  usize,
    auto_reject: usize,
    lambda:      f32,
    // RSS snapshots in MB after each stage (cross-platform via sysinfo).
    rss_after_compare_mb: Option<f64>,
    rss_after_em_mb:      Option<f64>,
    rss_after_score_mb:   Option<f64>,
}

/// Read current RSS (resident set size) in MB for this process using sysinfo.
/// Works on Linux, macOS, and Windows; returns None if the OS does not report it.
fn read_rss_mb() -> Option<f64> {
    let pid = sysinfo::get_current_pid().ok()?;
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::new().with_memory()),
    );
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::new().with_memory(),
    );
    sys.process(pid).map(|p| p.memory() as f64 / (1024.0 * 1024.0))
}

fn run_pipeline_timed(
    backend:  &Backend,
    records:  &[Record],
    pairs:    &[(usize, usize)],
    schema:   &Schema,
    em_iter:  usize,
    block_ms: u128,
    judge:    Option<&DebertaJudge>,
) -> PipelineMetrics {
    zer_prof::init();
    let comparator = Comparator::new(schema, backend);
    let scorer     = Scorer::new(backend);

    let t = Instant::now();
    let pool = RecordPool::from_records(records, schema);
    let setup_ms = t.elapsed().as_millis();

    let t = Instant::now();
    let vectors = comparator.compare_batch_from_pool(&pool, pairs, schema);
    let compare_ms = t.elapsed().as_millis();
    let rss_after_compare_mb = read_rss_mb();

    let t = Instant::now();
    let params = scorer
        .estimate_params(&vectors, None, em_iter)
        .expect("EM diverged, ensure candidate pairs include both matches and non-matches");
    let em_ms = t.elapsed().as_millis();
    let rss_after_em_mb = read_rss_mb();

    let t = Instant::now();
    let scored = scorer.score_batch(&vectors, &params);
    let score_ms = t.elapsed().as_millis();
    let rss_after_score_mb = read_rss_mb();

    let judge_ms = if let Some(j) = judge {
        let borderlines: Vec<ScoredPair> = scored.iter()
            .filter(|s| matches!(s.band, MatchBand::Borderline))
            .cloned()
            .collect();
        let t = Instant::now();
        if !borderlines.is_empty() {
            j.adjudicate(&borderlines).expect("judge adjudication failed");
        }
        Some(t.elapsed().as_millis())
    } else {
        None
    };

    let auto_match  = scored.iter().filter(|s| matches!(s.band, MatchBand::AutoMatch)).count();
    let borderline  = scored.iter().filter(|s| matches!(s.band, MatchBand::Borderline)).count();
    let auto_reject = scored.iter().filter(|s| matches!(s.band, MatchBand::AutoReject)).count();
    let lambda = params.log_prior_odds.exp() / (1.0 + params.log_prior_odds.exp());

    PipelineMetrics {
        block_ms, setup_ms, compare_ms, em_ms, score_ms, judge_ms,
        auto_match, borderline, auto_reject, lambda,
        rss_after_compare_mb, rss_after_em_mb, rss_after_score_mb,
    }
}

fn print_metrics(
    label:        &str,
    records:      &[Record],
    pairs:        &[(usize, usize)],
    backend:      &Backend,
    m:            &PipelineMetrics,
    out:          Option<&str>,
    judge_target: Option<&str>,
) {
    let n = pairs.len();
    let compare_pairs_s = if m.compare_ms > 0 { n as u64 * 1_000 / m.compare_ms as u64 } else { u64::MAX };
    let em_vectors_s    = if m.em_ms > 0      { n as u64 * 1_000 / m.em_ms      as u64 } else { u64::MAX };

    println!("preset\t\t{label}");
    println!("backend\t\t{}", backend.name());
    println!("records\t\t{}", records.len());
    println!("candidate_pairs\t{n}");
    println!("block_ms\t{}", m.block_ms);
    println!("setup_ms\t{}", m.setup_ms);
    println!("compare_ms\t{}", m.compare_ms);
    println!("em_ms\t\t{}", m.em_ms);
    println!("score_ms\t{}", m.score_ms);
    if let Some(ms) = m.judge_ms { println!("judge_ms\t{ms}"); }
    println!("compare_pairs_s\t{compare_pairs_s}");
    println!("em_vectors_s\t{em_vectors_s}");
    println!("auto_match\t{}", m.auto_match);
    println!("borderline\t{}", m.borderline);
    println!("auto_reject\t{}", m.auto_reject);
    println!("lambda_est\t{:.4}", m.lambda);
    if let Some(mb) = m.rss_after_compare_mb { println!("rss_compare_mb\t{mb:.1}"); }
    if let Some(mb) = m.rss_after_em_mb      { println!("rss_em_mb\t{mb:.1}"); }
    if let Some(mb) = m.rss_after_score_mb   { println!("rss_score_mb\t{mb:.1}"); }

    if let Some(dir) = out {
        if let Err(e) = write_summary(label, records.len(), pairs.len(), backend.name(), m, dir, judge_target) {
            eprintln!("warning: failed to write summary CSV: {e}");
        }
    }
}

fn write_summary(
    dataset:      &str,
    n_records:    usize,
    n_pairs:      usize,
    backend:      &str,
    m:            &PipelineMetrics,
    out_dir:      &str,
    judge_target: Option<&str>,
) -> anyhow::Result<()> {
    let out_path = resolve_out_dir(out_dir);
    std::fs::create_dir_all(&out_path)
        .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", out_path.display()))?;

    let elapsed_ms = m.block_ms + m.setup_ms + m.compare_ms + m.em_ms + m.score_ms + m.judge_ms.unwrap_or(0);
    let ts        = unix_secs_now();
    let run_id    = format!("{ts}");
    let timestamp = fmt_unix_secs(ts);
    let library = match judge_target {
        Some(jt) => format!("zer_{backend}+judge_{jt}"),
        None     => format!("zer_{backend}"),
    };

    let stem = format!("zer_throughput_{dataset}_{ts}");

    // ── CSV (shared summary schema for `compare`) ─────────────────────────────
    let csv_path = out_path.join(format!("{stem}_summary.csv"));
    let csv_file = std::fs::File::create(&csv_path)
        .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", csv_path.display()))?;
    let mut wtr = csv::Writer::from_writer(csv_file);
    wtr.write_record(&[
        "library","mode","dataset","run_id","timestamp",
        "total_records","candidate_pairs","auto_matched","borderline","auto_rejected",
        "elapsed_ms","true_pos","false_pos","false_neg","precision","recall","f1",
    ])?;
    wtr.write_record(&[
        &library, "throughput", dataset, &run_id, &timestamp,
        &n_records.to_string(), &n_pairs.to_string(),
        &m.auto_match.to_string(), &m.borderline.to_string(), &m.auto_reject.to_string(),
        &elapsed_ms.to_string(), "", "", "", "", "", "",
    ])?;
    wtr.flush()?;
    println!("csv written  path={}", csv_path.display());

    // ── JSON (full stage breakdown, human-readable) ───────────────────────────
    let pipeline_pairs_s = if elapsed_ms > 0   { n_pairs as u64 * 1_000 / elapsed_ms   as u64 } else { 0 };
    let compare_pairs_s  = if m.compare_ms > 0 { n_pairs as u64 * 1_000 / m.compare_ms as u64 } else { 0 };
    let em_vectors_s     = if m.em_ms > 0      { n_pairs as u64 * 1_000 / m.em_ms      as u64 } else { 0 };

    let round1 = |mb: f64| (mb * 10.0).round() / 10.0;
    let mem_after_compare = m.rss_after_compare_mb.map(round1);
    let mem_after_em      = m.rss_after_em_mb.map(round1);
    let mem_after_score   = m.rss_after_score_mb.map(round1);

    let json = serde_json::json!({
        "library":          library,
        "mode":             "throughput",
        "dataset":          dataset,
        "run_id":           run_id,
        "timestamp":        timestamp,
        "backend":          backend,
        "total_records":    n_records,
        "candidate_pairs":  n_pairs,
        // Common schema: block/compare/em/score/judge pipeline stages (non-init path only).
        "pipeline": {
            "block_ms":   m.block_ms,
            "setup_ms":   m.setup_ms,
            "compare_ms": m.compare_ms,
            "em_ms":      m.em_ms,
            "score_ms":   m.score_ms,
            "judge_ms":   m.judge_ms,
            "total_ms":   elapsed_ms,
        },
        "memory_mb": {},
        "throughput": {
            "pairs_per_s": pipeline_pairs_s,
        },
        "match_bands": {
            "auto_matched":  m.auto_match,
            "borderline":    m.borderline,
            "auto_rejected": m.auto_reject,
        },
        "lambda_est": (m.lambda * 10_000.0).round() / 10_000.0,
        // Library-specific raw stage data preserved for detailed analysis.
        "raw": {
            "stages": {
                "setup_ms":   m.setup_ms,
                "compare_ms": m.compare_ms,
                "em_ms":      m.em_ms,
                "score_ms":   m.score_ms,
                "total_ms":   elapsed_ms,
            },
            "throughput": {
                "compare_pairs_per_s": compare_pairs_s,
                "em_vectors_per_s":    em_vectors_s,
            },
            "memory_mb": {
                "after_compare": mem_after_compare,
                "after_em":      mem_after_em,
                "after_score":   mem_after_score,
            },
        },
    });

    let json_path = out_path.join(format!("{stem}_benchmark.json"));
    std::fs::write(&json_path, serde_json::to_string_pretty(&json)?)
        .map_err(|e| anyhow::anyhow!("cannot write {}: {e}", json_path.display()))?;
    println!("json written  path={}", json_path.display());

    Ok(())
}


// ── Device-vs-CPU scaling (replaces device_scale) ────────────────────────────

fn run_scale_comparison(
    records:  &[Record],
    pairs:    &[(usize, usize)],
    schema:   &Schema,
    device:   &Backend,
    em_iter:  usize,
) {
    zer_prof::init();
    let d = run_pipeline_timed(device,          records, pairs, schema, em_iter, 0, None);
    let c = run_pipeline_timed(&Backend::cpu(), records, pairs, schema, em_iter, 0, None);

    let speedup = |dev_ms: u128, cpu_ms: u128| -> String {
        if dev_ms == 0 { return "∞".into(); }
        format!("{:.2}×", cpu_ms as f64 / dev_ms as f64)
    };

    let d_total = d.compare_ms + d.em_ms + d.score_ms;
    let c_total = c.compare_ms + c.em_ms + c.score_ms;

    println!("records\t\t{}", records.len());
    println!("candidate_pairs\t{}", pairs.len());
    println!();
    println!("{:<16} {:>10} {:>10} {:>10}", "stage", device.name(), "cpu", "speedup");
    println!("{}", "-".repeat(50));
    println!("{:<16} {:>9}ms {:>9}ms {:>10}", "compare", d.compare_ms, c.compare_ms, speedup(d.compare_ms, c.compare_ms));
    println!("{:<16} {:>9}ms {:>9}ms {:>10}", "em",      d.em_ms,      c.em_ms,      speedup(d.em_ms,      c.em_ms));
    println!("{:<16} {:>9}ms {:>9}ms {:>10}", "score",   d.score_ms,   c.score_ms,   speedup(d.score_ms,   c.score_ms));
    println!("{}", "-".repeat(50));
    println!("{:<16} {:>9}ms {:>9}ms {:>10}", "total",   d_total,      c_total,      speedup(d_total, c_total));
}

// ── Profiling helper ──────────────────────────────────────────────────────────

fn print_profile_cmd(args: &ThroughputArgs) {
    let binary  = std::env::current_exe()
        .ok()
        .and_then(|p| p.to_str().map(str::to_owned))
        .unwrap_or_else(|| "zer-bench".into());
    let dataset = args.dataset.as_deref().unwrap_or("data/benchmarks/bench_dedup/source.csv");
    let target  = &args.target;

    if target.contains("cuda") {
        println!(
            "nsys profile --trace=cuda,nvtx --output=report_%q{{SLURM_JOB_ID}} \\\n  \
             {binary} throughput --dataset {dataset} --target {target}"
        );
        println!(
            "\n# For kernel-level metrics:\nncu --set full -o ncu_report \\\n  \
             {binary} throughput --dataset {dataset} --target {target}"
        );
    } else {
        println!(
            "perf record -g -- {binary} throughput --dataset {dataset} --target {target}"
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn resolve_backend(target: &str) -> Backend {
    Backend::from_target(target)
}

fn text_col(row: &csv::StringRecord, col: usize) -> FieldValue {
    FieldValue::Text(row.get(col).unwrap_or("").to_string())
}

