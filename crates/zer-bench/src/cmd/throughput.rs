//! `zer-bench throughput`, raw compare/EM/score throughput measurement.
//!
//! Measures compare/EM/score throughput against the canonical BRP benchmark
//! dataset (`data/benchmarks/bench_dedup/source.csv`).  Supply `--dataset`
//! to use a custom CSV instead.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Args;
use sysinfo::{ProcessRefreshKind, RefreshKind, System};

use zer::prelude::*;
use zer_adapters::time::{fmt_unix_secs, unix_secs_now};
use zer_judge::{DebertaJudge, DebertaJudgeConfig, JudgeBackend, MiniLmSpec};

use super::scenarios::{find_scenario, full_size_throughput_scenarios, throughput_scenarios};
use super::util::{log_trt_cache_status, resolve_out_dir};

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

    /// Scenario slug (e.g. `brp/dedupe`).  When provided, the slugified name
    /// (e.g. `brp_dedupe`) is used as the dataset label in output files so it
    /// matches the label written by competitor library scripts.
    /// Use `all` to run all full-size dedupe scenarios back-to-back.
    #[arg(long)]
    pub scenario: Option<String>,

    /// Output directory for the summary CSV (same schema as accuracy/library).
    #[arg(long, default_value = "bench_results")]
    pub out: String,

    /// List all throughput-eligible scenarios and exit.
    #[arg(long)]
    pub list_scenarios: bool,

    /// Comma-separated list of external libraries to benchmark alongside zer.
    /// Each library's throughput script is run after zer, then an inline
    /// comparison table is printed.  Example: `--compare-libs splink,foo`.
    #[arg(long, value_delimiter = ',')]
    pub compare_libs: Vec<String>,

    /// Root directory containing external library benchmark scripts.
    /// Scripts are resolved as `<dir>/<library>/throughput/run.py`.
    /// Falls back to the `ZER_EXTERNAL_BENCHMARKS_DIR` env var, then
    /// `benchmarks/` inside the workspace root.
    #[arg(long, env = "ZER_EXTERNAL_BENCHMARKS_DIR")]
    pub external_benchmarks_dir: Option<String>,

    /// Re-run setup.sh for each library even if the `.setup_done` sentinel exists.
    #[arg(long)]
    pub force_setup: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run(args: ThroughputArgs) -> anyhow::Result<()> {
    if args.list_scenarios {
        print_throughput_scenarios();
        return Ok(());
    }

    super::util::validate_compute_target(&args.target)?;
    if let Some(jt) = &args.judge_target {
        super::util::validate_judge_target(jt)?;
    }

    // Throughput only supports dedupe scenarios.
    if let Some(scenario) = args.scenario.as_deref() {
        if scenario != "all" {
            let mode_part = scenario.rsplit('/').next().unwrap_or(scenario);
            if mode_part != "dedupe" {
                anyhow::bail!(
                    "--type=throughput only supports 'dedupe' scenarios.\n  \
                     Scenario '{}' has mode '{}'.\n  \
                     Link and link_and_dedupe scenarios are not supported for throughput benchmarks.",
                    scenario, mode_part
                );
            }
        }
    }

    let run_start = std::time::SystemTime::now();
    let judge_target = args.judge_target.clone();
    let has_judge = judge_target.is_some();
    let has_libs = !args.compare_libs.is_empty();
    let n_zer = if has_judge { 2 } else { 1 };
    let total = n_zer + args.compare_libs.len();
    let idx = |i: usize| if total > 1 { Some((i, total)) } else { None };

    if args.scenario.as_deref() == Some("all") {
        let base_out = args.out.clone();
        for spec in full_size_throughput_scenarios() {
            let s_out = format!("{}/{}", base_out, spec.name.replace('/', "_"));
            std::fs::create_dir_all(&s_out)?;
            let path = resolve_dataset_path(&args, Some(spec.name));
            run_pass(&args, Some(spec.name), &path, &s_out, None, idx(1))?;
            if has_judge {
                run_pass(
                    &args,
                    Some(spec.name),
                    &path,
                    &s_out,
                    judge_target.as_deref(),
                    idx(2),
                )?;
            }
            if has_libs {
                run_compare_libs(
                    &args,
                    Some(spec.name),
                    &path,
                    &s_out,
                    &args.compare_libs,
                    n_zer + 1,
                    total,
                )?;
            }
            if has_judge || has_libs {
                super::compare::print_comparison_for_dir(&s_out, run_start)?;
            }
        }
        println!("\nDone. All scenario results in: {base_out}/");
        return Ok(());
    }

    std::fs::create_dir_all(&args.out)?;
    let path = resolve_dataset_path(&args, args.scenario.as_deref());
    run_pass(
        &args,
        args.scenario.as_deref(),
        &path,
        &args.out,
        None,
        idx(1),
    )?;
    if has_judge {
        run_pass(
            &args,
            args.scenario.as_deref(),
            &path,
            &args.out,
            judge_target.as_deref(),
            idx(2),
        )?;
    }
    if has_libs {
        run_compare_libs(
            &args,
            args.scenario.as_deref(),
            &path,
            &args.out,
            &args.compare_libs,
            n_zer + 1,
            total,
        )?;
    }
    if has_judge || has_libs {
        super::compare::print_comparison_for_dir(&args.out, run_start)?;
    }
    Ok(())
}

fn resolve_dataset_path(args: &ThroughputArgs, scenario: Option<&str>) -> String {
    let data_root = super::util::bench_data_root();
    args.dataset.clone().unwrap_or_else(|| {
        scenario
            .and_then(|slug| find_scenario(slug))
            .and_then(|spec| spec.sources.first())
            .map(|src| data_root.join(src.path).to_string_lossy().into_owned())
            .unwrap_or_else(|| {
                data_root
                    .join("benchmarks/bench_dedup/source.csv")
                    .to_string_lossy()
                    .into_owned()
            })
    })
}

fn run_pass(
    args: &ThroughputArgs,
    scenario: Option<&str>,
    path: &str,
    out: &str,
    judge_target: Option<&str>,
    run_index: Option<(usize, usize)>,
) -> anyhow::Result<()> {
    let label = scenario
        .map(|s| s.replace('/', "_"))
        .unwrap_or_else(|| "brp".to_string());

    let library_name = if judge_target.is_some() {
        format!("zer+judge_{}", judge_target.unwrap())
    } else {
        "zer".to_owned()
    };
    let header_label = match run_index {
        Some((i, n)) => format!("[{i}/{n}] {library_name}"),
        None => library_name,
    };
    super::util::print_bench_header(&[
        &header_label,
        "throughput",
        scenario.unwrap_or("brp/dedupe"),
        &args.target,
    ]);

    if label.starts_with("kvk") {
        run_kvk(path, &label, args, out, judge_target)?;
    } else {
        run_brp(path, &label, args, out, judge_target)?;
    }
    Ok(())
}

fn run_compare_libs(
    args: &ThroughputArgs,
    scenario: Option<&str>,
    path: &str,
    out: &str,
    compare_libs: &[String],
    start_index: usize,
    total: usize,
) -> anyhow::Result<()> {
    let bench_root =
        super::library::resolve_benchmarks_root(args.external_benchmarks_dir.as_deref());
    let datasets = [path];
    let scenario_disp = scenario.unwrap_or("brp/dedupe");
    let mut errors: Vec<String> = Vec::new();
    for (i, lib) in compare_libs.iter().enumerate() {
        let idx = start_index + i;
        super::util::print_bench_header(&[
            &format!("[{idx}/{total}] {lib}"),
            "throughput",
            scenario_disp,
        ]);
        println!("running library  library={lib}  mode=throughput");
        if let Err(e) = super::library::run_library(
            &bench_root,
            lib,
            "throughput",
            scenario,
            &datasets,
            None,
            out,
            None,
            args.force_setup,
        ) {
            eprintln!("warning: library failed  library={lib}  error={e}");
            errors.push(format!("{lib}: {e}"));
        }
    }
    if !errors.is_empty() {
        anyhow::bail!("some libraries failed:\n{}", errors.join("\n"));
    }
    Ok(())
}

// ── BRP scenario (replaces brp_throughput) ────────────────────────────────────

fn brp_schema() -> Schema {
    SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("straatnaam", FieldKind::Address)
        .build()
        .expect("BRP schema must not be empty")
}

fn kvk_schema() -> Schema {
    SchemaBuilder::new()
        .field("handelsnaam", FieldKind::Name)
        .field("voornamen", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("straatnaam", FieldKind::Address)
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
        records.push(
            Record::new(id)
                .insert("voornamen", text_col(&row, 1))
                .insert("achternaam", text_col(&row, 3))
                .insert("geboortedatum", text_col(&row, 4))
                .insert("straatnaam", text_col(&row, 9)),
        );
    }
    Ok((records, schema))
}

/// Load a CSV by matching field names from the schema to column headers.
/// Used for datasets (like KVK) whose column order differs from BRP.
fn load_by_headers(
    path: &str,
    schema: Schema,
    fields: &[&str],
) -> anyhow::Result<(Vec<Record>, Schema)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let col_idx: std::collections::HashMap<&str, usize> =
        headers.iter().enumerate().map(|(i, h)| (h, i)).collect();

    let id_col = col_idx.get("record_id").copied();
    let field_cols: Vec<(&str, Option<usize>)> = fields
        .iter()
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
            rec = rec.insert(
                name,
                FieldValue::Text(col.and_then(|c| row.get(c)).unwrap_or("").to_string()),
            );
        }
        records.push(rec);
    }
    Ok((records, schema))
}

fn load_kvk(path: &str) -> anyhow::Result<(Vec<Record>, Schema)> {
    load_by_headers(
        path,
        kvk_schema(),
        &[
            "handelsnaam",
            "voornamen",
            "geboortedatum",
            "straatnaam",
            "achternaam",
        ],
    )
}

fn run_brp(
    path: &str,
    label: &str,
    args: &ThroughputArgs,
    out: &str,
    judge_target: Option<&str>,
) -> anyhow::Result<()> {
    println!("loading records  path={path}");
    let (records, schema) = load_brp(path)?;
    let blocker = CompositeBlocker::new()
        .add(ExactFieldKey::new("achternaam"))
        .add(ExactFieldKey::new("geboortedatum"));
    run_scenario(records, schema, blocker, label, args, out, judge_target)?;
    Ok(())
}

fn run_kvk(
    path: &str,
    label: &str,
    args: &ThroughputArgs,
    out: &str,
    judge_target: Option<&str>,
) -> anyhow::Result<()> {
    println!("loading records  path={path}");
    let (records, schema) = load_kvk(path)?;
    let blocker = CompositeBlocker::new()
        .add(ExactFieldKey::new("achternaam"))
        .add(ExactFieldKey::new("geboortedatum"));
    run_scenario(records, schema, blocker, label, args, out, judge_target)?;
    Ok(())
}

fn run_scenario(
    records: Vec<Record>,
    schema: Schema,
    blocker: CompositeBlocker,
    label: &str,
    args: &ThroughputArgs,
    out: &str,
    judge_target: Option<&str>,
) -> anyhow::Result<()> {
    println!("records loaded  count={}", records.len());

    let t = Instant::now();
    let mut index = InvertedIndex::new();
    let mut id_to_idx = HashMap::with_capacity(records.len());
    for (pos, record) in records.iter().enumerate() {
        id_to_idx.insert(record.id, pos);
        blocker.index_record(record, &schema, &mut index);
    }
    let pairs = index.all_pairs(&id_to_idx, 0);
    let block_ms = t.elapsed().as_millis();
    println!(
        "blocking complete  pairs={}  block_ms={block_ms}",
        pairs.len()
    );

    let backend = resolve_backend(&args.target);
    println!("backend={}", backend.name());

    // Build judge outside the pipeline timer, init is not counted.
    let judge = build_judge(
        judge_target,
        args.judge_models_dir.as_deref(),
        &records,
        &schema,
    )?;

    let metrics = run_pipeline_timed(
        &backend,
        &records,
        &pairs,
        &schema,
        args.em_iter,
        block_ms,
        judge.as_ref(),
    );
    print_metrics(
        label,
        &records,
        &pairs,
        &backend,
        &metrics,
        out,
        judge_target,
    );
    Ok(())
}

fn build_judge(
    judge_target: Option<&str>,
    judge_models_dir: Option<&str>,
    records: &[Record],
    schema: &Schema,
) -> anyhow::Result<Option<DebertaJudge>> {
    let Some(jt) = judge_target else {
        return Ok(None);
    };

    if jt == "tensorrt" {
        log_trt_cache_status();
    }

    let record_store: Arc<dyn RecordStore> = Arc::new(VecRecordStore::new());
    for rec in records {
        record_store.insert(rec.clone());
    }

    let judge_backend = JudgeBackend::from_target(jt);
    let models_base = judge_models_dir.map(PathBuf::from).unwrap_or_else(|| {
        judge_backend.resolve_models_dir(&zer_judge::default_models_dir().join("nli-base"))
    });
    let minilm_dir = models_base.join("nli-minilm-onnx");
    let spec = MiniLmSpec::from_dir(&minilm_dir);

    println!("loading judge  target={jt}  path={}", minilm_dir.display());
    let t_load = Instant::now();
    let judge = DebertaJudge::new(
        &spec,
        &judge_backend,
        record_store,
        schema.clone(),
        DebertaJudgeConfig::default(),
    )
    .map_err(|e| anyhow::anyhow!("failed to load judge: {e}"))?;
    println!("judge ready  load_ms={}", t_load.elapsed().as_millis());

    Ok(Some(judge))
}

// ── Shared pipeline runner ────────────────────────────────────────────────────

struct PipelineMetrics {
    block_ms: u128,
    setup_ms: u128,
    compare_ms: u128,
    em_ms: u128,
    score_ms: u128,
    judge_ms: Option<u128>,
    auto_match: usize,
    borderline: usize,
    auto_reject: usize,
    lambda: f32,
    // RSS snapshots in MB after each stage (cross-platform via sysinfo).
    rss_after_compare_mb: Option<f64>,
    rss_after_em_mb: Option<f64>,
    rss_after_score_mb: Option<f64>,
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
    sys.process(pid)
        .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
}

fn run_pipeline_timed(
    backend: &Backend,
    records: &[Record],
    pairs: &[(usize, usize)],
    schema: &Schema,
    em_iter: usize,
    block_ms: u128,
    judge: Option<&DebertaJudge>,
) -> PipelineMetrics {
    let comparator = Comparator::new(schema, backend);
    let scorer = Scorer::new(backend);

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
        let borderlines: Vec<ScoredPair> = scored
            .iter()
            .filter(|s| matches!(s.band, MatchBand::Borderline))
            .cloned()
            .collect();
        let t = Instant::now();
        if !borderlines.is_empty() {
            j.adjudicate(&borderlines)
                .expect("judge adjudication failed");
        }
        Some(t.elapsed().as_millis())
    } else {
        None
    };

    let auto_match = scored
        .iter()
        .filter(|s| matches!(s.band, MatchBand::AutoMatch))
        .count();
    let borderline = scored
        .iter()
        .filter(|s| matches!(s.band, MatchBand::Borderline))
        .count();
    let auto_reject = scored
        .iter()
        .filter(|s| matches!(s.band, MatchBand::AutoReject))
        .count();
    let lambda = params.log_prior_odds.exp() / (1.0 + params.log_prior_odds.exp());

    PipelineMetrics {
        block_ms,
        setup_ms,
        compare_ms,
        em_ms,
        score_ms,
        judge_ms,
        auto_match,
        borderline,
        auto_reject,
        lambda,
        rss_after_compare_mb,
        rss_after_em_mb,
        rss_after_score_mb,
    }
}

fn print_metrics(
    label: &str,
    records: &[Record],
    pairs: &[(usize, usize)],
    backend: &Backend,
    m: &PipelineMetrics,
    out: &str,
    judge_target: Option<&str>,
) {
    let n = pairs.len();
    let compare_pairs_s = if m.compare_ms > 0 {
        n as u64 * 1_000 / m.compare_ms as u64
    } else {
        u64::MAX
    };
    let em_vectors_s = if m.em_ms > 0 {
        n as u64 * 1_000 / m.em_ms as u64
    } else {
        u64::MAX
    };

    println!("preset\t\t{label}");
    println!("backend\t\t{}", backend.name());
    println!("records\t\t{}", records.len());
    println!("candidate_pairs\t{n}");
    println!("block_ms\t{}", m.block_ms);
    println!("setup_ms\t{}", m.setup_ms);
    println!("compare_ms\t{}", m.compare_ms);
    println!("em_ms\t\t{}", m.em_ms);
    println!("score_ms\t{}", m.score_ms);
    if let Some(ms) = m.judge_ms {
        println!("judge_ms\t{ms}");
    }
    println!("compare_pairs_s\t{compare_pairs_s}");
    println!("em_vectors_s\t{em_vectors_s}");
    println!("auto_match\t{}", m.auto_match);
    println!("borderline\t{}", m.borderline);
    println!("auto_reject\t{}", m.auto_reject);
    println!("lambda_est\t{:.4}", m.lambda);
    if let Some(mb) = m.rss_after_compare_mb {
        println!("rss_compare_mb\t{mb:.1}");
    }
    if let Some(mb) = m.rss_after_em_mb {
        println!("rss_em_mb\t{mb:.1}");
    }
    if let Some(mb) = m.rss_after_score_mb {
        println!("rss_score_mb\t{mb:.1}");
    }

    if let Err(e) = write_summary(
        label,
        records.len(),
        pairs.len(),
        backend.name(),
        m,
        out,
        judge_target,
    ) {
        eprintln!("warning: failed to write summary CSV: {e}");
    }
}

fn write_summary(
    dataset: &str,
    n_records: usize,
    n_pairs: usize,
    backend: &str,
    m: &PipelineMetrics,
    out_dir: &str,
    judge_target: Option<&str>,
) -> anyhow::Result<()> {
    let out_path = resolve_out_dir(out_dir);
    std::fs::create_dir_all(&out_path)
        .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", out_path.display()))?;

    // elapsed_ms = full pipeline including blocking, matching splink's pipeline_ms definition
    // (splink's predict() call covers blocking+compare+score as a single DuckDB stage, so
    // their elapsed_ms always includes blocking; zer must match for a fair comparison).
    let elapsed_ms =
        m.block_ms + m.setup_ms + m.compare_ms + m.em_ms + m.score_ms + m.judge_ms.unwrap_or(0);
    let ts = unix_secs_now();
    let run_id = format!("{ts}");
    let timestamp = fmt_unix_secs(ts);
    let library = match judge_target {
        Some(jt) => format!("zer_{backend}+judge_{jt}"),
        None => format!("zer_{backend}"),
    };

    let stem = format!("zer_throughput_{dataset}_{ts}");

    // ── CSV (shared summary schema for `compare`) ─────────────────────────────
    let csv_path = out_path.join(format!("{stem}_summary.csv"));
    let csv_file = std::fs::File::create(&csv_path)
        .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", csv_path.display()))?;
    let mut wtr = csv::Writer::from_writer(csv_file);
    wtr.write_record(&[
        "library",
        "mode",
        "dataset",
        "run_id",
        "timestamp",
        "total_records",
        "candidate_pairs",
        "auto_matched",
        "borderline",
        "auto_rejected",
        "elapsed_ms",
        "true_pos",
        "false_pos",
        "false_neg",
        "precision",
        "recall",
        "f1",
    ])?;
    wtr.write_record(&[
        &library,
        "throughput",
        dataset,
        &run_id,
        &timestamp,
        &n_records.to_string(),
        &n_pairs.to_string(),
        &m.auto_match.to_string(),
        &m.borderline.to_string(),
        &m.auto_reject.to_string(),
        &elapsed_ms.to_string(),
        "",
        "",
        "",
        "",
        "",
        "",
    ])?;
    wtr.flush()?;
    println!("csv written  path={}", csv_path.display());

    // ── JSON (full stage breakdown, human-readable) ───────────────────────────
    let pipeline_pairs_s = if elapsed_ms > 0 {
        n_pairs as u64 * 1_000 / elapsed_ms as u64
    } else {
        0
    };
    let compare_pairs_s = if m.compare_ms > 0 {
        n_pairs as u64 * 1_000 / m.compare_ms as u64
    } else {
        0
    };
    let em_vectors_s = if m.em_ms > 0 {
        n_pairs as u64 * 1_000 / m.em_ms as u64
    } else {
        0
    };

    let round1 = |mb: f64| (mb * 10.0).round() / 10.0;
    let mem_after_compare = m.rss_after_compare_mb.map(round1);
    let mem_after_em = m.rss_after_em_mb.map(round1);
    let mem_after_score = m.rss_after_score_mb.map(round1);

    let json = serde_json::json!({
        "library":          library,
        "mode":             "throughput",
        "dataset":          dataset,
        "run_id":           run_id,
        "timestamp":        timestamp,
        "backend":          backend,
        "total_records":    n_records,
        "candidate_pairs":  n_pairs,
        // Pipeline stage breakdown.  total_ms = compute pipeline only (setup+compare+em+score+judge).
        // block_ms is reported separately; it is CPU-only inverted-index work independent of the backend.
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

// ── Scenario listing ─────────────────────────────────────────────────────────

fn print_throughput_scenarios() {
    println!("{:<35}  {}", "SCENARIO", "DESCRIPTION");
    println!("{}", "-".repeat(80));
    for s in throughput_scenarios() {
        println!("{:<35}  {}", s.name, s.description);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn resolve_backend(target: &str) -> Backend {
    Backend::from_target(target)
}

fn text_col(row: &csv::StringRecord, col: usize) -> FieldValue {
    FieldValue::Text(row.get(col).unwrap_or("").to_string())
}
