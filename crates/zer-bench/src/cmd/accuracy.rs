//! `zer-bench accuracy`, run the zer pipeline against a labeled dataset and
//! write a shared CSV summary.
//!
//! # Usage (preset)
//!
//! ```bash
//! # List available presets
//! zer-bench accuracy --list-presets
//!
//! # Run a named preset (datasets, sources, mode and ground truth are wired up automatically)
//! zer-bench accuracy --preset brp-dedupe-small
//!
//! # With neural judge on CUDA
//! zer-bench accuracy --preset brp-dedupe-small --judge --judge-target cuda
//! ```
//!
//! # Usage (manual)
//!
//! ```bash
//! # Deduplicate mode
//! zer-bench accuracy \
//!     --dataset data/benchmarks/brp_small/brp_persons.csv --source brp \
//!     --mode deduplicate \
//!     --ground-truth data/benchmarks/brp_small/ground_truth_pairs.csv
//!
//! # Link-only mode (two sources)
//! zer-bench accuracy \
//!     --dataset data/benchmarks/brp_small/brp_persons.csv  --source brp \
//!     --dataset data/benchmarks/hks/hks_records.csv        --source hks \
//!     --mode link-only
//! ```
//!
//! # Ground-truth format
//!
//! `record_id_a,record_id_b,is_match[,match_type]` (CSV, header required).
//! IDs may be numeric (BSN) or string (hex, prefixed).  String IDs are mapped
//! to internal u64 keys consistently between the record file and the ground
//! truth file so precision/recall is always correct.
//!
//! # Timing scope
//!
//! `elapsed_ms` covers **CSV loading → pipeline run → result extraction**.
//! Pipeline construction (ORT model loading, TRT engine build, CUDA warm-up)
//! happens *before* the timer starts, mirroring how Python libraries exclude
//! their import and model-loading time.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Args;
use tempfile::TempDir;

use zer::prelude::*;
use zer_core::{field_mapping::FieldMapping, record::derive_record_id};

use super::scenarios::{
    datasets_for_scenario, find_scenario, find_scenario_by_preset, full_size_scenarios,
    ALL_SCENARIOS,
};
use super::strategies;
use super::util::{log_trt_cache_status, resolve_out_dir};
use zer_adapters::{
    band_to_match, AccuracyMetrics, BenchBatchSummary, BenchResultWriter, PairRecord,
};
use zer_judge::{DebertaJudge, DebertaJudgeConfig, JudgeBackend, MiniLmSpec};
use zer_pipeline::{
    config::{LinkMode, PipelineConfig},
    pipeline::Pipeline,
    PipelineEvent,
};

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct AccuracyArgs {
    /// Run a named benchmark scenario (datasets, sources, mode, ground truth,
    /// and field mappings are configured automatically).
    /// Use --list-scenarios to see all options.
    #[arg(long)]
    pub scenario: Option<String>,

    /// List all available scenarios and exit.
    #[arg(long)]
    pub list_scenarios: bool,

    /// Run a benchmark scenario by tag (e.g. `dedupe`, `micro-dedupe`).
    /// Equivalent to --scenario with a tag search across ALL_SCENARIOS.
    /// Use --list-scenarios to see all options.
    #[arg(long)]
    pub preset: Option<String>,

    /// List all available scenarios and exit (alias for --list-scenarios).
    #[arg(long)]
    pub list_presets: bool,

    /// Input CSV dataset path.  May be specified multiple times for multi-source runs.
    /// Ignored when --preset is used.
    #[arg(long = "dataset")]
    pub datasets: Vec<String>,

    /// Source label for each dataset (must match --dataset order).
    /// Ignored when --preset is used.
    #[arg(long = "source")]
    pub sources: Vec<String>,

    /// Linking mode: deduplicate, link-only, link-and-dedupe.
    /// Ignored when --preset is used.
    #[arg(long, default_value = "deduplicate")]
    pub mode: String,

    /// Optional ground-truth labels CSV (`record_id_a,record_id_b,is_match`).
    /// Ignored when --preset is used.
    #[arg(long)]
    pub ground_truth: Option<String>,

    /// Output directory for the pairs NDJSON and summary CSV.
    #[arg(long, default_value = "bench_results")]
    pub out: String,

    /// Name of the dataset (used in the summary CSV `dataset` column).
    #[arg(long)]
    pub dataset_name: Option<String>,

    /// Maximum records to load from each source (0 = all).
    #[arg(long, default_value = "0")]
    pub max_records: usize,

    /// Compute backend for the zer pipeline (compare + EM + score).
    /// Valid values: auto, cpu, cuda, avx2, vulkan.
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

    /// Comma-separated list of external libraries to benchmark alongside zer.
    /// Each library's accuracy script is run after zer, then an inline
    /// comparison table is printed.  Example: `--compare-libs splink,foo`.
    #[arg(long, value_delimiter = ',')]
    pub compare_libs: Vec<String>,

    /// Root directory containing external library benchmark scripts.
    /// Scripts are resolved as `<dir>/<library>/<mode>/run.py`.
    /// Falls back to the `ZER_EXTERNAL_BENCHMARKS_DIR` env var, then
    /// `benchmarks/` inside the workspace root.
    #[arg(long, env = "ZER_EXTERNAL_BENCHMARKS_DIR")]
    pub external_benchmarks_dir: Option<String>,

    /// Re-run setup.sh for each library even if the `.setup_done` sentinel exists.
    #[arg(long)]
    pub force_setup: bool,
}

// ── Resolved run parameters (from preset or manual args) ──────────────────────

struct RunParams {
    datasets: Vec<String>,
    sources: Vec<Option<String>>,
    mode_str: String,
    ground_truth: Option<String>,
    dataset_name: String,
    field_mappings: Vec<FieldMapping>,
}

impl RunParams {
    fn from_args_with(args: &AccuracyArgs, scenario: Option<&str>) -> anyhow::Result<Self> {
        if let Some(scenario_name) = scenario {
            let spec = find_scenario(scenario_name).ok_or_else(|| {
                let names: Vec<&str> = ALL_SCENARIOS.iter().map(|s| s.name).collect();
                anyhow::anyhow!(
                    "unknown scenario {scenario_name:?}; available: {}",
                    names.join(", ")
                )
            })?;
            let root = super::util::bench_data_root();
            let (datasets, sources, gt) = datasets_for_scenario(spec, &root);
            Ok(Self {
                datasets,
                sources: sources.into_iter().map(Some).collect(),
                mode_str: spec.mode.as_str().to_owned(),
                ground_truth: Some(gt),
                dataset_name: args
                    .dataset_name
                    .clone()
                    .unwrap_or_else(|| spec.dataset_name.to_owned()),
                field_mappings: spec.to_field_mappings(),
            })
        } else if let Some(tag) = &args.preset {
            let spec = find_scenario_by_preset(tag.as_str()).ok_or_else(|| {
                anyhow::anyhow!(
                    "no scenario found for tag {tag:?}; use --list-scenarios to see options"
                )
            })?;
            let root = super::util::bench_data_root();
            let (datasets, sources, gt) = datasets_for_scenario(spec, &root);
            Ok(Self {
                datasets,
                sources: sources.into_iter().map(Some).collect(),
                mode_str: spec.mode.as_str().to_owned(),
                ground_truth: Some(gt),
                dataset_name: args
                    .dataset_name
                    .clone()
                    .unwrap_or_else(|| spec.dataset_name.to_owned()),
                field_mappings: spec.to_field_mappings(),
            })
        } else {
            if args.datasets.is_empty() {
                anyhow::bail!("either --scenario, --preset, or at least one --dataset is required");
            }
            let datasets: Vec<String> = args
                .datasets
                .iter()
                .map(|p| resolve_out_dir(p).to_string_lossy().into_owned())
                .collect();
            let sources: Vec<Option<String>> = (0..datasets.len())
                .map(|i| args.sources.get(i).cloned())
                .collect();
            let ground_truth = args
                .ground_truth
                .as_deref()
                .map(|p| resolve_out_dir(p).to_string_lossy().into_owned());
            let dataset_name = args
                .dataset_name
                .clone()
                .unwrap_or_else(|| infer_dataset_name(&datasets[0]));
            Ok(Self {
                datasets,
                sources,
                mode_str: args.mode.clone(),
                ground_truth,
                dataset_name,
                field_mappings: Vec::new(),
            })
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run(args: AccuracyArgs) -> anyhow::Result<()> {
    if args.list_scenarios {
        println!("{:<35}  {}", "SCENARIO", "DESCRIPTION");
        println!("{}", "-".repeat(80));
        for s in ALL_SCENARIOS {
            println!("{:<35}  {}", s.name, s.description);
        }
        return Ok(());
    }

    if args.list_presets {
        println!("{:<35}  {:<25}  {}", "SCENARIO", "TAGS", "DESCRIPTION");
        println!("{}", "-".repeat(100));
        for s in ALL_SCENARIOS {
            println!(
                "{:<35}  {:<25}  {}",
                s.name,
                s.tags.join(", "),
                s.description
            );
        }
        return Ok(());
    }

    super::util::validate_compute_target(&args.target)?;
    if let Some(jt) = &args.judge_target {
        super::util::validate_judge_target(jt)?;
    }

    let scenario_val = args.scenario.as_deref().unwrap_or("").to_owned();
    let judge_target = args.judge_target.clone();

    // ── --scenario=all: iterate over every full-size scenario ─────────────────
    if scenario_val == "all" {
        let base_out = args.out.clone();
        for spec in full_size_scenarios() {
            let s_out = format!("{}/{}", base_out, spec.name.replace('/', "_"));
            std::fs::create_dir_all(&s_out)?;
            let run_start = std::time::SystemTime::now();
            if judge_target.is_some() {
                run_pass(
                    &args,
                    Some(spec.name),
                    &s_out,
                    None,
                    &args.compare_libs,
                    run_start,
                )
                .await?;
                run_pass(
                    &args,
                    Some(spec.name),
                    &s_out,
                    judge_target.as_deref(),
                    &[],
                    run_start,
                )
                .await?;
                super::compare::print_comparison_for_dir(&s_out, run_start)?;
            } else {
                run_pass(
                    &args,
                    Some(spec.name),
                    &s_out,
                    None,
                    &args.compare_libs,
                    run_start,
                )
                .await?;
            }
        }
        println!("\nDone. All scenario results in: {base_out}/");
        return Ok(());
    }

    // ── Judge dual-pass: run without judge first, then with judge, then compare ─
    if judge_target.is_some() {
        std::fs::create_dir_all(&args.out)?;
        let run_start = std::time::SystemTime::now();
        run_pass(
            &args,
            args.scenario.as_deref(),
            &args.out,
            None,
            &args.compare_libs,
            run_start,
        )
        .await?;
        run_pass(
            &args,
            args.scenario.as_deref(),
            &args.out,
            judge_target.as_deref(),
            &[],
            run_start,
        )
        .await?;
        super::compare::print_comparison_for_dir(&args.out, run_start)?;
        return Ok(());
    }

    // ── Single pass (default) ─────────────────────────────────────────────────
    run_pass(
        &args,
        args.scenario.as_deref(),
        &args.out,
        None,
        &args.compare_libs,
        std::time::SystemTime::now(),
    )
    .await
}

async fn run_pass(
    args: &AccuracyArgs,
    scenario: Option<&str>,
    out: &str,
    judge_target: Option<&str>,
    compare_libs: &[String],
    run_start: std::time::SystemTime,
) -> anyhow::Result<()> {
    let params = RunParams::from_args_with(args, scenario)?;
    let link_mode = parse_link_mode(&params.mode_str)?;
    let use_judge = judge_target.is_some();
    let judge_target_str = judge_target.unwrap_or("cpu");
    let backend = Backend::from_target(&args.target);
    let library_name = if use_judge {
        format!("zer+judge_{judge_target_str}")
    } else {
        "zer".to_owned()
    };
    let run_id = make_run_id(&library_name, &params.mode_str, &params.dataset_name);

    let total = 1 + compare_libs.len();
    let scenario_disp = scenario.unwrap_or(&params.dataset_name);
    let zer_label = if compare_libs.is_empty() {
        library_name.clone()
    } else {
        format!("[1/{total}] {library_name}")
    };
    super::util::print_bench_header(&[&zer_label, "accuracy", scenario_disp, &args.target]);

    println!(
        "accuracy run  run_id={run_id}  library={library_name}  mode={}  target={}  out={}",
        params.mode_str,
        backend.name(),
        out
    );
    for (i, ds) in params.datasets.iter().enumerate() {
        println!("dataset  index={i}  path={}", ds.as_str());
    }

    // ── Schema inference from CSV headers ────────────────────────────────────
    let schema = infer_schema_from_headers(&params.datasets)?;

    // ── Progress channel ─────────────────────────────────────────────────────
    let verbose = cfg!(feature = "progress");
    let perf_mode = cfg!(feature = "perf-metrics");
    let (progress_tx, progress_handle) = if verbose || perf_mode {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PipelineEvent>();
        let handle = tokio::spawn(async move {
            let mut t_phase = Instant::now();
            while let Some(event) = rx.recv().await {
                let now = Instant::now();
                match event {
                    PipelineEvent::BlockingStarted { total_records } => {
                        t_phase = now;
                        if verbose {
                            println!("blocking started  total_records={total_records}");
                        }
                    }
                    PipelineEvent::CandidatesReady {
                        candidate_pairs,
                        cross_source,
                        within_source,
                    } => {
                        if perf_mode {
                            println!("blocking_ms={}", now.duration_since(t_phase).as_millis());
                        }
                        t_phase = now;
                        if verbose {
                            println!("candidates ready  candidate_pairs={candidate_pairs}  cross_source={cross_source}  within_source={within_source}");
                        }
                    }
                    PipelineEvent::ComparingPairs { candidate_pairs } => {
                        t_phase = now;
                        if verbose {
                            println!("comparing pairs  candidate_pairs={candidate_pairs}");
                        }
                    }
                    PipelineEvent::EmStarted {
                        startup_mode,
                        max_iterations,
                    } => {
                        if perf_mode {
                            println!("compare_ms={}", now.duration_since(t_phase).as_millis());
                        }
                        t_phase = now;
                        if verbose {
                            println!("EM started  startup_mode={startup_mode}  max_iterations={max_iterations}");
                        }
                    }
                    PipelineEvent::EmComplete { iterations } => {
                        if perf_mode {
                            println!("em_ms={}", now.duration_since(t_phase).as_millis());
                        }
                        t_phase = now;
                        if verbose {
                            println!("EM complete  iterations={iterations}");
                        }
                    }
                    PipelineEvent::ScoringComplete {
                        auto_matched,
                        borderline,
                        auto_rejected,
                    } => {
                        if perf_mode {
                            println!("score_ms={}", now.duration_since(t_phase).as_millis());
                        }
                        t_phase = now;
                        if verbose {
                            println!("scoring complete  auto_matched={auto_matched}  borderline={borderline}  auto_rejected={auto_rejected}");
                        }
                    }
                    PipelineEvent::JudgeStarted { borderline } => {
                        t_phase = now;
                        if verbose {
                            println!("judge started  borderline={borderline}");
                        }
                    }
                    PipelineEvent::JudgeComplete { promoted, demoted } => {
                        if perf_mode {
                            println!("judge_ms={}", now.duration_since(t_phase).as_millis());
                        }
                        t_phase = now;
                        if verbose {
                            println!("judge complete  promoted={promoted}  demoted={demoted}");
                        }
                    }
                    PipelineEvent::PersistingEntities => {
                        t_phase = now;
                        if verbose {
                            println!("clustering and persisting entities");
                        }
                    }
                    PipelineEvent::Done { elapsed_ms } => {
                        if perf_mode {
                            println!("persist_ms={}", now.duration_since(t_phase).as_millis());
                            println!("total_pipeline_ms={elapsed_ms}");
                        }
                        if verbose {
                            println!("pipeline done  elapsed_ms={elapsed_ms}");
                        }
                    }
                }
            }
        });
        (Some(tx), Some(handle))
    } else {
        (None, None)
    };

    // ── Per-scenario strategy (config overrides + optional custom blocker/comparator) ──
    let strategy = strategies::strategy_for(&params.dataset_name);
    if strategy.blocker_fn.is_some() {
        println!(
            "using custom blocker strategy  dataset={}",
            params.dataset_name
        );
    }
    if strategy.comparator_fn.is_some() {
        println!(
            "using custom comparator strategy  dataset={}",
            params.dataset_name
        );
    }

    // ── Pipeline construction (outside timer) ─────────────────────────────────
    let dir = TempDir::new()?;
    let cfg = strategy.apply_to_config(PipelineConfig {
        registry_path: dir.path().join("accuracy_run.zsm"),
        link_mode,
        field_mappings: params.field_mappings.clone(),
        ..PipelineConfig::default()
    });

    let pipeline: Arc<Pipeline> = if use_judge {
        if judge_target_str == "tensorrt" {
            log_trt_cache_status();
        }
        let record_store: Arc<dyn RecordStore> = Arc::new(VecRecordStore::new());
        let judge_backend = JudgeBackend::from_target(judge_target_str);
        let models_base = args
            .judge_models_dir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                judge_backend.resolve_models_dir(&zer_judge::default_models_dir().join("nli-base"))
            });
        let minilm_dir = models_base.join("nli-minilm-onnx");
        let spec = MiniLmSpec::from_dir(&minilm_dir);
        println!(
            "loading judge  target={judge_target_str}  path={}",
            minilm_dir.display()
        );
        let t_load = Instant::now();
        let judge = DebertaJudge::new(
            &spec,
            &judge_backend,
            Arc::clone(&record_store),
            schema.clone(),
            DebertaJudgeConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!("failed to load judge model: {e}"))?;
        println!("judge ready  load_ms={}", t_load.elapsed().as_millis());
        let mut b = Pipeline::builder()
            .schema(schema.clone())
            .comparator(match strategy.comparator_fn {
                Some(f) => Comparator::from_cpu(f(&schema)),
                None => Comparator::new(&schema, &backend),
            })
            .scorer(Scorer::new(&backend))
            .store(ZalEntityStore::open_in_memory()?)
            .record_store_arc(record_store)
            .judge(judge)
            .config(cfg);
        if let Some(blocker_fn) = strategy.blocker_fn {
            b = b.blocker(blocker_fn(&schema));
        }
        if let Some(tx) = progress_tx {
            b = b.progress(tx);
        }
        b.build()?
    } else {
        let mut b = Pipeline::builder()
            .schema(schema.clone())
            .comparator(match strategy.comparator_fn {
                Some(f) => Comparator::from_cpu(f(&schema)),
                None => Comparator::new(&schema, &backend),
            })
            .scorer(Scorer::new(&backend))
            .store(ZalEntityStore::open_in_memory()?)
            .config(cfg);
        if let Some(blocker_fn) = strategy.blocker_fn {
            b = b.blocker(blocker_fn(&schema));
        }
        if let Some(tx) = progress_tx {
            b = b.progress(tx);
        }
        b.build()?
    };

    // ── Wall-clock timer ──────────────────────────────────────────────────────
    let wall_start = Instant::now();

    // ── Load records ──────────────────────────────────────────────────────────
    let mut all_records: Vec<Record> = Vec::new();
    let mut id_map: HashMap<String, u64> = HashMap::new();
    let max = if args.max_records == 0 {
        usize::MAX
    } else {
        args.max_records
    };

    for (i, path) in params.datasets.iter().enumerate() {
        let source = params.sources.get(i).and_then(|s| s.as_deref());
        let records = load_csv_records(path, source, max, &mut id_map)?;
        println!(
            "records loaded  count={}  path={path}  source={source:?}",
            records.len()
        );
        all_records.extend(records);
    }
    println!("all records loaded  total={}", all_records.len());

    // ── Run pipeline ──────────────────────────────────────────────────────────
    println!("running pipeline");
    let report = pipeline.run_batch(all_records.clone()).await?;

    println!(
        "pipeline complete  candidate_pairs={}  cross_source_pairs={}  auto_matched={}  borderline={}  auto_rejected={}  pipeline_elapsed_ms={}",
        report.candidate_pairs, report.cross_source_pairs, report.auto_matched,
        report.borderline, report.auto_rejected, report.elapsed_ms,
    );

    // ── Extract results ───────────────────────────────────────────────────────
    let view = pipeline.cluster_view();
    let pairs = view.all_member_pairs();

    drop(pipeline);
    if let Some(h) = progress_handle {
        let _ = h.await;
    }

    let wall_elapsed_ms = wall_start.elapsed().as_millis() as u64;
    println!("wall time  wall_elapsed_ms={wall_elapsed_ms}");

    // Build a reverse map from internal RecordId to natural key for scored_pairs lookups.
    let rec_id_to_key: HashMap<u64, String> =
        all_records.iter().map(|r| (r.id, r.key.clone())).collect();

    let pair_records: Vec<PairRecord> = pairs
        .iter()
        .map(|lp| PairRecord {
            run_id: run_id.clone(),
            record_key_a: lp.record_key_a.clone(),
            source_a: lp.source_a.clone(),
            record_key_b: lp.record_key_b.clone(),
            source_b: lp.source_b.clone(),
            match_probability: lp.score,
            predicted_match: band_to_match(resolution_to_band(lp.method)),
        })
        .collect();

    // When the collect-pairs feature is enabled the pipeline holds every candidate
    // pair with its match probability; use that for an unbiased PR-AUC over the
    // full curve.  Without it, fall back to entity-store pairs only (faster but biased).
    let pr_auc_pairs: Vec<PairRecord> = if !report.scored_pairs.is_empty() {
        report
            .scored_pairs
            .iter()
            .map(|&(a, b, prob)| PairRecord {
                run_id: run_id.clone(),
                record_key_a: rec_id_to_key
                    .get(&a)
                    .cloned()
                    .unwrap_or_else(|| a.to_string()),
                source_a: None,
                record_key_b: rec_id_to_key
                    .get(&b)
                    .cloned()
                    .unwrap_or_else(|| b.to_string()),
                source_b: None,
                match_probability: prob,
                predicted_match: false,
            })
            .collect()
    } else {
        pair_records.clone()
    };

    // ── Compute accuracy metrics ──────────────────────────────────────────────
    let (pipeline_accuracy, opt_metrics, pr_auc, f1_max, cluster_recall, strat_rows, scored_pairs) =
        if let Some(gt_path) = &params.ground_truth {
            println!("loading ground truth  path={}", gt_path.as_str());
            let gt_map = load_ground_truth(gt_path, &id_map)?;
            println!("ground truth loaded  pairs={}", gt_map.len());

            let predicted: HashSet<(String, String)> = pair_records
                .iter()
                .filter(|p| p.predicted_match)
                .map(|p| canonical_pair(&p.record_key_a, &p.record_key_b))
                .collect();

            let gt_set: HashSet<(String, String)> = gt_map.keys().cloned().collect();
            let true_pos = predicted.intersection(&gt_set).count();
            let false_pos = predicted.difference(&gt_set).count();
            let false_neg = gt_set.difference(&predicted).count();

            let pipe_acc = AccuracyMetrics::from_counts(true_pos, false_pos, false_neg);
            println!("pipeline accuracy  precision={}  recall={}  f1={}  tp={true_pos}  fp={false_pos}  fn_={false_neg}", pipe_acc.precision, pipe_acc.recall, pipe_acc.f1);

            let all_m = compute_all_metrics(&pr_auc_pairs, &gt_map);
            let (opt, pr_auc_val, f1m) = match all_m {
                Some(ref m) => {
                    println!("optimal threshold metrics  precision={}  recall={}  f1={}  threshold={}  tp={}  fp={}  fn_={}  pr_auc={}  f1_max={}", m.best.precision, m.best.recall, m.best.f1, m.best.threshold, m.best.tp, m.best.fp, m.best.fn_, m.pr_auc, m.f1_max);
                    (
                        Some(ThresholdMetrics {
                            f1: m.best.f1,
                            precision: m.best.precision,
                            recall: m.best.recall,
                            threshold: m.best.threshold,
                            tp: m.best.tp,
                            fp: m.best.fp,
                            fn_: m.best.fn_,
                        }),
                        Some(m.pr_auc),
                        Some(m.f1_max),
                    )
                }
                None => (None, None, None),
            };

            let blk = compute_cluster_recall(&pair_records, &gt_map);
            println!("cluster recall  cluster_recall={blk}");

            let strat = compute_stratified_metrics(&pair_records, &gt_map);

            let tagged: Vec<(f32, bool)> = pr_auc_pairs
                .iter()
                .map(|p| {
                    let key = canonical_pair(&p.record_key_a, &p.record_key_b);
                    (p.match_probability, gt_map.contains_key(&key))
                })
                .collect();

            (
                Some(pipe_acc),
                opt,
                pr_auc_val,
                f1m,
                Some(blk),
                strat,
                Some(tagged),
            )
        } else {
            println!("no ground truth, accuracy columns will be empty");
            (None, None, None, None, None::<f32>, Vec::new(), None)
        };

    // ── Write output ──────────────────────────────────────────────────────────
    let writer = BenchResultWriter::new(resolve_out_dir(out).as_path(), &run_id)?;
    writer.write_pairs(&pair_records)?;
    if let Some(ref pairs) = scored_pairs {
        writer.write_scored_pairs_csv(pairs)?;
    }

    let summary = BenchBatchSummary {
        total_records: report.total_records,
        candidate_pairs: report.candidate_pairs,
        auto_matched: report.auto_matched,
        borderline: report.borderline,
        auto_rejected: report.auto_rejected,
        elapsed_ms: wall_elapsed_ms,
        link_mode: report.link_mode.as_str().to_owned(),
        dataset: params.dataset_name.clone(),
    };
    // Use optimal-threshold metrics as primary P/R/F1 in summary CSV so the
    // comparison table is on equal footing with splink (which also uses optimal threshold).
    // Pipeline-level metrics are preserved in the benchmark JSON as pipeline_*.
    let opt_acc: Option<AccuracyMetrics> = opt_metrics
        .as_ref()
        .map(|m| AccuracyMetrics::from_counts(m.tp, m.fp, m.fn_));
    writer.write_summary_with_library(&summary, opt_acc.as_ref(), &library_name)?;

    let has_types = strat_rows.iter().any(|r| !r.match_type.is_empty());
    if has_types {
        let strat_path = writer.out_dir().join(format!("{run_id}_strat.csv"));
        write_strat_csv(&strat_path, &strat_rows)?;
        println!("strat csv written  path={}", strat_path.display());
    }

    let scored_pairs_csv: Option<String> = scored_pairs
        .as_ref()
        .map(|_| format!("{run_id}_scored_pairs.csv"));

    let json_path = writer.out_dir().join(format!("{run_id}_benchmark.json"));
    write_benchmark_json(BenchmarkJsonArgs {
        path: &json_path,
        run_id: &run_id,
        library: &library_name,
        scenario: scenario,
        mode: &params.mode_str,
        dataset: &params.dataset_name,
        target: &args.target,
        total_records: report.total_records,
        candidate_pairs: report.candidate_pairs,
        auto_matched: report.auto_matched,
        borderline: report.borderline,
        auto_rejected: report.auto_rejected,
        opt_metrics: opt_metrics.as_ref(),
        pipeline_acc: pipeline_accuracy.as_ref(),
        pr_auc,
        f1_max,
        cluster_recall,
        strat_rows: &strat_rows,
        scored_pairs_csv: scored_pairs_csv.as_deref(),
    })?;
    println!("benchmark json written  path={}", json_path.display());

    println!("run_id:    {run_id}");
    println!("out_dir:   {}", writer.out_dir().display());
    println!("pairs:     {}", pair_records.len());
    if let Some(m) = &opt_metrics {
        println!("precision: {:.3}", m.precision);
        println!("recall:    {:.3}", m.recall);
        println!("f1:        {:.3}", m.f1);
        println!("opt_thr:   {:.6}", m.threshold);
    } else if let Some(acc) = &pipeline_accuracy {
        println!("precision: {:.3}", acc.precision);
        println!("recall:    {:.3}", acc.recall);
        println!("f1:        {:.3}", acc.f1);
    }
    if let Some(auc) = pr_auc {
        println!("pr_auc:    {:.4}", auc);
    }
    if let Some(f1m) = f1_max {
        println!("f1_max:    {:.4}", f1m);
    }
    if let Some(blk) = cluster_recall {
        println!("cluster_recall: {:.4}", blk);
    }

    // ── Run competitor libraries and print inline comparison ──────────────────
    if !compare_libs.is_empty() {
        let bench_root =
            super::library::resolve_benchmarks_root(args.external_benchmarks_dir.as_deref());
        let mode_dir = super::library::mode_dir_name(&params.mode_str);
        let dataset_refs: Vec<&str> = params.datasets.iter().map(String::as_str).collect();
        let gt = params.ground_truth.as_deref();
        let mut lib_errors: Vec<String> = Vec::new();
        for (i, lib) in compare_libs.iter().enumerate() {
            super::util::print_bench_header(&[
                &format!("[{}/{total}] {lib}", i + 2),
                "accuracy",
                scenario_disp,
            ]);
            println!("running library  library={lib}  mode={mode_dir}");
            if let Err(e) = super::library::run_library(
                &bench_root,
                lib,
                mode_dir,
                scenario,
                &dataset_refs,
                gt,
                out,
                None,
                args.force_setup,
            ) {
                eprintln!("warning: library failed  library={lib}  error={e}");
                lib_errors.push(format!("{lib}: {e}"));
            }
        }
        super::compare::print_comparison_for_dir(out, run_start)?;
        if !lib_errors.is_empty() {
            anyhow::bail!("some libraries failed:\n{}", lib_errors.join("\n"));
        }
    }

    Ok(())
}

// ── Record loading ────────────────────────────────────────────────────────────

/// Load records from a CSV file.
///
/// When `source` is `Some`, each record is created via [`Record::from_key`] so
/// that the natural key (column 0) is preserved in the `.zes` output.  The
/// `id_map` is populated with `raw_id → derived_hash` entries so that
/// [`load_ground_truth`] can resolve the same string IDs to the same u64s.
///
/// When `source` is `None`, the old numeric-or-sequential behaviour is kept
/// for backward compatibility.
fn load_csv_records(
    path: &str,
    source: Option<&str>,
    max: usize,
    id_map: &mut HashMap<String, u64>,
) -> anyhow::Result<Vec<Record>> {
    let mut rdr =
        csv::Reader::from_path(path).map_err(|e| anyhow::anyhow!("cannot open {path}: {e}"))?;

    let headers: Vec<String> = rdr.headers()?.iter().map(str::to_owned).collect();
    let mut records = Vec::new();

    for result in rdr.records().take(max) {
        let row = result?;
        let raw_id = row.get(0).unwrap_or("").to_string();

        let rec = if let Some(src) = source {
            // Natural-key path: hash(source:raw_id) → stable RecordId.
            let hash = derive_record_id(src, &raw_id);
            id_map.insert(raw_id.clone(), hash);
            let mut r = Record::from_key(src, &raw_id);
            for (j, header) in headers.iter().enumerate() {
                if let Some(val) = row.get(j) {
                    r = r.insert(header.as_str(), FieldValue::Text(val.to_string()));
                }
            }
            r
        } else {
            // No source label: parse as u64 or assign a sequential id.
            let id: u64 = if let Ok(n) = raw_id.parse::<u64>() {
                n
            } else {
                let next = id_map.len() as u64 + 1;
                *id_map.entry(raw_id.clone()).or_insert(next)
            };
            id_map.insert(raw_id.clone(), id);
            let mut r = Record::new(id);
            for (j, header) in headers.iter().enumerate() {
                if let Some(val) = row.get(j) {
                    r = r.insert(header.as_str(), FieldValue::Text(val.to_string()));
                }
            }
            r
        };

        records.push(rec);
    }
    Ok(records)
}

// ── Ground truth loading ──────────────────────────────────────────────────────

/// Load a ground truth CSV and return the map of canonical positive pairs to their `match_type`.
///
/// Column layout: `record_id_a, record_id_b, is_match[, match_type]`.
/// IDs are stored as raw strings so they match the natural keys from
/// [`load_csv_records`] directly. no numeric conversion needed.
/// The optional `match_type` column (index 3) enables stratified recall reporting.
fn load_ground_truth(
    path: &str,
    _id_map: &HashMap<String, u64>,
) -> anyhow::Result<HashMap<(String, String), String>> {
    let mut rdr = csv::Reader::from_path(path)
        .map_err(|e| anyhow::anyhow!("cannot open ground truth {path}: {e}"))?;

    let mut pairs = HashMap::new();

    for result in rdr.records() {
        let row = result?;
        let raw_a = row.get(0).unwrap_or("").to_string();
        let raw_b = row.get(1).unwrap_or("").to_string();
        let is_match: bool = row
            .get(2)
            .map(|s| matches!(s.to_lowercase().as_str(), "true" | "1" | "yes"))
            .unwrap_or(false);

        if !is_match || raw_a.is_empty() || raw_b.is_empty() {
            continue;
        }

        let match_type = row.get(3).unwrap_or("").to_owned();
        pairs.insert(canonical_pair(&raw_a, &raw_b), match_type);
    }

    Ok(pairs)
}

// ── Schema inference ──────────────────────────────────────────────────────────

fn infer_schema_from_headers(paths: &[String]) -> anyhow::Result<Schema> {
    let mut names: Vec<String> = Vec::new();
    for path in paths {
        let mut rdr =
            csv::Reader::from_path(path).map_err(|e| anyhow::anyhow!("cannot open {path}: {e}"))?;
        for h in rdr.headers()?.iter() {
            let s = h.to_owned();
            if !names.contains(&s) {
                names.push(s);
            }
        }
    }
    let mut builder = SchemaBuilder::new();
    for name in &names {
        builder = builder.field(name.as_str(), infer_field_kind(name));
    }
    Ok(builder.build().unwrap_or_else(|_| {
        SchemaBuilder::new()
            .field("id", FieldKind::Id)
            .build()
            .unwrap()
    }))
}

fn infer_field_kind(name: &str) -> FieldKind {
    let n = name.to_lowercase();
    // Company/org names must be checked before the generic "naam" rule to avoid
    // misclassifying them as person names, which would apply Jaro-Winkler blocking
    // keys and inflate cross-source candidate pairs.
    if n.contains("handelsnaam") || n.contains("rechtsvorm") || n.contains("kvk") {
        FieldKind::Categorical
    // Address patterns before "naam": "straatnaam" contains both "straat" and "naam";
    // the address check must win so it doesn't end up as the phonetic surname field.
    } else if n.contains("straat")
        || n.contains("adres")
        || n.contains("street")
        || n.contains("address")
        || n.contains("city")
        || n.contains("place")
        || n.contains("woon")
    {
        FieldKind::Address
    } else if n.contains("naam") || n.contains("name") || n.contains("nomen") || n.contains("alias")
    {
        FieldKind::Name
    } else if n.contains("datum") || n.contains("date") || n.contains("dob") || n.contains("birth")
    {
        FieldKind::Date
    } else if n.contains("id")
        || n.contains("nummer")
        || n.contains("bsn")
        || n.contains("number")
        || n.contains("code")
        || n.contains("postcode")
    {
        FieldKind::Id
    } else {
        FieldKind::Categorical
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_link_mode(s: &str) -> anyhow::Result<LinkMode> {
    match s.to_lowercase().replace('-', "_").as_str() {
        "deduplicate" | "dedupe" => Ok(LinkMode::Deduplicate),
        "link_only" | "link-only" => Ok(LinkMode::LinkOnly),
        "link_and_dedupe" | "link-and-dedupe" | "link_dedupe" => Ok(LinkMode::LinkAndDedupe),
        other => anyhow::bail!(
            "unknown link mode: {other:?}; valid: deduplicate, link-only, link-and-dedupe"
        ),
    }
}

fn infer_dataset_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_owned()
}

fn make_run_id(library: &str, mode: &str, dataset: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mode_clean = mode.replace('-', "_");
    let lib_clean = library.replace('+', "_plus_");
    format!("{lib_clean}_{mode_clean}_{dataset}_{ts}")
}

fn canonical_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

struct AllMetrics {
    pr_auc: f32,
    f1_max: f32,
    best: ThresholdMetrics,
}

/// Single-pass computation of PR-AUC, max-F1, and best-threshold metrics.
fn compute_all_metrics(
    pair_records: &[PairRecord],
    gt_map: &HashMap<(String, String), String>,
) -> Option<AllMetrics> {
    let n_pos = gt_map.len();
    if n_pos == 0 || pair_records.is_empty() {
        return None;
    }
    let mut tagged: Vec<(f32, bool)> = pair_records
        .iter()
        .map(|p| {
            let key = canonical_pair(&p.record_key_a, &p.record_key_b);
            (p.match_probability, gt_map.contains_key(&key))
        })
        .collect();
    tagged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let (mut tp, mut fp) = (0usize, 0usize);
    let mut fn_ = n_pos;
    let (mut auc, mut prev_recall) = (0.0f32, 0.0f32);
    let mut best_f1 = 0.0f32;
    let mut best = ThresholdMetrics {
        f1: 0.0,
        precision: 0.0,
        recall: 0.0,
        threshold: 1.0,
        tp: 0,
        fp: 0,
        fn_: n_pos,
    };

    for (score, is_match) in &tagged {
        if *is_match {
            tp += 1;
            fn_ -= 1;
        } else {
            fp += 1;
        }
        let precision = tp as f32 / (tp + fp) as f32;
        let recall = tp as f32 / n_pos as f32;
        auc += (recall - prev_recall) * precision;
        prev_recall = recall;
        let denom = 2 * tp + fp + fn_;
        if denom > 0 {
            let f1 = 2.0 * tp as f32 / denom as f32;
            if f1 > best_f1 {
                best_f1 = f1;
                best = ThresholdMetrics {
                    f1,
                    precision,
                    recall,
                    threshold: *score,
                    tp,
                    fp,
                    fn_,
                };
            }
        }
    }
    Some(AllMetrics {
        pr_auc: auc.clamp(0.0, 1.0),
        f1_max: best_f1,
        best,
    })
}

struct ThresholdMetrics {
    f1: f32,
    precision: f32,
    recall: f32,
    threshold: f32,
    tp: usize,
    fp: usize,
    fn_: usize,
}

/// Cluster recall: fraction of GT pairs whose two records ended up in the same cluster.
///
/// Computed from the entity-store view (all intra-cluster pairs), not raw blocking
/// candidates, so it reflects how many true matches survived comparison and clustering.
fn compute_cluster_recall(
    pair_records: &[PairRecord],
    gt_map: &HashMap<(String, String), String>,
) -> f32 {
    let n_pos = gt_map.len();
    if n_pos == 0 {
        return 1.0;
    }
    let candidate_set: HashSet<(String, String)> = pair_records
        .iter()
        .map(|p| canonical_pair(&p.record_key_a, &p.record_key_b))
        .collect();
    let found = gt_map.keys().filter(|k| candidate_set.contains(*k)).count();
    found as f32 / n_pos as f32
}

struct StratRow {
    match_type: String,
    count_gt: usize,
    true_pos: usize,
    false_neg: usize,
    recall: f32,
}

fn compute_stratified_metrics(
    pair_records: &[PairRecord],
    gt_map: &HashMap<(String, String), String>,
) -> Vec<StratRow> {
    let predicted: HashSet<(String, String)> = pair_records
        .iter()
        .filter(|p| p.predicted_match)
        .map(|p| canonical_pair(&p.record_key_a, &p.record_key_b))
        .collect();

    let mut gt_by_type: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (pair, mt) in gt_map {
        gt_by_type.entry(mt.clone()).or_default().push(pair.clone());
    }

    let mut rows: Vec<StratRow> = gt_by_type
        .into_iter()
        .map(|(mt, gt_pairs)| {
            let count_gt = gt_pairs.len();
            let true_pos = gt_pairs.iter().filter(|p| predicted.contains(*p)).count();
            let false_neg = count_gt - true_pos;
            let recall = if count_gt == 0 {
                0.0
            } else {
                true_pos as f32 / count_gt as f32
            };
            StratRow {
                match_type: mt,
                count_gt,
                true_pos,
                false_neg,
                recall,
            }
        })
        .collect();
    rows.sort_by(|a, b| a.match_type.cmp(&b.match_type));
    rows
}

fn write_strat_csv(path: &std::path::Path, rows: &[StratRow]) -> anyhow::Result<()> {
    let mut wtr = csv::Writer::from_path(path)
        .map_err(|e| anyhow::anyhow!("cannot create strat csv {}: {e}", path.display()))?;
    wtr.write_record(["match_type", "count_gt", "true_pos", "false_neg", "recall"])?;
    for r in rows {
        wtr.write_record([
            r.match_type.as_str(),
            &r.count_gt.to_string(),
            &r.true_pos.to_string(),
            &r.false_neg.to_string(),
            &format!("{:.4}", r.recall),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

struct BenchmarkJsonArgs<'a> {
    path: &'a std::path::Path,
    run_id: &'a str,
    library: &'a str,
    scenario: Option<&'a str>,
    mode: &'a str,
    dataset: &'a str,
    target: &'a str,
    total_records: usize,
    candidate_pairs: usize,
    auto_matched: usize,
    borderline: usize,
    auto_rejected: usize,
    opt_metrics: Option<&'a ThresholdMetrics>,
    pipeline_acc: Option<&'a AccuracyMetrics>,
    pr_auc: Option<f32>,
    f1_max: Option<f32>,
    cluster_recall: Option<f32>,
    strat_rows: &'a [StratRow],
    scored_pairs_csv: Option<&'a str>,
}

fn write_benchmark_json(a: BenchmarkJsonArgs<'_>) -> anyhow::Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let stem = a
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(a.run_id);
    let has_strat = a.strat_rows.iter().any(|r| !r.match_type.is_empty());

    let strat_json: Vec<serde_json::Value> = a
        .strat_rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "match_type": r.match_type,
                "count_gt":   r.count_gt,
                "true_pos":   r.true_pos,
                "false_neg":  r.false_neg,
                "recall":     (r.recall * 10000.0).round() / 10000.0,
            })
        })
        .collect();

    let round4 = |v: f32| (v * 10000.0).round() / 10000.0;
    let round3 = |v: f32| (v * 1000.0).round() / 1000.0;
    let round6 = |v: f32| (v * 1_000_000.0).round() / 1_000_000.0;

    let metrics = if let Some(opt) = a.opt_metrics {
        serde_json::json!({
            "total_records":       a.total_records,
            "candidate_pairs":     a.candidate_pairs,
            "auto_matched":        a.auto_matched,
            "borderline":          a.borderline,
            "auto_rejected":       a.auto_rejected,
            "precision":           round3(opt.precision),
            "recall":              round3(opt.recall),
            "f1":                  round3(opt.f1),
            "optimal_threshold":   round6(opt.threshold),
            "true_pos":            opt.tp,
            "false_pos":           opt.fp,
            "false_neg":           opt.fn_,
            "pipeline_precision":  a.pipeline_acc.map(|p| round3(p.precision)),
            "pipeline_recall":     a.pipeline_acc.map(|p| round3(p.recall)),
            "pipeline_f1":         a.pipeline_acc.map(|p| round3(p.f1)),
            "pipeline_true_pos":   a.pipeline_acc.map(|p| p.true_pos),
            "pipeline_false_pos":  a.pipeline_acc.map(|p| p.false_pos),
            "pipeline_false_neg":  a.pipeline_acc.map(|p| p.false_neg),
            "f1_max":              a.f1_max.map(round4),
            "pr_auc":              a.pr_auc.map(round4),
            "cluster_recall":      a.cluster_recall.map(round4),
        })
    } else {
        serde_json::json!({
            "total_records":       a.total_records,
            "candidate_pairs":     a.candidate_pairs,
            "auto_matched":        a.auto_matched,
            "borderline":          a.borderline,
            "auto_rejected":       a.auto_rejected,
            "precision":           null,
            "recall":              null,
            "f1":                  null,
            "optimal_threshold":   null,
            "true_pos":            null,
            "false_pos":           null,
            "false_neg":           null,
            "pipeline_precision":  null,
            "pipeline_recall":     null,
            "pipeline_f1":         null,
            "pipeline_true_pos":   null,
            "pipeline_false_pos":  null,
            "pipeline_false_neg":  null,
            "f1_max":              null,
            "pr_auc":              null,
            "cluster_recall":      null,
        })
    };

    let doc = serde_json::json!({
        "run_id":         a.run_id,
        "library":        a.library,
        "scenario":       a.scenario,
        "mode":           a.mode,
        "dataset":        a.dataset,
        "target":         a.target,
        "timestamp_unix": timestamp_unix,
        "files": {
            "summary_csv":      format!("{stem}_summary.csv"),
            "pairs_ndjson":     format!("{stem}_pairs.ndjson"),
            "strat_csv":        if has_strat { serde_json::Value::String(format!("{stem}_strat.csv")) } else { serde_json::Value::Null },
            "scored_pairs_csv": a.scored_pairs_csv.map_or(serde_json::Value::Null, |s| serde_json::Value::String(s.to_owned())),
        },
        "metrics": metrics,
        "strat": strat_json,
        "scored_pairs": serde_json::Value::Null,
    });

    let json_str = serde_json::to_string_pretty(&doc)
        .map_err(|e| anyhow::anyhow!("json serialization error: {e}"))?;
    std::fs::write(a.path, json_str)
        .map_err(|e| anyhow::anyhow!("cannot write benchmark json {}: {e}", a.path.display()))?;
    Ok(())
}

// ── ResolutionMethod → MatchBand ─────────────────────────────────────────────

use zer_core::entity::ResolutionMethod;

fn resolution_to_band(m: ResolutionMethod) -> MatchBand {
    match m {
        ResolutionMethod::AutoMatch => MatchBand::AutoMatch,
        ResolutionMethod::JudgePromoted => MatchBand::AutoMatch,
        ResolutionMethod::JudgeDemoted => MatchBand::AutoReject,
        ResolutionMethod::Manual => MatchBand::Borderline,
    }
}
