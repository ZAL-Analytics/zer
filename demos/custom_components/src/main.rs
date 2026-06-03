/// Custom components demo.
///
/// Shows how to implement custom pipeline components and plug them into zer:
///
///   1. `ThresholdJudge` , a Judge that promotes/demotes borderline pairs based
///      on a hard probability threshold, useful when human review is not available
///
///   2. `PostcodeBlocker`, a Blocker that only keys on the first four digits of
///      a Dutch postcode (very tight blocking, high precision, lower recall)
///
///   3. `MatchWeightScorer`, a minimal Scorer that returns match_weight
///      directly from a simple log-odds sum without the sigmoid transform,
///      illustrating how to swap out the scoring model
///
/// The demo wires each custom component into a pipeline and runs a small
/// synthetic dataset through it to show the results.
use demo_common::{init_tracing, section};
use zer_cluster::ZalEntityStore;
use zer_compare::FellegiSunterScorer;
use zer_core::{
    comparison::{ComparisonBatch, ComparisonVector},
    record::{Record, RecordId},
    schema::{FieldKind, Schema, SchemaBuilder},
    scoring::{MatchBand, ModelParams, ScoredPair},
    traits::{BlockIndex, Blocker, Judge, JudgeVerdict, Result, Scorer},
};
use zer_pipeline::{Pipeline, PipelineConfig};

// ── Custom Judge ──────────────────────────────────────────────────────────────

/// Promotes any borderline pair above `threshold` to IncreaseConfidence;
/// demotes pairs below `low_threshold` to DecreaseConfidence.
struct ThresholdJudge {
    high: f32,
    low: f32,
}

impl Judge for ThresholdJudge {
    fn adjudicate(&self, pairs: &[ScoredPair]) -> Result<Vec<JudgeVerdict>> {
        Ok(pairs
            .iter()
            .map(|p| {
                if p.match_probability >= self.high {
                    JudgeVerdict::IncreaseConfidence
                } else if p.match_probability < self.low {
                    JudgeVerdict::DecreaseConfidence
                } else {
                    JudgeVerdict::NoChange
                }
            })
            .collect())
    }
}

// ── Custom Blocker ────────────────────────────────────────────────────────────

/// Keys on the first four characters of a `postcode` field (e.g. "1234" from "1234AB").
/// Very selective: only compares persons in the same 4-digit Dutch postcode area.
struct PostcodeBlocker;

impl Blocker for PostcodeBlocker {
    fn blocking_keys(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        record
            .text("postcode")
            .map(|pc| {
                let digits: String = pc.chars().take(4).collect();
                if digits.len() == 4 {
                    vec![format!("pc4:{}", digits)]
                } else {
                    vec![]
                }
            })
            .unwrap_or_default()
    }

    fn index_record(&self, record: &Record, schema: &Schema, index: &mut dyn BlockIndex) {
        let keys = self.blocking_keys(record, schema);
        index.insert(record.id, keys);
    }

    fn candidates(
        &self,
        record: &Record,
        schema: &Schema,
        index: &dyn BlockIndex,
    ) -> Vec<RecordId> {
        let keys = self.blocking_keys(record, schema);
        index.lookup_union(&keys, record.id)
    }
}

// ── Custom Scorer ─────────────────────────────────────────────────────────────

/// A scorer that computes a simple uniform log-odds sum and returns it
/// as `match_weight`, with `match_probability = weight / (1 + weight.abs())`.
///
/// This is intentionally simplistic, it shows the interface, not best practice.
struct SimpleLogOddsScorer {
    match_log_ratio: f32,
    no_match_log_ratio: f32,
}

impl SimpleLogOddsScorer {
    fn new() -> Self {
        Self {
            match_log_ratio: (0.80_f32 / 0.05_f32).ln(),
            no_match_log_ratio: (0.05_f32 / 0.80_f32).ln(),
        }
    }
}

impl Scorer for SimpleLogOddsScorer {
    fn score(&self, vector: &ComparisonVector, params: &ModelParams) -> ScoredPair {
        use zer_core::comparison::ComparisonLevel;
        let weight: f32 = vector
            .levels
            .iter()
            .map(|&level| match level {
                ComparisonLevel::Exact | ComparisonLevel::Close => self.match_log_ratio,
                ComparisonLevel::Partial => 0.0,
                _ => self.no_match_log_ratio,
            })
            .sum();

        let prob = weight / (1.0 + weight.abs());
        let band = if prob >= params.upper_threshold {
            MatchBand::AutoMatch
        } else if prob < params.lower_threshold {
            MatchBand::AutoReject
        } else {
            MatchBand::Borderline
        };

        ScoredPair {
            record_a: vector.record_a,
            record_b: vector.record_b,
            match_weight: weight,
            match_probability: prob,
            vector: vector.clone(),
            band,
        }
    }

    fn estimate_params(
        &self,
        batch: &ComparisonBatch,
        _init: Option<ModelParams>,
        _max_iter: usize,
    ) -> Result<ModelParams> {
        // Delegate EM to the standard scorer since we only customise scoring.
        FellegiSunterScorer.estimate_params(batch, None, 10)
    }
}

// ── Synthetic dataset ─────────────────────────────────────────────────────────

fn person(id: u64, first: &str, last: &str, dob: &str, gender: &str, pc: &str) -> Record {
    Record::new(id)
        .insert("voornamen", first)
        .insert("achternaam", last)
        .insert("geboortedatum", dob)
        .insert("geslacht", gender)
        .insert("postcode", pc)
}

fn synthetic_records() -> Vec<Record> {
    vec![
        // Cluster 1: Jan de Vries, same postcode, name variant
        person(1, "Jan", "de Vries", "1985-03-22", "M", "1234AB"),
        person(2, "Johannes", "de Vries", "1985-03-22", "M", "1234CD"),
        // Cluster 2: Anna Bakker, same postcode, address move
        person(3, "Anna", "Bakker", "1990-07-15", "V", "5678EF"),
        person(4, "Anna", "Bakker", "1990-07-15", "V", "5678GH"),
        // Unique: different persons, same postcode area (should not match)
        person(5, "Pieter", "Smit", "1978-11-01", "M", "1234ZZ"),
        person(6, "Maria", "Janssen", "1962-09-08", "V", "1234WW"),
        // Unique: completely different postcode area
        person(7, "Robert", "Visser", "1975-04-30", "M", "9999XX"),
        person(8, "Fatima", "El-Amrani", "2000-01-20", "V", "8888YY"),
    ]
}

#[tokio::main]
async fn main() {
    init_tracing();

    let schema = SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("geslacht", FieldKind::FreeText)
        .field("postcode", FieldKind::FreeText)
        .build()
        .expect("build schema");

    let records = synthetic_records();

    // ── Pipeline with all custom components ───────────────────────────────────
    section("Wiring custom components");

    println!("Blocker : PostcodeBlocker (first 4 digits of postcode)");
    println!("Scorer  : SimpleLogOddsScorer (uniform log-odds sum)");
    println!("Judge   : ThresholdJudge (promotes ≥ 0.70, demotes < 0.30)");

    let store = ZalEntityStore::open_in_memory().expect("open entity store");
    let tmpdir = std::env::temp_dir();
    let zsm = tmpdir.join("demo_custom.zsm");

    let pipeline = Pipeline::builder()
        .schema(schema)
        .store(store)
        .blocker(PostcodeBlocker)
        .scorer(SimpleLogOddsScorer::new())
        .judge(ThresholdJudge {
            high: 0.70,
            low: 0.30,
        })
        .config(PipelineConfig {
            registry_path: zsm,
            ..PipelineConfig::default()
        })
        .build()
        .expect("build pipeline");

    // ── Run ───────────────────────────────────────────────────────────────────
    section("Running with custom components");

    let report = pipeline.run_batch(records).await.expect("run_batch");

    println!("total records   : {}", report.total_records);
    println!("candidate pairs : {}", report.candidate_pairs);
    println!("auto-matched    : {}", report.auto_matched);
    println!("borderline      : {}", report.borderline);
    println!("auto-rejected   : {}", report.auto_rejected);
    println!("judge promoted  : {}", report.judge_promoted);
    println!("judge demoted   : {}", report.judge_demoted);
    println!("entities created: {}", report.entities_created);

    // ── Summary ───────────────────────────────────────────────────────────────
    section("Key takeaways");
    println!("Custom components implement traits from zer-core::traits:");
    println!("  Blocker  → blocking_keys / index_record / candidates");
    println!("  Scorer   → score / score_batch / estimate_params");
    println!("  Judge    → adjudicate(&[ScoredPair]) → Vec<JudgeVerdict>");
    println!("Pass them to Pipeline::builder().blocker() / .scorer() / .judge().");

    section("Done");
}
