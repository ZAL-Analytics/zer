/// Benchmark result writer for zer accuracy runs.
///
/// Writes two output files per run:
/// - `<run_id>_pairs.ndjson`  , one JSON object per line (streaming pairs)
/// - `<run_id>_summary.csv`   , single-row CSV in the shared cross-library format
///
/// The shared CSV format makes side-by-side comparison with splink
/// trivial via the `zer-bench compare` subcommand.
use std::{
    fs::{self, File},
    io::{BufWriter, Write as IoWrite},
    path::{Path, PathBuf},
};

use zer_core::{error::ZerError, scoring::MatchBand};

/// Aggregate accuracy metrics computed against a ground-truth labels file.
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    pub true_pos: usize,
    pub false_pos: usize,
    pub false_neg: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

impl AccuracyMetrics {
    /// Compute from counts.  Returns a zero-valued struct when `tp + fp == 0`.
    pub fn from_counts(true_pos: usize, false_pos: usize, false_neg: usize) -> Self {
        let precision = if true_pos + false_pos > 0 {
            true_pos as f32 / (true_pos + false_pos) as f32
        } else {
            0.0
        };
        let recall = if true_pos + false_neg > 0 {
            true_pos as f32 / (true_pos + false_neg) as f32
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        Self {
            true_pos,
            false_pos,
            false_neg,
            precision,
            recall,
            f1,
        }
    }
}

/// A single scored pair as written to the NDJSON pairs file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PairRecord {
    pub run_id: String,
    pub record_key_a: String,
    pub source_a: Option<String>,
    pub record_key_b: String,
    pub source_b: Option<String>,
    pub match_probability: f32,
    pub predicted_match: bool,
}

/// Summary row shared with all benchmark libraries.
#[derive(Debug, Clone, serde::Serialize)]
struct SummaryRow {
    library: String,
    mode: String,
    dataset: String,
    run_id: String,
    timestamp: String,
    total_records: usize,
    candidate_pairs: usize,
    auto_matched: usize,
    borderline: usize,
    auto_rejected: usize,
    elapsed_ms: u64,
    true_pos: Option<usize>,
    false_pos: Option<usize>,
    false_neg: Option<usize>,
    precision: Option<f32>,
    recall: Option<f32>,
    f1: Option<f32>,
}

/// A lightweight `BatchReport`-like view used by `BenchResultWriter`.
///
/// This avoids a direct dependency on `zer-pipeline` from `zer-adapters`.
pub struct BenchBatchSummary {
    pub total_records: usize,
    pub candidate_pairs: usize,
    pub auto_matched: usize,
    pub borderline: usize,
    pub auto_rejected: usize,
    pub elapsed_ms: u64,
    pub link_mode: String,
    pub dataset: String,
}

pub struct BenchResultWriter {
    run_id: String,
    out_dir: PathBuf,
}

impl BenchResultWriter {
    /// Create a new writer.  `out_dir` is created if it does not yet exist.
    pub fn new(out_dir: &Path, run_id: &str) -> Result<Self, ZerError> {
        fs::create_dir_all(out_dir)
            .map_err(|e| ZerError::Store(format!("cannot create output dir: {e}")))?;
        Ok(Self {
            run_id: run_id.to_owned(),
            out_dir: out_dir.to_path_buf(),
        })
    }

    /// Write a streaming NDJSON pairs file.  One JSON object per line.
    pub fn write_pairs(&self, pairs: &[PairRecord]) -> Result<(), ZerError> {
        let path = self.out_dir.join(format!("{}_pairs.ndjson", self.run_id));
        let file = File::create(&path)
            .map_err(|e| ZerError::Store(format!("cannot create pairs file: {e}")))?;
        let mut w = BufWriter::new(file);
        for pair in pairs {
            let line = serde_json::to_string(pair)
                .map_err(|e| ZerError::Store(format!("JSON serialise error: {e}")))?;
            writeln!(w, "{line}").map_err(|e| ZerError::Store(format!("write error: {e}")))?;
        }
        Ok(())
    }

    /// Write a single-row summary CSV in the shared cross-library format.
    ///
    /// Accuracy columns (`true_pos`, `false_pos`, etc.) are left empty when
    /// `accuracy` is `None`, suitable for runs without a ground-truth file.
    /// Uses `"zer"` as the library name.  Call [`Self::write_summary_with_library`]
    /// when a different name is needed (e.g. `"zer+judge"`).
    pub fn write_summary(
        &self,
        summary: &BenchBatchSummary,
        accuracy: Option<&AccuracyMetrics>,
    ) -> Result<(), ZerError> {
        self.write_summary_with_library(summary, accuracy, "zer")
    }

    /// Like [`Self::write_summary`] but lets the caller set the `library` column.
    ///
    /// Use `"zer"` for the FS-only pipeline and `"zer+judge"` when the
    /// MiniLM neural judge is enabled, so the comparison table distinguishes
    /// both operating points.
    pub fn write_summary_with_library(
        &self,
        summary: &BenchBatchSummary,
        accuracy: Option<&AccuracyMetrics>,
        library: &str,
    ) -> Result<(), ZerError> {
        let path = self.out_dir.join(format!("{}_summary.csv", self.run_id));
        let file = File::create(&path)
            .map_err(|e| ZerError::Store(format!("cannot create summary file: {e}")))?;

        let timestamp = crate::time::utc_timestamp_iso();
        let row = SummaryRow {
            library: library.to_owned(),
            mode: summary.link_mode.to_lowercase(),
            dataset: summary.dataset.clone(),
            run_id: self.run_id.clone(),
            timestamp,
            total_records: summary.total_records,
            candidate_pairs: summary.candidate_pairs,
            auto_matched: summary.auto_matched,
            borderline: summary.borderline,
            auto_rejected: summary.auto_rejected,
            elapsed_ms: summary.elapsed_ms,
            true_pos: accuracy.map(|a| a.true_pos),
            false_pos: accuracy.map(|a| a.false_pos),
            false_neg: accuracy.map(|a| a.false_neg),
            precision: accuracy.map(|a| a.precision),
            recall: accuracy.map(|a| a.recall),
            f1: accuracy.map(|a| a.f1),
        };

        let mut wtr = csv::Writer::from_writer(file);
        wtr.serialize(&row)
            .map_err(|e| ZerError::Store(format!("CSV write error: {e}")))?;
        wtr.flush()
            .map_err(|e| ZerError::Store(format!("CSV flush error: {e}")))?;
        Ok(())
    }

    /// Write a scored-pairs CSV file sorted by score descending.
    ///
    /// The file is named `<run_id>_scored_pairs.csv` and contains two columns:
    /// `score` (f32) and `is_match` (0 or 1).  Separating this from the benchmark
    /// JSON keeps the JSON small and allows millions of rows without memory cost.
    pub fn write_scored_pairs_csv(&self, pairs: &[(f32, bool)]) -> Result<(), ZerError> {
        let path = self
            .out_dir
            .join(format!("{}_scored_pairs.csv", self.run_id));
        let file = File::create(&path)
            .map_err(|e| ZerError::Store(format!("cannot create scored pairs file: {e}")))?;
        let mut w = csv::Writer::from_writer(file);
        w.write_record(["score", "is_match"])
            .map_err(|e| ZerError::Store(format!("CSV write error: {e}")))?;
        let mut sorted: Vec<(f32, bool)> = pairs.to_vec();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (score, is_match) in &sorted {
            w.write_record(&[score.to_string(), (*is_match as u8).to_string()])
                .map_err(|e| ZerError::Store(format!("CSV write error: {e}")))?;
        }
        w.flush()
            .map_err(|e| ZerError::Store(format!("CSV flush error: {e}")))?;
        Ok(())
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }
}

/// Convert a `MatchBand` to a bool for the `predicted_match` column.
pub fn band_to_match(band: MatchBand) -> bool {
    matches!(band, MatchBand::AutoMatch)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_summary(_dir: &TempDir) -> BenchBatchSummary {
        BenchBatchSummary {
            total_records: 100,
            candidate_pairs: 500,
            auto_matched: 400,
            borderline: 50,
            auto_rejected: 50,
            elapsed_ms: 1200,
            link_mode: "deduplicate".into(),
            dataset: "test_dataset".into(),
        }
    }

    #[test]
    fn write_pairs_ndjson_line_count() {
        let dir = TempDir::new().unwrap();
        let writer = BenchResultWriter::new(dir.path(), "test_run").unwrap();

        let pairs: Vec<PairRecord> = (0..5)
            .map(|i| PairRecord {
                run_id: "test_run".into(),
                record_key_a: i.to_string(),
                source_a: Some("brp".into()),
                record_key_b: (i + 100).to_string(),
                source_b: Some("kvk".into()),
                match_probability: 0.9,
                predicted_match: true,
            })
            .collect();

        writer.write_pairs(&pairs).unwrap();

        let path = dir.path().join("test_run_pairs.ndjson");
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 5, "NDJSON file must have exactly N lines");

        // Each line must be valid JSON
        for line in &lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v.get("run_id").is_some());
            assert!(v.get("match_probability").is_some());
        }
    }

    #[test]
    fn write_summary_csv_no_accuracy() {
        let dir = TempDir::new().unwrap();
        let writer = BenchResultWriter::new(dir.path(), "run_no_acc").unwrap();
        let summary = sample_summary(&dir);

        writer.write_summary(&summary, None).unwrap();

        let path = dir.path().join("run_no_acc_summary.csv");
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("zer"), "library field must be 'zer'");
        assert!(content.contains("test_dataset"));
        assert!(content.contains("100")); // total_records
    }

    #[test]
    fn write_summary_csv_with_accuracy() {
        let dir = TempDir::new().unwrap();
        let writer = BenchResultWriter::new(dir.path(), "run_acc").unwrap();
        let summary = sample_summary(&dir);
        let acc = AccuracyMetrics::from_counts(96, 4, 2);

        writer.write_summary(&summary, Some(&acc)).unwrap();

        let path = dir.path().join("run_acc_summary.csv");
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("96")); // true_pos
    }

    #[test]
    fn accuracy_metrics_from_counts() {
        let acc = AccuracyMetrics::from_counts(90, 10, 5);
        assert!((acc.precision - 0.9).abs() < 0.001);
        assert!((acc.recall - (90.0 / 95.0)).abs() < 0.001);
        assert!(acc.f1 > 0.0 && acc.f1 < 1.0);
    }

    #[test]
    fn accuracy_metrics_zero_denominator() {
        let acc = AccuracyMetrics::from_counts(0, 0, 0);
        assert_eq!(acc.precision, 0.0);
        assert_eq!(acc.recall, 0.0);
        assert_eq!(acc.f1, 0.0);
    }
}
