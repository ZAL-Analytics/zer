/// Type adapters that bridge external data-frame / record-batch libraries to
/// `zer_core::record::Record` without a string round-trip.
///
/// # Feature flags
///
/// | flag     | what it adds                                      |
/// |----------|---------------------------------------------------|
/// | `polars` | `PolarsIngest` extension trait for Polars DataFrames |
/// | `arrow`  | `ArrowIngest` extension trait for Arrow RecordBatches |
///
/// Enable only the features you need to keep compile times low.
pub mod bench_writer;
pub mod time;

pub use bench_writer::{
    band_to_match, AccuracyMetrics, BenchBatchSummary, BenchResultWriter, PairRecord,
};
pub use time::{fmt_unix_secs, unix_secs_now, utc_timestamp_iso};

#[cfg(feature = "polars")]
pub mod polars;

#[cfg(feature = "arrow")]
pub mod arrow;

#[cfg(feature = "polars")]
pub use polars::PolarsIngest;

#[cfg(feature = "arrow")]
pub use arrow::ArrowIngest;

/// Configuration for loading a dataset from an external source.
///
/// Specifies which column holds the record's natural key (e.g. BSN, UUID,
/// or any primary-key column) and what source label to attach.
///
/// The adapter uses `key_column` to extract the natural key from each row,
/// then derives a stable `RecordId` via `FNV-1a(source:key)`.  This removes
/// the need for users to maintain sequential integer offsets across datasets.
///
/// # Example
///
/// ```rust
/// use zer_adapters::DatasetConfig;
///
/// let cfg = DatasetConfig::new("brp", "bsn");
/// assert_eq!(cfg.source, "brp");
/// assert_eq!(cfg.key_column, "bsn");
/// ```
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Source label attached to every record (e.g. `"brp"`, `"kvk"`).
    pub source: String,
    /// Name of the column whose value is the record's natural key.
    pub key_column: String,
}

impl DatasetConfig {
    pub fn new(source: impl Into<String>, key_column: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            key_column: key_column.into(),
        }
    }
}
