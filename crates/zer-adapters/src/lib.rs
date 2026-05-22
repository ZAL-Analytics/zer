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

pub use bench_writer::{AccuracyMetrics, BenchBatchSummary, BenchResultWriter, PairRecord, band_to_match};
pub use time::{fmt_unix_secs, unix_secs_now, utc_timestamp_iso};

#[cfg(feature = "polars")]
pub mod polars;

#[cfg(feature = "arrow")]
pub mod arrow;

#[cfg(feature = "polars")]
pub use polars::PolarsIngest;

#[cfg(feature = "arrow")]
pub use arrow::ArrowIngest;
