# zer-adapters

Type adapters for the [zer](https://crates.io/crates/zer) entity-resolution library, bridging Polars DataFrames and Arrow RecordBatches to `zer-core` Records without a string round-trip.

## Feature flags

| Flag     | Adds |
|----------|------|
| `polars` | `PolarsIngest` extension trait for Polars `DataFrame` |
| `arrow`  | `ArrowIngest` extension trait for Arrow `RecordBatch` |

Enable only the features you need to keep compile times low.

## Usage

```rust
use zer_adapters::PolarsIngest;

// Convert a Polars DataFrame into zer Records directly.
let records = df.into_zer_records(&schema)?;
```

This crate also provides `BenchResultWriter`, `AccuracyMetrics`, and timestamp utilities used by `zer-bench` (internal tooling, not part of the public API).

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
