# zer-adapters

Type adapters for the zer entity-resolution library, bridging Polars DataFrames and Arrow RecordBatches to `zer-core` Records without a string round-trip.

- **Documentation**: [docs.zal-analytics.ch](https://docs.zal-analytics.ch)
- **Website**: [www.zal-analytics.ch](https://www.zal-analytics.ch)
- **Support & feedback**: [info@zal-analytics.ch](mailto:info@zal-analytics.ch)

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
