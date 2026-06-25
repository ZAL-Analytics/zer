# zer-adapters

Type adapters for the zer entity-resolution library, bridging Polars DataFrames and Arrow RecordBatches to `zer-core` Records without a string round-trip.

- **Documentation**: [docs.zal-analytics.nl](https://docs.zal-analytics.nl)
- **Website**: [www.zal-analytics.nl](https://www.zal-analytics.nl)
- **Support & feedback**: [info@zal-analytics.nl](mailto:info@zal-analytics.nl)

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

## Breaking changes

### v1.1

**`into_records` signature changed, `DatasetConfig` replaces `id_start`**

Both `PolarsIngest::into_records` and `ArrowIngest::into_records` now take `&DatasetConfig` instead of an integer `id_start`. `DatasetConfig` names the source label and the column to use as each record's natural key; IDs are derived via `FNV-1a(source:key)`.

```rust
// v1.0
let records_a = df_a.into_records(1);
let records_b = df_b.into_records(n_a + 1);  // manual offset to avoid collisions

// v1.1
let records_a = df_a.into_records(&DatasetConfig::new("A", "bsn"));
let records_b = df_b.into_records(&DatasetConfig::new("B", "record_id"));
// no offset needed, source label is part of the hash
```

**`DatasetConfig`** is a new public struct in this crate.

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
