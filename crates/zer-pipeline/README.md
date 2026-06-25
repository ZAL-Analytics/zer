# zer-pipeline

End-to-end entity resolution pipeline orchestration for the zer library.

Ties together ingestion, blocking, comparison, scoring, and clustering into a single `Pipeline` type. Supports deduplication (single source), linkage (two or more sources), and combined link-and-dedupe modes, with optional Tokio-based async progress events.

- **Documentation**: [docs.zal-analytics.nl](https://docs.zal-analytics.nl)
- **Website**: [www.zal-analytics.nl](https://www.zal-analytics.nl)
- **Support & feedback**: [info@zal-analytics.nl](mailto:info@zal-analytics.nl)

## What it provides

| Item | Description |
|------|-------------|
| `Pipeline` / `PipelineBuilder` | Fluent builder and executor for the full ER pipeline |
| `Ingester` / `IngestResult` | Loads records from CSV or any `IntoRecord` source |
| `PipelineConfig` / `LinkMode` | Controls link vs. dedupe mode, batch size, score threshold |
| `PipelineEvent` | Async progress events (blocking done, comparison done, etc.) |
| `ClusterView` / `ClusterIter` | Iterates over resolved entity clusters |
| `BatchReport` | Per-batch statistics (pairs generated, matched, time) |

## Feature flags

| Flag            | Effect |
|-----------------|--------|
| `collect-pairs` | Keeps all scored pairs in memory after judging for PR-AUC analysis; incurs allocation cost proportional to candidate count |

## Usage

```rust
use zer::pipeline::{Pipeline, PipelineBuilder, PipelineConfig, LinkMode};

let pipeline = PipelineBuilder::new(schema)
    .mode(LinkMode::Dedupe)
    .config(PipelineConfig::default())
    .build(blocker, comparator, scorer, clusterer);

let report = pipeline.run(ingester).await?;
```

## Breaking changes

### v1.1

**`LinkedPair`, `record_id_a/b` replaced by `record_key_a/b`**

`LinkedPair` no longer exposes raw numeric record IDs. The fields `record_id_a: RecordId` and `record_id_b: RecordId` are replaced by `record_key_a: String` and `record_key_b: String`, which hold the natural key values (e.g. BSN, KvK number) as supplied via `DatasetConfig` at ingestion time.

```rust
// v1.0
println!("{} ↔ {}", pair.record_id_a, pair.record_id_b);

// v1.1
println!("{} ↔ {}", pair.record_key_a, pair.record_key_b);
```

Evaluation code that built `HashSet<(u64, u64)>` from ground-truth integer IDs must be updated to `HashSet<(String, String)>` using natural key pairs.

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
