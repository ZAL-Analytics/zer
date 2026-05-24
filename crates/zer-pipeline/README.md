# zer-pipeline

End-to-end entity resolution pipeline for the [zer](https://crates.io/crates/zer) library.

Ties together ingestion, blocking, comparison, scoring, and clustering into a single `Pipeline` type. Supports deduplication (single source), linkage (two sources), and combined link-and-dedupe modes, with optional Tokio-based async progress events.

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

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
