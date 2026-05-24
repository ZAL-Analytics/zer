# zer-cluster

Connected-components clustering and entity storage for the [zer](https://crates.io/crates/zer) entity-resolution library.

After the comparison and scoring phase produces a set of match/non-match decisions, this crate groups the matched record pairs into entity clusters using a connected-components algorithm, and persists the resolved entities to a SQLite store.

## What it provides

| Item | Description |
|------|-------------|
| `ConnectedComponentsClusterer` | Groups matched pairs into entity clusters |
| `ClusterGraph` / `ClusterConfig` | Petgraph-based pair graph with configurable match threshold |
| `ZalEntityStore` | SQLite-backed persistent store for resolved entity clusters |
| `ResolutionEvent` | Provenance record: which records were merged, when, and why |
| `partition_by_band` / `BandedPairs` | LSH-style banded partitioning for large pair sets |

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
