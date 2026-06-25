# zer-cluster

Connected-components clustering and entity storage for the zer entity-resolution library.

After the comparison and scoring phase produces a set of match/non-match decisions, this crate groups the matched record pairs into entity clusters using a connected-components algorithm, and persists the resolved entities to a SQLite store.

- **Documentation**: [docs.zal-analytics.nl](https://docs.zal-analytics.nl)
- **Website**: [www.zal-analytics.nl](https://www.zal-analytics.nl)
- **Support & feedback**: [info@zal-analytics.nl](mailto:info@zal-analytics.nl)

## What it provides

| Item | Description |
|------|-------------|
| `ConnectedComponentsClusterer` | Groups matched pairs into entity clusters |
| `ClusterGraph` / `ClusterConfig` | Petgraph-based pair graph with configurable match threshold |
| `ZalEntityStore` | SQLite-backed persistent store for resolved entity clusters |
| `ResolutionEvent` | Provenance record: which records were merged, when, and why |
| `partition_by_band` / `BandedPairs` | LSH-style banded partitioning for large pair sets |

## Breaking changes

### v1.1

**`EntityMember`, new `record_key` field**

`EntityMember` now has a `record_key: String` field storing the record's natural key (the value from whichever column was nominated as the identity column when loading the dataset). Any code constructing `EntityMember` directly must supply it.

**`.zes` store schema changed, v1.0 stores are not compatible**

The `entity_members` table has a new `record_key TEXT NOT NULL` column. Existing `.zes` files created with v1.0 cannot be opened by v1.1, delete and regenerate them.

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
