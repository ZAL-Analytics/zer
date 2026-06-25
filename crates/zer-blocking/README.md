# zer-blocking

Blocking strategies and inverted index for candidate pair generation in the zer entity-resolution library.

Blocking drastically reduces the number of record pairs that need to be compared by grouping records into blocks; only pairs within the same block are compared. This crate provides a composable, pluggable set of blocking keys, with built-in keys optimized for Dutch administrative data.

- **Documentation**: [docs.zal-analytics.nl](https://docs.zal-analytics.nl)
- **Website**: [www.zal-analytics.nl](https://www.zal-analytics.nl)
- **Support & feedback**: [info@zal-analytics.nl](mailto:info@zal-analytics.nl)

## What it provides

| Item | Description |
|------|-------------|
| `CompositeBlocker` | Combines multiple blocking keys; a pair is a candidate if it matches on any key |
| `InvertedIndex` | Efficient inverted index mapping blocking keys to record IDs |
| `BlockerFactory` | Builds a `CompositeBlocker` from a `Schema` automatically |
| `SchemaCategory` | Enum describing the dataset type (BRP, KvK, ANPR, …) |

### Built-in blocking keys

| Key | Description |
|-----|-------------|
| Phonetic | Soundex / Double Metaphone on name fields |
| Date | Exact-match on date fields (birth date, registration date) |
| Suffix | Last-N-characters of string fields |
| Alias | Normalised alias / nickname matching |
| Transliterated | ASCII transliteration of accented Dutch characters |
| Vehicle | Licence plate normalisation for ANPR data |

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
