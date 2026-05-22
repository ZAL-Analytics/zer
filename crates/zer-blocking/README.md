# zer-blocking

Blocking strategies and inverted index for candidate pair generation in the [zer](https://crates.io/crates/zer) entity-resolution library.

Blocking drastically reduces the number of record pairs that need to be compared by grouping records into blocks, only pairs within the same block are compared. This crate provides a composable set of blocking keys tailored to Dutch administrative data.

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

## Usage

```toml
[dependencies]
zer = { version = "1.0" }  # includes zer-blocking
# or directly:
zer-blocking = "0.1"
```

## Part of the zer ecosystem

[`zer`](https://crates.io/crates/zer) · [GitHub](https://github.com/ZAL-Analytics/zer)
