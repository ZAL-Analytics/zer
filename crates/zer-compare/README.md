# zer-compare

Fellegi-Sunter field comparison, similarity functions, and EM scoring for the zer entity-resolution library.

This crate is the CPU implementation of the comparison and scoring pipeline. For GPU-accelerated equivalents see [`zer-compute`](https://crates.io/crates/zer-compute).

- **Documentation**: [docs.zal-analytics.nl](https://docs.zal-analytics.nl)
- **Website**: [www.zal-analytics.nl](https://www.zal-analytics.nl)
- **Support & feedback**: [info@zal-analytics.nl](mailto:info@zal-analytics.nl)

## What it provides

| Item | Description |
|------|-------------|
| `FieldComparator` | Compares record pairs field-by-field, producing comparison vectors |
| `FellegiSunterScorer` | Computes match probability from comparison vectors using trained m/u parameters |
| `SimilarityFn` | Trait for plugging in custom similarity functions |
| `JaroWinklerSimilarity` | Jaro-Winkler string distance, good for short names |
| `LevenshteinSimilarity` | Edit distance similarity |
| `PhoneticEqualitySimilarity` | Phonetic matching via Soundex / Double Metaphone |
| `TokenOverlapSimilarity` | Jaccard-based token overlap, good for addresses |
| `AddressTokenOverlap` | Dutch address-aware token comparison |
| `run_em` / `auto_calibrate_thresholds` | EM algorithm for unsupervised m/u parameter estimation |
| `LevelThresholds` | Discretises continuous similarity scores into comparison levels |

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
