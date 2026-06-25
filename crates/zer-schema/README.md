# zer-schema

Schema inference and Fellegi-Sunter model registry for the zer entity-resolution library.

This crate handles two concerns: automatically detecting the schema of an incoming dataset, and persisting trained Fellegi-Sunter parameters so they can be reused across pipeline runs without re-running EM.

- **Documentation**: [docs.zal-analytics.nl](https://docs.zal-analytics.nl)
- **Website**: [www.zal-analytics.nl](https://www.zal-analytics.nl)
- **Support & feedback**: [info@zal-analytics.nl](mailto:info@zal-analytics.nl)

## What it provides

| Item | Description |
|------|-------------|
| `SchemaInferrer` | Detects `FieldKind` from column names and value patterns; no manual schema definition needed |
| `SchemaFingerprint` | Compact SHA-256 identity for a schema plus its data distribution |
| `SchemaRegistry` | `sled`-backed persistent store for trained `ModelArtifact`s |
| `ModelArtifact` | Serialisable container for m/u parameters, thresholds, and metadata |
| `StartupMode` | Decision enum: load exact match, warm-start EM, or run full EM from priors |

## Usage

```rust
use zer_schema::{SchemaInferrer, SchemaRegistry};

let schema   = SchemaInferrer::default().infer_from_csv_headers(&headers)?;
let registry = SchemaRegistry::open("./model-store")?;
let mode     = registry.lookup_startup_mode(&schema, &fingerprint)?;
```

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
