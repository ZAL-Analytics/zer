# zer-core

Core traits and types for the zer entity-resolution library.

This crate defines the fundamental building blocks that every other `zer-*` crate builds on top of. If you are using zer through the top-level `zer-lib` crate you do not need to depend on this directly.

- **Documentation**: [docs.zal-analytics.ch](https://docs.zal-analytics.ch)
- **Website**: [www.zal-analytics.ch](https://www.zal-analytics.ch)
- **Support & feedback**: [info@zal-analytics.ch](mailto:info@zal-analytics.ch)

## What it provides

| Item | Description |
|------|-------------|
| `Record` / `IntoRecord` | The universal row type and conversion trait |
| `RecordStore` / `VecRecordStore` | In-memory storage for ingested records |
| `RecordPool` | Thread-safe pool for candidate pair buffering |
| `Schema` / `FieldKind` | Schema definition: field names, types, and null handling |
| `FieldMapping` / `NullPolicy` | Controls how source columns map to schema fields |
| `Comparator` / `Scorer` traits | Interfaces implemented by `zer-compare` and `zer-compute` |
| `ZerError` | Unified error type for the whole ecosystem |

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
