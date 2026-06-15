# zer-core

Core traits and types for the zer entity-resolution library.

This crate defines the fundamental building blocks that every other `zer-*` crate builds on top of. If you are using zer through the top-level `zer` crate you do not need to depend on this directly.

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

## Breaking changes

### v1.1

**`Record`, new `key` field and `from_key` constructor**

`Record` now carries a `key: String` alongside its internal numeric `id`. Use `Record::from_key(source, key)` when loading real data, it derives `id` deterministically via `FNV-1a(source:key)` and stores the natural key for output. `Record::new(id)` still exists for synthetic/test records and sets `key = id.to_string()`.

```rust
// v1.0
let r = Record::new(42).with_source("brp");

// v1.1
let r = Record::from_key("brp", "893479421");
```

**`derive_record_id(source, key) -> RecordId`** is now public if you need to pre-compute IDs.

**`EntityMember`, new `record_key` field**

`EntityMember` now has a `record_key: String` field (the natural key of the member record). Any code constructing `EntityMember` directly must supply it. Existing `.zes` entity stores from v1.0 are **not compatible**, regenerate or migrate them.

## License

Apache-2.0 · [GitHub](https://github.com/ZAL-Analytics/zer)
