# zer

**Zero-shot probabilistic entity resolution.** Given multiple datasets with records about the same people, vehicles, or organisations, but no shared unique key and noisy data, zer finds which records belong together.

[![crates.io](https://img.shields.io/crates/v/zer.svg)](https://crates.io/crates/zer)
[![docs](https://img.shields.io/badge/docs-docs.zal--analytics.ch-blue)](https://docs.zal-analytics.ch/zer)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

---

## What is zer?

Entity resolution (also called record linkage or deduplication) is the problem of deciding that two records refer to the same real-world entity even when there is no shared identifier. The same person might appear in multiple registries under slightly different names, with different address formats, and with OCR errors throughout.

zer solves this with a six-stage pipeline:
- **Schema → Blocker → Comparator → Scorer → Clusterer → Entity Store**

Every stage is pluggable: swap in a custom blocker, similarity function, comparator, or storage backend. zer ships with built-in support for Dutch administrative data (BRP, KvK, SIS II, ANPR), phonetic name encoding, tussenvoegsel normalisation, and licence-plate variants, but works with any domain.

Fellegi-Sunter probabilistic scoring drives match decisions, with an optional neural cross-encoder judge for borderline pairs. No labelled training data is required; EM parameter estimation runs unsupervised from your data and is cached in a `.zsm` model file for incremental updates.

---

## Quick start

Add to `Cargo.toml`:

```toml
[dependencies]
zer = { version = "1.1", features = ["pipeline"] }
```

```rust
use zer_core::{
    record::{FieldValue, Record},
    schema::{FieldKind, SchemaBuilder},
};
use zer_cluster::ZalEntityStore;
use zer_pipeline::{
    config::{LinkMode, PipelineConfig},
    pipeline::Pipeline,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let schema = SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("postcode",      FieldKind::Id)
        .build()?;

    let pipeline = Pipeline::builder()
        .schema(schema)
        .store(ZalEntityStore::open_in_memory()?)
        .config(PipelineConfig {
            registry_path: "model.zsm".into(),
            link_mode: LinkMode::Dedupe,
            ..PipelineConfig::default()
        })
        .build()?;

    let records: Vec<Record> = vec![
        Record::from_key("brp", "893479421")
            .insert("voornamen",     FieldValue::Text("Jan".into()))
            .insert("achternaam",    FieldValue::Text("de Vries".into()))
            .insert("geboortedatum", FieldValue::Text("1985-03-15".into()))
            .insert("postcode",      FieldValue::Text("1011AB".into())),
        Record::from_key("brp", "891234567")
            .insert("voornamen",     FieldValue::Text("J.".into()))
            .insert("achternaam",    FieldValue::Text("De Vries".into()))
            .insert("geboortedatum", FieldValue::Text("1985-03-15".into()))
            .insert("postcode",      FieldValue::Text("1011 AB".into())),
    ];

    let report = pipeline.run_batch(records).await?;
    println!("entities created: {}", report.entities_created);

    for (entity, members) in &pipeline.cluster_view() {
        println!("entity {},{} records", entity.id, members.len());
    }

    Ok(())
}
```

The `LinkMode` controls what pairs are considered:

| Mode | Behaviour |
|---|---|
| `Dedupe` | Within-source deduplication only |
| `LinkOnly` | Cross-source pairs only |
| `LinkAndDedupe` | Both within-source and cross-source |

---

## Feature flags

### Compute backend

| Feature | What it enables |
|---|---|
| `pipeline` | Full pipeline (`zer-pipeline`, `zer-cluster`, all core crates). Start here. |
| `cuda` | CUDA GPU acceleration for scoring. Requires CUDA Toolkit 13.1+ and an Ampere GPU (SM 8.6+). Falls back to CPU at runtime when no device is found. |
| `vulkan` | Vulkan compute backend. Requires Vulkan 1.3 driver and `slangc` on `PATH` at build time. |
| `avx2` | AVX2 SIMD backend for x86-64 CPU servers. ~4 times  throughput vs the generic CPU backend. No GPU required. |

### Neural judge

| Feature | What it enables |
|---|---|
| `judge_cpu` | DeBERTa/MiniLM NLI cross-encoder for borderline pairs, running on CPU via ONNX Runtime (downloaded at build time). |
| `judge_cuda` | Same judge on the CUDA ORT execution provider. Requires CUDA Toolkit 13.1+. |
| `judge_rocm` | Same judge on the ROCm ORT execution provider. |
| `judge_tensorrt` | TensorRT FP16 inference with engine caching. Requires TensorRT 8.0+ and CUDA. |

CPU-only builds (`pipeline` only) have no external dependencies beyond Rust itself.

---

## System requirements

Rust ≥ 1.75 is always required. GPU and neural judge features need CUDA Toolkit ≥ 13.1, Vulkan SDK ≥ 1.3, and/or TensorRT ≥ 8.0 depending on the flags selected. CPU-only builds (`pipeline` only) have no external dependencies.

See the [full installation guide](https://docs.zal-analytics.ch/zer/introduction/installation.html) for per-flag dependency tables, Linux package install commands, and GPU driver requirements.

---

## Neural judge models

The `judge_*` features require ONNX model files not bundled with the crate. Download from Hugging Face:

```bash
hf download arsalan-anwari/zjudge --local-dir ~/.cache/zer/models
```

Set `ZER_MODEL_DIR` to override the default search path. Full model setup and layout details are in the [installation guide](https://docs.zal-analytics.ch/zer/introduction/installation.html).

---

## Benchmarks

`zer-bench` is the standalone benchmark harness. Install it from crates.io:

```bash
cargo install zer-bench                   # CPU backend
cargo install zer-bench --features avx2   # x86-64 AVX2 SIMD
cargo install zer-bench --features cuda   # NVIDIA CUDA (requires CUDA Toolkit 13.1+)
cargo install zer-bench --features vulkan # Vulkan 1.3 compute
```

Download the benchmark datasets from HuggingFace and set `ZER_DATASET_DIR`:

```bash
hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
    --repo-type dataset --local-dir ~/datasets/zer
export ZER_DATASET_DIR=~/datasets/zer
```

| Variable | Default | Description |
|---|---|---|
| `ZER_DATASET_DIR` | `<workspace>/data` | Root directory for benchmark datasets downloaded from HuggingFace |
| `ZER_MODEL_DIR` | `~/.cache/zer/models` | Directory containing neural judge ONNX model files |
| `ZER_EXTERNAL_BENCHMARKS_DIR` | `<workspace>/benchmarks` | Root directory for external library benchmark scripts (`library` subcommand) |

Pass `--judge-target` to enable the neural judge and select its ONNX Runtime execution provider. Each target requires the matching feature flag at install time:

| `--judge-target` | Feature flag | Notes |
|---|---|---|
| `cpu` | *(none)* | Default |
| `cuda` | `judge_cuda` | NVIDIA CUDA |
| `tensorrt` | `judge_tensorrt` | TensorRT FP16, engine cached after first run |
| `rocm` | `judge_rocm` | AMD ROCm |
| `directml` | `judge_directml` | Windows DirectML |
| `openvino` | `judge_openvino` | Intel OpenVINO |

See the [benchmarks reference](https://docs.zal-analytics.ch/zer/reference/benchmarks.html) for throughput figures, accuracy tables, and full CLI documentation.

---

## Demos

The demos live in the repository and are not published to crates.io. Clone the repo to run them:

```bash
git clone https://github.com/ZAL-Analytics/zer
cd zer
```

Download the synthetic benchmark datasets (required by all demos):

```bash
hf download arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset \
    --repo-type dataset --local-dir data/
```

Generate demo-specific data and run (requires `data/base/`, see the [dataset generation guide](https://docs.zal-analytics.ch/zer/contribution/datasets.html) for setup steps):

```bash
./scripts/generate_data.sh --demos
cargo run -p hello-backend          # sanity check,no data needed
cargo run -p person-deduplication   # single-source dedup, ~1 000 records
cargo run -p cross-source-linkage   # two-source record linkage
cargo run -p multi-source-linkage   # LinkOnly vs LinkAndDedupe side-by-side
cargo run -p blocking-explorer      # inspect blocking keys and bucket sizes
cargo run -p scoring-walkthrough    # field-by-field comparison vectors
cargo run -p custom-components      # plug in a custom comparator or blocking key
```

GPU demos pass a feature flag:

```bash
cargo run -p person-deduplication --features cuda
```

See [demos/README.md](demos/README.md) for the recommended reading order and what each demo teaches. For full dataset generation prerequisites and options, see the [dataset generation guide](https://docs.zal-analytics.ch/zer/contribution/datasets.html).

> The datasets contain **synthetic records only**, generated from statistical distributions of Dutch administrative data. No real personal information is included.

---

## Documentation

Full documentation is at **[docs.zal-analytics.ch/zer](https://docs.zal-analytics.ch/zer)**, including:

- Installation guide (all feature flags, GPU drivers, ORT configuration)
- Tutorials (deduplication, cross-source linkage, custom components)
- Explanation (entity resolution theory, Fellegi-Sunter, EM estimation)
- How-to guides (GPU backend, neural judge, Polars/Arrow adapters, schema tuning)
- API reference

---

## License

Apache-2.0. See [LICENSE](LICENSE).
