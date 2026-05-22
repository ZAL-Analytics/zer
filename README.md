# zer

**Zero-shot entity resolution for Dutch-centric data.** Given multiple datasets with records about the same people, vehicles, or organisations,but no shared unique key and noisy data,zer finds which records belong together.

[![crates.io](https://img.shields.io/crates/v/zer-lib.svg)](https://crates.io/crates/zer-lib)
[![docs](https://img.shields.io/badge/docs-docs.zal--analytics.ch-blue)](https://docs.zal-analytics.ch/zer)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

---

## What is zer?

Entity resolution (also called record linkage or deduplication) is the problem of deciding that two records refer to the same real-world entity even when there is no shared identifier. The same person might appear in a BRP register, a KvK extract, and a benefits system under slightly different names, with different address formats, and with OCR errors throughout.

zer solves this with a six-stage pipeline,**Schema → Blocker → Comparator → Scorer → Clusterer → Entity Store**,using Dutch-specific blocking keys (phonetic name encoding, tussenvoegsel normalisation, licence-plate variants), Fellegi-Sunter probabilistic scoring, and an optional neural cross-encoder judge for borderline pairs.

No labelled training data is required. The EM parameters are estimated from the data itself and cached in a `.zsm` model file for incremental updates.

---

## Quick start

Add to `Cargo.toml`:

```toml
[dependencies]
zer-lib = { version = "1.0", features = ["pipeline"] }
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
        Record::new(1)
            .insert("voornamen",     FieldValue::Text("Jan".into()))
            .insert("achternaam",    FieldValue::Text("de Vries".into()))
            .insert("geboortedatum", FieldValue::Text("1985-03-15".into()))
            .insert("postcode",      FieldValue::Text("1011AB".into())),
        Record::new(2)
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
| `avx2` | AVX2 SIMD backend for x86-64 CPU servers. ~4× throughput vs the generic CPU backend. No GPU required. |

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

| Requirement | Version | When needed |
|---|---|---|
| Rust (stable) | ≥ 1.75 | always |
| CUDA Toolkit | ≥ 13.1 | `cuda`, `judge_cuda`, `judge_tensorrt` |
| Vulkan SDK + `slangc` | ≥ 1.3 | `vulkan` |
| TensorRT | ≥ 8.0 | `judge_tensorrt` |
| Python | ≥ 3.10 | demo data generators only |

See the [full installation guide](https://docs.zal-analytics.ch/zer/introduction/installation.html) for GPU driver versions, ORT configuration, and per-flag build instructions.

---

## Neural judge models

The `judge_*` features require ONNX model files that are not bundled with the crate. Download them from Hugging Face:

```bash
# Using the huggingface_hub CLI (pip install huggingface_hub[cli])
hf download arsalan-anwari/zjudge --local-dir ~/.cache/zer/models

# Or with git-lfs
git lfs install
git clone https://huggingface.co/arsalan-anwari/zjudge ~/.cache/zer/models
```

Expected layout after download:

```
~/.cache/zer/models/
  zjudge/
    nli-base/
      base/        # FP32,CPU / CUDA
        model.onnx
        tokenizer.json
      fp16/        # FP16,GPU / TensorRT
        model.onnx
        tokenizer.json
```

Override the default path with `ZER_MODEL_DIR`:

```bash
export ZER_MODEL_DIR=/data/zer/models
```

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

Generate demo-specific data and run:

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

See [demos/README.md](demos/README.md) for the recommended reading order and what each demo teaches.

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
