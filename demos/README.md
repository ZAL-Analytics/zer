# Zer Demos

These self-contained demos walk you through the key capabilities of the `zer` entity-resolution framework. Work through them in the order listed below, each demo builds on concepts introduced by the previous ones.

All demos read their data from `data/v1.1/demos/` (git-ignored). Generate the datasets first; see the [dataset generation guide](https://docs.zal-analytics.ch/zer/contribution/datasets.html) for prerequisites (`data/base/` must be present) and full instructions.

Then run any demo from the repo root:

```bash
cargo run -p <package-name>
```

---

## 1. `hello_backend`, Minimal pipeline sanity check

**Package:** `hello-backend`

The smallest possible zer program: a handful of hard-coded records fed through a `Deduplicate` pipeline. No CSV, no schema complexity. Read this first to understand how `Pipeline`, `Record`, and `BatchReport` fit together.

```bash
cargo run -p hello-backend
```

---

## 2. `person_deduplication`, Single-source deduplication

**Package:** `person-deduplication`  
**Data:** `data/v1.1/demos/persons/`

Deduplicates a synthetic Dutch population register (`~1 000 records`) where roughly 10 % of persons appear more than once with name variants or address changes. Shows how blocking, comparison, and scoring produce deduplicated entity clusters, and prints precision / recall / F1 against a ground-truth file.

```bash
cargo run -p person-deduplication
```

---

## 3. `cross_source_linkage`, Two-source record linkage

**Package:** `cross-source-linkage`  
**Data:** `data/v1.1/demos/linkage/`

Links a municipal register (source A) to a downstream benefits system (source B). The two sources share ~40 % of persons but use different data-quality profiles (name variants, address lag, occasional DOB drift). Uses `LinkMode::LinkOnly`, only cross-source candidate pairs are considered. Demonstrates ID-offset strategy, `linked_pairs()`, and evaluation against ground truth.

```bash
cargo run -p cross-source-linkage
```

---

## 4. `multi_source_linkage`, LinkOnly vs LinkAndDedupe, side by side

**Package:** `multi-source-linkage`  
**Data:** `data/v1.1/demos/multi_source/`

Runs the **same** BRP + KvK dataset through two pipeline modes and compares the results:

- **LinkOnly**, finds cross-source matches between the municipal register and company director extract only.
- **LinkAndDedupe**, simultaneously deduplicates each source *and* links across sources.

Ground truth covers both `cross_source` and `within_source` match types. The final section prints a side-by-side metric table so you can see exactly what changes between the two modes.

```bash
cargo run -p multi-source-linkage
```

---

## 5. `blocking_explorer`, Inspect the blocking layer

**Package:** `blocking-explorer`

Visualises how the blocking step works: which blocking keys are generated for a record, how many candidates each key produces, and what the bucket-size distribution looks like. Use this to understand why certain pairs are or are not considered as candidates.

```bash
cargo run -p blocking-explorer
```

---

## 6. `scoring_walkthrough`, Field-by-field comparison vectors

**Package:** `scoring-walkthrough`

Steps through a small set of record pairs and prints the raw comparison vector (one score per field) alongside the final Fellegi-Sunter match probability. Useful for understanding how individual field similarities combine into an overall score.

```bash
cargo run -p scoring-walkthrough
```

---

## 7. `custom_components`, Plugging in custom logic

**Package:** `custom-components`

Shows how to register a custom comparator or blocking key outside the built-in set. The entry point for users who need domain-specific matching logic (e.g., structured identifiers, non-Latin scripts, specialised date formats).

```bash
cargo run -p custom-components
```

---

## Suggested reading order

| Step | Demo | Why |
|------|------|-----|
| 1 | `hello_backend` | Understand the basic API surface |
| 2 | `person_deduplication` | See a real deduplication workflow end-to-end |
| 3 | `cross_source_linkage` | Add a second source and cross-source evaluation |
| 4 | `multi_source_linkage` | Compare pipeline modes on the same data |
| 5 | `blocking_explorer` | Dig into the blocking layer if results are unexpected |
| 6 | `scoring_walkthrough` | Understand how scores are built from field comparisons |
| 7 | `custom_components` | Extend zer with your own logic |
