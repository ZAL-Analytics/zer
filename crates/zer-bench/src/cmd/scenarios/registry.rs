//! Static scenario registry.
//!
//! Scenarios are organised as `{domain}/{mode}`, e.g. `brp/dedupe` or
//! `brp_sis/link`.  Micro variants share the same domain but are stored under
//! `micro/{domain}/{mode}`.
//!
//! Field mappings for cross-schema scenarios are declared as const arrays of
//! [`FieldMappingDef`] and inlined directly in the registry, no external TOML
//! files are required.

use zer_core::field_mapping::NullPolicy;
use zer_pipeline::config::LinkMode;

// ── Types ─────────────────────────────────────────────────────────────────────

/// Const-compatible field mapping definition (inlined in the scenario registry).
/// Use [`ScenarioSpec::to_field_mappings`] to convert to [`zer_core::field_mapping::FieldMapping`].
#[derive(Debug, Clone)]
pub struct FieldMappingDef {
    pub a_field:     &'static str,
    pub b_field:     &'static str,
    pub null_policy: NullPolicy,
}

/// A single dataset source within a scenario.
#[derive(Debug, Clone)]
pub struct SourceSpec {
    /// Workspace-relative path to the CSV file.
    pub path:   &'static str,
    /// Source label assigned to every record loaded from this file.
    pub source: &'static str,
}

/// Full specification of one benchmark scenario.
#[derive(Debug, Clone)]
pub struct ScenarioSpec {
    /// Short slug used on the CLI: e.g. `"brp/dedupe"`, `"brp_sis/link"`.
    pub name:           &'static str,
    /// Human-readable description shown by `--list-scenarios`.
    pub description:    &'static str,
    /// Ordered list of source files.
    pub sources:        &'static [SourceSpec],
    /// Workspace-relative path to the ground-truth CSV.
    pub ground_truth:   &'static str,
    /// Pipeline link mode.
    pub mode:           LinkMode,
    /// Field mappings for cross-schema scenarios; empty for same-schema scenarios.
    pub field_mappings: &'static [FieldMappingDef],
    /// Short token used in output filenames / summary CSV `dataset` column.
    pub dataset_name:   &'static str,
    /// Tags for filtering and preset-name lookup (e.g. `"dedupe"`, `"micro"`).
    pub tags:           &'static [&'static str],
}

impl ScenarioSpec {
    /// Convert the static field-mapping definitions to owned [`zer_core::field_mapping::FieldMapping`] values.
    pub fn to_field_mappings(&self) -> Vec<zer_core::field_mapping::FieldMapping> {
        self.field_mappings.iter().map(|d| zer_core::field_mapping::FieldMapping {
            a_field:     d.a_field.to_owned(),
            b_field:     d.b_field.to_owned(),
            null_policy: d.null_policy.clone(),
        }).collect()
    }
}

// ── Cross-schema field mapping tables ─────────────────────────────────────────

const BRP_KVK_MAPPINGS: &[FieldMappingDef] = &[
    FieldMappingDef { a_field: "voornamen",     b_field: "voornamen",     null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "tussenvoegsel", b_field: "tussenvoegsel", null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "achternaam",    b_field: "achternaam",    null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboortedatum", b_field: "geboortedatum", null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "woonplaats",    b_field: "woonplaats",    null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "postcode",      b_field: "postcode",      null_policy: NullPolicy::Skip },
];

const BRP_SIS_MAPPINGS: &[FieldMappingDef] = &[
    FieldMappingDef { a_field: "voornamen",      b_field: "voornamen",      null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "achternaam",     b_field: "achternaam",     null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboortedatum",  b_field: "geboortedatum",  null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboorteplaats", b_field: "geboorteplaats", null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboorteland",   b_field: "geboorteland",   null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "nationaliteit",  b_field: "nationaliteit",  null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geslacht",       b_field: "geslacht",       null_policy: NullPolicy::Skip },
];

const BRP_HKS_MAPPINGS: &[FieldMappingDef] = &[
    FieldMappingDef { a_field: "voornamen",      b_field: "voornamen",      null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "tussenvoegsel",  b_field: "tussenvoegsel",  null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "achternaam",     b_field: "achternaam",     null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboortedatum",  b_field: "geboortedatum",  null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboorteplaats", b_field: "geboorteplaats", null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboorteland",   b_field: "geboorteland",   null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "nationaliteit",  b_field: "nationaliteit",  null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geslacht",       b_field: "geslacht",       null_policy: NullPolicy::Skip },
];

const BRP_KVK_HKS_MAPPINGS: &[FieldMappingDef] = &[
    FieldMappingDef { a_field: "voornamen",      b_field: "voornamen",      null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "tussenvoegsel",  b_field: "tussenvoegsel",  null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "achternaam",     b_field: "achternaam",     null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboortedatum",  b_field: "geboortedatum",  null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboorteplaats", b_field: "geboorteplaats", null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geboorteland",   b_field: "geboorteland",   null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "nationaliteit",  b_field: "nationaliteit",  null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "geslacht",       b_field: "geslacht",       null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "woonplaats",     b_field: "woonplaats",     null_policy: NullPolicy::Skip },
    FieldMappingDef { a_field: "postcode",       b_field: "postcode",       null_policy: NullPolicy::Skip },
];

// ── Full-size scenarios ───────────────────────────────────────────────────────

pub static ALL_SCENARIOS: &[ScenarioSpec] = &[
    // ── BRP single-source dedupe ───────────────────────────────────────────────
    ScenarioSpec {
        name:           "brp/dedupe",
        description:    "BRP single-source dedupe, ~11K records, ~1K true pairs",
        sources:        &[SourceSpec { path: "benchmarks/brp/dedupe/source.csv", source: "brp" }],
        ground_truth:   "benchmarks/brp/dedupe/ground_truth.csv",
        mode:           LinkMode::Deduplicate,
        field_mappings: &[],
        dataset_name:   "brp_dedupe",
        tags:           &["dedupe"],
    },
    // ── BRP cross-registry link ───────────────────────────────────────────────
    ScenarioSpec {
        name:           "brp/link",
        description:    "BRP cross-source linkage, two ~6K registries, ~2K cross-source pairs",
        sources:        &[
            SourceSpec { path: "benchmarks/brp/link/source_a.csv", source: "brp_a" },
            SourceSpec { path: "benchmarks/brp/link/source_b.csv", source: "brp_b" },
        ],
        ground_truth:   "benchmarks/brp/link/ground_truth.csv",
        mode:           LinkMode::LinkOnly,
        field_mappings: &[],
        dataset_name:   "brp_link",
        tags:           &["link"],
    },
    // ── BRP link + intra-source dedupe ────────────────────────────────────────
    ScenarioSpec {
        name:           "brp/link_and_dedupe",
        description:    "BRP combined linkage + dedupe, two sources with intra-source dups",
        sources:        &[
            SourceSpec { path: "benchmarks/brp/link_and_dedupe/source_a.csv", source: "brp_a" },
            SourceSpec { path: "benchmarks/brp/link_and_dedupe/source_b.csv", source: "brp_b" },
        ],
        ground_truth:   "benchmarks/brp/link_and_dedupe/ground_truth.csv",
        mode:           LinkMode::LinkAndDedupe,
        field_mappings: &[],
        dataset_name:   "brp_link_and_dedupe",
        tags:           &["link-and-dedupe"],
    },
    // ── BRP  times  KvK cross-schema link ──────────────────────────────────────────
    ScenarioSpec {
        name:           "brp_kvk/link",
        description:    "BRP  times  KvK cross-schema linkage, natural-person ↔ company-contact records",
        sources:        &[
            SourceSpec { path: "benchmarks/brp_kvk/link/source_brp.csv", source: "brp" },
            SourceSpec { path: "benchmarks/brp_kvk/link/source_kvk.csv", source: "kvk" },
        ],
        ground_truth:   "benchmarks/brp_kvk/link/ground_truth.csv",
        mode:           LinkMode::LinkOnly,
        field_mappings: BRP_KVK_MAPPINGS,
        dataset_name:   "brp_kvk_link",
        tags:           &["cross-schema", "kvk"],
    },
    // ── BRP  times  SIS cross-schema link ──────────────────────────────────────────
    ScenarioSpec {
        name:           "brp_sis/link",
        description:    "BRP  times  SIS cross-schema linkage, civil registry ↔ law-enforcement records",
        sources:        &[
            SourceSpec { path: "benchmarks/brp_sis/link/source_brp.csv", source: "brp" },
            SourceSpec { path: "benchmarks/brp_sis/link/source_sis.csv", source: "sis" },
        ],
        ground_truth:   "benchmarks/brp_sis/link/ground_truth.csv",
        mode:           LinkMode::LinkOnly,
        field_mappings: BRP_SIS_MAPPINGS,
        dataset_name:   "brp_sis_link",
        tags:           &["cross-schema", "sis"],
    },
    // ── BRP  times  HKS cross-schema link ──────────────────────────────────────────
    ScenarioSpec {
        name:           "brp_hks/link",
        description:    "BRP  times  HKS cross-schema linkage, civil registry ↔ criminal-history records",
        sources:        &[
            SourceSpec { path: "benchmarks/brp_hks/link/source_brp.csv", source: "brp" },
            SourceSpec { path: "benchmarks/brp_hks/link/source_hks.csv", source: "hks" },
        ],
        ground_truth:   "benchmarks/brp_hks/link/ground_truth.csv",
        mode:           LinkMode::LinkOnly,
        field_mappings: BRP_HKS_MAPPINGS,
        dataset_name:   "brp_hks_link",
        tags:           &["cross-schema", "hks"],
    },
    // ── BRP  times  KvK  times  HKS three-source link+dedupe ─────────────────────────────
    ScenarioSpec {
        name:           "brp_kvk_hks/link_and_dedupe",
        description:    "BRP  times  KvK  times  HKS three-source link+dedupe, cross-domain with intra-source dups",
        sources:        &[
            SourceSpec { path: "benchmarks/brp_kvk_hks/link_and_dedupe/source_brp.csv", source: "brp" },
            SourceSpec { path: "benchmarks/brp_kvk_hks/link_and_dedupe/source_kvk.csv", source: "kvk" },
            SourceSpec { path: "benchmarks/brp_kvk_hks/link_and_dedupe/source_hks.csv", source: "hks" },
        ],
        ground_truth:   "benchmarks/brp_kvk_hks/link_and_dedupe/ground_truth.csv",
        mode:           LinkMode::LinkAndDedupe,
        field_mappings: BRP_KVK_HKS_MAPPINGS,
        dataset_name:   "brp_kvk_hks_link_and_dedupe",
        tags:           &["cross-schema", "kvk", "hks"],
    },
    // ── KvK single-source dedupe ──────────────────────────────────────────────
    ScenarioSpec {
        name:           "kvk/dedupe",
        description:    "KvK single-source dedupe, company registry with duplicate contacts",
        sources:        &[SourceSpec { path: "benchmarks/kvk/dedupe/source.csv", source: "kvk" }],
        ground_truth:   "benchmarks/kvk/dedupe/ground_truth.csv",
        mode:           LinkMode::Deduplicate,
        field_mappings: &[],
        dataset_name:   "kvk_dedupe",
        tags:           &["kvk"],
    },

    // ── Micro variants (CI smoke tests) ──────────────────────────────────────
    ScenarioSpec {
        name:           "micro/brp/dedupe",
        description:    "BRP dedupe micro, ~1.1K records (CI smoke test)",
        sources:        &[SourceSpec { path: "benchmarks/micro/brp/dedupe/source.csv", source: "brp" }],
        ground_truth:   "benchmarks/micro/brp/dedupe/ground_truth.csv",
        mode:           LinkMode::Deduplicate,
        field_mappings: &[],
        dataset_name:   "micro_brp_dedupe",
        tags:           &["micro", "micro-dedupe"],
    },
    ScenarioSpec {
        name:           "micro/brp/link",
        description:    "BRP link micro, two ~600-record sources (CI smoke test)",
        sources:        &[
            SourceSpec { path: "benchmarks/micro/brp/link/source_a.csv", source: "brp_a" },
            SourceSpec { path: "benchmarks/micro/brp/link/source_b.csv", source: "brp_b" },
        ],
        ground_truth:   "benchmarks/micro/brp/link/ground_truth.csv",
        mode:           LinkMode::LinkOnly,
        field_mappings: &[],
        dataset_name:   "micro_brp_link",
        tags:           &["micro", "micro-link"],
    },
    ScenarioSpec {
        name:           "micro/brp/link_and_dedupe",
        description:    "BRP link+dedupe micro, two sources with intra-source dups (CI smoke test)",
        sources:        &[
            SourceSpec { path: "benchmarks/micro/brp/link_and_dedupe/source_a.csv", source: "brp_a" },
            SourceSpec { path: "benchmarks/micro/brp/link_and_dedupe/source_b.csv", source: "brp_b" },
        ],
        ground_truth:   "benchmarks/micro/brp/link_and_dedupe/ground_truth.csv",
        mode:           LinkMode::LinkAndDedupe,
        field_mappings: &[],
        dataset_name:   "micro_brp_link_and_dedupe",
        tags:           &["micro", "micro-link-and-dedupe"],
    },
    ScenarioSpec {
        name:           "micro/brp_sis/link",
        description:    "BRP  times  SIS link micro (CI smoke test)",
        sources:        &[
            SourceSpec { path: "benchmarks/micro/brp_sis/link/source_brp.csv", source: "brp" },
            SourceSpec { path: "benchmarks/micro/brp_sis/link/source_sis.csv", source: "sis" },
        ],
        ground_truth:   "benchmarks/micro/brp_sis/link/ground_truth.csv",
        mode:           LinkMode::LinkOnly,
        field_mappings: BRP_SIS_MAPPINGS,
        dataset_name:   "micro_brp_sis_link",
        tags:           &["micro", "cross-schema", "sis"],
    },
];

// ── Lookup helpers ────────────────────────────────────────────────────────────

pub fn find_scenario(name: &str) -> Option<&'static ScenarioSpec> {
    ALL_SCENARIOS.iter().find(|s| s.name == name)
}

/// Find a scenario by name or by tag (first match wins).
/// Used to implement the `--preset` CLI alias.
pub fn find_scenario_by_preset(tag: &str) -> Option<&'static ScenarioSpec> {
    ALL_SCENARIOS.iter().find(|s| s.name == tag || s.tags.contains(&tag))
}

/// Returns an iterator over scenarios eligible for throughput benchmarks
/// (deduplicate-mode only; link modes are not applicable to throughput).
pub fn throughput_scenarios() -> impl Iterator<Item = &'static ScenarioSpec> {
    ALL_SCENARIOS.iter().filter(|s| s.mode.as_str() == "deduplicate")
}

/// Full-size (non-micro) scenarios the 8 production-scale scenarios used by `--scenario=all`.
pub fn full_size_scenarios() -> impl Iterator<Item = &'static ScenarioSpec> {
    ALL_SCENARIOS.iter().filter(|s| !s.name.starts_with("micro/"))
}

/// Full-size throughput-eligible scenarios: non-micro dedupe scenarios only.
pub fn full_size_throughput_scenarios() -> impl Iterator<Item = &'static ScenarioSpec> {
    ALL_SCENARIOS.iter().filter(|s| !s.name.starts_with("micro/") && s.mode.as_str() == "deduplicate")
}

/// Returns `(dataset_paths, source_labels, ground_truth_path)` for a scenario,
/// all rooted at `workspace_root`.
pub fn datasets_for_scenario(
    spec: &ScenarioSpec,
    root: &std::path::Path,
) -> (Vec<String>, Vec<String>, String) {
    let datasets: Vec<String> = spec.sources.iter()
        .map(|s| root.join(s.path).to_string_lossy().into_owned())
        .collect();
    let sources: Vec<String> = spec.sources.iter()
        .map(|s| s.source.to_owned())
        .collect();
    let gt = root.join(spec.ground_truth).to_string_lossy().into_owned();
    (datasets, sources, gt)
}
