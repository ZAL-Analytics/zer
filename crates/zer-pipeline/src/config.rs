use std::path::PathBuf;

use zer_cluster::ClusterConfig;
use zer_core::field_mapping::FieldMapping;

/// How the pipeline started relative to its stored schema artifact.
///
/// Mirrors the variants of `zer_schema::StartupMode` but as a plain `Copy`
/// enum without any associated data, suitable for storing in [`crate::batch::BatchReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum BatchStartupMode {
    /// No prior artifact existed; EM ran from scratch with default priors.
    ColdStart,
    /// An artifact with an identical schema fingerprint was found; EM was skipped.
    WarmLoad,
    /// An artifact with a similar (but not identical) schema was found; EM ran
    /// for a few refinement iterations.
    WarmStart,
}

impl std::fmt::Display for BatchStartupMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColdStart => write!(f, "ColdStart"),
            Self::WarmLoad => write!(f, "WarmLoad"),
            Self::WarmStart => write!(f, "WarmStart"),
        }
    }
}

/// Controls which record pairs the pipeline generates candidates for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LinkMode {
    /// Find duplicates within a single dataset.  All candidate pairs allowed.
    #[default]
    Deduplicate,

    /// Link records across datasets.  Only pairs where the source labels differ
    /// are generated.  Within-source pairs are never compared or scored.
    ///
    /// Use when you have two or more curated datasets (e.g. BRP + KvK) and want
    /// to find cross-source matches without disturbing each source's internal
    /// integrity.
    LinkOnly,

    /// Simultaneously deduplicate within each source and link across sources.
    /// All candidate pairs are generated regardless of source label.
    LinkAndDedupe,
}

impl LinkMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deduplicate => "deduplicate",
            Self::LinkOnly => "link-only",
            Self::LinkAndDedupe => "link-and-dedupe",
        }
    }
}

/// Rate-adaptive threshold configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RateConfig {
    /// Records/sec below which processing is fully synchronous.
    pub slow_threshold: f32,
    /// Records/sec above which the auto-match threshold is widened.
    pub fast_threshold: f32,
    /// Threshold divisor applied to `upper_threshold` during bulk load.
    pub bulk_threshold_multiplier: f32,
}

impl Default for RateConfig {
    fn default() -> Self {
        Self {
            slow_threshold: 1.0,
            fast_threshold: 100.0,
            bulk_threshold_multiplier: 1.05,
        }
    }
}

/// All tunable parameters for a [`crate::pipeline::Pipeline`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineConfig {
    /// Path to the `.zsm` schema-registry file.
    pub registry_path: PathBuf,

    /// Maximum EM iterations for a cold start (no prior artifact).
    pub em_max_iter_cold: usize,

    /// Maximum EM iterations for a warm start (similar prior artifact).
    pub em_max_iter_warm: usize,

    /// Clustering shape parameters.
    pub cluster_config: ClusterConfig,

    /// Minimum pair-count to prefer GPU comparison (when available).
    pub gpu_min_batch: usize,

    /// Rate-adaptive threshold tuning.
    pub rate_config: RateConfig,

    /// Override the EM-estimated Fellegi-Sunter upper (auto-match) threshold.
    ///
    /// When `Some(t)`, pairs with a match probability ≥ `t` are auto-matched
    /// regardless of what EM produces.  Use to tighten precision on high-stakes
    /// pipelines or to force a specific operating point for benchmarking.
    /// `None` (default) defers entirely to the EM estimate.
    #[serde(default)]
    pub upper_threshold: Option<f32>,

    /// Override the EM-estimated Fellegi-Sunter lower (auto-reject) threshold.
    ///
    /// When `Some(t)`, pairs with a match probability ≤ `t` are auto-rejected.
    /// `None` (default) defers entirely to the EM estimate.
    #[serde(default)]
    pub lower_threshold: Option<f32>,

    /// Controls which record pairs are generated during blocking.
    ///
    /// `Deduplicate` (default) generates all candidate pairs.
    /// `LinkOnly` skips pairs where both records share the same source label.
    /// `LinkAndDedupe` is identical to `Deduplicate` at the pair-generation
    /// level but is reported differently in `BatchReport`.
    #[serde(default)]
    pub link_mode: LinkMode,

    /// Maximum number of records allowed in a blocking bucket before it is
    /// skipped during candidate-pair generation.
    ///
    /// Buckets larger than this threshold have poor selectivity (e.g. a common
    /// birth year-month) and produce O(n²) spurious pairs that exhaust memory.
    /// Setting to `0` disables the cap entirely (not recommended for datasets
    /// larger than a few thousand records).
    ///
    /// Default: 300.
    #[serde(default = "default_max_bucket_size")]
    pub max_bucket_size: usize,

    /// Explicit field-to-field mappings for cross-schema linkage.
    ///
    /// When non-empty the pipeline uses `FieldComparator::compare_pair_mapped`
    /// and `CompositeBlocker` source remaps instead of the standard single-
    /// schema path.  Leave empty (default) for same-schema dedupe/link runs.
    #[serde(default)]
    pub field_mappings: Vec<FieldMapping>,
}

const DEFAULT_MAX_BUCKET_SIZE: usize = 300;

fn default_max_bucket_size() -> usize {
    DEFAULT_MAX_BUCKET_SIZE
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            registry_path: PathBuf::from("schema.zsm"),
            em_max_iter_cold: 25,
            em_max_iter_warm: 3,
            cluster_config: ClusterConfig::default(),
            gpu_min_batch: 1_000,
            rate_config: RateConfig::default(),
            upper_threshold: None,
            lower_threshold: None,
            link_mode: LinkMode::Deduplicate,
            max_bucket_size: DEFAULT_MAX_BUCKET_SIZE,
            field_mappings: Vec::new(),
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.em_max_iter_cold, 25);
        assert_eq!(cfg.em_max_iter_warm, 3);
        assert_eq!(cfg.gpu_min_batch, 1_000);
    }

    #[test]
    fn default_threshold_overrides_are_none() {
        let cfg = PipelineConfig::default();
        assert!(
            cfg.upper_threshold.is_none(),
            "upper_threshold must default to None"
        );
        assert!(
            cfg.lower_threshold.is_none(),
            "lower_threshold must default to None"
        );
    }

    #[test]
    fn threshold_overrides_round_trip_json() {
        let cfg = PipelineConfig {
            upper_threshold: Some(0.92),
            lower_threshold: Some(0.08),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: PipelineConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.upper_threshold, Some(0.92));
        assert_eq!(back.lower_threshold, Some(0.08));
    }

    #[test]
    fn threshold_override_none_round_trips_from_json_without_field() {
        // Old configs without the field must deserialize to None (serde default).
        let json = r#"{"registry_path":"schema.zsm","em_max_iter_cold":25,"em_max_iter_warm":3,"cluster_config":{"max_cluster_size":50,"within_cluster_min":0.85},"gpu_min_batch":1000,"rate_config":{"slow_threshold":1.0,"fast_threshold":100.0,"bulk_threshold_multiplier":1.05}}"#;
        let cfg: PipelineConfig = serde_json::from_str(json).expect("deserialize");
        assert!(cfg.upper_threshold.is_none());
        assert!(cfg.lower_threshold.is_none());
        // link_mode must default to Deduplicate when absent from old configs
        assert_eq!(cfg.link_mode, LinkMode::Deduplicate);
        // max_bucket_size must default to 300 when absent from old configs
        assert_eq!(cfg.max_bucket_size, 300);
        // field_mappings must default to empty
        assert!(cfg.field_mappings.is_empty());
    }

    #[test]
    fn max_bucket_size_default_is_300() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.max_bucket_size, 300);
    }

    #[test]
    fn max_bucket_size_round_trips_json() {
        let cfg = PipelineConfig {
            max_bucket_size: 500,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: PipelineConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.max_bucket_size, 500);
    }

    #[test]
    fn link_mode_default_is_deduplicate() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.link_mode, LinkMode::Deduplicate);
    }

    #[test]
    fn link_mode_round_trips_json() {
        let cfg = PipelineConfig {
            link_mode: LinkMode::LinkOnly,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: PipelineConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.link_mode, LinkMode::LinkOnly);
    }

    #[test]
    fn link_mode_link_and_dedupe_round_trips_json() {
        let cfg = PipelineConfig {
            link_mode: LinkMode::LinkAndDedupe,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: PipelineConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.link_mode, LinkMode::LinkAndDedupe);
    }

    #[test]
    fn default_rate_config_thresholds_ordered() {
        let r = RateConfig::default();
        assert!(r.slow_threshold < r.fast_threshold);
        assert!(r.bulk_threshold_multiplier > 1.0);
    }

    #[test]
    fn pipeline_config_roundtrip_json() {
        let cfg = PipelineConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: PipelineConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg.em_max_iter_cold, back.em_max_iter_cold);
        assert_eq!(cfg.em_max_iter_warm, back.em_max_iter_warm);
        assert_eq!(
            cfg.rate_config.fast_threshold,
            back.rate_config.fast_threshold
        );
    }

    #[test]
    fn cluster_config_default_reasonable() {
        let cfg = PipelineConfig::default();
        assert!(cfg.cluster_config.max_cluster_size > 0);
        assert!(cfg.cluster_config.within_cluster_min > 0.0);
        assert!(cfg.cluster_config.within_cluster_min < 1.0);
    }
}
