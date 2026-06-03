//! Per-scenario accuracy tuning strategies.
//!
//! Each scenario can register a custom [`ScenarioStrategy`] that overrides
//! `PipelineConfig` fields and/or supplies a custom blocker factory.
//! Call [`strategy_for`] with the `dataset_name` (e.g. `"brp_hks_link"`) to
//! get the right strategy; unknown names fall back to
//! [`ScenarioStrategy::default`] (all overrides are `None`).

mod brp_dedupe;
mod brp_hks_link;
mod brp_kvk_hks_link_and_dedupe;
mod brp_sis_link;

use zer::blocking::CustomSchemaCategory;
use zer::prelude::{BlockerFactory, CompositeBlocker, FieldComparator};
use zer_core::schema::Schema;
use zer_pipeline::config::PipelineConfig;

/// Tuning overrides for a single benchmark scenario.
///
/// All fields are `Option`, `None` means "use the pipeline's own default".
/// Add a new file under `strategies/` for each scenario that needs custom
/// blocking or config tweaks, then register it in [`strategy_for`].
pub struct ScenarioStrategy {
    /// Custom blocker factory.  When `None`, the pipeline falls back to
    /// `BlockerFactory::from_schema`.
    pub blocker_fn: Option<fn(&Schema) -> CompositeBlocker>,
    /// Custom comparator factory.  When `None`, the pipeline uses
    /// `Comparator::new(&schema, &backend)` with default similarity functions.
    pub comparator_fn: Option<fn(&Schema) -> FieldComparator>,
    pub em_max_iter_cold: Option<usize>,
    pub max_bucket_size: Option<usize>,
    pub upper_threshold: Option<f32>,
    pub lower_threshold: Option<f32>,
}

impl Default for ScenarioStrategy {
    fn default() -> Self {
        Self {
            blocker_fn: None,
            comparator_fn: None,
            em_max_iter_cold: None,
            max_bucket_size: None,
            upper_threshold: None,
            lower_threshold: None,
        }
    }
}

impl ScenarioStrategy {
    /// Apply config overrides to `cfg`, leaving fields untouched where `None`.
    pub fn apply_to_config(&self, mut cfg: PipelineConfig) -> PipelineConfig {
        if let Some(v) = self.em_max_iter_cold {
            cfg.em_max_iter_cold = v;
        }
        if let Some(v) = self.max_bucket_size {
            cfg.max_bucket_size = v;
        }
        if let Some(v) = self.upper_threshold {
            cfg.upper_threshold = Some(v);
        }
        if let Some(v) = self.lower_threshold {
            cfg.lower_threshold = Some(v);
        }
        cfg
    }
}

/// Shared blocker for scenarios where `voornamen` may contain initials.
///
/// Replaces the default `DateFragmentKey(YearMonth)` secondary key with
/// `PhoneticNameDobInitialKey` (soundex(achternaam) + first_initial + birth_year).
/// This eliminates same-month false pairs caused by initial-only first names
/// while retaining all true matches.
pub(crate) fn phonetic_name_dob_initial_blocker(schema: &Schema) -> CompositeBlocker {
    BlockerFactory::from_custom_category(
        schema,
        CustomSchemaCategory::new().with_phonetic_name_dob_initial(),
    )
}

/// Return the [`ScenarioStrategy`] for the given `dataset_name`.
///
/// To add a new per-scenario strategy:
/// 1. Create `strategies/<dataset_name>.rs` that exports `pub fn strategy() -> ScenarioStrategy`.
/// 2. Add a `mod <dataset_name>;` declaration above.
/// 3. Add a match arm below.
pub fn strategy_for(dataset_name: &str) -> ScenarioStrategy {
    match dataset_name {
        "brp_dedupe" | "micro_brp_dedupe" | "kvk_dedupe" => brp_dedupe::strategy(),
        "brp_hks_link" => brp_hks_link::strategy(),
        "brp_sis_link" | "micro_brp_sis_link" => brp_sis_link::strategy(),
        "brp_kvk_hks_link_and_dedupe" => brp_kvk_hks_link_and_dedupe::strategy(),
        _ => ScenarioStrategy::default(),
    }
}
