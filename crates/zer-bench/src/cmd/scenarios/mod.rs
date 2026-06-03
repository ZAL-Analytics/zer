//! Scenario registry for zer-bench.
//!
//! Each `ScenarioSpec` describes a benchmark scenario: which CSV files to load,
//! which source labels to assign, what pipeline mode to use, where the ground-
//! truth file is, and (for cross-schema runs) where the mapping TOML lives.
//!
//! The flat list `ALL_SCENARIOS` is the single source of truth consumed by both
//! `accuracy` (runs the zer pipeline) and `library` (runs competitor scripts).

pub mod registry;

#[allow(unused_imports)]
pub use registry::{
    datasets_for_scenario, find_scenario, find_scenario_by_preset, full_size_scenarios,
    full_size_throughput_scenarios, throughput_scenarios, ScenarioSpec, SourceSpec, ALL_SCENARIOS,
};
