//! Accuracy strategy for `brp_sis/link` (and `micro/brp_sis/link`).
//!
//! Same root cause as `brp_hks/link`: SIS records contain ~17 % initials in
//! `voornamen`, and the default `DateFragmentKey(YearMonth)` secondary key
//! floods the candidate set with high-FP pairs.  See `brp_hks_link.rs` for a
//! full analysis.
//!
//! Fix: replace the default blocker with `PhoneticNameDobInitialKey` only.

use super::{phonetic_name_dob_initial_blocker, ScenarioStrategy};

pub fn strategy() -> ScenarioStrategy {
    ScenarioStrategy {
        blocker_fn: Some(phonetic_name_dob_initial_blocker),
        ..ScenarioStrategy::default()
    }
}
