//! Accuracy strategy for `brp_kvk_hks/link_and_dedupe`.
//!
//! Three-source scenario: BRP (full names) × KvK (company-contact) × HKS
//! (criminal-history).  HKS records contain ~11 % initials in `voornamen`,
//! which causes the default `DateFragmentKey(YearMonth)` to generate excessive
//! false candidate pairs and miscalibrate EM, same root cause as `brp_hks/link`.
//!
//! Fix: use only `PhoneticNameDobInitialKey` (soundex(achternaam) +
//! first_initial + birth_year), eliminating same-month false pairs while
//! retaining true matches.

use super::{phonetic_name_dob_initial_blocker, ScenarioStrategy};

pub fn strategy() -> ScenarioStrategy {
    ScenarioStrategy {
        blocker_fn: Some(phonetic_name_dob_initial_blocker),
        ..ScenarioStrategy::default()
    }
}
