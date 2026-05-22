//! Accuracy strategy for `brp_hks/link`.
//!
//! Root cause of low F1: the default `BlockerFactory::from_schema` adds a
//! secondary `DateFragmentKey(YearMonth)` key that generates ~130 K candidates
//! for all records sharing a birth year-month.  HKS records contain ~11 %
//! initials in `voornamen`, so Jaro-Winkler on first names cannot distinguish
//! "J. Jansen" from any other "J. Jansen" born in the same month.  EM then
//! trains on a dataset that is ~98 % false pairs and becomes overconfident,
//! assigning maximum scores to spurious collisions.
//!
//! Fix: use only `PhoneticNameDobInitialKey` (soundex(achternaam) + first_initial
//! + birth_year).  This AND-style key requires all three components to agree,
//! eliminating same-month false pairs while retaining true matches (a full first
//! name's initial always agrees with an abbreviated initial).

use super::{phonetic_name_dob_initial_blocker, ScenarioStrategy};

pub fn strategy() -> ScenarioStrategy {
    ScenarioStrategy {
        blocker_fn: Some(phonetic_name_dob_initial_blocker),
        ..ScenarioStrategy::default()
    }
}
