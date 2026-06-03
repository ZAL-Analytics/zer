//! Field comparison, similarity functions, and Fellegi-Sunter scoring.

pub mod comparator;
pub mod discretize;
pub mod em;
pub mod scoring;
pub mod similarity;

pub use comparator::FieldComparator;
pub use discretize::LevelThresholds;
pub use em::{auto_calibrate_thresholds, e_step, estimate_lambda, run_em};
pub use scoring::FellegiSunterScorer;
pub use similarity::address::{AddressTokenOverlap, StreetNumberEditDistance};
pub use similarity::name::{
    JaroWinklerSimilarity, LevenshteinSimilarity, PhoneticEqualitySimilarity,
    TokenOverlapSimilarity,
};
pub use similarity::NullSimilarity;
pub use similarity::SimilarityFn;
