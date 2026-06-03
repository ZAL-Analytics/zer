use crate::{comparison::ComparisonVector, record::RecordId};

/// Coarse classification of a scored pair based on match probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MatchBand {
    AutoMatch,
    Borderline,
    AutoReject,
}

/// Learned Fellegi-Sunter m/u parameters and classification thresholds for one schema.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelParams {
    pub m: Vec<Vec<f32>>,
    pub u: Vec<Vec<f32>>,
    pub log_prior_odds: f32,
    pub upper_threshold: f32,
    pub lower_threshold: f32,
}

/// A candidate pair annotated with its match weight, probability, and band.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoredPair {
    pub record_a: RecordId,
    pub record_b: RecordId,
    pub match_weight: f32,
    pub match_probability: f32,
    pub vector: ComparisonVector,
    pub band: MatchBand,
}
