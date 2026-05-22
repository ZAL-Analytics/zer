use zer_core::scoring::{MatchBand, ModelParams, ScoredPair};

/// Pairs partitioned by their match band.
pub struct BandedPairs {
    pub auto_match:  Vec<ScoredPair>,
    pub borderline:  Vec<ScoredPair>,
    pub auto_reject: Vec<ScoredPair>,
}

/// Classify each pair by `match_probability` vs the upper/lower thresholds in
/// `params`. A pair is `AutoMatch` if `prob >= upper_threshold`, `AutoReject`
/// if `prob < lower_threshold`, and `Borderline` otherwise.
///
/// The band already stored in `ScoredPair::band` is used directly, it must
/// have been assigned by the same `ModelParams` that are passed here. If the
/// stored band disagrees with the thresholds (e.g., params were updated after
/// scoring), the stored band takes precedence so that provenance is preserved.
pub fn partition_by_band(pairs: Vec<ScoredPair>, _params: &ModelParams) -> BandedPairs {
    let mut auto_match  = Vec::new();
    let mut borderline  = Vec::new();
    let mut auto_reject = Vec::new();

    for pair in pairs {
        match pair.band {
            MatchBand::AutoMatch  => auto_match.push(pair),
            MatchBand::Borderline => borderline.push(pair),
            MatchBand::AutoReject => auto_reject.push(pair),
        }
    }

    BandedPairs { auto_match, borderline, auto_reject }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{comparison::ComparisonVector, scoring::MatchBand};

    fn params() -> ModelParams {
        ModelParams {
            m: vec![],
            u: vec![],
            log_prior_odds: 0.0,
            upper_threshold: 0.8,
            lower_threshold: 0.2,
        }
    }

    fn pair(a: u64, b: u64, prob: f32, band: MatchBand) -> ScoredPair {
        ScoredPair {
            record_a:          a,
            record_b:          b,
            match_weight:      0.0,
            match_probability: prob,
            vector:            ComparisonVector { record_a: a, record_b: b, levels: vec![] },
            band,
        }
    }

    #[test]
    fn empty_input_returns_empty_partitions() {
        let result = partition_by_band(vec![], &params());
        assert!(result.auto_match.is_empty());
        assert!(result.borderline.is_empty());
        assert!(result.auto_reject.is_empty());
    }

    #[test]
    fn all_auto_match() {
        let pairs = vec![
            pair(1, 2, 0.95, MatchBand::AutoMatch),
            pair(3, 4, 0.90, MatchBand::AutoMatch),
        ];
        let result = partition_by_band(pairs, &params());
        assert_eq!(result.auto_match.len(), 2);
        assert!(result.borderline.is_empty());
        assert!(result.auto_reject.is_empty());
    }

    #[test]
    fn all_auto_reject() {
        let pairs = vec![
            pair(1, 2, 0.05, MatchBand::AutoReject),
            pair(3, 4, 0.10, MatchBand::AutoReject),
        ];
        let result = partition_by_band(pairs, &params());
        assert!(result.auto_match.is_empty());
        assert!(result.borderline.is_empty());
        assert_eq!(result.auto_reject.len(), 2);
    }

    #[test]
    fn mixed_bands() {
        let pairs = vec![
            pair(1, 2, 0.95, MatchBand::AutoMatch),
            pair(2, 3, 0.50, MatchBand::Borderline),
            pair(4, 5, 0.05, MatchBand::AutoReject),
        ];
        let result = partition_by_band(pairs, &params());
        assert_eq!(result.auto_match.len(), 1);
        assert_eq!(result.borderline.len(), 1);
        assert_eq!(result.auto_reject.len(), 1);
    }
}
