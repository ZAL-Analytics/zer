use zer_cluster::partition_by_band;
use zer_core::{comparison::ComparisonVector, scoring::{MatchBand, ModelParams, ScoredPair}};

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
fn empty_input_gives_empty_partitions() {
    let r = partition_by_band(vec![], &params());
    assert!(r.auto_match.is_empty());
    assert!(r.borderline.is_empty());
    assert!(r.auto_reject.is_empty());
}

#[test]
fn all_auto_match_goes_to_match_bucket() {
    let pairs = vec![
        pair(1, 2, 0.95, MatchBand::AutoMatch),
        pair(3, 4, 0.90, MatchBand::AutoMatch),
        pair(5, 6, 0.85, MatchBand::AutoMatch),
    ];
    let r = partition_by_band(pairs, &params());
    assert_eq!(r.auto_match.len(), 3);
    assert!(r.borderline.is_empty());
    assert!(r.auto_reject.is_empty());
}

#[test]
fn all_auto_reject_goes_to_reject_bucket() {
    let pairs = vec![
        pair(1, 2, 0.10, MatchBand::AutoReject),
        pair(3, 4, 0.05, MatchBand::AutoReject),
    ];
    let r = partition_by_band(pairs, &params());
    assert!(r.auto_match.is_empty());
    assert!(r.borderline.is_empty());
    assert_eq!(r.auto_reject.len(), 2);
}

#[test]
fn mixed_bands_routed_correctly() {
    let pairs = vec![
        pair(1, 2, 0.95, MatchBand::AutoMatch),
        pair(2, 3, 0.50, MatchBand::Borderline),
        pair(4, 5, 0.05, MatchBand::AutoReject),
        pair(6, 7, 0.91, MatchBand::AutoMatch),
        pair(8, 9, 0.30, MatchBand::Borderline),
    ];
    let r = partition_by_band(pairs, &params());
    assert_eq!(r.auto_match.len(), 2);
    assert_eq!(r.borderline.len(), 2);
    assert_eq!(r.auto_reject.len(), 1);
}

#[test]
fn borderline_pairs_excluded_from_clustering() {
    // Borderlines must NOT end up in auto_match, they require judge adjudication.
    let pairs = vec![
        pair(1, 2, 0.95, MatchBand::AutoMatch),
        pair(3, 4, 0.55, MatchBand::Borderline),
    ];
    let r = partition_by_band(pairs, &params());
    assert_eq!(r.auto_match.len(), 1);
    assert_eq!(r.auto_match[0].record_a, 1);
    assert_eq!(r.auto_match[0].record_b, 2);
}
