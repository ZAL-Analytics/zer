/// A no-op judge that unconditionally promotes every borderline pair.
///
/// Useful for testing pipeline wiring and verifying that all components work
/// without requiring an ONNX model on disk.  Suitable for CI, demos, and
/// development environments where a real model is not available.
///
/// # Example
///
/// ```rust
/// use zer_judge::DummyJudge;
/// use zer_core::{
///     comparison::ComparisonVector,
///     scoring::{MatchBand, ScoredPair},
///     traits::Judge,
/// };
///
/// let judge = DummyJudge;
/// let pairs = vec![ScoredPair {
///     record_a: 1,
///     record_b: 2,
///     match_weight: 0.0,
///     match_probability: 0.5,
///     vector: ComparisonVector::new(1, 2, vec![]),
///     band: MatchBand::Borderline,
/// }];
/// let verdicts = judge.adjudicate(&pairs).unwrap();
/// assert_eq!(verdicts.len(), 1);
/// ```
use zer_core::{
    scoring::ScoredPair,
    traits::{Judge, JudgeVerdict},
};

/// No-op judge: promotes every pair unconditionally.
///
/// Install this in place of [`crate::judge::DebertaJudge`] when you want to verify that your
/// pipeline is wired correctly before committing to a full ONNX deployment.
pub struct DummyJudge;

impl Judge for DummyJudge {
    fn adjudicate(
        &self,
        pairs: &[ScoredPair],
    ) -> zer_core::traits::Result<Vec<JudgeVerdict>> {
        Ok(vec![JudgeVerdict::IncreaseConfidence; pairs.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{comparison::ComparisonVector, scoring::MatchBand};

    #[test]
    fn empty_input_returns_empty() {
        let verdicts = DummyJudge.adjudicate(&[]).unwrap();
        assert!(verdicts.is_empty());
    }

    #[test]
    fn all_verdicts_are_increase_confidence() {
        let pairs = vec![
            ScoredPair {
                record_a: 1,
                record_b: 2,
                match_weight: 0.0,
                match_probability: 0.4,
                vector: ComparisonVector::new(1, 2, vec![]),
                band: MatchBand::Borderline,
            },
            ScoredPair {
                record_a: 3,
                record_b: 4,
                match_weight: 0.0,
                match_probability: 0.55,
                vector: ComparisonVector::new(3, 4, vec![]),
                band: MatchBand::Borderline,
            },
        ];
        let verdicts = DummyJudge.adjudicate(&pairs).unwrap();
        assert_eq!(verdicts.len(), 2);
        assert!(verdicts.iter().all(|v| matches!(v, JudgeVerdict::IncreaseConfidence)));
    }

    #[test]
    fn dummy_judge_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DummyJudge>();
    }
}
