/// Bayesian odds-space calibration table.
///
/// Converts raw Fellegi-Sunter `match_probability` values into calibrated
/// posteriors by applying likelihood-ratio updates from judge verdicts.
///
/// The update rule in odds space:
/// ```text
/// posterior_odds = prior_odds  times  lr
/// ```
/// where `prior_odds = p / (1 - p)` and `p = match_probability`.

use zer_core::traits::JudgeVerdict;

/// Per-decision likelihood ratios applied by the judge.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CalibrationTable {
    /// LR applied when the judge says `IncreaseConfidence` (match).
    pub lr_increase:   f64,
    /// LR applied when the judge says `DecreaseConfidence` (non-match).
    pub lr_decrease:   f64,
    /// LR applied when the judge says `NoChange`.
    pub lr_no_change:  f64,
}

impl Default for CalibrationTable {
    fn default() -> Self {
        Self {
            lr_increase:  4.5,
            lr_decrease:  0.12,
            lr_no_change: 1.0,
        }
    }
}

impl CalibrationTable {
    pub fn new(lr_increase: f64, lr_decrease: f64, lr_no_change: f64) -> Self {
        Self { lr_increase, lr_decrease, lr_no_change }
    }

    /// Apply `IncreaseConfidence` update to a raw probability.
    pub fn update_increase(&self, p: f32) -> f32 {
        self.apply(p, self.lr_increase)
    }

    /// Apply `DecreaseConfidence` update to a raw probability.
    pub fn update_decrease(&self, p: f32) -> f32 {
        self.apply(p, self.lr_decrease)
    }

    /// Apply `NoChange` update (identity for `lr_no_change = 1.0`).
    pub fn update_no_change(&self, p: f32) -> f32 {
        self.apply(p, self.lr_no_change)
    }

    /// Dispatch to the correct update function based on a judge verdict.
    pub fn update_probability(&self, prior_prob: f32, verdict: &JudgeVerdict) -> f32 {
        match verdict {
            JudgeVerdict::IncreaseConfidence => self.update_increase(prior_prob),
            JudgeVerdict::DecreaseConfidence => self.update_decrease(prior_prob),
            JudgeVerdict::NoChange           => self.update_no_change(prior_prob),
        }
    }

    fn apply(&self, p: f32, lr: f64) -> f32 {
        let p = p.clamp(1e-6, 1.0 - 1e-6) as f64;
        let prior_odds  = p / (1.0 - p);
        let post_odds   = prior_odds * lr;
        let post_p      = post_odds / (1.0 + post_odds);
        post_p.clamp(0.0, 1.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increase_moves_probability_up() {
        let cal = CalibrationTable::default();
        let p_before = 0.60_f32;
        let p_after  = cal.update_increase(p_before);
        assert!(p_after > p_before, "increase should raise probability: {p_after} <= {p_before}");
        assert!(p_after <= 1.0);
    }

    #[test]
    fn decrease_moves_probability_down() {
        let cal = CalibrationTable::default();
        let p_before = 0.60_f32;
        let p_after  = cal.update_decrease(p_before);
        assert!(p_after < p_before, "decrease should lower probability: {p_after} >= {p_before}");
        assert!(p_after >= 0.0);
    }

    #[test]
    fn no_change_with_unit_lr_is_identity() {
        let cal = CalibrationTable::default();
        let p = 0.75_f32;
        let p_after = cal.update_no_change(p);
        assert!((p_after - p).abs() < 1e-4, "no_change should be near-identity: {p_after} != {p}");
    }

    #[test]
    fn extreme_probabilities_are_clamped() {
        let cal = CalibrationTable::default();
        // Very high probability shouldn't overflow to exactly 1.0
        let p_high = cal.update_increase(0.9999_f32);
        assert!(p_high < 1.0);
        // Very low probability shouldn't underflow to exactly 0.0
        let p_low = cal.update_decrease(0.0001_f32);
        assert!(p_low > 0.0);
    }

    #[test]
    fn update_probability_dispatches_correctly() {
        let cal = CalibrationTable::default();
        let p = 0.6_f32;
        assert_eq!(cal.update_probability(p, &JudgeVerdict::IncreaseConfidence), cal.update_increase(p));
        assert_eq!(cal.update_probability(p, &JudgeVerdict::DecreaseConfidence), cal.update_decrease(p));
        assert_eq!(cal.update_probability(p, &JudgeVerdict::NoChange),           cal.update_no_change(p));
    }

    #[test]
    fn update_probability_increase_raises() {
        let cal = CalibrationTable::default();
        let p = 0.5_f32;
        assert!(cal.update_probability(p, &JudgeVerdict::IncreaseConfidence) > p);
    }

    #[test]
    fn update_probability_decrease_lowers() {
        let cal = CalibrationTable::default();
        let p = 0.5_f32;
        assert!(cal.update_probability(p, &JudgeVerdict::DecreaseConfidence) < p);
    }
}
