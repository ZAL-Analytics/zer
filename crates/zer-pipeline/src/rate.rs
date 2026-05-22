use std::time::Instant;

use zer_core::scoring::ModelParams;

use crate::config::RateConfig;

/// Tracks ingestion rate and adjusts scoring thresholds during bulk loads.
pub struct RateAdapter {
    window_start: Instant,
    record_count: usize,
    config:       RateConfig,
}

impl RateAdapter {
    pub fn new(config: RateConfig) -> Self {
        Self { window_start: Instant::now(), record_count: 0, config }
    }

    /// Increment the record counter by one.
    pub fn tick(&mut self) {
        self.record_count += 1;
    }

    /// Records per second since the adapter was created.
    pub fn current_rate(&self) -> f32 {
        let elapsed = self.window_start.elapsed().as_secs_f32();
        if elapsed < 1e-9 { 0.0 } else { self.record_count as f32 / elapsed }
    }

    /// Return a (possibly widened) copy of `base` for the current ingestion rate.
    ///
    /// When the rate exceeds `fast_threshold`, the `upper_threshold` is divided
    /// by `bulk_threshold_multiplier` so slightly-lower-confidence pairs are
    /// also auto-matched during bulk load.
    pub fn adjusted_params(&self, base: &ModelParams) -> ModelParams {
        let rate = self.current_rate();
        if rate >= self.config.fast_threshold {
            let m = self.config.bulk_threshold_multiplier;
            ModelParams {
                upper_threshold: (base.upper_threshold / m).min(1.0),
                ..base.clone()
            }
        } else {
            base.clone()
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::scoring::ModelParams;

    fn base_params() -> ModelParams {
        ModelParams {
            m:               vec![vec![0.01, 0.09, 0.30, 0.60]],
            u:               vec![vec![0.70, 0.20, 0.07, 0.03]],
            log_prior_odds:  -2.0,
            upper_threshold: 0.9,
            lower_threshold: 0.1,
        }
    }

    #[test]
    fn fresh_adapter_rate_is_zero_or_low() {
        let adapter = RateAdapter::new(RateConfig::default());
        let rate = adapter.current_rate();
        assert!(rate < 1.0, "brand-new adapter should report near-zero rate");
    }

    #[test]
    fn tick_increments_record_count() {
        let mut adapter = RateAdapter::new(RateConfig::default());
        for _ in 0..10 {
            adapter.tick();
        }
        assert_eq!(adapter.record_count, 10);
    }

    #[test]
    fn slow_rate_returns_base_params() {
        let adapter = RateAdapter::new(RateConfig::default());
        let base    = base_params();
        let adj     = adapter.adjusted_params(&base);
        assert!(
            (adj.upper_threshold - base.upper_threshold).abs() < 1e-6,
            "at slow rate, threshold must remain unchanged"
        );
    }

    #[test]
    fn fast_rate_widens_upper_threshold() {
        let config = RateConfig {
            fast_threshold:            0.0, // always triggers fast path
            bulk_threshold_multiplier: 1.05,
            slow_threshold:            0.0,
        };
        let adapter = RateAdapter::new(config);
        let base = base_params();
        let adj  = adapter.adjusted_params(&base);
        assert!(
            adj.upper_threshold < base.upper_threshold,
            "fast rate must widen (lower) the upper threshold"
        );
    }

    #[test]
    fn adjusted_params_never_exceeds_one() {
        // Even with a multiplier < 1 (widening), result must be capped at 1.0.
        let config = RateConfig {
            fast_threshold:            0.0,
            bulk_threshold_multiplier: 0.5, // dividing by 0.5 doubles the threshold, but .min(1.0) caps it
            slow_threshold:            0.0,
        };
        let adapter = RateAdapter::new(config);
        let base    = base_params();
        let adj     = adapter.adjusted_params(&base);
        assert!(adj.upper_threshold <= 1.0);
    }
}
