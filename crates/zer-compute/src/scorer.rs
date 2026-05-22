//! `DeviceScorer`, implements the `Scorer` trait with GPU-accelerated EM.

use std::sync::Arc;

use zer_core::{
    comparison::{ComparisonBatch, ComparisonVector},
    scoring::{ModelParams, ScoredPair},
    traits::{Result, Scorer},
};

use crate::{
    backend::{cpu::CpuFallbackScorer, DeviceBackend},
    error::GpuError,
};

/// Minimum pairs before the GPU EM path is used.
pub(crate) const EM_GPU_MIN_PAIRS: usize = 50_000;

pub struct DeviceScorer {
    backend:      Arc<DeviceBackend>,
    cpu_fallback: CpuFallbackScorer,
}

impl DeviceScorer {
    pub fn new(backend: Arc<DeviceBackend>) -> Self {
        Self { backend, cpu_fallback: CpuFallbackScorer }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }
}

impl Scorer for DeviceScorer {
    fn score(&self, vector: &ComparisonVector, params: &ModelParams) -> ScoredPair {
        self.cpu_fallback.score(vector, params)
    }

    fn score_batch(&self, batch: &ComparisonBatch, params: &ModelParams) -> Vec<ScoredPair> {
        self.cpu_fallback.score_batch(batch, params)
    }

    fn estimate_params(
        &self,
        batch:    &ComparisonBatch,
        init:     Option<ModelParams>,
        max_iter: usize,
    ) -> Result<ModelParams> {
        if self.backend.is_accelerated() && batch.n_pairs >= EM_GPU_MIN_PAIRS {
            let result = zer_prof::trace!("zer_compute::estimate_params_accelerated", {
                gpu_em_estimate(&self.backend, batch, init.clone(), max_iter)
            });
            match result {
                Ok(params) => {
                    tracing::info!(backend = %self.backend.name(), "EM converged via accelerated backend");
                    return Ok(params);
                }
                Err(e) => {
                    tracing::warn!(%e, backend = %self.backend.name(), "accelerated EM failed, falling back to CPU");
                }
            }
        } else if self.backend.is_accelerated() {
            tracing::debug!(
                n_pairs = batch.n_pairs,
                threshold = EM_GPU_MIN_PAIRS,
                "EM: batch below GPU threshold, using CPU path"
            );
        }
        self.cpu_fallback.estimate_params(batch, init, max_iter)
    }
}

// ── GPU EM loop ───────────────────────────────────────────────────────────────

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "avx2"))]
fn build_estep_weights(params: &ModelParams, n_fields: usize) -> Vec<f32> {
    const LEVELS: usize = 4;
    let mut w = Vec::with_capacity(n_fields * LEVELS);
    for f in 0..n_fields {
        for l in 0..LEVELS {
            let m = params.m[f][l].max(1e-15_f32);
            let u = params.u[f][l].max(1e-15_f32);
            w.push((m / u).ln());
        }
    }
    w
}

/// Run the full EM algorithm on the GPU backend.
///
/// `comparison_levels` is uploaded once before the loop as a trivial
/// `u8 → u32` cast, the `ComparisonBatch` is already field-major, so no
/// transposition is needed.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "avx2"))]
fn gpu_em_estimate(
    backend:  &DeviceBackend,
    batch:    &ComparisonBatch,
    init:     Option<ModelParams>,
    max_iter: usize,
) -> std::result::Result<ModelParams, GpuError> {
    if batch.n_pairs == 0 {
        return Err(GpuError::LaunchFailed("EM requires at least one comparison pair".into()));
    }

    if !backend.is_gpu() {
        return crate::backend::cpu::cpu_estimate_params(batch, init, max_iter)
            .map_err(|e| GpuError::LaunchFailed(e.to_string()));
    }

    let n_fields = batch.n_fields;
    let n_pairs  = batch.n_pairs;

    // ComparisonBatch.levels is already field-major u8, just widen to u32.
    let comparison_levels: Vec<u32> = batch.levels.iter().map(|&l| l as u32).collect();

    let mut params = init.unwrap_or_else(|| {
        let lambda = zer_compare::em::estimate_lambda(batch);
        let log_prior_odds = (lambda / (1.0 - lambda)).ln();
        ModelParams {
            m:               vec![vec![0.02, 0.06, 0.12, 0.80]; n_fields],
            u:               vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
            log_prior_odds,
            upper_threshold: 0.9,
            lower_threshold: 0.1,
        }
    });

    let mut session = zer_prof::trace!("zer_compute::em_init_session", {
        backend.em_init_session(&comparison_levels, n_pairs, n_fields)
    })?;

    // Closure ensures em_drop_session runs even if an iteration returns Err.
    let result: std::result::Result<ModelParams, GpuError> = (|| {
        for _iter in 0..max_iter {
            let weights = build_estep_weights(&params, n_fields);

            let out = zer_prof::trace!("zer_compute::em_full_iteration", {
                backend.em_run_iteration(&mut session, &weights, params.log_prior_odds)
            })?;

            let new_params = em_normalize(
                &out.m_counts, &out.u_counts,
                out.total_match, out.total_nonmatch,
                n_fields,
            );

            if em_converged(&params, &new_params, n_fields) {
                return Ok(new_params);
            }
            params = new_params;
        }
        Ok(params)
    })();

    backend.em_drop_session(session);
    result
}

#[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "avx2")))]
fn gpu_em_estimate(
    _backend:  &DeviceBackend,
    _batch:    &ComparisonBatch,
    _init:     Option<ModelParams>,
    _max_iter: usize,
) -> std::result::Result<ModelParams, GpuError> {
    Err(GpuError::BackendUnavailable(
        "full-GPU EM requires the cuda or vulkan feature".into(),
    ))
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "avx2"))]
fn em_normalize(
    m_counts:       &[f32],
    u_counts:       &[f32],
    total_match:    f32,
    total_nonmatch: f32,
    n_fields:       usize,
) -> ModelParams {
    const ALPHA: f32 = 1e-3;
    const LEVELS: usize = 4;

    let denom_m = (total_match    + LEVELS as f32 * ALPHA).max(1e-9_f32);
    let denom_u = (total_nonmatch + LEVELS as f32 * ALPHA).max(1e-9_f32);

    let m: Vec<Vec<f32>> = (0..n_fields)
        .map(|f| (0..LEVELS).map(|l| (m_counts[f * LEVELS + l] + ALPHA) / denom_m).collect())
        .collect();
    let u: Vec<Vec<f32>> = (0..n_fields)
        .map(|f| (0..LEVELS).map(|l| (u_counts[f * LEVELS + l] + ALPHA) / denom_u).collect())
        .collect();

    let n_total = (total_match + total_nonmatch).max(1.0_f32);
    let lambda  = (total_match / n_total).max(0.001_f32).min(0.999_f32);
    let log_prior_odds = (lambda / (1.0 - lambda)).ln();

    ModelParams { m, u, log_prior_odds, upper_threshold: 0.9, lower_threshold: 0.1 }
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "avx2"))]
fn em_converged(old: &ModelParams, new: &ModelParams, n_fields: usize) -> bool {
    const TOL: f32 = 1e-6;
    const LEVELS: usize = 4;
    let mut max_delta = 0.0_f32;
    for f in 0..n_fields {
        for l in 0..LEVELS {
            let dm = (old.m[f][l] - new.m[f][l]).abs();
            let du = (old.u[f][l] - new.u[f][l]).abs();
            max_delta = max_delta.max(dm).max(du);
        }
    }
    max_delta < TOL
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{
        comparison::{ComparisonBatch, ComparisonLevel, ComparisonVector},
        scoring::{MatchBand, ModelParams},
    };

    fn uniform_params(n_fields: usize) -> ModelParams {
        ModelParams {
            m:               vec![vec![0.05, 0.10, 0.15, 0.70]; n_fields],
            u:               vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
            log_prior_odds:  0.0,
            upper_threshold: 0.9,
            lower_threshold: 0.1,
        }
    }

    fn all_exact_vector(n_fields: usize) -> ComparisonVector {
        ComparisonVector::new(1, 2, vec![ComparisonLevel::Exact; n_fields])
    }

    fn all_none_vector(n_fields: usize) -> ComparisonVector {
        ComparisonVector::new(3, 4, vec![ComparisonLevel::None; n_fields])
    }

    fn separable_batch(n_matches: usize, n_nonmatches: usize, n_fields: usize) -> ComparisonBatch {
        let mut v = Vec::with_capacity(n_matches + n_nonmatches);
        for i in 0..n_matches as u64 {
            v.push(ComparisonVector::new(i * 2, i * 2 + 1, vec![ComparisonLevel::Exact; n_fields]));
        }
        let off = (n_matches as u64) * 2;
        for i in 0..n_nonmatches as u64 {
            v.push(ComparisonVector::new(off + i * 2, off + i * 2 + 1, vec![ComparisonLevel::None; n_fields]));
        }
        ComparisonBatch::from_vectors(&v)
    }

    #[test]
    fn score_exact_match_gives_high_probability() {
        let scorer = DeviceScorer::new(Arc::new(DeviceBackend::cpu()));
        let params = uniform_params(3);
        let v      = all_exact_vector(3);
        let pair   = scorer.score(&v, &params);

        assert!(pair.match_probability > 0.9,
            "all-Exact vector should have high match_probability, got {}", pair.match_probability);
        assert_eq!(pair.band, MatchBand::AutoMatch);
    }

    #[test]
    fn score_none_gives_low_probability() {
        let scorer = DeviceScorer::new(Arc::new(DeviceBackend::cpu()));
        let params = uniform_params(3);
        let v      = all_none_vector(3);
        let pair   = scorer.score(&v, &params);

        assert!(pair.match_probability < 0.1,
            "all-None vector should have low match_probability, got {}", pair.match_probability);
        assert_eq!(pair.band, MatchBand::AutoReject);
    }

    #[test]
    fn score_batch_matches_individual_scores() {
        let scorer  = DeviceScorer::new(Arc::new(DeviceBackend::cpu()));
        let params  = uniform_params(4);
        let vectors = vec![
            all_exact_vector(4),
            all_none_vector(4),
            ComparisonVector::new(5, 6, vec![
                ComparisonLevel::Exact,
                ComparisonLevel::None,
                ComparisonLevel::Close,
                ComparisonLevel::Partial,
            ]),
        ];
        let batch = ComparisonBatch::from_vectors(&vectors);
        let batch_results = scorer.score_batch(&batch, &params);

        for (v, br) in vectors.iter().zip(batch_results.iter()) {
            let single = scorer.score(v, &params);
            assert!(
                (single.match_probability - br.match_probability).abs() < 1e-6,
                "batch and individual scores must agree"
            );
        }
    }

    #[test]
    fn estimate_params_converges_on_separable_data() {
        let scorer   = DeviceScorer::new(Arc::new(DeviceBackend::cpu()));
        let n_fields = 4;
        let batch    = separable_batch(200, 1_000, n_fields);

        let params = scorer.estimate_params(&batch, None, 30)
            .expect("EM should not return an error");

        for f in 0..n_fields {
            assert!(params.m[f][3] > params.u[f][3],
                "m[Exact] should exceed u[Exact] for separable data (field {f})");
        }
    }

    #[test]
    fn estimate_params_returns_error_on_empty_input() {
        let scorer = DeviceScorer::new(Arc::new(DeviceBackend::cpu()));
        let batch  = ComparisonBatch::new(0, 0, vec![]);
        let result = scorer.estimate_params(&batch, None, 10);
        assert!(result.is_err(), "empty input should return an error");
    }

    #[test]
    fn weight_table_is_consistent_with_params() {
        use crate::soa::build_weight_table;

        let params = uniform_params(3);
        let table  = build_weight_table(&params);

        let weight_exact = table[0 * 4 + 3];
        let expected     = (0.70_f32 / 0.05_f32).ln();
        assert!(
            (weight_exact - expected).abs() < 1e-5,
            "weight_table Exact entry mismatch: {weight_exact} vs {expected}"
        );
    }

    #[test]
    fn em_cpu_path_correct_below_threshold() {
        let batch = separable_batch(200, 800, 4);
        assert!(batch.n_pairs < EM_GPU_MIN_PAIRS);

        let scorer = DeviceScorer::new(Arc::new(DeviceBackend::cpu()));
        let params = scorer.estimate_params(&batch, None, 30).unwrap();
        for f in 0..4 {
            assert!(params.m[f][3] > params.u[f][3], "field {f}: m[Exact] must exceed u[Exact]");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn em_gpu_path_correct_above_threshold() {
        let n_fields     = 4;
        let n_matches    = EM_GPU_MIN_PAIRS / 5;
        let n_nonmatches = EM_GPU_MIN_PAIRS;
        let batch        = separable_batch(n_matches, n_nonmatches, n_fields);
        assert!(batch.n_pairs >= EM_GPU_MIN_PAIRS);

        let params = gpu_em_estimate(&DeviceBackend::auto_detect(), &batch, None, 50)
            .expect("gpu_em_estimate must not fail");
        for f in 0..n_fields {
            assert!(params.m[f][3] > params.u[f][3], "field {f}: m[Exact] must exceed u[Exact]");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn em_gpu_cpu_agree_on_key_parameters() {
        let n_fields     = 4;
        let n_matches    = EM_GPU_MIN_PAIRS / 5;
        let n_nonmatches = EM_GPU_MIN_PAIRS;
        let batch        = separable_batch(n_matches, n_nonmatches, n_fields);
        assert!(batch.n_pairs >= EM_GPU_MIN_PAIRS);

        let cpu_params = gpu_em_estimate(&DeviceBackend::cpu(), &batch, None, 50).unwrap();
        let gpu_params = gpu_em_estimate(&DeviceBackend::auto_detect(), &batch, None, 50).unwrap();

        for f in 0..n_fields {
            assert!(cpu_params.m[f][3] > cpu_params.u[f][3],
                "CPU path field {f}: m[Exact] must exceed u[Exact]");
            assert!(gpu_params.m[f][3] > gpu_params.u[f][3],
                "GPU path field {f}: m[Exact] must exceed u[Exact]");

            // Both paths must agree that Exact is a strong match signal.
            // Allow ≤ 0.15 absolute difference in m[Exact] and u[Exact].
            let dm_exact = (cpu_params.m[f][3] - gpu_params.m[f][3]).abs();
            let du_exact = (cpu_params.u[f][3] - gpu_params.u[f][3]).abs();
            assert!(dm_exact < 0.15,
                "field {f}: CPU/GPU m[Exact] differ by {dm_exact:.4} (cpu={:.4}, gpu={:.4})",
                cpu_params.m[f][3], gpu_params.m[f][3]);
            assert!(du_exact < 0.15,
                "field {f}: CPU/GPU u[Exact] differ by {du_exact:.4} (cpu={:.4}, gpu={:.4})",
                cpu_params.u[f][3], gpu_params.u[f][3]);
        }

        // Both paths must produce a negative log_prior_odds for a 1:5 match rate.
        assert!(cpu_params.log_prior_odds < 0.0,
            "CPU log_prior_odds should be negative for rare matches: {}", cpu_params.log_prior_odds);
        assert!(gpu_params.log_prior_odds < 0.0,
            "GPU log_prior_odds should be negative for rare matches: {}", gpu_params.log_prior_odds);
        let dlpo = (cpu_params.log_prior_odds - gpu_params.log_prior_odds).abs();
        assert!(dlpo < 1.0,
            "log_prior_odds differ too much: cpu={:.4}, gpu={:.4}",
            cpu_params.log_prior_odds, gpu_params.log_prior_odds);
    }

    #[test]
    fn em_cpu_log_prior_odds_tracks_match_rate() {
        // 1 match in 10 pairs → lambda ≈ 0.1 → log_prior_odds ≈ ln(0.1/0.9) ≈ -2.2
        let n_fields = 2;
        let batch    = separable_batch(100, 900, n_fields);
        let scorer   = DeviceScorer::new(Arc::new(DeviceBackend::cpu()));
        let params   = scorer.estimate_params(&batch, None, 50).unwrap();

        assert!(params.log_prior_odds < 0.0,
            "log_prior_odds must be negative for 10% match rate: {}", params.log_prior_odds);
        assert!(params.log_prior_odds > -5.0,
            "log_prior_odds too negative for 10% match rate: {}", params.log_prior_odds);
    }

    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "avx2"))]
    #[test]
    fn em_normalize_updates_log_prior_odds() {
        // Verify em_normalize computes log_prior_odds from total_match/total_nonmatch.
        // With total_match=100, total_nonmatch=900, lambda=0.1, log_prior_odds≈-2.2.
        let m_counts = vec![25.0_f32, 25.0, 25.0, 25.0];  // uniform over levels
        let u_counts = vec![225.0_f32, 225.0, 225.0, 225.0];
        let total_match    = 100.0_f32;
        let total_nonmatch = 900.0_f32;
        let params = em_normalize(&m_counts, &u_counts, total_match, total_nonmatch, 1);

        let expected_lpo = (0.1_f32 / 0.9_f32).ln();
        assert!(
            (params.log_prior_odds - expected_lpo).abs() < 0.01,
            "log_prior_odds mismatch: got {:.4}, expected {:.4}",
            params.log_prior_odds, expected_lpo
        );
    }

    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "avx2"))]
    #[test]
    fn em_converged_uses_raw_delta() {
        let n_fields = 2;

        // Params that differ by < 1e-6, should be considered converged.
        let p1 = ModelParams {
            m:               vec![vec![0.02, 0.06, 0.12, 0.80]; n_fields],
            u:               vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
            log_prior_odds:  -2.0,
            upper_threshold: 0.9,
            lower_threshold: 0.1,
        };
        let mut p2 = p1.clone();
        p2.m[0][3] += 5e-7;  // tiny delta
        assert!(em_converged(&p1, &p2, n_fields), "should converge for delta < 1e-6");

        // Params that differ by > 1e-6, should not converge.
        let mut p3 = p1.clone();
        p3.m[0][3] += 2e-6;
        assert!(!em_converged(&p1, &p3, n_fields), "should not converge for delta > 1e-6");
    }
}
