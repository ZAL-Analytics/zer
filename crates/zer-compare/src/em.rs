use zer_core::{
    comparison::{ComparisonBatch, ComparisonLevel, ComparisonVector},
    error::ZerError,
    scoring::ModelParams,
};

const N_LEVELS: usize = 4; // None=0, Partial=1, Close=2, Exact=3


// ── E-step ────────────────────────────────────────────────────────────────────

/// Compute P(match | comparison_vector) for a single pair given current params.
pub fn e_step(vector: &ComparisonVector, params: &ModelParams) -> f32 {
    let log_odds: f32 = params.log_prior_odds
        + vector.levels.iter().enumerate()
            .map(|(i, &level)| {
                if level == ComparisonLevel::Null { return 0.0_f32; }
                let l = level as usize;
                let m = params.m[i][l].max(1e-9_f32);
                let u = params.u[i][l].max(1e-9_f32);
                (m / u).ln()
            })
            .sum::<f32>();
    1.0 / (1.0 + (-log_odds).exp())
}

#[inline]
fn e_step_p(batch: &ComparisonBatch, p: usize, params: &ModelParams) -> f32 {
    let n_pairs = batch.n_pairs;
    let log_odds: f32 = params.log_prior_odds
        + (0..batch.n_fields)
            .map(|f| {
                let l_u8 = batch.levels[f * n_pairs + p];
                if l_u8 == 255 { return 0.0_f32; } // ComparisonLevel::Null, skip
                let l = l_u8 as usize;
                let m = params.m[f][l].max(1e-9_f32);
                let u = params.u[f][l].max(1e-9_f32);
                (m / u).ln()
            })
            .sum::<f32>();
    1.0 / (1.0 + (-log_odds).exp())
}

// ── M-step ────────────────────────────────────────────────────────────────────

fn m_step(
    batch:     &ComparisonBatch,
    posteriors: &[f32],
    prev:      &ModelParams,
) -> ModelParams {
    let n_fields = batch.n_fields;
    let n_pairs  = batch.n_pairs;

    let mut m_num = vec![vec![0.0f32; N_LEVELS]; n_fields];
    let mut u_num = vec![vec![0.0f32; N_LEVELS]; n_fields];

    let mut total_match    = 0.0f32;
    let mut total_nonmatch = 0.0f32;

    for p in 0..n_pairs {
        total_match    += posteriors[p];
        total_nonmatch += 1.0 - posteriors[p];
    }

    // Field-outer, pair-inner: sequential reads of levels[f*n_pairs+p].
    // This layout lets the compiler auto-vectorize the inner accumulation.
    // Null (255) fields are skipped, they carry no m/u evidence.
    for f in 0..n_fields {
        let field_slice = &batch.levels[f * n_pairs..(f + 1) * n_pairs];
        for p in 0..n_pairs {
            let l_u8 = field_slice[p];
            if l_u8 == 255 { continue; } // ComparisonLevel::Null
            let l = l_u8 as usize;
            m_num[f][l] += posteriors[p];
            u_num[f][l] += 1.0 - posteriors[p];
        }
    }

    let total_match    = total_match.max(1e-9);
    let total_nonmatch = total_nonmatch.max(1e-9);

    let mut m = vec![vec![1e-9f32; N_LEVELS]; n_fields];
    let mut u = vec![vec![1e-9f32; N_LEVELS]; n_fields];

    for f in 0..n_fields {
        for l in 0..N_LEVELS {
            m[f][l] = (m_num[f][l] / total_match).max(1e-9);
            u[f][l] = (u_num[f][l] / total_nonmatch).max(1e-9);
        }
        let m_sum: f32 = m[f].iter().sum();
        let u_sum: f32 = u[f].iter().sum();
        for l in 0..N_LEVELS {
            m[f][l] /= m_sum;
            u[f][l] /= u_sum;
        }
    }

    let lambda    = (total_match / n_pairs as f32).max(0.001).min(0.999);
    let log_prior = (lambda / (1.0 - lambda)).ln();

    ModelParams {
        m,
        u,
        log_prior_odds:  log_prior,
        upper_threshold: prev.upper_threshold,
        lower_threshold: prev.lower_threshold,
    }
}

// ── Delta ─────────────────────────────────────────────────────────────────────

fn params_delta(a: &ModelParams, b: &ModelParams) -> f32 {
    let mut max_delta = 0.0f32;
    for (am, bm) in a.m.iter().zip(b.m.iter()) {
        for (&av, &bv) in am.iter().zip(bm.iter()) {
            max_delta = max_delta.max((av - bv).abs());
        }
    }
    for (au, bu) in a.u.iter().zip(b.u.iter()) {
        for (&av, &bv) in au.iter().zip(bu.iter()) {
            max_delta = max_delta.max((av - bv).abs());
        }
    }
    max_delta
}

// ── Initialization ────────────────────────────────────────────────────────────

fn init_from_priors(n_fields: usize) -> ModelParams {
    let m = vec![vec![0.02, 0.06, 0.12, 0.80]; n_fields];
    let u = vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields];
    ModelParams {
        m,
        u,
        log_prior_odds:  0.0,
        upper_threshold: 0.9,
        lower_threshold: 0.1,
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Estimate the prior match rate λ = P(true match in candidate set).
pub fn estimate_lambda(batch: &ComparisonBatch) -> f32 {
    if batch.n_pairs == 0 { return 0.01; }
    let exact = ComparisonLevel::Exact as u8;
    let n_pairs = batch.n_pairs;
    let high_sim_count = (0..n_pairs)
        .filter(|&p| {
            (0..batch.n_fields).any(|f| batch.levels[f * n_pairs + p] == exact)
        })
        .count();
    let raw = high_sim_count as f32 / n_pairs as f32;
    raw.max(0.001).min(0.5)
}

/// Auto-calibrate upper/lower thresholds after EM converges.
pub fn auto_calibrate_thresholds(scores: &[f32]) -> (f32, f32) {
    if scores.is_empty() { return (0.9, 0.1); }

    let high: Vec<f32> = scores.iter().copied().filter(|&s| s >= 0.7).collect();
    let low:  Vec<f32> = scores.iter().copied().filter(|&s| s <= 0.3).collect();

    let upper = if high.len() >= 10 {
        let mut sorted = high.clone();
        sorted.sort_by(f32::total_cmp);
        sorted[(sorted.len() as f32 * 0.05) as usize].max(0.85)
    } else {
        0.9
    };

    let lower = if low.len() >= 10 {
        let mut sorted = low.clone();
        sorted.sort_by(f32::total_cmp);
        sorted[(sorted.len() as f32 * 0.95) as usize].min(0.15)
    } else {
        0.1
    };

    (upper, lower)
}

/// Run the EM algorithm to learn m/u parameters without labels.
pub fn run_em(
    batch:    &ComparisonBatch,
    init:     Option<ModelParams>,
    max_iter: usize,
) -> Result<ModelParams, ZerError> {
    if batch.n_pairs == 0 {
        return Err(ZerError::SchemaMismatch { expected: 1, got: 0 });
    }

    let n_fields = batch.n_fields;
    if n_fields == 0 {
        return Err(ZerError::EmptySchema);
    }

    let mut params = init.unwrap_or_else(|| {
        let mut p = init_from_priors(n_fields);
        let lambda = estimate_lambda(batch);
        p.log_prior_odds = (lambda / (1.0 - lambda)).ln();
        tracing::debug!(lambda, "auto-estimated prior match rate");
        p
    });

    for iter in 0..max_iter {
        let posteriors: Vec<f32> = (0..batch.n_pairs)
            .map(|p| e_step_p(batch, p, &params))
            .collect();

        let new_params = m_step(batch, &posteriors, &params);
        let delta      = params_delta(&params, &new_params);

        params = new_params;
        tracing::debug!(iter, delta, "EM iteration");

        if delta < 1e-6 {
            tracing::info!(iter, "EM converged");
            break;
        }
    }

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::comparison::{ComparisonBatch, ComparisonLevel, ComparisonVector};

    fn uniform_vector(id_a: u64, id_b: u64, n_fields: usize, level: ComparisonLevel) -> ComparisonVector {
        ComparisonVector::new(id_a, id_b, vec![level; n_fields])
    }

    fn synthetic_batch(n_match: usize, n_nonmatch: usize, n_fields: usize) -> ComparisonBatch {
        let mut vecs = Vec::with_capacity(n_match + n_nonmatch);
        for i in 0..n_match {
            vecs.push(uniform_vector(i as u64, (i + 1_000_000) as u64, n_fields, ComparisonLevel::Exact));
        }
        for i in 0..n_nonmatch {
            vecs.push(uniform_vector((i + 2_000_000) as u64, (i + 3_000_000) as u64, n_fields, ComparisonLevel::None));
        }
        ComparisonBatch::from_vectors(&vecs)
    }

    #[test]
    fn em_converges_on_synthetic_data() {
        let batch  = synthetic_batch(200, 800, 4);
        let params = run_em(&batch, None, 100).expect("EM should succeed");
        for f in 0..4 {
            let exact_idx = ComparisonLevel::Exact as usize;
            assert!(
                params.m[f][exact_idx] > params.u[f][exact_idx],
                "m[Exact] should exceed u[Exact] for field {f}: m={}, u={}",
                params.m[f][exact_idx], params.u[f][exact_idx]
            );
        }
    }

    #[test]
    fn em_warm_start_converges_faster() {
        let batch = synthetic_batch(200, 800, 3);

        let warm = ModelParams {
            m: vec![vec![0.02, 0.06, 0.12, 0.78]; 3],
            u: vec![vec![0.75, 0.12, 0.08, 0.05]; 3],
            log_prior_odds:  (0.2_f32 / 0.8_f32).ln(),
            upper_threshold: 0.9,
            lower_threshold: 0.1,
        };

        let params = run_em(&batch, Some(warm), 5).expect("warm start EM should succeed");
        for f in 0..3 {
            let exact_idx = ComparisonLevel::Exact as usize;
            assert!(params.m[f][exact_idx] > params.u[f][exact_idx],
                "warm-start: m[Exact] should exceed u[Exact] for field {f}");
        }
    }

    #[test]
    fn em_empty_batch_returns_error() {
        let batch = ComparisonBatch::new(0, 0, vec![]);
        let result = run_em(&batch, None, 50);
        assert!(result.is_err(), "empty batch should return an error");
    }

    #[test]
    fn estimate_lambda_all_exact() {
        let batch  = synthetic_batch(100, 0, 2);
        let lambda = estimate_lambda(&batch);
        assert_eq!(lambda, 0.5);
    }

    #[test]
    fn estimate_lambda_all_none() {
        let batch  = synthetic_batch(0, 100, 2);
        let lambda = estimate_lambda(&batch);
        assert_eq!(lambda, 0.001);
    }

    #[test]
    fn auto_calibrate_bimodal_distribution() {
        let mut scores = vec![];
        for _ in 0..50  { scores.push(0.95_f32); }
        for _ in 0..200 { scores.push(0.05_f32); }
        let (upper, lower) = auto_calibrate_thresholds(&scores);
        assert!(upper >= 0.85, "upper threshold should be ≥ 0.85, got {upper}");
        assert!(lower <= 0.15, "lower threshold should be ≤ 0.15, got {lower}");
    }

    #[test]
    fn auto_calibrate_empty_returns_defaults() {
        let (upper, lower) = auto_calibrate_thresholds(&[]);
        assert_eq!(upper, 0.9);
        assert_eq!(lower, 0.1);
    }
}
