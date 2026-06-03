use zer_core::{
    comparison::{ComparisonBatch, ComparisonLevel, ComparisonVector},
    scoring::{MatchBand, ModelParams, ScoredPair},
    traits::Scorer,
};

const NULL_LEVEL_BYTE: u8 = ComparisonLevel::Null as u8;

/// Fellegi-Sunter scorer.
pub struct FellegiSunterScorer;

impl FellegiSunterScorer {
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

#[inline]
fn classify(prob: f32, params: &ModelParams) -> MatchBand {
    if prob >= params.upper_threshold {
        MatchBand::AutoMatch
    } else if prob < params.lower_threshold {
        MatchBand::AutoReject
    } else {
        MatchBand::Borderline
    }
}

impl Scorer for FellegiSunterScorer {
    fn score(&self, vector: &ComparisonVector, params: &ModelParams) -> ScoredPair {
        let match_weight: f32 = vector
            .levels
            .iter()
            .enumerate()
            .map(|(i, &level)| {
                if level == ComparisonLevel::Null {
                    return 0.0_f32;
                }
                let l = level as usize;
                let m = params.m[i][l].max(1e-9);
                let u = params.u[i][l].max(1e-9);
                (m / u).ln()
            })
            .sum();

        let match_probability = Self::sigmoid(match_weight + params.log_prior_odds);
        let band = classify(match_probability, params);

        ScoredPair {
            record_a: vector.record_a,
            record_b: vector.record_b,
            match_weight,
            match_probability,
            vector: vector.clone(),
            band,
        }
    }

    /// Batch scoring over all pairs in the `ComparisonBatch`.
    fn score_batch(&self, batch: &ComparisonBatch, params: &ModelParams) -> Vec<ScoredPair> {
        let n_pairs = batch.n_pairs;
        let n_fields = batch.n_fields;

        // Flatten the weight table: weight_flat[f*4 + l] = ln(m[f][l] / u[f][l]).
        // Storing it flat lets the inner loop use a single indexed load.
        let mut weight_flat = vec![0.0f32; n_fields * 4];
        for f in 0..n_fields {
            for l in 0..4 {
                let m = params.m[f][l].max(1e-9_f32);
                let u = params.u[f][l].max(1e-9_f32);
                weight_flat[f * 4 + l] = (m / u).ln();
            }
        }

        let mut match_weights = vec![0.0f32; n_pairs];

        // Field-outer / pair-inner, sequential reads, auto-vectorizable inner loop.
        for f in 0..n_fields {
            let field_levels = &batch.levels[f * n_pairs..(f + 1) * n_pairs];
            let field_weights = &weight_flat[f * 4..(f + 1) * 4];
            for p in 0..n_pairs {
                let l = field_levels[p];
                if l != NULL_LEVEL_BYTE {
                    match_weights[p] += field_weights[l as usize];
                }
            }
        }

        (0..n_pairs)
            .map(|p| {
                let (a, b) = batch.pair_ids[p];
                let match_weight = match_weights[p];
                let match_probability = Self::sigmoid(match_weight + params.log_prior_odds);
                let band = classify(match_probability, params);
                ScoredPair {
                    record_a: a,
                    record_b: b,
                    match_weight,
                    match_probability,
                    vector: batch.pair_as_vector(p),
                    band,
                }
            })
            .collect()
    }

    fn estimate_params(
        &self,
        batch: &ComparisonBatch,
        init: Option<ModelParams>,
        max_iter: usize,
    ) -> zer_core::traits::Result<ModelParams> {
        let mut params = crate::em::run_em(batch, init, max_iter)?;

        let scores: Vec<f32> = self
            .score_batch(batch, &params)
            .into_iter()
            .map(|sp| sp.match_probability)
            .collect();
        let (upper, lower) = crate::em::auto_calibrate_thresholds(&scores);
        params.upper_threshold = upper;
        params.lower_threshold = lower;
        tracing::info!(upper, lower, "auto-calibrated thresholds");

        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use zer_core::comparison::{ComparisonBatch, ComparisonLevel, ComparisonVector};

    use super::*;

    fn default_params(n_fields: usize) -> ModelParams {
        ModelParams {
            m: vec![vec![0.02, 0.06, 0.12, 0.80]; n_fields],
            u: vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
            log_prior_odds: (0.1_f32 / 0.9_f32).ln(),
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

    #[test]
    fn all_exact_produces_high_probability() {
        let scorer = FellegiSunterScorer;
        let params = default_params(4);
        let cv = all_exact_vector(4);
        let scored = scorer.score(&cv, &params);
        assert!(
            scored.match_probability > 0.9,
            "all-Exact vector should score > 0.9, got {}",
            scored.match_probability
        );
        assert_eq!(scored.band, MatchBand::AutoMatch);
    }

    #[test]
    fn all_none_produces_low_probability() {
        let scorer = FellegiSunterScorer;
        let params = default_params(4);
        let cv = all_none_vector(4);
        let scored = scorer.score(&cv, &params);
        assert!(
            scored.match_probability < 0.1,
            "all-None vector should score < 0.1, got {}",
            scored.match_probability
        );
        assert_eq!(scored.band, MatchBand::AutoReject);
    }

    #[test]
    fn mixed_vector_scores_between_extremes() {
        let scorer = FellegiSunterScorer;
        let params = default_params(4);
        let all_exact = scorer.score(&all_exact_vector(4), &params);
        let all_none = scorer.score(&all_none_vector(4), &params);

        let cv = ComparisonVector::new(
            5,
            6,
            vec![
                ComparisonLevel::Exact,
                ComparisonLevel::Exact,
                ComparisonLevel::None,
                ComparisonLevel::None,
            ],
        );
        let scored = scorer.score(&cv, &params);

        assert!(
            scored.match_probability > all_none.match_probability
                && scored.match_probability < all_exact.match_probability,
            "mixed vector ({}) should be between all-None ({}) and all-Exact ({})",
            scored.match_probability,
            all_none.match_probability,
            all_exact.match_probability
        );
        assert_ne!(
            scored.band,
            MatchBand::AutoMatch,
            "mixed vector should not be AutoMatch"
        );
    }

    #[test]
    fn score_batch_matches_individual_scores() {
        let scorer = FellegiSunterScorer;
        let params = default_params(3);
        let vectors = vec![all_exact_vector(3), all_none_vector(3)];
        let batch = ComparisonBatch::from_vectors(&vectors);
        let scored = scorer.score_batch(&batch, &params);
        let ind: Vec<ScoredPair> = vectors.iter().map(|v| scorer.score(v, &params)).collect();

        for (b, i) in scored.iter().zip(ind.iter()) {
            assert_eq!(b.band, i.band);
            assert!((b.match_probability - i.match_probability).abs() < 1e-6);
        }
    }

    #[test]
    fn estimate_params_converges_from_mixed_data() {
        let scorer = FellegiSunterScorer;
        let mut vectors = vec![];
        for i in 0..100u64 {
            vectors.push(ComparisonVector::new(
                i,
                i + 10000,
                vec![ComparisonLevel::Exact; 3],
            ));
        }
        for i in 0..400u64 {
            vectors.push(ComparisonVector::new(
                i + 20000,
                i + 30000,
                vec![ComparisonLevel::None; 3],
            ));
        }
        let batch = ComparisonBatch::from_vectors(&vectors);

        let params = scorer
            .estimate_params(&batch, None, 100)
            .expect("estimate_params should succeed");

        for f in 0..3 {
            assert!(
                params.m[f][ComparisonLevel::Exact as usize]
                    > params.u[f][ComparisonLevel::Exact as usize],
                "after EM, m[Exact] should exceed u[Exact] for field {f}"
            );
        }
    }
}
