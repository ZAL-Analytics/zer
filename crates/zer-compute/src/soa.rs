/// Fixed stride per string in the interleaved byte buffers.
/// Used by `BatchSizer` to estimate buffer sizes.
pub const STRING_STRIDE: usize = 64;

/// Pre-compute `ln(m[f][l] / u[f][l])` weight table for GPU upload.
pub fn build_weight_table(params: &zer_core::scoring::ModelParams) -> Vec<f32> {
    let n_fields = params.m.len();
    let n_levels = if n_fields > 0 { params.m[0].len() } else { 0 };
    let mut table = Vec::with_capacity(n_fields * n_levels);
    for f in 0..n_fields {
        for l in 0..n_levels {
            let m = params.m[f][l].max(1e-9_f32);
            let u = params.u[f][l].max(1e-9_f32);
            table.push((m / u).ln());
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_table_log_ratios_are_finite() {
        use zer_core::scoring::ModelParams;
        let params = ModelParams {
            m: vec![vec![0.05, 0.10, 0.15, 0.70]; 3],
            u: vec![vec![0.70, 0.15, 0.10, 0.05]; 3],
            log_prior_odds: 0.0,
            upper_threshold: 0.9,
            lower_threshold: 0.1,
        };
        let table = build_weight_table(&params);
        assert_eq!(table.len(), 3 * 4);
        for v in &table {
            assert!(v.is_finite(), "weight table must not contain NaN/Inf: {v}");
        }
        assert!(
            table[0 * 4 + 3] > 0.0,
            "exact match should have positive weight"
        );
        assert!(
            table[0 * 4 + 0] < 0.0,
            "none match should have negative weight"
        );
    }
}
