use crate::kernel::Kernel;

/// Marker for the GPU M-step reduction kernel.
///
/// One `EmReduce` dispatch corresponds to one M-step of the EM algorithm:
/// given `match_probs` from the E-step it accumulates weighted level counts
/// that the caller normalises into updated `m`/`u` probability tables.
pub struct EmReduce;

/// Input to one M-step reduction pass.
pub struct EmReduceInput<'a> {
    /// P(match | vector) for each pair, E-step output, computed on CPU.
    pub match_probs: &'a [f32],
    /// Field-major comparison levels: `levels[field * n_pairs + pair]`, values 0–3.
    pub comparison_levels: &'a [u32],
    pub n_pairs:  usize,
    pub n_fields: usize,
}

/// Raw M-step counts returned from one reduction pass.
///
/// Normalize to get updated Fellegi-Sunter probability tables:
/// ```text
/// m[f][l] = (m_counts[f*4 + l] + smoothing) / (total_match    + 4*smoothing)
/// u[f][l] = (u_counts[f*4 + l] + smoothing) / (total_nonmatch + 4*smoothing)
/// ```
pub struct EmReduceOutput {
    /// Unnormalized m-counts: `m_counts[f*4 + l] = Σ_pairs P(match)  times  1[level==l]`.
    /// Length is `n_fields * 4`.
    pub m_counts: Vec<f32>,
    /// Unnormalized u-counts: `u_counts[f*4 + l] = Σ_pairs P(nonmatch)  times  1[level==l]`.
    /// Length is `n_fields * 4`.
    pub u_counts: Vec<f32>,
    /// Σ P(match) across all pairs.
    pub total_match: f32,
    /// Σ (1 - P(match)) across all pairs.
    pub total_nonmatch: f32,
}

impl Kernel for EmReduce {
    type Input<'a> = EmReduceInput<'a>;
    type Output    = EmReduceOutput;
}
