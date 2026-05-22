//! AVX2 dispatch for [`EmReduce`], vectorized E-step + M-step with Rayon parallelism.
//!
//! # E-step (AVX2 gather + Rayon)
//!
//! For each chunk of 8 pairs, `_mm256_i32gather_ps` fetches the weight
//! `weights[f*4 + level[f][p]]` for all 8 pairs simultaneously, one gather
//! per field.  The gathered values are accumulated into a running log-odds
//! vector, then a scalar sigmoid is applied to each of the 8 sums.  Rayon
//! `par_chunks_mut` distributes chunks across cores.
//!
//! # M-step (AVX2 mask-accumulate + Rayon)
//!
//! Rayon `par_chunks` splits pairs across threads; each thread maintains
//! thread-local `m_counts`/`u_counts` arrays (n_fields × 4 floats each).
//! Within each chunk, `_mm256_cmpeq_epi32` / `_mm256_and_ps` accumulates
//! per-level bins without branching.  Thread-local results are merged via
//! tree reduction, input arrays are never copied.

use rayon::prelude::*;

use crate::{
    backend::avx2::device::Avx2Device,
    error::GpuError,
    kernel::KernelDispatch,
    kernels::em_reduce::{EmReduce, EmReduceInput, EmReduceOutput},
};

const NUM_LEVELS: usize = 4;

// ── Session ───────────────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers held across EM iterations.
///
/// `levels_u32` is uploaded (widened u8 → u32) once in `em_init_session`
/// and reused every iteration.  `match_probs` is the E-step output scratch.
pub(crate) struct Avx2EmSession {
    pub levels_u32:  Vec<u32>,
    pub match_probs: Vec<f32>,
    pub n_pairs:     usize,
    pub n_fields:    usize,
}

impl Avx2Device {
    pub(crate) fn em_init_session(
        comparison_levels: &[u32],
        n_pairs:           usize,
        n_fields:          usize,
    ) -> Avx2EmSession {
        Avx2EmSession {
            levels_u32:  comparison_levels.to_vec(),
            match_probs: vec![0.0f32; n_pairs],
            n_pairs,
            n_fields,
        }
    }

    /// Run one full EM iteration (E-step + M-step) using AVX2 + Rayon.
    pub(crate) fn em_run_iteration(
        session:        &mut Avx2EmSession,
        weights:        &[f32],
        log_prior_odds: f32,
    ) -> Result<EmReduceOutput, GpuError> {
        let n_pairs  = session.n_pairs;
        let n_fields = session.n_fields;

        if n_pairs == 0 || n_fields == 0 {
            let zeros = vec![0.0f32; n_fields * NUM_LEVELS];
            return Ok(EmReduceOutput {
                m_counts: zeros.clone(), u_counts: zeros,
                total_match: 0.0, total_nonmatch: 0.0,
            });
        }

        run_estep(&mut session.match_probs, &session.levels_u32, weights, n_pairs, n_fields, log_prior_odds);

        let (m_counts, u_counts, total_match, total_nonmatch) =
            run_mstep(&session.match_probs, &session.levels_u32, n_pairs, n_fields);

        Ok(EmReduceOutput { m_counts, u_counts, total_match, total_nonmatch })
    }
}

// ── KernelDispatch impl (stateless, match_probs provided by caller) ──────────

impl KernelDispatch<EmReduce> for Avx2Device {
    fn dispatch(&self, input: EmReduceInput<'_>) -> Result<EmReduceOutput, GpuError> {
        let EmReduceInput { match_probs, comparison_levels, n_pairs, n_fields } = input;

        if n_pairs == 0 || n_fields == 0 {
            let zeros = vec![0.0f32; n_fields * NUM_LEVELS];
            return Ok(EmReduceOutput {
                m_counts: zeros.clone(), u_counts: zeros,
                total_match: 0.0, total_nonmatch: 0.0,
            });
        }

        let (m_counts, u_counts, total_match, total_nonmatch) =
            run_mstep(match_probs, comparison_levels, n_pairs, n_fields);

        Ok(EmReduceOutput { m_counts, u_counts, total_match, total_nonmatch })
    }
}

// ── Chunk sizing ──────────────────────────────────────────────────────────────

/// Chunk size that is a multiple of 8 and balances work across Rayon threads.
fn compute_chunk_size(n_pairs: usize) -> usize {
    let n_threads = rayon::current_num_threads().max(1);
    ((n_pairs / n_threads + 7) / 8 * 8).max(8)
}

// ── E-step ────────────────────────────────────────────────────────────────────

fn run_estep(
    match_probs:       &mut [f32],
    comparison_levels: &[u32],
    weights:           &[f32],
    n_pairs:           usize,
    n_fields:          usize,
    log_prior_odds:    f32,
) {
    let chunk_size = compute_chunk_size(n_pairs);
    match_probs
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let pair_start = chunk_idx * chunk_size;
            run_estep_chunk(chunk, pair_start, comparison_levels, weights, n_pairs, n_fields, log_prior_odds);
        });
}

fn run_estep_chunk(
    out_probs:         &mut [f32],
    pair_start:        usize,
    comparison_levels: &[u32],
    weights:           &[f32],
    n_pairs:           usize,
    n_fields:          usize,
    log_prior_odds:    f32,
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe {
            run_estep_chunk_avx2(out_probs, pair_start, comparison_levels, weights, n_pairs, n_fields, log_prior_odds)
        };
    }
    run_estep_chunk_scalar(out_probs, pair_start, comparison_levels, weights, n_pairs, n_fields, log_prior_odds);
}

fn run_estep_chunk_scalar(
    out_probs:         &mut [f32],
    pair_start:        usize,
    comparison_levels: &[u32],
    weights:           &[f32],
    n_pairs:           usize,
    n_fields:          usize,
    log_prior_odds:    f32,
) {
    for (i, p) in out_probs.iter_mut().enumerate() {
        let pair = pair_start + i;
        let mut log_odds = log_prior_odds;
        for f in 0..n_fields {
            let level = comparison_levels[f * n_pairs + pair] as usize;
            log_odds += weights[f * NUM_LEVELS + level];
        }
        *p = 1.0 / (1.0 + (-log_odds).exp());
    }
}

/// E-step chunk using `_mm256_i32gather_ps` to fetch all 8 weights per field in one instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn run_estep_chunk_avx2(
    out_probs:         &mut [f32],
    pair_start:        usize,
    comparison_levels: &[u32],
    weights:           &[f32],
    n_pairs:           usize,
    n_fields:          usize,
    log_prior_odds:    f32,
) {
    use std::arch::x86_64::*;

    let chunk_len = out_probs.len();
    let chunks    = chunk_len / 8;
    let prior     = _mm256_set1_ps(log_prior_odds);

    for c in 0..chunks {
        let base = pair_start + c * 8;
        let mut sum = prior;

        for f in 0..n_fields {
            let lv_ptr  = comparison_levels.as_ptr().add(f * n_pairs + base);
            let lvs     = _mm256_loadu_si256(lv_ptr as *const __m256i);
            let f_off   = _mm256_set1_epi32((f * NUM_LEVELS) as i32);
            let indices = _mm256_add_epi32(lvs, f_off);
            // Gather: for each of 8 lanes, load weights[f*4 + level[p]]
            let w = _mm256_i32gather_ps(weights.as_ptr(), indices, 4);
            sum   = _mm256_add_ps(sum, w);
        }

        let mut sums = [0.0f32; 8];
        _mm256_storeu_ps(sums.as_mut_ptr(), sum);
        for i in 0..8 {
            out_probs[c * 8 + i] = 1.0 / (1.0 + (-sums[i]).exp());
        }
    }

    // Scalar tail
    let tail_start = chunks * 8;
    run_estep_chunk_scalar(
        &mut out_probs[tail_start..],
        pair_start + tail_start,
        comparison_levels, weights, n_pairs, n_fields, log_prior_odds,
    );
}

// ── M-step ────────────────────────────────────────────────────────────────────

fn run_mstep(
    match_probs:       &[f32],
    comparison_levels: &[u32],
    n_pairs:           usize,
    n_fields:          usize,
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    let n_cells    = n_fields * NUM_LEVELS;
    let chunk_size = compute_chunk_size(n_pairs);

    match_probs
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let pair_start = chunk_idx * chunk_size;
            let mut m = vec![0.0f32; n_cells];
            let mut u = vec![0.0f32; n_cells];
            let (tm, tnm) = accumulate_chunk(chunk, pair_start, comparison_levels, n_pairs, n_fields, &mut m, &mut u);
            (m, u, tm, tnm)
        })
        .reduce(
            || (vec![0.0f32; n_cells], vec![0.0f32; n_cells], 0.0f32, 0.0f32),
            |(mut m1, mut u1, tm1, tnm1), (m2, u2, tm2, tnm2)| {
                for i in 0..n_cells {
                    m1[i] += m2[i];
                    u1[i] += u2[i];
                }
                (m1, u1, tm1 + tm2, tnm1 + tnm2)
            },
        )
}

fn accumulate_chunk(
    match_probs_chunk: &[f32],
    pair_start:        usize,
    comparison_levels: &[u32],
    n_pairs:           usize,
    n_fields:          usize,
    m_counts:          &mut [f32],
    u_counts:          &mut [f32],
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe {
            accumulate_chunk_avx2(match_probs_chunk, pair_start, comparison_levels, n_pairs, n_fields, m_counts, u_counts)
        };
    }
    accumulate_chunk_scalar(match_probs_chunk, pair_start, comparison_levels, n_pairs, n_fields, m_counts, u_counts)
}

fn accumulate_chunk_scalar(
    match_probs_chunk: &[f32],
    pair_start:        usize,
    comparison_levels: &[u32],
    n_pairs:           usize,
    n_fields:          usize,
    m_counts:          &mut [f32],
    u_counts:          &mut [f32],
) -> (f32, f32) {
    let mut tm  = 0.0f32;
    let mut tnm = 0.0f32;
    for (i, &pm) in match_probs_chunk.iter().enumerate() {
        let pair = pair_start + i;
        let pnm  = 1.0 - pm;
        tm  += pm;
        tnm += pnm;
        for f in 0..n_fields {
            let level = comparison_levels[f * n_pairs + pair] as usize;
            let idx   = f * NUM_LEVELS + level;
            m_counts[idx] += pm;
            u_counts[idx] += pnm;
        }
    }
    (tm, tnm)
}

/// Per-chunk M-step using AVX2 mask-accumulate for level bins + scalar reduction for totals.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_chunk_avx2(
    match_probs_chunk: &[f32],
    pair_start:        usize,
    comparison_levels: &[u32],
    n_pairs:           usize,
    n_fields:          usize,
    m_counts:          &mut [f32],
    u_counts:          &mut [f32],
) -> (f32, f32) {
    use std::arch::x86_64::*;

    let chunk_len = match_probs_chunk.len();
    let chunks    = chunk_len / 8;
    let ones      = _mm256_set1_ps(1.0);

    // Running totals (only need one pass over match_probs)
    let mut sum_m  = _mm256_setzero_ps();
    let mut sum_nm = _mm256_setzero_ps();
    for c in 0..chunks {
        let pm  = _mm256_loadu_ps(match_probs_chunk.as_ptr().add(c * 8));
        let pnm = _mm256_sub_ps(ones, pm);
        sum_m  = _mm256_add_ps(sum_m,  pm);
        sum_nm = _mm256_add_ps(sum_nm, pnm);
    }
    let mut tm  = hsum256_ps(sum_m);
    let mut tnm = hsum256_ps(sum_nm);
    for &pm in &match_probs_chunk[chunks * 8..] {
        tm  += pm;
        tnm += 1.0 - pm;
    }

    // Per-field level accumulation
    for f in 0..n_fields {
        let lv_base = f * n_pairs + pair_start;
        let mut acc_m  = [_mm256_setzero_ps(); NUM_LEVELS];
        let mut acc_nm = [_mm256_setzero_ps(); NUM_LEVELS];

        for c in 0..chunks {
            let base = c * 8;
            let pm   = _mm256_loadu_ps(match_probs_chunk.as_ptr().add(base));
            let pnm  = _mm256_sub_ps(ones, pm);
            let lv   = _mm256_loadu_si256(
                comparison_levels.as_ptr().add(lv_base + base) as *const __m256i,
            );

            for level in 0..NUM_LEVELS {
                let lvl_vec = _mm256_set1_epi32(level as i32);
                let mask    = _mm256_castsi256_ps(_mm256_cmpeq_epi32(lv, lvl_vec));
                acc_m[level]  = _mm256_add_ps(acc_m[level],  _mm256_and_ps(pm,  mask));
                acc_nm[level] = _mm256_add_ps(acc_nm[level], _mm256_and_ps(pnm, mask));
            }
        }

        for level in 0..NUM_LEVELS {
            let idx = f * NUM_LEVELS + level;
            m_counts[idx] += hsum256_ps(acc_m[level]);
            u_counts[idx] += hsum256_ps(acc_nm[level]);
        }

        // Scalar tail
        for i in (chunks * 8)..chunk_len {
            let level = comparison_levels[lv_base + i] as usize;
            let idx   = f * NUM_LEVELS + level;
            let pm    = match_probs_chunk[i];
            m_counts[idx] += pm;
            u_counts[idx] += 1.0 - pm;
        }
    }

    (tm, tnm)
}

// ── Horizontal sum ────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi   = _mm256_extractf128_ps(v, 1);
    let lo   = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let s64  = _mm_add_ps(s128, shuf);
    let s32  = _mm_add_ss(s64, _mm_movehl_ps(shuf, s64));
    _mm_cvtss_f32(s32)
}
