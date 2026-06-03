//! CPU dispatch for [`EmReduce`], pure-CPU M-step reduction.

use crate::{
    backend::cpu::device::CpuDevice,
    error::GpuError,
    kernel::KernelDispatch,
    kernels::em_reduce::{EmReduce, EmReduceInput, EmReduceOutput},
};

const NUM_LEVELS: usize = 4;

impl KernelDispatch<EmReduce> for CpuDevice {
    fn dispatch(&self, input: EmReduceInput<'_>) -> Result<EmReduceOutput, GpuError> {
        let EmReduceInput {
            match_probs,
            comparison_levels,
            n_pairs,
            n_fields,
        } = input;

        if n_pairs == 0 || n_fields == 0 {
            let zeros = vec![0.0f32; n_fields * NUM_LEVELS];
            return Ok(EmReduceOutput {
                m_counts: zeros.clone(),
                u_counts: zeros,
                total_match: 0.0,
                total_nonmatch: 0.0,
            });
        }

        let mut m_counts = vec![0.0f32; n_fields * NUM_LEVELS];
        let mut u_counts = vec![0.0f32; n_fields * NUM_LEVELS];
        let mut total_match = 0.0f32;
        let mut total_nonmatch = 0.0f32;

        for p in 0..n_pairs {
            let pm = match_probs[p];
            let pnm = 1.0 - pm;
            total_match += pm;
            total_nonmatch += pnm;

            for f in 0..n_fields {
                let level = comparison_levels[f * n_pairs + p] as usize;
                let idx = f * NUM_LEVELS + level;
                m_counts[idx] += pm;
                u_counts[idx] += pnm;
            }
        }

        Ok(EmReduceOutput {
            m_counts,
            u_counts,
            total_match,
            total_nonmatch,
        })
    }
}
