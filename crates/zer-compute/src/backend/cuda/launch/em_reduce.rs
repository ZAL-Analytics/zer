//! CUDA dispatch for [`EmReduce`], two-pass GPU M-step reduction.
//!
//! Pass 1 (`em_reduce_kernel`): each block reduces 256 pairs into per-block
//! partial m/u counts using per-warp private shared memory (no atomics).
//! Partials are written in CELL-MAJOR order: `m_partials[cell * num_blocks + b]`.
//!
//! Pass 2 (`em_reduce_final_kernel`): `num_cells + 2` blocks (one per output
//! cell, two for scalar totals) cooperatively tree-reduce all partial sums.
//! All SMs are utilised; pass-2 reads are sequential (coalesced) per cell.
//!
//! `MAX_FIELDS = 16`, `NUM_LEVELS = 4` must stay in sync with `em_reduce.cu`.

// ── Kernel spec (consumed by CudaDevice::init) ───────────────────────────────

pub(crate) static PTX_SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/em_reduce.ptx"));

pub(crate) const ESTEP_FN:   &str = "em_estep_kernel";
pub(crate) const PARTIAL_FN: &str = "em_reduce_kernel";
pub(crate) const FINAL_FN:   &str = "em_reduce_final_kernel";

// ── Dispatch ─────────────────────────────────────────────────────────────────

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::{
    backend::cuda::{
        buffers::{alloc_zeros, download, upload},
        device::{CudaDevice, BLOCK_DIM},
    },
    error::GpuError,
    kernel::KernelDispatch,
    kernels::em_reduce::{EmReduce, EmReduceInput, EmReduceOutput},
};

const MAX_FIELDS: usize = 32;
const NUM_LEVELS: usize = 4;

// ── Session-based full-GPU EM (E-step + M-step, data stays on device) ────────

/// Device buffers allocated once before the EM loop and reused every iteration.
///
/// `comparison_levels` is uploaded once in `em_init_session` and never
/// re-transferred.  Only the 160-byte weight table crosses PCIe per iteration.
pub(crate) struct CudaEmSession {
    pub d_levels:          CudaSlice<u32>,
    pub d_weights:         CudaSlice<f32>,  // ln(m/u) table, updated cheaply each iter
    pub d_match_probs:     CudaSlice<f32>,
    pub d_m_partials:      CudaSlice<f32>,
    pub d_u_partials:      CudaSlice<f32>,
    pub d_match_totals:    CudaSlice<f32>,
    pub d_nonmatch_totals: CudaSlice<f32>,
    pub d_m_out:           CudaSlice<f32>,
    pub d_u_out:           CudaSlice<f32>,
    pub d_total_match:     CudaSlice<f32>,
    pub d_total_nonmatch:  CudaSlice<f32>,
    pub n_pairs:    usize,
    pub n_fields:   usize,
    pub num_blocks: u32,
}

impl CudaDevice {
    /// Upload `comparison_levels` once and pre-allocate all EM device buffers.
    pub(crate) fn em_init_session(
        &self,
        comparison_levels: &[u32],
        n_pairs:  usize,
        n_fields: usize,
    ) -> Result<CudaEmSession, GpuError> {
        let num_blocks    = ((n_pairs as u32) + BLOCK_DIM - 1) / BLOCK_DIM;
        let partial_elems = MAX_FIELDS * NUM_LEVELS * num_blocks as usize;
        let out_elems     = MAX_FIELDS * NUM_LEVELS;

        macro_rules! dev {
            ($T:ty, $n:expr) => {
                alloc_zeros::<$T>(&self.stream, $n)?
            };
        }

        Ok(CudaEmSession {
            d_levels:          upload(&self.stream, comparison_levels)?,
            d_weights:         dev!(f32, n_fields * NUM_LEVELS),
            d_match_probs:     dev!(f32, n_pairs),
            d_m_partials:      dev!(f32, partial_elems),
            d_u_partials:      dev!(f32, partial_elems),
            d_match_totals:    dev!(f32, num_blocks as usize),
            d_nonmatch_totals: dev!(f32, num_blocks as usize),
            d_m_out:           dev!(f32, out_elems),
            d_u_out:           dev!(f32, out_elems),
            d_total_match:     dev!(f32, 1),
            d_total_nonmatch:  dev!(f32, 1),
            n_pairs,
            n_fields,
            num_blocks,
        })
    }

    /// Run one EM iteration (E-step + M-step) entirely on GPU.
    ///
    /// `weights` is `ln(m[f][l] / u[f][l])` for each field  times  level (160 bytes
    /// for 10 fields).  Only this tiny table crosses PCIe per iteration.
    /// Returns the raw M-step counts used to update `ModelParams` on the host.
    pub(crate) fn em_run_iteration(
        &self,
        session:        &mut CudaEmSession,
        weights:        &[f32],
        log_prior_odds: f32,
    ) -> Result<EmReduceOutput, GpuError> {
        let n_pairs    = session.n_pairs;
        let n_fields   = session.n_fields;
        let num_blocks = session.num_blocks;
        let n_cells    = (n_fields * NUM_LEVELS) as u32;

        // Upload weight table (160 bytes for 10 fields, negligible PCIe cost).
        {
            let mut dst = session.d_weights.slice_mut(..weights.len());
            self.stream
                .memcpy_htod(weights, &mut dst)
                .map_err(|e| GpuError::TransferFailed(e.to_string()))?;
        }

        // ── E-step kernel ─────────────────────────────────────────────────────
        let estep_cfg = LaunchConfig {
            grid_dim:         (num_blocks, 1, 1),
            block_dim:        (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        zer_prof::trace_cuda!("em_estep_kernel", {
            unsafe {
                self.stream
                    .launch_builder(&self.em.estep_fn)
                    .arg(&session.d_levels)
                    .arg(&session.d_weights)
                    .arg(&mut session.d_match_probs)
                    .arg(&log_prior_odds)
                    .arg(&(n_pairs as u32))
                    .arg(&(n_fields as u32))
                    .launch(estep_cfg)
            }
            .map_err(|e| GpuError::LaunchFailed(format!("em_estep: {e}")))
        })?;

        // ── M-step pass 1 ─────────────────────────────────────────────────────
        let cfg1 = LaunchConfig {
            grid_dim:         (num_blocks, 1, 1),
            block_dim:        (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        zer_prof::trace_cuda!("em_reduce_kernel", {
            unsafe {
                self.stream
                    .launch_builder(&self.em.partial_fn)
                    .arg(&session.d_match_probs)
                    .arg(&session.d_levels)
                    .arg(&mut session.d_m_partials)
                    .arg(&mut session.d_u_partials)
                    .arg(&mut session.d_match_totals)
                    .arg(&mut session.d_nonmatch_totals)
                    .arg(&(n_pairs as u32))
                    .arg(&(n_fields as u32))
                    .arg(&num_blocks)
                    .launch(cfg1)
            }
            .map_err(|e| GpuError::LaunchFailed(format!("em_reduce pass 1: {e}")))
        })?;

        // ── M-step pass 2 ─────────────────────────────────────────────────────
        let cfg2 = LaunchConfig {
            grid_dim:         (n_cells + 2, 1, 1),
            block_dim:        (BLOCK_DIM, 1, 1),
            shared_mem_bytes: BLOCK_DIM * 4,
        };
        zer_prof::trace_cuda!("em_reduce_final_kernel", {
            unsafe {
                self.stream
                    .launch_builder(&self.em.final_fn)
                    .arg(&session.d_m_partials)
                    .arg(&session.d_u_partials)
                    .arg(&session.d_match_totals)
                    .arg(&session.d_nonmatch_totals)
                    .arg(&mut session.d_m_out)
                    .arg(&mut session.d_u_out)
                    .arg(&mut session.d_total_match)
                    .arg(&mut session.d_total_nonmatch)
                    .arg(&num_blocks)
                    .arg(&n_cells)
                    .launch(cfg2)
            }
            .map_err(|e| GpuError::LaunchFailed(format!("em_reduce pass 2: {e}")))
        })?;

        self.stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("em sync: {e}")))?;

        // Download ~80 bytes of counts.
        let used          = n_fields * NUM_LEVELS;
        let h_m_out       = download(&self.stream, &session.d_m_out)?;
        let h_u_out       = download(&self.stream, &session.d_u_out)?;
        let h_total_match    = download(&self.stream, &session.d_total_match)?;
        let h_total_nonmatch = download(&self.stream, &session.d_total_nonmatch)?;

        Ok(EmReduceOutput {
            m_counts:       h_m_out[..used].to_vec(),
            u_counts:       h_u_out[..used].to_vec(),
            total_match:    h_total_match[0],
            total_nonmatch: h_total_nonmatch[0],
        })
    }
}

impl KernelDispatch<EmReduce> for CudaDevice {
    fn dispatch(&self, input: EmReduceInput<'_>) -> Result<EmReduceOutput, GpuError> {
        let EmReduceInput { match_probs, comparison_levels, n_pairs, n_fields } = input;

        if n_pairs == 0 || n_fields == 0 {
            let zeros = vec![0.0f32; n_fields * NUM_LEVELS];
            return Ok(EmReduceOutput {
                m_counts: zeros.clone(), u_counts: zeros,
                total_match: 0.0, total_nonmatch: 0.0,
            });
        }
        if n_fields > MAX_FIELDS {
            return Err(GpuError::LaunchFailed(format!(
                "schema has {n_fields} fields but CUDA kernel supports at most {MAX_FIELDS}; \
                 increase MAX_FIELDS in em_reduce.cu and recompile"
            )));
        }

        let num_blocks = ((n_pairs as u32) + BLOCK_DIM - 1) / BLOCK_DIM;
        let n_cells    = (n_fields * NUM_LEVELS) as u32;

        // ── Upload inputs ─────────────────────────────────────────────────────
        let d_probs  = upload(&self.stream, match_probs)?;
        let d_levels = upload(&self.stream, comparison_levels)?;

        // ── Allocate pass-1 output buffers (cell-major layout) ────────────────
        // m_partials[cell * num_blocks + block]: n_cells rows  times  num_blocks cols.
        // Allocate MAX_FIELDS * NUM_LEVELS rows so the kernel can use the
        // compile-time SHARED_SZ constant without bounds checking.
        let partial_elems = (MAX_FIELDS * NUM_LEVELS) * num_blocks as usize;
        let mut d_m_partials      = alloc_zeros::<f32>(&self.stream, partial_elems)?;
        let mut d_u_partials      = alloc_zeros::<f32>(&self.stream, partial_elems)?;
        let mut d_match_totals    = alloc_zeros::<f32>(&self.stream, num_blocks as usize)?;
        let mut d_nonmatch_totals = alloc_zeros::<f32>(&self.stream, num_blocks as usize)?;

        // ── Pass 1 ────────────────────────────────────────────────────────────
        let cfg1 = LaunchConfig {
            grid_dim:         (num_blocks, 1, 1),
            block_dim:        (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        zer_prof::trace_cuda!("em_reduce_kernel", {
            unsafe {
                self.stream
                    .launch_builder(&self.em.partial_fn)
                    .arg(&d_probs)
                    .arg(&d_levels)
                    .arg(&mut d_m_partials)
                    .arg(&mut d_u_partials)
                    .arg(&mut d_match_totals)
                    .arg(&mut d_nonmatch_totals)
                    .arg(&(n_pairs as u32))
                    .arg(&(n_fields as u32))
                    .arg(&num_blocks)   // needed for cell-major stride
                    .launch(cfg1)
            }
            .map_err(|e| GpuError::LaunchFailed(format!("em_reduce pass 1: {e}")))
        })?;

        // ── Allocate pass-2 output buffers ────────────────────────────────────
        let out_elems = MAX_FIELDS * NUM_LEVELS;
        let mut d_m_out           = alloc_zeros::<f32>(&self.stream, out_elems)?;
        let mut d_u_out           = alloc_zeros::<f32>(&self.stream, out_elems)?;
        let mut d_total_match     = alloc_zeros::<f32>(&self.stream, 1)?;
        let mut d_total_nonmatch  = alloc_zeros::<f32>(&self.stream, 1)?;

        // ── Pass 2 ────────────────────────────────────────────────────────────
        // grid = (n_cells + 2, 1, 1): one block per cell + two scalar blocks.
        // shared_mem_bytes = BLOCK_DIM * sizeof(float): smem for tree reduction.
        let cfg2 = LaunchConfig {
            grid_dim:         (n_cells + 2, 1, 1),
            block_dim:        (BLOCK_DIM, 1, 1),
            shared_mem_bytes: BLOCK_DIM * 4,
        };
        zer_prof::trace_cuda!("em_reduce_final_kernel", {
            unsafe {
                self.stream
                    .launch_builder(&self.em.final_fn)
                    .arg(&d_m_partials)
                    .arg(&d_u_partials)
                    .arg(&d_match_totals)
                    .arg(&d_nonmatch_totals)
                    .arg(&mut d_m_out)
                    .arg(&mut d_u_out)
                    .arg(&mut d_total_match)
                    .arg(&mut d_total_nonmatch)
                    .arg(&num_blocks)
                    .arg(&n_cells)
                    .launch(cfg2)
            }
            .map_err(|e| GpuError::LaunchFailed(format!("em_reduce pass 2: {e}")))
        })?;

        self.stream.synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("em_reduce sync: {e}")))?;

        // ── Download and trim to n_fields*4 ──────────────────────────────────
        let h_m_out          = download(&self.stream, &d_m_out)?;
        let h_u_out          = download(&self.stream, &d_u_out)?;
        let h_total_match    = download(&self.stream, &d_total_match)?;
        let h_total_nonmatch = download(&self.stream, &d_total_nonmatch)?;

        let used = n_fields * NUM_LEVELS;
        Ok(EmReduceOutput {
            m_counts:       h_m_out[..used].to_vec(),
            u_counts:       h_u_out[..used].to_vec(),
            total_match:    h_total_match[0],
            total_nonmatch: h_total_nonmatch[0],
        })
    }
}
