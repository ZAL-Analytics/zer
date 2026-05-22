use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::{
    backend::vulkan::{
        buffers::{VulkanBuffer, buffer_barrier, cmd_copy_buffer, one_shot_submit},
        device::VulkanDevice,
    },
    error::GpuError,
    kernel::KernelDispatch,
    kernels::em_reduce::{EmReduce, EmReduceInput, EmReduceOutput},
};

const MAX_FIELDS: usize = 16;
const NUM_LEVELS: usize = 4;
const BLOCK_DIM:  u32   = 256;

// ── KernelDispatch ────────────────────────────────────────────────────────────

impl KernelDispatch<EmReduce> for VulkanDevice {
    fn dispatch(&self, input: EmReduceInput<'_>) -> Result<EmReduceOutput, GpuError> {
        let EmReduceInput { match_probs, comparison_levels, n_pairs, n_fields } = input;

        if n_pairs == 0 || n_fields == 0 {
            let zeros = vec![0.0f32; n_fields * NUM_LEVELS];
            return Ok(EmReduceOutput {
                m_counts: zeros.clone(), u_counts: zeros,
                total_match: 0.0, total_nonmatch: 0.0,
            });
        }

        let vk_err = |ctx: &str, e: vk::Result| GpuError::LaunchFailed(format!("{ctx}: {e}"));

        let num_blocks = (n_pairs as u32 + BLOCK_DIM - 1) / BLOCK_DIM;
        let n_cells    = (n_fields * NUM_LEVELS) as u32;
        let partial_elems = (MAX_FIELDS * NUM_LEVELS) * num_blocks as usize;
        let out_elems     = MAX_FIELDS * NUM_LEVELS;

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;

        // ── Allocate all buffers (single lock) ────────────────────────────────
        let (
            staging_probs, staging_levels,
            d_probs, d_levels,
            d_m_partials, d_u_partials, d_match_totals, d_nonmatch_totals,
            d_m_out, d_u_out, d_total_match, d_total_nonmatch,
            staging_m_out, staging_u_out, staging_total_match, staging_total_nonmatch,
        ) = {
            let mut a = self.allocator.lock().unwrap();
            macro_rules! staging {
                ($n:expr, $name:expr) => {
                    VulkanBuffer::new_staging(&self.device, &mut a, $n, $name)?
                };
            }
            macro_rules! local {
                ($n:expr, $name:expr) => {
                    VulkanBuffer::new_device_local(&self.device, &mut a, $n, storage, $name)?
                };
            }

            (
                staging!(n_pairs as u64 * 4,              "em_staging_probs"),
                staging!((n_fields * n_pairs) as u64 * 4, "em_staging_levels"),
                local!(n_pairs as u64 * 4,                "em_d_probs"),
                local!((n_fields * n_pairs) as u64 * 4,   "em_d_levels"),
                local!(partial_elems as u64 * 4,          "em_d_m_partials"),
                local!(partial_elems as u64 * 4,          "em_d_u_partials"),
                local!(num_blocks as u64 * 4,             "em_d_match_totals"),
                local!(num_blocks as u64 * 4,             "em_d_nonmatch_totals"),
                local!(out_elems as u64 * 4,              "em_d_m_out"),
                local!(out_elems as u64 * 4,              "em_d_u_out"),
                local!(4,                                  "em_d_total_match"),
                local!(4,                                  "em_d_total_nonmatch"),
                staging!(out_elems as u64 * 4,            "em_staging_m_out"),
                staging!(out_elems as u64 * 4,            "em_staging_u_out"),
                staging!(4,                                "em_staging_total_match"),
                staging!(4,                                "em_staging_total_nonmatch"),
            )
        };

        // ── Write inputs to staging ───────────────────────────────────────────
        staging_probs.write(match_probs);
        staging_levels.write(comparison_levels);

        // ── Allocate descriptor sets ──────────────────────────────────────────
        let alloc_partial = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.em.partial_dsl));
        let alloc_final = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.em.final_dsl));

        let ds_partial = unsafe { self.device.allocate_descriptor_sets(&alloc_partial) }
            .map_err(|e| vk_err("allocate_descriptor_sets (em_partial)", e))?[0];
        let ds_final = unsafe { self.device.allocate_descriptor_sets(&alloc_final) }
            .map_err(|e| vk_err("allocate_descriptor_sets (em_final)", e))?[0];

        // ── Update descriptor sets ────────────────────────────────────────────
        let make_buf_info = |buf: &VulkanBuffer| {
            vk::DescriptorBufferInfo::default().buffer(buf.buffer).offset(0).range(vk::WHOLE_SIZE)
        };

        // partial: bindings 0-5
        let pi: [vk::DescriptorBufferInfo; 6] = [
            make_buf_info(&d_probs),
            make_buf_info(&d_levels),
            make_buf_info(&d_m_partials),
            make_buf_info(&d_u_partials),
            make_buf_info(&d_match_totals),
            make_buf_info(&d_nonmatch_totals),
        ];
        let partial_writes: Vec<vk::WriteDescriptorSet> = pi
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds_partial)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        // final: bindings 0-7
        let fi: [vk::DescriptorBufferInfo; 8] = [
            make_buf_info(&d_m_partials),
            make_buf_info(&d_u_partials),
            make_buf_info(&d_match_totals),
            make_buf_info(&d_nonmatch_totals),
            make_buf_info(&d_m_out),
            make_buf_info(&d_u_out),
            make_buf_info(&d_total_match),
            make_buf_info(&d_total_nonmatch),
        ];
        let final_writes: Vec<vk::WriteDescriptorSet> = fi
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds_final)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        let mut all_writes = partial_writes;
        all_writes.extend(final_writes);
        unsafe { self.device.update_descriptor_sets(&all_writes, &[]) };

        // ── M-step push constants ─────────────────────────────────────────────
        // MstepPush { n_pairs: u32, n_fields: u32, num_blocks: u32 }
        let mstep_pc = {
            let mut b = [0u8; 12];
            b[0..4].copy_from_slice(&(n_pairs as u32).to_ne_bytes());
            b[4..8].copy_from_slice(&(n_fields as u32).to_ne_bytes());
            b[8..12].copy_from_slice(&num_blocks.to_ne_bytes());
            b
        };
        // FinalPush { num_blocks: u32, num_cells: u32 }
        let final_pc = {
            let mut b = [0u8; 8];
            b[0..4].copy_from_slice(&num_blocks.to_ne_bytes());
            b[4..8].copy_from_slice(&n_cells.to_ne_bytes());
            b
        };

        // ── Record, submit, wait ──────────────────────────────────────────────
        let device           = &self.device;
        let partial_pipeline = self.em.partial_pipeline;
        let partial_layout   = self.em.partial_layout;
        let final_pipeline   = self.em.final_pipeline;
        let final_layout     = self.em.final_layout;

        one_shot_submit(device, self.command_pool, self.compute_queue, |cmd| {
            unsafe {
                // H2D: staging → device buffers
                cmd_copy_buffer(device, cmd, &staging_probs,  &d_probs);
                cmd_copy_buffer(device, cmd, &staging_levels, &d_levels);

                // Barrier: transfer write → shader read
                for buf in [&d_probs, &d_levels] {
                    buffer_barrier(
                        device, cmd, buf,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::AccessFlags2::SHADER_STORAGE_READ,
                        vk::PipelineStageFlags2::COPY,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                    );
                }

                // ── Pass 1: em_mstep_partial ──────────────────────────────────
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, partial_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, partial_layout, 0, &[ds_partial], &[],
                );
                device.cmd_push_constants(
                    cmd, partial_layout, vk::ShaderStageFlags::COMPUTE, 0, &mstep_pc,
                );
                device.cmd_dispatch(cmd, num_blocks, 1, 1);

                // Barrier: shader write (partials, totals) → shader read (pass 2)
                for buf in [&d_m_partials, &d_u_partials, &d_match_totals, &d_nonmatch_totals] {
                    buffer_barrier(
                        device, cmd, buf,
                        vk::AccessFlags2::SHADER_STORAGE_WRITE,
                        vk::AccessFlags2::SHADER_STORAGE_READ,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                    );
                }

                // ── Pass 2: em_mstep_final ────────────────────────────────────
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, final_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, final_layout, 0, &[ds_final], &[],
                );
                device.cmd_push_constants(
                    cmd, final_layout, vk::ShaderStageFlags::COMPUTE, 0, &final_pc,
                );
                device.cmd_dispatch(cmd, n_cells + 2, 1, 1);

                // Barrier: shader write (final outputs) → transfer read (D2H)
                for buf in [&d_m_out, &d_u_out, &d_total_match, &d_total_nonmatch] {
                    buffer_barrier(
                        device, cmd, buf,
                        vk::AccessFlags2::SHADER_STORAGE_WRITE,
                        vk::AccessFlags2::TRANSFER_READ,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::PipelineStageFlags2::COPY,
                    );
                }

                // D2H: device → readback staging
                cmd_copy_buffer(device, cmd, &d_m_out,           &staging_m_out);
                cmd_copy_buffer(device, cmd, &d_u_out,           &staging_u_out);
                cmd_copy_buffer(device, cmd, &d_total_match,     &staging_total_match);
                cmd_copy_buffer(device, cmd, &d_total_nonmatch,  &staging_total_nonmatch);
            }
        })?;

        // ── Read back ─────────────────────────────────────────────────────────
        let h_m_out          = staging_m_out.read::<f32>(out_elems);
        let h_u_out          = staging_u_out.read::<f32>(out_elems);
        let h_total_match    = staging_total_match.read::<f32>(1);
        let h_total_nonmatch = staging_total_nonmatch.read::<f32>(1);

        let used = n_fields * NUM_LEVELS;

        // ── Cleanup ───────────────────────────────────────────────────────────
        unsafe {
            let _ = self.device.free_descriptor_sets(
                self.descriptor_pool, &[ds_partial, ds_final],
            );
        }
        {
            let mut a = self.allocator.lock().unwrap();
            staging_probs.destroy(&self.device, &mut a);
            staging_levels.destroy(&self.device, &mut a);
            d_probs.destroy(&self.device, &mut a);
            d_levels.destroy(&self.device, &mut a);
            d_m_partials.destroy(&self.device, &mut a);
            d_u_partials.destroy(&self.device, &mut a);
            d_match_totals.destroy(&self.device, &mut a);
            d_nonmatch_totals.destroy(&self.device, &mut a);
            d_m_out.destroy(&self.device, &mut a);
            d_u_out.destroy(&self.device, &mut a);
            d_total_match.destroy(&self.device, &mut a);
            d_total_nonmatch.destroy(&self.device, &mut a);
            staging_m_out.destroy(&self.device, &mut a);
            staging_u_out.destroy(&self.device, &mut a);
            staging_total_match.destroy(&self.device, &mut a);
            staging_total_nonmatch.destroy(&self.device, &mut a);
        }

        Ok(EmReduceOutput {
            m_counts:       h_m_out[..used].to_vec(),
            u_counts:       h_u_out[..used].to_vec(),
            total_match:    h_total_match[0],
            total_nonmatch: h_total_nonmatch[0],
        })
    }
}

// ── Session-based full-GPU EM ─────────────────────────────────────────────────

/// Pre-allocated GPU buffers and Vulkan objects for the full EM session.
///
/// Allocated once via `VulkanDevice::em_init_session`, reused every iteration.
/// Only the ~256-byte weight table is transferred per iteration.
///
/// Call `destroy` when done, `VulkanBuffer` has no `Drop` impl.
pub(crate) struct VulkanEmSession {
    // Device-local buffers (persistent across iterations).
    d_levels:          VulkanBuffer,
    d_weights:         VulkanBuffer,
    d_probs:           VulkanBuffer,
    d_m_partials:      VulkanBuffer,
    d_u_partials:      VulkanBuffer,
    d_match_totals:    VulkanBuffer,
    d_nonmatch_totals: VulkanBuffer,
    d_m_out:           VulkanBuffer,
    d_u_out:           VulkanBuffer,
    d_total_match:     VulkanBuffer,
    d_total_nonmatch:  VulkanBuffer,
    // Host-visible staging buffers.
    staging_weights:        VulkanBuffer,
    staging_m_out:          VulkanBuffer,
    staging_u_out:          VulkanBuffer,
    staging_total_match:    VulkanBuffer,
    staging_total_nonmatch: VulkanBuffer,
    // Persistent Vulkan objects (freed in destroy).
    ds_estep:        vk::DescriptorSet,
    ds_partial:      vk::DescriptorSet,
    ds_final:        vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,
    cmd:             vk::CommandBuffer,
    command_pool:    vk::CommandPool,
    fence:           vk::Fence,
    // Dimensions.
    n_pairs:    usize,
    n_fields:   usize,
    num_blocks: u32,
    n_cells:    u32,
    out_elems:  usize,
}

impl VulkanEmSession {
    pub(crate) fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            let _ = device.free_descriptor_sets(
                self.descriptor_pool, &[self.ds_estep, self.ds_partial, self.ds_final],
            );
            device.free_command_buffers(self.command_pool, &[self.cmd]);
            device.destroy_fence(self.fence, None);
        }
        self.d_levels.destroy(device, allocator);
        self.d_weights.destroy(device, allocator);
        self.d_probs.destroy(device, allocator);
        self.d_m_partials.destroy(device, allocator);
        self.d_u_partials.destroy(device, allocator);
        self.d_match_totals.destroy(device, allocator);
        self.d_nonmatch_totals.destroy(device, allocator);
        self.d_m_out.destroy(device, allocator);
        self.d_u_out.destroy(device, allocator);
        self.d_total_match.destroy(device, allocator);
        self.d_total_nonmatch.destroy(device, allocator);
        self.staging_weights.destroy(device, allocator);
        self.staging_m_out.destroy(device, allocator);
        self.staging_u_out.destroy(device, allocator);
        self.staging_total_match.destroy(device, allocator);
        self.staging_total_nonmatch.destroy(device, allocator);
    }
}

impl VulkanDevice {
    /// Allocate all EM GPU buffers and upload `comparison_levels` once.
    ///
    /// The returned session is reused across all EM iterations; only the
    /// ~256-byte weight table is transferred per iteration.
    pub(crate) fn em_init_session(
        &self,
        comparison_levels: &[u32],
        n_pairs:  usize,
        n_fields: usize,
    ) -> Result<VulkanEmSession, GpuError> {
        let vk_err = |ctx: &str, e: vk::Result| GpuError::LaunchFailed(format!("{ctx}: {e}"));

        let num_blocks    = (n_pairs as u32 + BLOCK_DIM - 1) / BLOCK_DIM;
        let n_cells       = (n_fields * NUM_LEVELS) as u32;
        let partial_elems = MAX_FIELDS * NUM_LEVELS * num_blocks as usize;
        let out_elems     = MAX_FIELDS * NUM_LEVELS;
        let storage       = vk::BufferUsageFlags::STORAGE_BUFFER;

        // ── Allocate all persistent buffers (single allocator lock) ───────
        let (
            staging_levels_init,
            d_levels, d_weights, d_probs,
            d_m_partials, d_u_partials, d_match_totals, d_nonmatch_totals,
            d_m_out, d_u_out, d_total_match, d_total_nonmatch,
            staging_weights,
            staging_m_out, staging_u_out, staging_total_match, staging_total_nonmatch,
        ) = {
            let mut a = self.allocator.lock().unwrap();
            macro_rules! staging {
                ($n:expr, $name:expr) => { VulkanBuffer::new_staging(&self.device, &mut a, $n, $name)? };
            }
            macro_rules! local {
                ($n:expr, $name:expr) => {
                    VulkanBuffer::new_device_local(&self.device, &mut a, $n, storage, $name)?
                };
            }
            (
                staging!((n_fields * n_pairs) as u64 * 4,      "em_sess_staging_levels"),
                local!((n_fields * n_pairs) as u64 * 4,        "em_sess_d_levels"),
                local!((n_fields * NUM_LEVELS) as u64 * 4,     "em_sess_d_weights"),
                local!(n_pairs as u64 * 4,                     "em_sess_d_probs"),
                local!(partial_elems as u64 * 4,               "em_sess_d_m_partials"),
                local!(partial_elems as u64 * 4,               "em_sess_d_u_partials"),
                local!(num_blocks as u64 * 4,                  "em_sess_d_match_totals"),
                local!(num_blocks as u64 * 4,                  "em_sess_d_nonmatch_totals"),
                local!(out_elems as u64 * 4,                   "em_sess_d_m_out"),
                local!(out_elems as u64 * 4,                   "em_sess_d_u_out"),
                local!(4,                                       "em_sess_d_total_match"),
                local!(4,                                       "em_sess_d_total_nonmatch"),
                staging!((n_fields * NUM_LEVELS) as u64 * 4,   "em_sess_staging_weights"),
                staging!(out_elems as u64 * 4,                 "em_sess_staging_m_out"),
                staging!(out_elems as u64 * 4,                 "em_sess_staging_u_out"),
                staging!(4,                                     "em_sess_staging_total_match"),
                staging!(4,                                     "em_sess_staging_total_nonmatch"),
            )
        };

        // ── Upload comparison_levels once, then free the temporary staging ─
        staging_levels_init.write(comparison_levels);
        one_shot_submit(&self.device, self.command_pool, self.compute_queue, |cmd| {
            cmd_copy_buffer(&self.device, cmd, &staging_levels_init, &d_levels);
        })?;
        {
            let mut a = self.allocator.lock().unwrap();
            staging_levels_init.destroy(&self.device, &mut a);
        }

        // ── Allocate persistent descriptor sets ───────────────────────────
        let ds_estep = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(std::slice::from_ref(&self.em.estep_dsl)),
            )
        }
        .map_err(|e| vk_err("allocate ds_estep", e))?[0];

        let ds_partial = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(std::slice::from_ref(&self.em.partial_dsl)),
            )
        }
        .map_err(|e| vk_err("allocate ds_partial", e))?[0];

        let ds_final = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(std::slice::from_ref(&self.em.final_dsl)),
            )
        }
        .map_err(|e| vk_err("allocate ds_final", e))?[0];

        // ── Update descriptor sets ────────────────────────────────────────
        let make_buf_info = |buf: &VulkanBuffer| {
            vk::DescriptorBufferInfo::default().buffer(buf.buffer).offset(0).range(vk::WHOLE_SIZE)
        };

        // E-step: bindings 0=levels, 1=weights, 2=probs
        let ei: [vk::DescriptorBufferInfo; 3] = [
            make_buf_info(&d_levels),
            make_buf_info(&d_weights),
            make_buf_info(&d_probs),
        ];
        let estep_writes: Vec<vk::WriteDescriptorSet> = ei.iter().enumerate().map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds_estep).dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        }).collect();

        // M-step partial: bindings 0=probs, 1=levels, 2=m_partials, 3=u_partials, 4=match_totals, 5=nonmatch_totals
        let pi: [vk::DescriptorBufferInfo; 6] = [
            make_buf_info(&d_probs),
            make_buf_info(&d_levels),
            make_buf_info(&d_m_partials),
            make_buf_info(&d_u_partials),
            make_buf_info(&d_match_totals),
            make_buf_info(&d_nonmatch_totals),
        ];
        let partial_writes: Vec<vk::WriteDescriptorSet> = pi.iter().enumerate().map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds_partial).dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        }).collect();

        // M-step final: bindings 0-3=partials+totals (read), 4-7=outputs (write)
        let fi: [vk::DescriptorBufferInfo; 8] = [
            make_buf_info(&d_m_partials),
            make_buf_info(&d_u_partials),
            make_buf_info(&d_match_totals),
            make_buf_info(&d_nonmatch_totals),
            make_buf_info(&d_m_out),
            make_buf_info(&d_u_out),
            make_buf_info(&d_total_match),
            make_buf_info(&d_total_nonmatch),
        ];
        let final_writes: Vec<vk::WriteDescriptorSet> = fi.iter().enumerate().map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds_final).dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        }).collect();

        let mut all_writes = estep_writes;
        all_writes.extend(partial_writes);
        all_writes.extend(final_writes);
        unsafe { self.device.update_descriptor_sets(&all_writes, &[]) };

        // ── Persistent command buffer + fence ─────────────────────────────
        // Pool has RESET_COMMAND_BUFFER, individual reset is legal.
        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| vk_err("allocate em session cmd", e))?[0];

        let fence = unsafe {
            self.device.create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| vk_err("create em session fence", e))?;

        Ok(VulkanEmSession {
            d_levels, d_weights, d_probs,
            d_m_partials, d_u_partials, d_match_totals, d_nonmatch_totals,
            d_m_out, d_u_out, d_total_match, d_total_nonmatch,
            staging_weights, staging_m_out, staging_u_out,
            staging_total_match, staging_total_nonmatch,
            ds_estep, ds_partial, ds_final,
            descriptor_pool: self.descriptor_pool,
            cmd,
            command_pool: self.command_pool,
            fence,
            n_pairs, n_fields, num_blocks, n_cells,
            out_elems,
        })
    }

    /// Run one EM iteration (E-step + M-step) entirely on GPU.
    ///
    /// Only the `n_fields * 4` weight floats cross PCIe; all other data stays on device.
    pub(crate) fn em_run_iteration(
        &self,
        session:        &mut VulkanEmSession,
        weights:        &[f32],
        log_prior_odds: f32,
    ) -> Result<EmReduceOutput, GpuError> {
        let vk_err = |ctx: &str, e: vk::Result| GpuError::LaunchFailed(format!("{ctx}: {e}"));

        let n_pairs    = session.n_pairs as u32;
        let n_fields   = session.n_fields as u32;
        let num_blocks = session.num_blocks;
        let n_cells    = session.n_cells;

        // Write weight table to host-visible staging (n_fields * 4 floats ≈ 160 bytes).
        session.staging_weights.write(weights);

        // Push constant payloads.
        let estep_pc = {
            let mut b = [0u8; 12];
            b[0..4].copy_from_slice(&n_pairs.to_ne_bytes());
            b[4..8].copy_from_slice(&n_fields.to_ne_bytes());
            b[8..12].copy_from_slice(&log_prior_odds.to_ne_bytes());
            b
        };
        let mstep_pc = {
            let mut b = [0u8; 12];
            b[0..4].copy_from_slice(&n_pairs.to_ne_bytes());
            b[4..8].copy_from_slice(&n_fields.to_ne_bytes());
            b[8..12].copy_from_slice(&num_blocks.to_ne_bytes());
            b
        };
        let final_pc = {
            let mut b = [0u8; 8];
            b[0..4].copy_from_slice(&num_blocks.to_ne_bytes());
            b[4..8].copy_from_slice(&n_cells.to_ne_bytes());
            b
        };

        let device          = &self.device;
        let cmd             = session.cmd;
        let estep_pipeline  = self.em.estep_pipeline;
        let estep_layout    = self.em.estep_layout;
        let partial_pipeline = self.em.partial_pipeline;
        let partial_layout  = self.em.partial_layout;
        let final_pipeline  = self.em.final_pipeline;
        let final_layout    = self.em.final_layout;

        // ── Re-record command buffer ──────────────────────────────────────
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| vk_err("reset_command_buffer", e))?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| vk_err("begin_command_buffer", e))?;

            // H2D: staging_weights → d_weights (only PCIe transfer per iteration)
            cmd_copy_buffer(device, cmd, &session.staging_weights, &session.d_weights);
            buffer_barrier(device, cmd, &session.d_weights,
                vk::AccessFlags2::TRANSFER_WRITE,    vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::PipelineStageFlags2::COPY,       vk::PipelineStageFlags2::COMPUTE_SHADER);

            // E-step: levels + weights → probs
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, estep_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, estep_layout, 0, &[session.ds_estep], &[]);
            device.cmd_push_constants(
                cmd, estep_layout, vk::ShaderStageFlags::COMPUTE, 0, &estep_pc);
            device.cmd_dispatch(cmd, num_blocks, 1, 1);

            buffer_barrier(device, cmd, &session.d_probs,
                vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::PipelineStageFlags2::COMPUTE_SHADER);

            // M-step pass 1: probs + levels → per-block partials
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, partial_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, partial_layout, 0, &[session.ds_partial], &[]);
            device.cmd_push_constants(
                cmd, partial_layout, vk::ShaderStageFlags::COMPUTE, 0, &mstep_pc);
            device.cmd_dispatch(cmd, num_blocks, 1, 1);

            for buf in [&session.d_m_partials, &session.d_u_partials,
                        &session.d_match_totals, &session.d_nonmatch_totals] {
                buffer_barrier(device, cmd, buf,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::AccessFlags2::SHADER_STORAGE_READ,
                    vk::PipelineStageFlags2::COMPUTE_SHADER, vk::PipelineStageFlags2::COMPUTE_SHADER);
            }

            // M-step pass 2: partials → final m/u counts
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, final_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, final_layout, 0, &[session.ds_final], &[]);
            device.cmd_push_constants(
                cmd, final_layout, vk::ShaderStageFlags::COMPUTE, 0, &final_pc);
            device.cmd_dispatch(cmd, n_cells + 2, 1, 1);

            for buf in [&session.d_m_out, &session.d_u_out,
                        &session.d_total_match, &session.d_total_nonmatch] {
                buffer_barrier(device, cmd, buf,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::AccessFlags2::TRANSFER_READ,
                    vk::PipelineStageFlags2::COMPUTE_SHADER, vk::PipelineStageFlags2::COPY);
            }

            // D2H: device outputs → readback staging
            cmd_copy_buffer(device, cmd, &session.d_m_out,          &session.staging_m_out);
            cmd_copy_buffer(device, cmd, &session.d_u_out,          &session.staging_u_out);
            cmd_copy_buffer(device, cmd, &session.d_total_match,    &session.staging_total_match);
            cmd_copy_buffer(device, cmd, &session.d_total_nonmatch, &session.staging_total_nonmatch);

            device.end_command_buffer(cmd)
                .map_err(|e| vk_err("end_command_buffer", e))?;
        }

        // ── Submit and wait ───────────────────────────────────────────────
        unsafe {
            device.reset_fences(&[session.fence])
                .map_err(|e| vk_err("reset_fences", e))?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&session.cmd));
            device.queue_submit(self.compute_queue, std::slice::from_ref(&submit), session.fence)
                .map_err(|e| vk_err("queue_submit", e))?;
            device.wait_for_fences(&[session.fence], true, u64::MAX)
                .map_err(|e| vk_err("wait_for_fences", e))?;
        }

        // ── Read back ─────────────────────────────────────────────────────
        let out_elems = session.out_elems;
        let used      = session.n_fields * NUM_LEVELS;
        let h_m_out          = session.staging_m_out.read::<f32>(out_elems);
        let h_u_out          = session.staging_u_out.read::<f32>(out_elems);
        let h_total_match    = session.staging_total_match.read::<f32>(1);
        let h_total_nonmatch = session.staging_total_nonmatch.read::<f32>(1);

        Ok(EmReduceOutput {
            m_counts:       h_m_out[..used].to_vec(),
            u_counts:       h_u_out[..used].to_vec(),
            total_match:    h_total_match[0],
            total_nonmatch: h_total_nonmatch[0],
        })
    }
}
