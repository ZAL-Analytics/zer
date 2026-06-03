//! Vulkan buffer management, allocation, upload, download, and the pre-allocated
//! EM pool.

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::error::GpuError;

// ── VulkanBuffer ──────────────────────────────────────────────────────────────

/// RAII wrapper around a `VkBuffer` + a `gpu_allocator` allocation.
pub(crate) struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
}

impl VulkanBuffer {
    /// Allocate a host-visible, host-coherent staging buffer.
    pub fn new_staging(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        name: &str,
    ) -> Result<Self, GpuError> {
        Self::new_inner(
            device,
            allocator,
            size,
            name,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
        )
    }

    /// Allocate a device-local buffer (not host-visible).
    pub fn new_device_local(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        name: &str,
    ) -> Result<Self, GpuError> {
        Self::new_inner(
            device,
            allocator,
            size,
            name,
            usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
        )
    }

    fn new_inner(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        name: &str,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self, GpuError> {
        let buf_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buf_ci, None) }.map_err(|e| {
            GpuError::AllocationFailed {
                requested_bytes: size,
                detail: format!("vkCreateBuffer: {e}"),
            }
        })?;

        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name,
                requirements: reqs,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| GpuError::AllocationFailed {
                requested_bytes: size,
                detail: format!("gpu_allocator: {e}"),
            })?;

        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
            .map_err(|e| GpuError::AllocationFailed {
                requested_bytes: size,
                detail: format!("vkBindBufferMemory: {e}"),
            })?;

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    /// Map the buffer (host-visible only) and return a raw pointer for writing.
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.allocation
            .as_ref()?
            .mapped_ptr()
            .map(|p| p.as_ptr() as *mut u8)
    }

    /// Write `data` into the host-visible mapping.
    ///
    /// # Panics
    /// Panics if the buffer is not host-visible or `data.len()` exceeds the buffer size.
    pub fn write<T: Copy>(&self, data: &[T]) {
        let ptr = self.mapped_ptr().expect("buffer is not host-visible");
        let byte_len = data.len() * std::mem::size_of::<T>();
        assert!(
            byte_len as u64 <= self.size,
            "write exceeds buffer capacity"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr, byte_len);
        }
    }

    /// Read back `count` elements of type T from the host-visible mapping.
    pub fn read<T: Copy>(&self, count: usize) -> Vec<T> {
        let ptr = self.mapped_ptr().expect("buffer is not host-visible");
        let byte_len = count * std::mem::size_of::<T>();
        assert!(byte_len as u64 <= self.size, "read exceeds buffer size");
        let mut out = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr() as *mut u8, byte_len);
            out.set_len(count);
        }
        out
    }

    /// Destroy the buffer and free the allocation.
    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
        unsafe { device.destroy_buffer(self.buffer, None) };
    }
}

// ── Upload / download helpers ────────────────────────────────────────────────

/// Record a full-buffer host-to-device copy on `cmd`.
pub(crate) fn cmd_copy_buffer(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
) {
    let region = vk::BufferCopy::default()
        .src_offset(0)
        .dst_offset(0)
        .size(src.size.min(dst.size));
    unsafe { device.cmd_copy_buffer(cmd, src.buffer, dst.buffer, std::slice::from_ref(&region)) };
}

/// Insert a buffer memory barrier that makes a preceding write visible to subsequent reads.
pub(crate) fn buffer_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    buffer: &VulkanBuffer,
    src_access: vk::AccessFlags2,
    dst_access: vk::AccessFlags2,
    src_stage: vk::PipelineStageFlags2,
    dst_stage: vk::PipelineStageFlags2,
) {
    let barrier = vk::BufferMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .buffer(buffer.buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    let dep = vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };
}

// ── Single-command-buffer submit helper ───────────────────────────────────────

/// Allocate, record, submit, and wait on a one-shot command buffer.
///
/// `record_fn` receives the command buffer; after it returns, the buffer is
/// submitted to `queue` and the host blocks until the GPU finishes.
pub(crate) fn one_shot_submit<F>(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    record_fn: F,
) -> Result<(), GpuError>
where
    F: FnOnce(vk::CommandBuffer),
{
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmds = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| GpuError::LaunchFailed(format!("allocate_command_buffers: {e}")))?;
    let cmd = cmds[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info) }
        .map_err(|e| GpuError::LaunchFailed(format!("begin_command_buffer: {e}")))?;

    record_fn(cmd);

    unsafe { device.end_command_buffer(cmd) }
        .map_err(|e| GpuError::LaunchFailed(format!("end_command_buffer: {e}")))?;

    let fence_ci = vk::FenceCreateInfo::default();
    let fence = unsafe { device.create_fence(&fence_ci, None) }
        .map_err(|e| GpuError::LaunchFailed(format!("create_fence: {e}")))?;

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    unsafe { device.queue_submit(queue, std::slice::from_ref(&submit_info), fence) }
        .map_err(|e| GpuError::LaunchFailed(format!("queue_submit: {e}")))?;

    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX) }
        .map_err(|e| GpuError::LaunchFailed(format!("wait_for_fences: {e}")))?;

    unsafe {
        device.destroy_fence(fence, None);
        device.free_command_buffers(pool, &cmds);
    }
    Ok(())
}
