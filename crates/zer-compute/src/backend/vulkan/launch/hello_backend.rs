use ash::vk;

use crate::{
    backend::vulkan::{
        buffers::{buffer_barrier, cmd_copy_buffer, one_shot_submit, VulkanBuffer},
        device::VulkanDevice,
    },
    error::GpuError,
    kernel::KernelDispatch,
    kernels::hello_backend::{HelloBackend, HelloBackendInput, HelloBackendOutput},
};

impl KernelDispatch<HelloBackend> for VulkanDevice {
    fn dispatch(&self, _input: HelloBackendInput) -> Result<HelloBackendOutput, GpuError> {
        let vk_err = |ctx: &str, e: vk::Result| GpuError::LaunchFailed(format!("{ctx}: {e}"));

        let (d_out, staging_out) = {
            let mut alloc = self.allocator.lock().unwrap();
            let d = VulkanBuffer::new_device_local(
                &self.device,
                &mut alloc,
                4,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                "hello_d_out",
            )?;
            let s = VulkanBuffer::new_staging(&self.device, &mut alloc, 4, "hello_staging_out")?;
            (d, s)
        };

        let ds = {
            let ai = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.hello.descriptor_set_layout));
            unsafe { self.device.allocate_descriptor_sets(&ai) }
                .map_err(|e| vk_err("allocate_descriptor_sets (hello)", e))?[0]
        };

        let out_buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(d_out.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&out_buf_info);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };

        let device = &self.device;
        let pipeline = self.hello.pipeline;
        let layout = self.hello.pipeline_layout;

        one_shot_submit(
            device,
            self.command_pool,
            self.compute_queue,
            |cmd| unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[ds],
                    &[],
                );
                let dummy: u32 = 0;
                device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &dummy.to_ne_bytes(),
                );
                device.cmd_dispatch(cmd, 1, 1, 1);

                buffer_barrier(
                    device,
                    cmd,
                    &d_out,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::AccessFlags2::TRANSFER_READ,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::PipelineStageFlags2::COPY,
                );
                cmd_copy_buffer(device, cmd, &d_out, &staging_out);
            },
        )?;

        let token = staging_out.read::<u32>(1)[0];

        unsafe {
            let _ = self
                .device
                .free_descriptor_sets(self.descriptor_pool, &[ds]);
        }
        {
            let mut alloc = self.allocator.lock().unwrap();
            d_out.destroy(&self.device, &mut alloc);
            staging_out.destroy(&self.device, &mut alloc);
        }

        Ok(HelloBackendOutput { token })
    }
}
