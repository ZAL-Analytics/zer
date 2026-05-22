//! Vulkan device initialisation.
//!
//! `VulkanDevice` owns the full Vulkan object graph (instance, logical device,
//! allocator, pools, pipelines) and keeps per-kernel pipeline resources in
//! typed structs so that `device.rs` stays free of dispatch logic.
//!
//! The actual per-kernel dispatch logic lives in `launch/` submodules as
//! `impl KernelDispatch<K> for VulkanDevice`.

use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use std::sync::Mutex;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::error::GpuError;

// ── Per-kernel pipeline bundles ───────────────────────────────────────────────

pub(crate) struct HelloPipelines {
    pub shader_module:         vk::ShaderModule,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout:       vk::PipelineLayout,
    pub pipeline:              vk::Pipeline,
}

pub(crate) struct EmPipelines {
    pub estep_shader:          vk::ShaderModule,
    pub partial_shader:        vk::ShaderModule,
    pub final_shader:          vk::ShaderModule,
    pub estep_dsl:             vk::DescriptorSetLayout,
    pub partial_dsl:           vk::DescriptorSetLayout,
    pub final_dsl:             vk::DescriptorSetLayout,
    pub estep_layout:          vk::PipelineLayout,
    pub partial_layout:        vk::PipelineLayout,
    pub final_layout:          vk::PipelineLayout,
    pub estep_pipeline:        vk::Pipeline,
    pub partial_pipeline:      vk::Pipeline,
    pub final_pipeline:        vk::Pipeline,
}

// ── VulkanDevice ──────────────────────────────────────────────────────────────

/// Wrapper around a Vulkan logical device with pre-built compute pipelines.
///
/// Fields are `pub(crate)` so `launch/` submodules can access resources directly.
pub struct VulkanDevice {
    #[allow(dead_code)] // must outlive instance (Vulkan loader lifetime)
    pub(crate) entry:                ash::Entry,
    pub(crate) instance:             ash::Instance,
    #[allow(dead_code)] // needed for memory budget queries
    pub(crate) physical_device:      vk::PhysicalDevice,
    pub(crate) device:               ash::Device,
    pub(crate) compute_queue:        vk::Queue,
    #[allow(dead_code)] // needed for multi-queue and barrier operations
    pub(crate) compute_queue_family: u32,
    pub(crate) allocator:            ManuallyDrop<Mutex<Allocator>>,
    pub(crate) command_pool:         vk::CommandPool,
    pub(crate) descriptor_pool:      vk::DescriptorPool,
    pub(crate) pipeline_cache:       vk::PipelineCache,
    pub(crate) hello:                HelloPipelines,
    pub(crate) em:                   EmPipelines,

    #[cfg(debug_assertions)]
    debug_utils:    ash::ext::debug_utils::Instance,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,

    device_name: String,
    total_vram:  u64,
}

impl VulkanDevice {
    pub fn init() -> Result<Self, GpuError> {
        let vk_err = |ctx: &str, e: vk::Result| -> GpuError {
            GpuError::Vulkan(format!("{ctx}: {e}"))
        };

        // ── Load Vulkan entry point ───────────────────────────────────────────
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| GpuError::Vulkan(format!("failed to load Vulkan: {e}")))?;

        // ── Instance ──────────────────────────────────────────────────────────
        let app_name    = CString::new("zaggr").unwrap();
        let engine_name = CString::new("zer-compute").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::API_VERSION_1_3);

        let mut instance_extensions: Vec<*const i8> = vec![];

        #[cfg(debug_assertions)]
        let layers: Vec<*const i8> = {
            instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            static VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";
            let available = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
            if available.iter().any(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) } == VALIDATION) {
                vec![VALIDATION.as_ptr()]
            } else {
                vec![]
            }
        };
        #[cfg(not(debug_assertions))]
        let layers: Vec<*const i8> = vec![];

        instance_extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&layers);

        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(|e| vk_err("vkCreateInstance", e))?;

        // ── Debug messenger (debug builds only) ───────────────────────────────
        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) = {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let messenger = unsafe { du.create_debug_utils_messenger(&ci, None) }
                .map_err(|e| vk_err("vkCreateDebugUtilsMessenger", e))?;
            (du, messenger)
        };

        // ── Physical device selection ──────────────────────────────────────────
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| vk_err("vkEnumeratePhysicalDevices", e))?;

        if physical_devices.is_empty() {
            return Err(GpuError::Vulkan(
                "no Vulkan-capable GPU found".into(),
            ));
        }

        let (physical_device, compute_queue_family) =
            select_physical_device(&instance, &physical_devices)?;

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let total_vram: u64 = (0..mem_props.memory_heap_count as usize)
            .filter(|&i| {
                mem_props.memory_heaps[i].flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|i| mem_props.memory_heaps[i].size)
            .sum();

        tracing::info!(
            device = %device_name,
            vram_mib = total_vram / (1024 * 1024),
            "Vulkan: selected physical device"
        );

        // ── Logical device ────────────────────────────────────────────────────
        let queue_prios = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_prios);

        let device_extensions: Vec<*const i8> = vec![];

        // Enable VkPhysicalDeviceSynchronization2Features (required in Vk 1.3).
        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(true);
        // Core Vulkan 1.2 features, timeline_semaphore subsumes the standalone
        // VkPhysicalDeviceTimelineSemaphoreFeatures struct (VUID-VkDeviceCreateInfo-pNext-02830).
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .buffer_device_address(false);

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&device_extensions)
            .push_next(&mut sync2_features)
            .push_next(&mut vk12_features);

        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(|e| vk_err("vkCreateDevice", e))?;

        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        // ── gpu-allocator ─────────────────────────────────────────────────────
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance:             instance.clone(),
            device:               device.clone(),
            physical_device,
            debug_settings:       Default::default(),
            buffer_device_address: false,
            allocation_sizes:     Default::default(),
        })
        .map_err(|e| GpuError::Vulkan(format!("gpu_allocator::new: {e}")))?;

        // ── Command pool ──────────────────────────────────────────────────────
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| vk_err("vkCreateCommandPool", e))?;

        // ── Descriptor pool ───────────────────────────────────────────────────
        // 3 kernels × 3 descriptor sets each, plus some headroom.
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 64,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 8,
            },
        ];
        let desc_pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(32)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&desc_pool_ci, None) }
            .map_err(|e| vk_err("vkCreateDescriptorPool", e))?;

        // ── Pipeline cache ────────────────────────────────────────────────────
        let cache_ci = vk::PipelineCacheCreateInfo::default();
        let pipeline_cache = unsafe { device.create_pipeline_cache(&cache_ci, None) }
            .map_err(|e| vk_err("vkCreatePipelineCache", e))?;

        // ── Build pipelines ───────────────────────────────────────────────────
        let hello = build_hello_pipeline(&device, pipeline_cache)
            .map_err(|e| GpuError::Vulkan(format!("hello pipeline: {e}")))?;
        let em    = build_em_pipelines(&device, pipeline_cache)
            .map_err(|e| GpuError::Vulkan(format!("em pipelines: {e}")))?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            compute_queue_family,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            command_pool,
            descriptor_pool,
            pipeline_cache,
            hello,
            em,
            #[cfg(debug_assertions)]
            debug_utils,
            #[cfg(debug_assertions)]
            debug_messenger,
            device_name,
            total_vram,
        })
    }

    pub fn name(&self) -> &str { &self.device_name }
    pub fn total_vram_bytes(&self) -> u64 { self.total_vram }

    /// Remaining device-local memory estimated from gpu-allocator statistics.
    pub fn available_vram_bytes(&self) -> Option<u64> {
        let guard = self.allocator.lock().unwrap();
        let report = guard.generate_report();
        // Sum device-local heaps from the allocation report.
        let used: u64 = report.allocations.iter()
            .map(|a| a.size)
            .sum();
        Some(self.total_vram.saturating_sub(used))
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            // Destroy pipelines (hello).
            self.device.destroy_pipeline(self.hello.pipeline, None);
            self.device.destroy_pipeline_layout(self.hello.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.hello.descriptor_set_layout, None);
            self.device.destroy_shader_module(self.hello.shader_module, None);

            // Destroy pipelines (em).
            for &pipeline in &[self.em.estep_pipeline, self.em.partial_pipeline, self.em.final_pipeline] {
                self.device.destroy_pipeline(pipeline, None);
            }
            for &layout in &[self.em.estep_layout, self.em.partial_layout, self.em.final_layout] {
                self.device.destroy_pipeline_layout(layout, None);
            }
            for &dsl in &[self.em.estep_dsl, self.em.partial_dsl, self.em.final_dsl] {
                self.device.destroy_descriptor_set_layout(dsl, None);
            }
            for &sm in &[self.em.estep_shader, self.em.partial_shader, self.em.final_shader] {
                self.device.destroy_shader_module(sm, None);
            }

            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);

            // gpu-allocator must be dropped before vkDestroyDevice.
            // ManuallyDrop::drop runs the Allocator destructor (which frees all
            // VkDeviceMemory blocks) before we call destroy_device below.
            ManuallyDrop::drop(&mut self.allocator);

            self.device.destroy_device(None);

            #[cfg(debug_assertions)]
            self.debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);

            self.instance.destroy_instance(None);
        }
    }
}

// ── Physical device selection ─────────────────────────────────────────────────

fn select_physical_device(
    instance: &ash::Instance,
    devices:  &[vk::PhysicalDevice],
) -> Result<(vk::PhysicalDevice, u32), GpuError> {
    for &pd in devices {
        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(pd)
        };
        for (idx, qf) in queue_families.iter().enumerate() {
            if !qf.queue_flags.contains(vk::QueueFlags::COMPUTE) { continue; }

            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.limits.timestamp_compute_and_graphics == 0 { continue; }

            return Ok((pd, idx as u32));
        }
    }
    Err(GpuError::Vulkan(
        "no Vulkan physical device found with a compute queue and timestamp support".into(),
    ))
}

// ── SPIR-V shader module loader ───────────────────────────────────────────────

fn load_shader(device: &ash::Device, spv_bytes: &[u8]) -> Result<vk::ShaderModule, vk::Result> {
    assert!(spv_bytes.len() % 4 == 0, "SPIR-V must be 4-byte aligned");
    let code: Vec<u32> = spv_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ci = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&ci, None) }
}

// ── Pipeline builders ─────────────────────────────────────────────────────────

fn build_hello_pipeline(
    device:         &ash::Device,
    pipeline_cache: vk::PipelineCache,
) -> Result<HelloPipelines, vk::Result> {
    let spv = include_bytes!(concat!(env!("OUT_DIR"), "/hello_backend.spv"));
    let shader_module = load_shader(device, spv)?;

    // Descriptor set layout: binding 0 = storage buffer (output uint[]).
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&dsl_ci, None) }?;

    // Push constants: 4 bytes (dummy uint).
    let pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(4);
    let dsls = [descriptor_set_layout];
    let layout_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&dsls)
        .push_constant_ranges(std::slice::from_ref(&pc_range));
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_ci, None) }?;

    let entry = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry);
    let ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    let pipelines = unsafe {
        device.create_compute_pipelines(pipeline_cache, std::slice::from_ref(&ci), None)
    }
    .map_err(|(_, e)| e)?;

    Ok(HelloPipelines { shader_module, descriptor_set_layout, pipeline_layout, pipeline: pipelines[0] })
}

fn build_em_pipelines(
    device:         &ash::Device,
    pipeline_cache: vk::PipelineCache,
) -> Result<EmPipelines, vk::Result> {
    // ── E-step ────────────────────────────────────────────────────────────────
    let estep_spv    = include_bytes!(concat!(env!("OUT_DIR"), "/em_estep.spv"));
    let estep_shader = load_shader(device, estep_spv)?;

    let estep_bindings = [
        make_storage_binding(0), // levels
        make_storage_binding(1), // weights
        make_storage_binding(2), // probs output
    ];
    let estep_dsl = make_dsl(device, &estep_bindings)?;
    let estep_pc  = push_range(12); // EstepPush: n_pairs(4) + n_fields(4) + log_prior_odds(4)
    let estep_layout = make_pipeline_layout(device, estep_dsl, Some(estep_pc))?;
    let estep_pipeline = make_compute_pipeline(device, pipeline_cache, estep_shader, "main", estep_layout)?;

    // ── M-step pass 1 ─────────────────────────────────────────────────────────
    let partial_spv    = include_bytes!(concat!(env!("OUT_DIR"), "/em_mstep_partial.spv"));
    let partial_shader = load_shader(device, partial_spv)?;

    let partial_bindings = [
        make_storage_binding(0), // match_probs
        make_storage_binding(1), // comparison_levels
        make_storage_binding(2), // m_partials
        make_storage_binding(3), // u_partials
        make_storage_binding(4), // match_totals
        make_storage_binding(5), // nonmatch_totals
    ];
    let partial_dsl    = make_dsl(device, &partial_bindings)?;
    let partial_pc     = push_range(12); // MstepPush: n_pairs(4) + n_fields(4) + num_blocks(4)
    let partial_layout = make_pipeline_layout(device, partial_dsl, Some(partial_pc))?;
    let partial_pipeline = make_compute_pipeline(device, pipeline_cache, partial_shader, "main", partial_layout)?;

    // ── M-step pass 2 ─────────────────────────────────────────────────────────
    let final_spv    = include_bytes!(concat!(env!("OUT_DIR"), "/em_mstep_final.spv"));
    let final_shader = load_shader(device, final_spv)?;

    let final_bindings = [
        make_storage_binding(0), // m_partials
        make_storage_binding(1), // u_partials
        make_storage_binding(2), // match_totals
        make_storage_binding(3), // nonmatch_totals
        make_storage_binding(4), // m_out
        make_storage_binding(5), // u_out
        make_storage_binding(6), // total_match
        make_storage_binding(7), // total_nonmatch
    ];
    let final_dsl    = make_dsl(device, &final_bindings)?;
    let final_pc     = push_range(8); // FinalPush: num_blocks(4) + num_cells(4)
    let final_layout = make_pipeline_layout(device, final_dsl, Some(final_pc))?;
    let final_pipeline = make_compute_pipeline(device, pipeline_cache, final_shader, "main", final_layout)?;

    Ok(EmPipelines {
        estep_shader, partial_shader, final_shader,
        estep_dsl, partial_dsl, final_dsl,
        estep_layout, partial_layout, final_layout,
        estep_pipeline, partial_pipeline, final_pipeline,
    })
}

// ── Pipeline building helpers ─────────────────────────────────────────────────

fn make_storage_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn make_dsl(
    device:   &ash::Device,
    bindings: &[vk::DescriptorSetLayoutBinding<'_>],
) -> Result<vk::DescriptorSetLayout, vk::Result> {
    let ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
    unsafe { device.create_descriptor_set_layout(&ci, None) }
}

fn push_range(size: u32) -> vk::PushConstantRange {
    vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(size)
}

fn make_pipeline_layout(
    device: &ash::Device,
    dsl:    vk::DescriptorSetLayout,
    pc:     Option<vk::PushConstantRange>,
) -> Result<vk::PipelineLayout, vk::Result> {
    let dsls = [dsl];
    let mut ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls);
    if let Some(range) = pc.as_ref() {
        ci = ci.push_constant_ranges(std::slice::from_ref(range));
    }
    unsafe { device.create_pipeline_layout(&ci, None) }
}

fn make_compute_pipeline(
    device:         &ash::Device,
    cache:          vk::PipelineCache,
    shader_module:  vk::ShaderModule,
    entry_name:     &str,
    layout:         vk::PipelineLayout,
) -> Result<vk::Pipeline, vk::Result> {
    let entry = CString::new(entry_name).unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry);
    let ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipelines = unsafe {
        device.create_compute_pipelines(cache, std::slice::from_ref(&ci), None)
    }
    .map_err(|(_, e)| e)?;
    Ok(pipelines[0])
}

// ── Debug callback ────────────────────────────────────────────────────────────

#[cfg(debug_assertions)]
unsafe extern "system" fn vulkan_debug_callback(
    severity:    vk::DebugUtilsMessageSeverityFlagsEXT,
    _msg_type:   vk::DebugUtilsMessageTypeFlagsEXT,
    data:        *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data:  *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe {
        CStr::from_ptr((*data).p_message).to_string_lossy()
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!(target: "vulkan", "{msg}");
    } else {
        tracing::warn!(target: "vulkan", "{msg}");
    }
    vk::FALSE
}
