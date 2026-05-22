use crate::{
    error::GpuError,
    kernel::KernelDispatch,
    kernels::hello_backend::{HelloBackend, HelloBackendInput, HelloBackendOutput},
};

use super::super::device::CpuDevice;

const CPU_TOKEN: u32 = 0xC09F_CAFE;

impl KernelDispatch<HelloBackend> for CpuDevice {
    fn dispatch(&self, _input: HelloBackendInput) -> Result<HelloBackendOutput, GpuError> {
        Ok(HelloBackendOutput { token: CPU_TOKEN })
    }
}
