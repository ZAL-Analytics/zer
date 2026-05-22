use crate::{
    backend::avx2::device::Avx2Device,
    error::GpuError,
    kernel::KernelDispatch,
    kernels::hello_backend::{HelloBackend, HelloBackendInput, HelloBackendOutput},
};

const AVX2_TOKEN: u32 = 0xA4F2_7E01;

impl KernelDispatch<HelloBackend> for Avx2Device {
    fn dispatch(&self, _input: HelloBackendInput) -> Result<HelloBackendOutput, GpuError> {
        Ok(HelloBackendOutput { token: AVX2_TOKEN })
    }
}
