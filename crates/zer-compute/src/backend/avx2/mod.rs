//! AVX2 SIMD backend, x86_64 only, no external toolchain required.
//!
//! Enabled with `--features avx2`.  Availability is verified at runtime via
//! `is_x86_feature_detected!("avx2")` before any intrinsic is called.

pub mod device;
pub mod launch;

pub use device::Avx2Device;
