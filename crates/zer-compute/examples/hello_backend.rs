//! Verify that the correct GPU kernel/shader actually executed.
//!
//! Each backend writes a **different magic token** baked into its kernel source:
//!
//!   CPU fallback → private to cpu/launch/hello_backend.rs
//!   CUDA kernel  → private to hello_backend.cu
//!   Vulkan shader → private to hello_backend.glsl
//!
//! The token values are intentionally not exported from Rust, the only way to
//! receive a non-zero token is for the actual kernel to have run on the device.
//! A zero token means the kernel never executed.
//!
//! Run with:
//!   cargo run -p zer-compute --example hello_backend
//!   cargo run -p zer-compute --features avx2   --example hello_backend
//!   cargo run -p zer-compute --features cuda   --example hello_backend
//!   cargo run -p zer-compute --features vulkan --example hello_backend

use zer_compute::{
    backend::{BackendPreference, DeviceBackend},
    kernels::hello_backend::{HelloBackend, HelloBackendInput},
};

fn main() {
    // ── Auto-detect (CUDA → Vulkan → AVX2 → CPU) ─────────────────────────────
    println!("=== Auto-detect ===");
    run_hello(&DeviceBackend::auto_detect());

    // ── Forced CPU ────────────────────────────────────────────────────────────
    println!("\n=== CPU (forced) ===");
    match DeviceBackend::from_preference(BackendPreference::Cpu) {
        Ok(b) => run_hello(&b),
        Err(e) => println!("  error: {e}"),
    }

    // ── Forced CUDA (requires --features cuda) ────────────────────────────────
    println!("\n=== CUDA (forced) ===");
    match DeviceBackend::from_preference(BackendPreference::Cuda) {
        Ok(b) => run_hello(&b),
        Err(e) => println!("  not available: {e}"),
    }

    // ── Forced Vulkan (requires --features vulkan) ────────────────────────────
    println!("\n=== Vulkan (forced) ===");
    match DeviceBackend::from_preference(BackendPreference::Vulkan) {
        Ok(b) => run_hello(&b),
        Err(e) => println!("  not available: {e}"),
    }

    // ── Forced AVX2 (requires --features avx2 and x86_64 CPU with AVX2) ──────
    println!("\n=== AVX2 (forced) ===");
    match DeviceBackend::from_preference(BackendPreference::Avx2) {
        Ok(b) => run_hello(&b),
        Err(e) => println!("  not available: {e}"),
    }
}

fn run_hello(backend: &DeviceBackend) {
    print!("  backend={:<6}  ", backend.name());

    match backend.run::<HelloBackend>(HelloBackendInput) {
        Ok(out) if out.token != 0 => {
            println!("token={:#010X}  ✓ kernel confirmed", out.token);
        }
        Ok(out) => {
            println!(
                "token={:#010X}  ✗ zero token, kernel did not execute",
                out.token
            );
        }
        Err(e) => println!("dispatch error: {e}"),
    }
}
