//! `HelloBackend` kernel, trivial diagnostic that proves a GPU kernel/shader
//! was actually invoked on the device.
//!
//! Each backend writes a **different** magic token into a 4-byte output buffer.
//! The token value is baked into each backend's kernel/shader source, only
//! that specific kernel running on the device can produce the correct non-zero
//! value.  A zero token means the kernel never executed.
//!
//! The per-backend magic values are intentionally private to their respective
//! launch modules so that a buggy Rust stub cannot accidentally return the
//! "right" answer without the kernel having run.
//!
//! See `examples/hello_backend.rs` for the canonical usage.

use crate::kernel::Kernel;

/// Marker for the diagnostic hello kernel.
pub struct HelloBackend;

/// No input needed, the kernel only writes a constant.
pub struct HelloBackendInput;

/// Output of the hello kernel.
pub struct HelloBackendOutput {
    /// Magic token written by the kernel.  Non-zero proves the kernel
    /// executed; zero means the kernel never ran or the output buffer was
    /// not written.  The exact value is private to each backend's launch
    /// module and should not be compared against a Rust-side constant.
    pub token: u32,
}

impl Kernel for HelloBackend {
    type Input<'a> = HelloBackendInput;
    type Output = HelloBackendOutput;
}
