//! Typed kernel descriptors, one module per compute operation.
//!
//! Each submodule defines:
//! - A zero-sized marker struct (e.g. `CompareScore`)
//! - Typed `Input` and `Output` structs
//! - `impl Kernel for <Marker>` binding them together
//!
//! The actual dispatch logic (upload / launch / download) lives in the
//! per-backend `launch/` modules; these files only carry types.

pub mod em_reduce;
pub mod hello_backend;

pub use em_reduce::EmReduce;
pub use hello_backend::HelloBackend;
