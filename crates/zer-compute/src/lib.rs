//! `zer-compute`, hardware-accelerated backend for entity resolution.
//!
//! Provides [`DeviceComparator`] and [`DeviceScorer`] as drop-in replacements for
//! the CPU-only counterparts in `zer-compare`.  Both implement the same
//! [`zer_core::traits::Comparator`] and [`zer_core::traits::Scorer`]
//! traits, so the rest of the pipeline is backend-agnostic.
//!
//! # Backend selection
//!
//! ```rust
//! use std::sync::Arc;
//! use zer_compute::{GpuBackend, DeviceComparator, DeviceScorer};
//! use zer_core::schema::{FieldKind, SchemaBuilder};
//!
//! let schema = SchemaBuilder::new()
//!     .field("naam",  FieldKind::Name)
//!     .field("datum", FieldKind::Date)
//!     .build()
//!     .unwrap();
//!
//! // Auto-detect: tries CUDA → AVX2 → CPU in order.
//! let backend    = Arc::new(GpuBackend::auto_detect());
//! let comparator = DeviceComparator::new(Arc::clone(&backend), &schema).unwrap();
//! let scorer     = DeviceScorer::new(Arc::clone(&backend));
//! ```
//!
//! # Feature flags
//!
//! | Flag     | Description |
//! |----------|---|
//! | `cuda`   | NVIDIA CUDA via `cudarc`, requires CUDA toolkit at build time |
//! | `avx2`   | x86_64 AVX2 SIMD via `std::arch`, no external toolchain required |
//!
//! When no flag is set the crate compiles and runs normally using the
//! always-available scalar CPU fallback backed by `zer-compare`.

pub mod backend;
pub mod batch_sizer;
pub mod comparator;
pub mod error;
pub mod kernel;
pub mod kernels;
pub mod scorer;
pub mod soa;

pub use backend::{BackendPreference, DeviceBackend, GpuBackend};
pub use batch_sizer::BatchSizer;
pub use comparator::DeviceComparator;
pub use error::GpuError;
pub use scorer::DeviceScorer;
