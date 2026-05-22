//! `zer-lib`, unified entity resolution library.
//!
//! Provides [`Comparator`], [`Scorer`], and a [`Backend`] abstraction that
//! selects GPU acceleration automatically when compiled with the `cuda` or
//! `vulkan` features and suitable hardware is present.  Without those features
//! the crate compiles and runs entirely on CPU via `zer-compare`.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use zer_lib::prelude::*;
//!
//! let schema = SchemaBuilder::new()
//!     .field("naam",  FieldKind::Name)
//!     .field("datum", FieldKind::Date)
//!     .build().unwrap();
//!
//! let backend    = Backend::auto_detect();        // CUDA → Vulkan → AVX2 → CPU
//! let comparator = Comparator::new(&schema, &backend);
//! let scorer     = Scorer::new(&backend);
//! ```
//!
//! # Feature flags
//!
//! **Compute backends** (mutually exclusive in practice; pick one):
//!
//! | Flag             | Description                                                              |
//! |------------------|--------------------------------------------------------------------------|
//! | `cuda`           | NVIDIA CUDA via `zer-compute`, requires CUDA Toolkit 13.1+ and `nvcc`   |
//! | `vulkan`         | Vulkan 1.3 compute via `zer-compute`, requires `slangc` on `PATH`        |
//! | `avx2`           | x86_64 AVX2 SIMD via `zer-compute`, no external toolchain required       |
//! | `cpu`            | Explicit scalar CPU path via `zer-compute` (Rayon parallel)              |
//! | `debug-shaders`  | Embed debug info in CUDA kernels for `cuda-gdb` / Nsight (needs `cuda`) |
//!
//! **Pipeline integration:**
//!
//! | Flag       | Description                                                              |
//! |------------|--------------------------------------------------------------------------|
//! | `pipeline` | Enable `Pipeline`, `Ingester`, and related types from `zer-pipeline`     |
//!
//! **Neural judge ORT execution providers** (independent of compute backend):
//!
//! | Flag             | Description                                                              |
//! |------------------|--------------------------------------------------------------------------|
//! | `judge_cpu`      | Scalar CPU execution provider for ORT (no extra dependencies)            |
//! | `judge_cuda`     | NVIDIA CUDA execution provider for ORT                                   |
//! | `judge_rocm`     | AMD ROCm execution provider for ORT                                      |
//! | `judge_directml` | Windows DirectML execution provider for ORT                              |
//! | `judge_openvino` | Intel OpenVINO execution provider for ORT                                |
//!
//! # CPU-only usage
//!
//! Users who never need GPU can depend on `zer-compare` directly and never
//! import this crate.  `zer_compare::FieldComparator` and
//! `zer_compare::FellegiSunterScorer` are the raw CPU implementations.

#[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
use std::sync::Arc;

use zer_core::{
    comparison::{ComparisonBatch, ComparisonVector},
    record::Record,
    record_pool::RecordPool,
    schema::Schema,
    scoring::{ModelParams, ScoredPair},
    traits::{Comparator as ComparatorTrait, Result as ZerResult, Scorer as ScorerTrait},
};

// ── Backend ───────────────────────────────────────────────────────────────────

enum BackendInner {
    Cpu,
    #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
    Gpu(Arc<zer_compute::DeviceBackend>),
}

/// Opaque compute backend handle.
///
/// Create once and share between [`Comparator`] and [`Scorer`] so both use the
/// same underlying GPU device.
///
/// ```rust,no_run
/// use zer_lib::prelude::*;
///
/// let schema     = SchemaBuilder::new().field("naam", FieldKind::Name).build().unwrap();
/// let backend    = Backend::auto_detect();
/// let comparator = Comparator::new(&schema, &backend);
/// let scorer     = Scorer::new(&backend);
/// ```
pub struct Backend {
    inner: BackendInner,
    name:  &'static str,
}

impl Backend {
    /// Read `--target=<name>` from process args and return the matching backend.
    ///
    /// Falls back to CPU when the flag is absent, no hardware probing.
    /// Pass `--target=auto` to restore the hardware-detection order
    /// (CUDA → Vulkan → AVX2 → CPU).
    pub fn auto_detect() -> Self {
        match std::env::args()
            .find_map(|a| a.strip_prefix("--target=").map(str::to_owned))
            .as_deref()
        {
            Some(t) => Self::from_target(t),
            None    => Self::cpu(),
        }
    }

    /// Force the CPU backend regardless of available hardware.
    pub fn cpu() -> Self {
        Self { inner: BackendInner::Cpu, name: "cpu" }
    }

    /// Select a backend by name, called by `auto_detect()` to resolve `--target=<name>`.
    ///
    /// Accepted values: `"auto"` (hardware-detect), `"cpu"`, `"cuda"`, `"avx2"`, `"vulkan"`.
    ///
    /// Exits with a diagnostic if the target is unknown, not compiled in, or hardware init fails.
    pub fn from_target(target: &str) -> Self {
        if target == "cpu" {
            return Self::cpu();
        }

        #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
        {
            let pref = match target {
                "auto"   => zer_compute::BackendPreference::Auto,
                "cuda"   => zer_compute::BackendPreference::Cuda,
                "vulkan" => zer_compute::BackendPreference::Vulkan,
                "avx2"   => zer_compute::BackendPreference::Avx2,
                other => {
                    tracing::error!(target = other, "unknown --target; valid: auto, cpu, avx2, cuda, vulkan");
                    std::process::exit(1);
                }
            };
            return match zer_compute::DeviceBackend::from_preference(pref) {
                Ok(dev) => {
                    let name = dev.name();
                    if dev.is_accelerated() {
                        Self { inner: BackendInner::Gpu(Arc::new(dev)), name }
                    } else {
                        Self { inner: BackendInner::Cpu, name: "cpu" }
                    }
                }
                Err(e) => {
                    tracing::error!(target, error = %e, "--target unavailable");
                    std::process::exit(1);
                }
            };
        }

        #[allow(unreachable_code)]
        {
            if target == "auto" {
                return Self::cpu();
            }
            tracing::error!(target, "unknown --target; valid values when built without GPU features: auto, cpu");
            std::process::exit(1);
        }
    }

    /// Human-readable name of the active backend: `"cpu"`, `"cuda"`, or `"avx2"`.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// `true` when a GPU backend is active.
    pub fn is_gpu(&self) -> bool {
        !matches!(self.inner, BackendInner::Cpu)
    }
}

// ── Comparator ────────────────────────────────────────────────────────────────

enum ComparatorInner {
    Cpu(zer_compare::FieldComparator),
    #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
    Gpu(zer_compute::DeviceComparator),
}

/// Pairwise record comparator with automatic GPU/CPU selection.
///
/// Wraps `FieldComparator` (CPU) or `DeviceComparator` (GPU) depending on the
/// [`Backend`].  Implements [`ComparatorTrait`] identically in both cases.
pub struct Comparator {
    inner: ComparatorInner,
}

impl Comparator {
    /// Wrap an already-constructed [`zer_compare::FieldComparator`] directly.
    ///
    /// Use this when you want to override default similarity functions via
    /// [`zer_compare::FieldComparator::with_fns`] before creating the comparator.
    /// Always uses the CPU path; GPU acceleration is not available this way.
    pub fn from_cpu(fc: zer_compare::FieldComparator) -> Self {
        Self { inner: ComparatorInner::Cpu(fc) }
    }

    /// Build a comparator from a schema and backend.
    pub fn new(schema: &Schema, backend: &Backend) -> Self {
        match &backend.inner {
            BackendInner::Cpu => Self {
                inner: ComparatorInner::Cpu(
                    zer_compare::FieldComparator::from_schema(schema),
                ),
            },
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            BackendInner::Gpu(dev) => Self {
                inner: ComparatorInner::Gpu(
                    zer_compute::DeviceComparator::new(Arc::clone(dev), schema).unwrap(),
                ),
            },
        }
    }

    /// Name of the active backend, for diagnostics.
    pub fn backend_name(&self) -> &'static str {
        match &self.inner {
            ComparatorInner::Cpu(_) => "cpu",
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            ComparatorInner::Gpu(c) => c.backend_name(),
        }
    }

    /// Primary hot-path: pool-native batch comparison.
    ///
    /// `pool` is a `RecordPool` built from the candidate records; `pair_indices`
    /// holds `(i, j)` pairs where `i` and `j` are indices into the pool.
    /// Avoids all `Record::clone()` and `HashMap` lookups, the fastest path for
    /// large BRP-style jobs where records are already loaded into a pool.
    pub fn compare_batch_from_pool(
        &self,
        pool:         &RecordPool,
        pair_indices: &[(usize, usize)],
        schema:       &Schema,
    ) -> ComparisonBatch {
        match &self.inner {
            ComparatorInner::Cpu(c) => c.compare_batch_from_pool(pool, pair_indices, schema),
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            ComparatorInner::Gpu(c) => c.compare_batch_from_pool(pool, pair_indices, schema),
        }
    }

    /// Convenience wrapper: builds a pool from a flat `records` slice and compares
    /// the `pair_indices` pairs.  No `Record::clone()`.
    pub fn compare_batch_indexed(
        &self,
        records:      &[Record],
        pair_indices: &[(usize, usize)],
        schema:       &Schema,
    ) -> ComparisonBatch {
        let pool = RecordPool::from_records(records, schema);
        self.compare_batch_from_pool(&pool, pair_indices, schema)
    }
}

impl ComparatorTrait for Comparator {
    fn compare(&self, a: &Record, b: &Record, schema: &Schema) -> ComparisonVector {
        match &self.inner {
            ComparatorInner::Cpu(c) => c.compare(a, b, schema),
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            ComparatorInner::Gpu(c) => c.compare(a, b, schema),
        }
    }

    fn compare_batch_from_pool(
        &self,
        pool:    &RecordPool,
        indices: &[(usize, usize)],
        schema:  &Schema,
    ) -> ComparisonBatch {
        self.compare_batch_from_pool(pool, indices, schema)
    }
}

// ── Scorer ────────────────────────────────────────────────────────────────────

enum ScorerInner {
    Cpu(zer_compare::FellegiSunterScorer),
    #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
    Gpu(zer_compute::DeviceScorer),
}

/// Fellegi-Sunter scorer with automatic GPU/CPU EM acceleration.
///
/// `score` / `score_batch` always run on CPU, no kernel overhead for small
/// operations.  `estimate_params` uses the GPU EM kernel when the backend is
/// GPU and the batch exceeds the transfer break-even threshold; otherwise it
/// falls back to `zer_compare::run_em` on the CPU.
pub struct Scorer {
    inner: ScorerInner,
}

impl Scorer {
    /// Build a scorer using the given backend.
    pub fn new(backend: &Backend) -> Self {
        match &backend.inner {
            BackendInner::Cpu => Self {
                inner: ScorerInner::Cpu(zer_compare::FellegiSunterScorer),
            },
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            BackendInner::Gpu(dev) => Self {
                inner: ScorerInner::Gpu(zer_compute::DeviceScorer::new(Arc::clone(dev))),
            },
        }
    }

    /// Name of the active backend, for diagnostics.
    pub fn backend_name(&self) -> &'static str {
        match &self.inner {
            ScorerInner::Cpu(_) => "cpu",
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            ScorerInner::Gpu(s) => s.backend_name(),
        }
    }
}

impl ScorerTrait for Scorer {
    fn score(&self, vector: &ComparisonVector, params: &ModelParams) -> ScoredPair {
        match &self.inner {
            ScorerInner::Cpu(s) => s.score(vector, params),
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            ScorerInner::Gpu(s) => s.score(vector, params),
        }
    }

    fn score_batch(
        &self,
        batch:  &ComparisonBatch,
        params: &ModelParams,
    ) -> Vec<ScoredPair> {
        match &self.inner {
            ScorerInner::Cpu(s) => s.score_batch(batch, params),
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            ScorerInner::Gpu(s) => s.score_batch(batch, params),
        }
    }

    fn estimate_params(
        &self,
        batch:    &ComparisonBatch,
        init:     Option<ModelParams>,
        max_iter: usize,
    ) -> ZerResult<ModelParams> {
        match &self.inner {
            ScorerInner::Cpu(s) => s.estimate_params(batch, init, max_iter),
            #[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
            ScorerInner::Gpu(s) => s.estimate_params(batch, init, max_iter),
        }
    }
}

// ── Low-level kernel access for power users ───────────────────────────────────

/// Raw GPU kernel dispatch, for users writing custom kernels.
///
/// Requires the `cuda` or `avx2` feature.  Most users should use
/// [`Comparator`] and [`Scorer`] instead.
///
/// # Writing a custom kernel
///
/// 1. Define a zero-sized marker struct and `impl Kernel for It`.
/// 2. `impl KernelDispatch<It> for zer_compute::backend::cpu::CpuDevice`, CPU fallback.
/// 3. `impl KernelDispatch<It> for zer_compute::backend::cuda::CudaDevice`, CUDA path.
/// 4. Add the `impl KernelDispatch<It> for DeviceBackend` match in
///    `zer_compute::backend::mod`.
/// 5. Access the raw device via `zer::compute::DeviceBackend`.
#[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
pub mod kernel {
    pub use zer_compute::{
        backend::DeviceBackend,
        error::GpuError,
        kernel::{Kernel, KernelDispatch},
    };
}

// ── Crate re-exports ──────────────────────────────────────────────────────────

pub use zer_blocking as blocking;
pub use zer_compare  as compare;
pub use zer_core     as core;
pub use zer_schema   as schema;
pub use zer_cluster  as cluster;

#[cfg(feature = "pipeline")]
pub use zer_pipeline as pipeline;

#[cfg(any(feature = "cuda", feature = "avx2", feature = "vulkan"))]
pub use zer_compute as compute;

// ── Prelude ───────────────────────────────────────────────────────────────────

pub mod prelude {
    // Concrete auto-detecting types, primary user-facing API
    pub use crate::{Backend, Comparator, Scorer};

    // Core data types
    pub use zer_core::{
        comparison::{ComparisonBatch, ComparisonLevel, ComparisonVector},
        entity::{Entity, EntityId, EntityMember, ResolutionMethod},
        error::ZerError,
        record::{FieldValue, Record, RecordId},
        record_pool::RecordPool,
        schema::{FieldKind, Schema, SchemaBuilder},
        scoring::{MatchBand, ModelParams, ScoredPair},
        traits::{
            BlockIndex, Blocker, Clusterer, EntityStore, Judge, JudgeVerdict, RecordStore,
            // Renamed to avoid shadowing the concrete Comparator / Scorer structs above
            Comparator as ComparatorTrait,
            Scorer as ScorerTrait,
        },
        VecRecordStore,
    };

    // Blocking
    pub use zer_blocking::{
        BlockerFactory, CompositeBlocker, InvertedIndex, SchemaCategory,
        keys::{
            AddressInitialKey, AliasPhoneticKey, CameraTimeWindowKey, DateFragmentKey,
            DateGranularity, DocumentDigitSuffixKey, DocumentSuffixKey, ExactFieldKey,
            FuzzyYearKey, GeoGridKey, LicensePlateNormKey, PhoneticAlgo, PhoneticNameDobKey,
            PlateOCRFuzzyKey, SuffixKey, TransliteratedPhoneticKey,
        },
    };

    // CPU implementations, available directly for users who want the raw types
    pub use zer_compare::{
        FellegiSunterScorer, FieldComparator, LevelThresholds, SimilarityFn,
        JaroWinklerSimilarity, PhoneticEqualitySimilarity, TokenOverlapSimilarity,
        AddressTokenOverlap, StreetNumberEditDistance,
    };

    // Schema registry and artifact management (Phase 6)
    pub use zer_schema::{ModelArtifact, SchemaFingerprint, SchemaInferrer, SchemaRegistry, StartupMode};

    // Clustering and entity store (Phase 6)
    pub use zer_cluster::{ClusterConfig, ConnectedComponentsClusterer, ZalEntityStore};

    // Pipeline types, available with the `pipeline` feature (no polars required)
    #[cfg(feature = "pipeline")]
    pub use zer_pipeline::{
        BatchReport, ClusterIter, ClusterView, IngestResult, Ingester,
        Pipeline, PipelineBuilder, PipelineConfig, RateConfig,
    };

}
