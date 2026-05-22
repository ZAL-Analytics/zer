//! Schema registry and model persistence for zer.
//!
//! This crate provides three cooperating components:
//!
//! 1. **[`SchemaInferrer`]**, automatic `FieldKind` detection from column names
//!    and value patterns; produces a [`Schema`] without requiring the caller to
//!    know the dataset structure upfront.
//!
//! 2. **[`SchemaFingerprint`]**, a compact identity for a schema plus its data
//!    distribution (SHA-256 hash of field names/kinds, per-field null rates,
//!    cardinalities).
//!
//! 3. **[`SchemaRegistry`]**, a `sled`-backed persistent store for
//!    [`ModelArtifact`]s (trained Fellegi-Sunter parameters). On startup the
//!    pipeline calls [`SchemaRegistry::lookup_startup_mode`] to decide whether
//!    to load params directly (exact match), warm-start EM (similar schema), or
//!    run full EM from priors (new/incompatible schema).
//!
//! [`Schema`]: zer_core::schema::Schema

pub mod artifact;
pub mod config;
pub mod fingerprint;
pub mod infer;
pub mod registry;
pub mod similarity;

pub use artifact::ModelArtifact;
pub use config::{NameHeuristics, ValuePatterns};
pub use fingerprint::{FieldStats, SchemaFingerprint};
pub use infer::SchemaInferrer;
pub use registry::{SchemaRegistry, StartupMode};
pub use similarity::{fingerprint_distance, EXACT_MATCH_THRESHOLD, WARM_START_THRESHOLD};
