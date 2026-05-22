use std::sync::Arc;

use zer_compare::{FieldComparator, FellegiSunterScorer};
use zer_core::{
    comparison::{ComparisonBatch, ComparisonVector},
    error::ZerError,
    record::Record,
    record_pool::RecordPool,
    schema::Schema,
    scoring::{ModelParams, ScoredPair},
    traits::{Comparator, Result as ZerResult, Scorer},
};

// ── CpuDevice ─────────────────────────────────────────────────────────────────

pub struct CpuDevice;

// ── CpuFallbackComparator ─────────────────────────────────────────────────────

/// CPU-side comparator wrapping `zer_compare::FieldComparator`.
#[derive(Clone)]
pub struct CpuFallbackComparator {
    inner: Arc<FieldComparator>,
}

impl CpuFallbackComparator {
    pub fn from_schema(schema: &Schema) -> Self {
        Self { inner: Arc::new(FieldComparator::from_schema(schema)) }
    }
}

impl Comparator for CpuFallbackComparator {
    fn compare(&self, a: &Record, b: &Record, schema: &Schema) -> ComparisonVector {
        self.inner.compare(a, b, schema)
    }

    fn compare_batch_from_pool(
        &self,
        pool:    &RecordPool,
        indices: &[(usize, usize)],
        schema:  &Schema,
    ) -> ComparisonBatch {
        self.inner.compare_batch_from_pool(pool, indices, schema)
    }
}

// ── CpuFallbackScorer ─────────────────────────────────────────────────────────

/// CPU-side Fellegi-Sunter scorer wrapping `zer_compare::FellegiSunterScorer`.
#[derive(Clone)]
pub struct CpuFallbackScorer;

impl Scorer for CpuFallbackScorer {
    fn score(&self, vector: &ComparisonVector, params: &ModelParams) -> ScoredPair {
        FellegiSunterScorer.score(vector, params)
    }

    fn score_batch(&self, batch: &ComparisonBatch, params: &ModelParams) -> Vec<ScoredPair> {
        FellegiSunterScorer.score_batch(batch, params)
    }

    fn estimate_params(
        &self,
        batch:    &ComparisonBatch,
        init:     Option<ModelParams>,
        max_iter: usize,
    ) -> ZerResult<ModelParams> {
        FellegiSunterScorer.estimate_params(batch, init, max_iter)
    }
}

/// Convenience wrapper for `DeviceScorer::estimate_params` CPU fallback.
pub fn cpu_estimate_params(
    batch:    &ComparisonBatch,
    init:     Option<ModelParams>,
    max_iter: usize,
) -> ZerResult<ModelParams> {
    zer_compare::run_em(batch, init, max_iter).map_err(ZerError::from)
}
