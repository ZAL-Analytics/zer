use crate::record::RecordId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash,
         serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum ComparisonLevel {
    None    = 0,
    Partial = 1,
    Close   = 2,
    Exact   = 3,
    /// Field structurally absent on one or both sides (cross-schema linkage).
    /// Never fed to EM; both E-step and M-step skip pairs where any field carries this level.
    Null    = 255,
}

impl ComparisonLevel {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            1   => Self::Partial,
            2   => Self::Close,
            3   => Self::Exact,
            255 => Self::Null,
            _   => Self::None,
        }
    }
}

impl PartialOrd for ComparisonLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ComparisonLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_u8().cmp(&other.as_u8())
    }
}

// ── ComparisonVector (single-pair) ────────────────────────────────────────────

/// Comparison result for a single candidate pair.
///
/// Used for single-pair comparisons (`Comparator::compare`) and as the
/// per-pair view stored in `ScoredPair`.  For batch operations use
/// `ComparisonBatch`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComparisonVector {
    pub record_a: RecordId,
    pub record_b: RecordId,
    pub levels:   Vec<ComparisonLevel>,
}

impl ComparisonVector {
    pub fn new(record_a: RecordId, record_b: RecordId, levels: Vec<ComparisonLevel>) -> Self {
        Self { record_a, record_b, levels }
    }
}

// ── ComparisonBatch (field-major SoA) ─────────────────────────────────────────

/// Field-major SoA batch of comparison results for many pairs.
///
/// # Layout
///
/// ```text
/// levels[field_idx * n_pairs + pair_idx] = ComparisonLevel as u8
/// ```
///
/// All values for field 0 across every pair are contiguous, then field 1, etc.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComparisonBatch {
    pub n_pairs:  usize,
    pub n_fields: usize,
    /// `(record_a_id, record_b_id)` for pair `p`.
    pub pair_ids: Vec<(RecordId, RecordId)>,
    /// Field-major levels: `levels[f * n_pairs + p]`.
    pub levels:   Vec<u8>,
}

impl ComparisonBatch {
    /// Allocate a zeroed batch (all levels = `ComparisonLevel::None`).
    pub fn new(n_pairs: usize, n_fields: usize, pair_ids: Vec<(RecordId, RecordId)>) -> Self {
        Self {
            n_pairs,
            n_fields,
            pair_ids,
            levels: vec![0u8; n_fields * n_pairs],
        }
    }

    /// Read the level for `(field, pair)`.
    #[inline]
    pub fn level(&self, field: usize, pair: usize) -> ComparisonLevel {
        ComparisonLevel::from_u8(self.levels[field * self.n_pairs + pair])
    }

    /// Write the level for `(field, pair)`.
    #[inline]
    pub fn set_level(&mut self, field: usize, pair: usize, level: ComparisonLevel) {
        self.levels[field * self.n_pairs + pair] = level as u8;
    }

    /// Reconstruct a `ComparisonVector` for pair `p`.
    pub fn pair_as_vector(&self, pair_idx: usize) -> ComparisonVector {
        let (a, b) = self.pair_ids[pair_idx];
        let levels = (0..self.n_fields)
            .map(|f| self.level(f, pair_idx))
            .collect();
        ComparisonVector::new(a, b, levels)
    }

    /// Build from an existing `Vec<ComparisonVector>` for migration / tests.
    pub fn from_vectors(vectors: &[ComparisonVector]) -> Self {
        if vectors.is_empty() {
            return Self::new(0, 0, vec![]);
        }
        let n_pairs  = vectors.len();
        let n_fields = vectors[0].levels.len();
        let pair_ids = vectors.iter().map(|v| (v.record_a, v.record_b)).collect();
        let mut batch = Self::new(n_pairs, n_fields, pair_ids);
        for (p, v) in vectors.iter().enumerate() {
            for (f, &level) in v.levels.iter().enumerate() {
                batch.set_level(f, p, level);
            }
        }
        batch
    }

    /// Convert back to `Vec<ComparisonVector>` for callers that still need it.
    pub fn into_vectors(&self) -> Vec<ComparisonVector> {
        (0..self.n_pairs).map(|p| self.pair_as_vector(p)).collect()
    }

    /// Concatenate multiple field-major batches (same `n_fields`) into one.
    ///
    /// Each chunk may have a different `n_pairs`.  The merged layout remains
    /// field-major with `n_pairs_total = sum of all chunk n_pairs`.
    pub fn concat(chunks: &[Self]) -> Self {
        let chunks: Vec<&Self> = chunks.iter().filter(|c| c.n_pairs > 0).collect();
        if chunks.is_empty() {
            return Self::new(0, 0, vec![]);
        }
        let n_fields = chunks[0].n_fields;
        let n_total: usize = chunks.iter().map(|c| c.n_pairs).sum();

        let mut pair_ids = Vec::with_capacity(n_total);
        let mut levels = vec![0u8; n_fields * n_total];

        let mut offset = 0usize;
        for chunk in &chunks {
            pair_ids.extend_from_slice(&chunk.pair_ids);
            for f in 0..n_fields {
                let dst = f * n_total + offset;
                let src = f * chunk.n_pairs;
                levels[dst..dst + chunk.n_pairs]
                    .copy_from_slice(&chunk.levels[src..src + chunk.n_pairs]);
            }
            offset += chunk.n_pairs;
        }

        Self { n_pairs: n_total, n_fields, pair_ids, levels }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_level_ordering() {
        assert!(ComparisonLevel::Exact   > ComparisonLevel::Close);
        assert!(ComparisonLevel::Close   > ComparisonLevel::Partial);
        assert!(ComparisonLevel::Partial > ComparisonLevel::None);
    }

    #[test]
    fn comparison_level_repr_values() {
        assert_eq!(ComparisonLevel::Exact.as_u8(),   3);
        assert_eq!(ComparisonLevel::Close.as_u8(),   2);
        assert_eq!(ComparisonLevel::Partial.as_u8(), 1);
        assert_eq!(ComparisonLevel::None.as_u8(),    0);
    }

    #[test]
    fn comparison_level_round_trip() {
        for &l in &[
            ComparisonLevel::None,
            ComparisonLevel::Partial,
            ComparisonLevel::Close,
            ComparisonLevel::Exact,
            ComparisonLevel::Null,
        ] {
            assert_eq!(ComparisonLevel::from_u8(l.as_u8()), l);
        }
        assert_eq!(ComparisonLevel::from_u8(99), ComparisonLevel::None);
    }

    #[test]
    fn batch_field_major_layout() {
        // n_fields=2, n_pairs=3
        let pair_ids = vec![(1, 2), (3, 4), (5, 6)];
        let mut batch = ComparisonBatch::new(3, 2, pair_ids);

        // Set levels in a known pattern
        batch.set_level(0, 0, ComparisonLevel::Exact);   // field 0, pair 0
        batch.set_level(0, 1, ComparisonLevel::Close);   // field 0, pair 1
        batch.set_level(0, 2, ComparisonLevel::Partial); // field 0, pair 2
        batch.set_level(1, 0, ComparisonLevel::None);    // field 1, pair 0
        batch.set_level(1, 1, ComparisonLevel::Exact);   // field 1, pair 1
        batch.set_level(1, 2, ComparisonLevel::Close);   // field 1, pair 2

        // Field-major: field 0 values at indices 0,1,2; field 1 at 3,4,5
        assert_eq!(batch.levels[0], ComparisonLevel::Exact   as u8);
        assert_eq!(batch.levels[1], ComparisonLevel::Close   as u8);
        assert_eq!(batch.levels[2], ComparisonLevel::Partial as u8);
        assert_eq!(batch.levels[3], ComparisonLevel::None    as u8);
        assert_eq!(batch.levels[4], ComparisonLevel::Exact   as u8);
        assert_eq!(batch.levels[5], ComparisonLevel::Close   as u8);

        // pair_as_vector reconstructs correctly
        let v = batch.pair_as_vector(1); // pair index 1
        assert_eq!(v.record_a, 3);
        assert_eq!(v.record_b, 4);
        assert_eq!(v.levels, vec![ComparisonLevel::Close, ComparisonLevel::Exact]);
    }

    #[test]
    fn batch_from_vectors_round_trips() {
        let vectors = vec![
            ComparisonVector::new(1, 2, vec![ComparisonLevel::Exact, ComparisonLevel::None]),
            ComparisonVector::new(3, 4, vec![ComparisonLevel::Partial, ComparisonLevel::Close]),
        ];
        let batch = ComparisonBatch::from_vectors(&vectors);
        assert_eq!(batch.n_pairs, 2);
        assert_eq!(batch.n_fields, 2);

        let back = batch.into_vectors();
        for (orig, got) in vectors.iter().zip(back.iter()) {
            assert_eq!(orig.record_a, got.record_a);
            assert_eq!(orig.record_b, got.record_b);
            assert_eq!(orig.levels,   got.levels);
        }
    }

    #[test]
    fn batch_empty_is_valid() {
        let batch = ComparisonBatch::from_vectors(&[]);
        assert_eq!(batch.n_pairs, 0);
        assert_eq!(batch.n_fields, 0);
        assert!(batch.levels.is_empty());
    }
}
