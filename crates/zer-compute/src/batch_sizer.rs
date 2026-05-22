//! Auto-tunes the GPU batch size from available VRAM using the exact buffer layout.

use crate::soa::STRING_STRIDE;

/// Fraction of VRAM to target by default. Leaves headroom for the DeBERTa judge
/// (Phase 7), which can occupy ~1–2 GB depending on the model variant.
const DEFAULT_VRAM_UTILIZATION: f32 = 0.75;

/// Minimum number of pairs required to justify the GPU kernel launch overhead.
/// Below this threshold `DeviceComparator` silently falls back to the CPU path.
pub const GPU_BATCH_MIN: usize = 1_000;

/// Computes the maximum batch size that fits safely within VRAM.
///
/// Uses the exact per-pair device memory layout of the GPU compare kernel, no
/// field-length estimation required because every string is padded to
/// [`STRING_STRIDE`] bytes on the device regardless of actual content.
///
/// # Per-pair device memory layout
///
/// | Buffer | Bytes per pair |
/// |---|---|
/// | `d_data_a` + `d_data_b` (u8, STRING_STRIDE each) | `2 × n_fields × 64` |
/// | `d_lens_a` + `d_lens_b` (u16) | `2 × n_fields × 2` |
/// | `d_ids_a`  + `d_ids_b`  (u64) | `2 × 8` |
/// | `d_weights` + `d_probs` (f32) | `2 × 4` |
/// | `d_levels`              (u32) | `n_fields × 4` |
///
/// Total: `n_fields × 136 + 24` bytes per pair (exact, no estimation).
///
/// # Example
///
/// ```
/// use zer_compute::batch_sizer::BatchSizer;
///
/// let sizer = BatchSizer::new();
/// // 3 GB available VRAM (e.g. after OS + model overhead), 10 fields
/// let available = 3u64 * 1024 * 1024 * 1024;
/// let max = sizer.max_batch_size(available, 10);
/// assert!(max > 1_000_000, "should easily fit millions of pairs");
/// ```
#[derive(Debug, Clone)]
pub struct BatchSizer {
    /// Fraction of available VRAM to commit to the comparison batch. Default: 0.75.
    pub vram_utilization: f32,
}

impl Default for BatchSizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchSizer {
    pub fn new() -> Self {
        Self { vram_utilization: DEFAULT_VRAM_UTILIZATION }
    }

    /// Override the utilization fraction (0.0 < fraction ≤ 1.0).
    pub fn with_utilization(mut self, fraction: f32) -> Self {
        assert!(fraction > 0.0 && fraction <= 1.0, "utilization must be in (0, 1]");
        self.vram_utilization = fraction;
        self
    }

    /// Compute the maximum number of pairs that fit in `available_vram_bytes` VRAM
    /// for a schema with `num_fields` fields.
    ///
    /// The formula matches the GPU compare kernel buffer layout exactly, no avg_field_len
    /// estimate is needed because device buffers always use `STRING_STRIDE` bytes per string.
    ///
    /// Returns at least 1 so callers never divide by zero.
    pub fn max_batch_size(
        &self,
        available_vram_bytes: u64,
        num_fields: usize,
    ) -> usize {
        let bytes_per_pair: usize =
              2 * num_fields * STRING_STRIDE   // d_data_a + d_data_b (u8)
            + 2 * num_fields * 2              // d_lens_a + d_lens_b (u16)
            + 2 * 8                           // d_ids_a  + d_ids_b  (u64)
            + 2 * 4                           // d_weights + d_probs (f32)
            + num_fields * 4;                 // d_levels            (u32)

        let usable = (available_vram_bytes as f64 * self.vram_utilization as f64) as u64;
        (usable / bytes_per_pair as u64).max(1) as usize
    }

    /// Minimum batch size to justify a GPU kernel launch. Batches smaller than
    /// this are routed to the CPU path transparently.
    pub const fn min_batch_for_gpu() -> usize {
        GPU_BATCH_MIN
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_batch_grows_with_vram() {
        let sizer = BatchSizer::new();
        let small = sizer.max_batch_size(1 * 1024 * 1024 * 1024, 10);
        let large = sizer.max_batch_size(8 * 1024 * 1024 * 1024, 10);
        assert!(large > small);
    }

    #[test]
    fn max_batch_never_zero() {
        let sizer = BatchSizer::new();
        // Even with absurdly many fields it must return at least 1
        let r = sizer.max_batch_size(1, 1000);
        assert_eq!(r, 1);
    }

    #[test]
    fn three_gb_vram_fits_millions() {
        let sizer     = BatchSizer::new();
        // 3 GB is a realistic headroom figure after OS + model overhead
        // 10 fields × 136 + 24 = 1,384 bytes/pair → >2M pairs in 3 GB at 75%
        let available = 3u64 * 1024 * 1024 * 1024;
        let max       = sizer.max_batch_size(available, 10);
        assert!(max > 1_000_000, "expected >1M pairs, got {max}");
    }

    #[test]
    fn min_batch_constant_is_positive() {
        assert!(BatchSizer::min_batch_for_gpu() > 0);
    }

    #[test]
    fn utilization_scales_result() {
        let full = BatchSizer::new().with_utilization(1.0).max_batch_size(1_000_000, 5);
        let half = BatchSizer::new().with_utilization(0.5).max_batch_size(1_000_000, 5);
        assert!(full > half);
    }

    #[test]
    fn formula_matches_compare_pool_layout() {
        // Verify the formula matches the GPU compare kernel buffer layout.
        // For n=1 field, 1 pair the device allocates:
        //   d_data_a/b: 2 × 1 × 64 = 128 bytes
        //   d_lens_a/b: 2 × 1 × 2  =   4 bytes
        //   d_ids_a/b:  2 × 8       =  16 bytes
        //   d_weights/probs: 2 × 4  =   8 bytes
        //   d_levels:   1 × 4       =   4 bytes
        //   total = 160 bytes/pair
        let bytes_per_pair_1field = 2 * 1 * STRING_STRIDE + 2 * 1 * 2 + 16 + 8 + 1 * 4;
        assert_eq!(bytes_per_pair_1field, 160);

        let sizer = BatchSizer::new().with_utilization(1.0);
        let max   = sizer.max_batch_size(160, 1);
        assert_eq!(max, 1, "exactly one pair should fit in 160 bytes for 1 field");
    }
}
