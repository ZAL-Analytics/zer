/// Shared utilities for all zer demos.
///
/// Provides section headers and ASCII visualisers so every demo has the same
/// look and feel.
pub mod viz;

pub use viz::{
    print_block_histogram, print_cluster_tree, print_comparison_vectors, print_pair_table,
    print_score_histogram,
};

/// No-op, kept so existing demo `main` functions compile unchanged.
pub fn init_tracing() {}

// ── Section header ────────────────────────────────────────────────────────────

/// Print a styled section header.
pub fn section(label: &str) {
    let fill = 60usize.saturating_sub(label.len() + 5);
    let bar = "─".repeat(fill);
    println!("─── {} {}", label, bar);
}
