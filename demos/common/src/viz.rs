/// ASCII visualisers for demo output.
// ── Block size histogram ───────────────────────────────────────────────────────

/// Print a horizontal bar chart of how many blocks fall in each size bucket.
///
/// `block_sizes` is a slice of (block_key_str, member_count) pairs, exactly
/// what you get from iterating a blocking index.
pub fn print_block_histogram(label: &str, block_sizes: &[usize]) {
    if block_sizes.is_empty() {
        println!("{}: no blocks", label);
        return;
    }

    let buckets: &[(usize, usize, &str)] = &[
        (2,  4,   "2–4   "),
        (5,  9,   "5–9   "),
        (10, 24,  "10–24 "),
        (25, 99,  "25–99 "),
        (100, usize::MAX, "100+  "),
    ];

    let mut counts = vec![0usize; buckets.len()];
    for &sz in block_sizes {
        for (i, &(lo, hi, _)) in buckets.iter().enumerate() {
            if sz >= lo && sz <= hi {
                counts[i] += 1;
                break;
            }
        }
    }

    let max_count = counts.iter().copied().max().unwrap_or(1).max(1);
    let bar_width = 40usize;

    println!("  {} block size distribution:", label);
    for (i, &(_, _, lbl)) in buckets.iter().enumerate() {
        let c   = counts[i];
        let bar = bar_width * c / max_count;
        let filled = "█".repeat(bar);
        let empty  = "░".repeat(bar_width - bar);
        println!("    {}│{}{}│ {}", lbl, filled, empty, c);
    }
    println!(
        "  total blocks: {}  total candidates: {}",
        block_sizes.len(),
        block_sizes.iter().map(|&s| s * (s - 1) / 2).sum::<usize>()
    );
}

// ── Entity cluster tree ────────────────────────────────────────────────────────

/// One resolved entity shown in the cluster tree.
pub struct ClusterEntry {
    pub entity_id:   u64,
    pub record_ids:  Vec<u64>,
    pub best_scores: Vec<f32>,
    pub labels:      Vec<String>,
}

/// Print resolved entities as an indented tree.
///
/// Shows up to `max_clusters` entities; truncates the rest with a summary line.
pub fn print_cluster_tree(entries: &[ClusterEntry], max_clusters: usize) {
    let shown = entries.len().min(max_clusters);
    println!("  entity clusters ({} total):", entries.len());
    for entry in entries.iter().take(shown) {
        println!("  ├─ entity #{}", entry.entity_id);
        let n = entry.record_ids.len();
        for (i, (rid, label)) in entry.record_ids.iter().zip(entry.labels.iter()).enumerate() {
            let score_str = entry.best_scores
                .get(i)
                .map(|s| format!(" [{:.2}]", s))
                .unwrap_or_default();
            let connector = if i + 1 == n { "└──" } else { "├──" };
            println!("  │  {} record #{}: {}{}", connector, rid, label, score_str);
        }
    }
    if entries.len() > shown {
        println!("  └─ … and {} more", entries.len() - shown);
    }
}

// ── Matched pair table ─────────────────────────────────────────────────────────

/// One row in the side-by-side pair table.
pub struct PairRow {
    pub score:    f32,
    pub a_fields: Vec<(String, String)>,
    pub b_fields: Vec<(String, String)>,
}

/// Print matched pairs side-by-side: source A fields | source B fields | score.
pub fn print_pair_table(rows: &[PairRow], max_rows: usize) {
    if rows.is_empty() {
        println!("  no matched pairs");
        return;
    }
    let shown = rows.len().min(max_rows);
    println!("  matched pairs ({} total, showing {}):", rows.len(), shown);
    println!("  {:─<30} {:─<30} {:─<6}", "", "", "");
    println!("  {:<30} {:<30} {:<6}", "Source A", "Source B", "Score");
    println!("  {:─<30} {:─<30} {:─<6}", "", "", "");

    for row in rows.iter().take(shown) {
        let fields = row.a_fields.len().max(row.b_fields.len());
        for fi in 0..fields {
            let a = row.a_fields.get(fi)
                .map(|(k, v)| format!("{}: {}", k, v))
                .unwrap_or_default();
            let b = row.b_fields.get(fi)
                .map(|(k, v)| format!("{}: {}", k, v))
                .unwrap_or_default();
            let score_col = if fi == 0 {
                format!("{:.3}", row.score)
            } else {
                String::new()
            };
            println!("  {:<30} {:<30} {:<6}", a, b, score_col);
        }
        println!("  {:─<30} {:─<30} {:─<6}", "", "", "");
    }
    if rows.len() > shown {
        println!("  … and {} more pairs", rows.len() - shown);
    }
}

// ── Comparison vector fill-bars ────────────────────────────────────────────────

/// One field's comparison result.
pub struct FieldComparison {
    pub field:      String,
    pub similarity: f32,
}

/// Print comparison vectors as fill-bar rows.
///
/// Each row shows the field name, a `████░░░` bar, and the raw similarity.
pub fn print_comparison_vectors(header: &str, fields: &[FieldComparison]) {
    const WIDTH: usize = 20;
    println!("  {}:", header);
    for fc in fields {
        let filled = ((fc.similarity.clamp(0.0, 1.0) * WIDTH as f32).round() as usize).min(WIDTH);
        let bar    = format!("{}{}", "█".repeat(filled), "░".repeat(WIDTH - filled));
        println!("    {:<20} │{}│ {:.3}", fc.field, bar, fc.similarity);
    }
}

// ── Score distribution histogram ───────────────────────────────────────────────

/// Print a score distribution histogram with threshold markers.
///
/// `scores` is a list of match probabilities in [0, 1].
/// `auto_match` and `auto_reject` are the threshold lines drawn on the chart.
pub fn print_score_histogram(scores: &[f32], auto_match: f32, auto_reject: f32) {
    if scores.is_empty() {
        println!("  no scores to display");
        return;
    }

    let n_buckets = 20usize;
    let mut buckets = vec![0usize; n_buckets];
    for &s in scores {
        let idx = ((s.clamp(0.0, 1.0) * n_buckets as f32) as usize).min(n_buckets - 1);
        buckets[idx] += 1;
    }
    let max_count = buckets.iter().copied().max().unwrap_or(1).max(1);
    let bar_height = 8usize;

    let mut grid: Vec<Vec<char>> = vec![vec![' '; n_buckets]; bar_height];
    for (col, &cnt) in buckets.iter().enumerate() {
        let bar = (bar_height * cnt / max_count).min(bar_height);
        for row in (bar_height - bar)..bar_height {
            grid[row][col] = '█';
        }
    }

    let reject_col  = ((auto_reject * n_buckets as f32) as usize).min(n_buckets - 1);
    let promote_col = ((auto_match  * n_buckets as f32) as usize).min(n_buckets - 1);

    println!("  score distribution  [reject<{:.2}  match>{:.2}]:", auto_reject, auto_match);
    for (row_idx, row) in grid.iter().enumerate() {
        let line: String = row.iter().enumerate().map(|(col, &ch)| {
            if col == reject_col  { '|' }
            else if col == promote_col { '|' }
            else { ch }
        }).collect();
        let label = if row_idx == 0 { format!("{:>4}", max_count) } else { "    ".to_string() };
        println!("  {}│{}│", label, line);
    }
    let axis: String = (0..n_buckets).map(|i| {
        if i == 0            { '0' }
        else if i == n_buckets - 1 { '1' }
        else if i == reject_col    { 'R' }
        else if i == promote_col   { 'M' }
        else                       { '─' }
    }).collect();
    println!("      └{}┘", axis);
    println!(
        "  n={} | auto-rejected: {} | borderline: {} | auto-matched: {}",
        scores.len(),
        scores.iter().filter(|&&s| s < auto_reject).count(),
        scores.iter().filter(|&&s| s >= auto_reject && s <= auto_match).count(),
        scores.iter().filter(|&&s| s > auto_match).count(),
    );
}
