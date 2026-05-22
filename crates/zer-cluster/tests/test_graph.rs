use zer_cluster::{ClusterConfig, ClusterGraph};
use zer_core::{comparison::ComparisonVector, scoring::{MatchBand, ScoredPair}};

fn auto_match(a: u64, b: u64, prob: f32) -> ScoredPair {
    ScoredPair {
        record_a:          a,
        record_b:          b,
        match_weight:      0.0,
        match_probability: prob,
        vector:            ComparisonVector { record_a: a, record_b: b, levels: vec![] },
        band:              MatchBand::AutoMatch,
    }
}

fn config() -> ClusterConfig {
    ClusterConfig { max_cluster_size: 50, within_cluster_min: 0.85 }
}

// ── Connected-component basics ─────────────────────────────────────────────────

#[test]
fn single_pair_yields_one_cluster_of_two() {
    let mut g = ClusterGraph::new();
    g.add_pairs(&[auto_match(1, 2, 0.95)]);
    let clusters = g.compute_clusters(&config());
    assert_eq!(clusters.len(), 1);
    let mut c = clusters[0].clone();
    c.sort();
    assert_eq!(c, vec![1, 2]);
}

#[test]
fn three_nodes_in_chain_form_one_cluster() {
    let mut g = ClusterGraph::new();
    g.add_pairs(&[auto_match(1, 2, 0.95), auto_match(2, 3, 0.95)]);
    let clusters = g.compute_clusters(&config());
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn two_disconnected_pairs_form_two_clusters() {
    let mut g = ClusterGraph::new();
    g.add_pairs(&[auto_match(1, 2, 0.95), auto_match(3, 4, 0.95)]);
    let clusters = g.compute_clusters(&config());
    assert_eq!(clusters.len(), 2);
}

#[test]
fn empty_graph_returns_no_clusters() {
    let g = ClusterGraph::new();
    assert!(g.compute_clusters(&config()).is_empty());
}

// ── Weak-edge removal / chain-breaking ────────────────────────────────────────

#[test]
fn weak_bridge_splits_chain_into_two_clusters() {
    // A -[0.95]- B -[0.28]- C -[0.95]- D
    // B-C edge is below within_cluster_min (0.85) → chain breaks
    let mut g = ClusterGraph::new();
    g.add_pairs(&[
        auto_match(1, 2, 0.95), // A-B strong
        auto_match(2, 3, 0.28), // B-C weak
        auto_match(3, 4, 0.95), // C-D strong
    ]);
    let mut clusters = g.compute_clusters(&config());
    clusters.sort_by_key(|c| *c.iter().min().unwrap());

    assert_eq!(clusters.len(), 2, "weak bridge must produce 2 clusters");

    let mut c0 = clusters[0].clone(); c0.sort();
    let mut c1 = clusters[1].clone(); c1.sort();
    assert_eq!(c0, vec![1, 2]);
    assert_eq!(c1, vec![3, 4]);
}

#[test]
fn strong_bridge_keeps_chain_intact() {
    // A -[0.95]- B -[0.90]- C  (0.90 >= 0.85 → no split)
    let mut g = ClusterGraph::new();
    g.add_pairs(&[auto_match(1, 2, 0.95), auto_match(2, 3, 0.90)]);
    let clusters = g.compute_clusters(&config());
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn all_weak_edges_below_min_no_clusters() {
    // All edges below min → every edge removed → no component of size ≥ 2
    let mut g = ClusterGraph::new();
    g.add_pairs(&[
        auto_match(1, 2, 0.50),
        auto_match(2, 3, 0.60),
        auto_match(3, 4, 0.70),
    ]);
    let clusters = g.compute_clusters(&config());
    assert!(clusters.is_empty(), "all-weak edges should yield no clusters");
}

// ── Star pruning ───────────────────────────────────────────────────────────────

#[test]
fn star_pruning_handles_oversized_cluster_without_panic() {
    // Hub (0) connected to 60 satellites, exceeds max_cluster_size=50
    let cfg = ClusterConfig { max_cluster_size: 50, within_cluster_min: 0.85 };
    let mut g = ClusterGraph::new();
    let pairs: Vec<_> = (1u64..=60).map(|i| auto_match(0, i, 0.95)).collect();
    g.add_pairs(&pairs);

    let clusters = g.compute_clusters(&cfg);
    // Star pruning must complete without panic; result must cover non-trivial groups.
    let total: usize = clusters.iter().map(|c| c.len()).sum();
    assert!(total >= 2);
}

#[test]
fn duplicate_pairs_do_not_add_duplicate_edges() {
    let mut g = ClusterGraph::new();
    // Add the same pair twice.
    g.add_pairs(&[auto_match(1, 2, 0.95), auto_match(1, 2, 0.90)]);
    let clusters = g.compute_clusters(&config());
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 2);
}
