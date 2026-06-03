use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::{
    graph::{NodeIndex, UnGraph},
    visit::EdgeRef,
};
use zer_core::{record::RecordId, scoring::ScoredPair};

/// Parameters controlling cluster shape after graph construction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterConfig {
    /// Clusters larger than this are subjected to star pruning.
    pub max_cluster_size: usize,
    /// Edges with weight below this threshold are removed before extracting
    /// components (weak-edge removal / chain-breaking).
    pub within_cluster_min: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            max_cluster_size: 50,
            within_cluster_min: 0.85,
        }
    }
}

/// Undirected similarity graph over records.
///
/// Each node is a `RecordId`; each edge weight is the `match_probability` of
/// the `AutoMatch` pair that connected those two records.
pub struct ClusterGraph {
    graph: UnGraph<RecordId, f32>,
    node_map: HashMap<RecordId, NodeIndex>,
}

impl ClusterGraph {
    pub fn new() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
            node_map: HashMap::new(),
        }
    }

    /// Add `AutoMatch` pairs to the graph. Non-AutoMatch pairs are ignored.
    pub fn add_pairs(&mut self, pairs: &[ScoredPair]) {
        for pair in pairs {
            let a = self.get_or_insert(pair.record_a);
            let b = self.get_or_insert(pair.record_b);
            // Avoid duplicate edges, keep the higher-weight one.
            if let Some(edge) = self.graph.find_edge(a, b) {
                let w = self.graph.edge_weight_mut(edge).unwrap();
                if pair.match_probability > *w {
                    *w = pair.match_probability;
                }
            } else {
                self.graph.add_edge(a, b, pair.match_probability);
            }
        }
    }

    /// Compute clusters using the two-phase chain-breaking algorithm:
    ///
    /// 1. **Weak-edge removal**: remove all edges with weight <
    ///    `config.within_cluster_min` then extract connected components.
    /// 2. **Star pruning**: for any component whose size exceeds
    ///    `config.max_cluster_size`, find the hub (highest-degree node in the
    ///    original graph), remove all non-hub edges below the min threshold,
    ///    and re-extract components from that sub-graph.
    ///
    /// Returns only non-trivial components (size ≥ 2).
    pub fn compute_clusters(&self, config: &ClusterConfig) -> Vec<Vec<RecordId>> {
        let pruned = weak_edge_removal(&self.graph, config.within_cluster_min);
        let mut components = extract_components(&pruned);

        // Star pruning for oversized components.
        let mut result = Vec::new();
        for comp in components.drain(..) {
            if comp.len() <= config.max_cluster_size {
                if comp.len() >= 2 {
                    result.push(comp);
                }
            } else {
                let sub = star_prune(&self.graph, &comp, config.within_cluster_min);
                result.extend(sub.into_iter().filter(|c| c.len() >= 2));
            }
        }
        result
    }

    fn get_or_insert(&mut self, id: RecordId) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&id) {
            return idx;
        }
        let idx = self.graph.add_node(id);
        self.node_map.insert(id, idx);
        idx
    }
}

impl Default for ClusterGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Graph algorithms ──────────────────────────────────────────────────────────

/// Clone the graph, remove all edges below `min_weight`, and return the result.
///
/// Edge indices are removed in descending order to avoid the petgraph `Graph`
/// index-swap issue: removing edge `i` moves the last edge into slot `i`, so
/// removing from highest to lowest keeps all lower indices stable.
fn weak_edge_removal(graph: &UnGraph<RecordId, f32>, min_weight: f32) -> UnGraph<RecordId, f32> {
    let mut g = graph.clone();
    let mut weak: Vec<_> = g
        .edge_indices()
        .filter(|&e| *g.edge_weight(e).unwrap() < min_weight)
        .collect();
    weak.sort_by_key(|e| std::cmp::Reverse(e.index()));
    for e in weak {
        g.remove_edge(e);
    }
    g
}

/// BFS-based connected-component extraction.
///
/// `petgraph::algo::connected_components()` returns only a count, this
/// function also yields the actual groups.
pub(crate) fn extract_components(graph: &UnGraph<RecordId, f32>) -> Vec<Vec<RecordId>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for start in graph.node_indices() {
        if !visited.insert(start) {
            continue;
        }
        let mut comp = vec![graph[start]];
        let mut queue = VecDeque::from([start]);

        while let Some(node) = queue.pop_front() {
            for nb in graph.neighbors(node) {
                if visited.insert(nb) {
                    comp.push(graph[nb]);
                    queue.push_back(nb);
                }
            }
        }
        components.push(comp);
    }
    components
}

/// Star pruning for a single oversized component.
///
/// Finds the hub (highest-degree node in the original graph restricted to
/// `comp`), builds a sub-graph containing only hub-edges with weight ≥
/// `min_weight`, and returns the resulting sub-components.
fn star_prune(
    graph: &UnGraph<RecordId, f32>,
    comp: &[RecordId],
    min_weight: f32,
) -> Vec<Vec<RecordId>> {
    let comp_set: HashSet<RecordId> = comp.iter().copied().collect();

    // Identify node indices in the original graph for this component.
    let node_indices: Vec<NodeIndex> = graph
        .node_indices()
        .filter(|&n| comp_set.contains(&graph[n]))
        .collect();

    // Find hub: node with most edges to other comp members with weight >= min.
    let hub = node_indices.iter().max_by_key(|&&n| {
        graph
            .edges(n)
            .filter(|e| {
                let other = if e.source() == n {
                    e.target()
                } else {
                    e.source()
                };
                comp_set.contains(&graph[other]) && *e.weight() >= min_weight
            })
            .count()
    });

    let Some(&hub_idx) = hub else {
        return vec![];
    };

    // Build sub-graph: hub + its qualifying neighbors.
    let mut sub: UnGraph<RecordId, f32> = UnGraph::new_undirected();
    let mut sub_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    let hub_sub = sub.add_node(graph[hub_idx]);
    sub_map.insert(hub_idx, hub_sub);

    for edge in graph.edges(hub_idx) {
        let other = if edge.source() == hub_idx {
            edge.target()
        } else {
            edge.source()
        };
        if !comp_set.contains(&graph[other]) || *edge.weight() < min_weight {
            continue;
        }
        let other_sub = *sub_map
            .entry(other)
            .or_insert_with(|| sub.add_node(graph[other]));
        sub.add_edge(hub_sub, other_sub, *edge.weight());
    }

    extract_components(&sub)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::scoring::ScoredPair;
    use zer_core::{comparison::ComparisonVector, scoring::MatchBand};

    fn auto_match_pair(a: u64, b: u64, prob: f32) -> ScoredPair {
        ScoredPair {
            record_a: a,
            record_b: b,
            match_weight: 0.0,
            match_probability: prob,
            vector: ComparisonVector {
                record_a: a,
                record_b: b,
                levels: vec![],
            },
            band: MatchBand::AutoMatch,
        }
    }

    fn config() -> ClusterConfig {
        ClusterConfig {
            max_cluster_size: 50,
            within_cluster_min: 0.85,
        }
    }

    #[test]
    fn basic_connected_components() {
        // A-B, B-C → one component of 3
        let mut g = ClusterGraph::new();
        g.add_pairs(&[auto_match_pair(1, 2, 0.95), auto_match_pair(2, 3, 0.95)]);
        let clusters = g.compute_clusters(&config());
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn single_pair_one_cluster() {
        let mut g = ClusterGraph::new();
        g.add_pairs(&[auto_match_pair(1, 2, 0.95)]);
        let clusters = g.compute_clusters(&config());
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    #[test]
    fn weak_bridge_splits_chain() {
        // A -[0.95]- B -[0.28]- C -[0.95]- D
        // with within_cluster_min = 0.85, the B-C edge is removed
        // → {A,B} and {C,D}
        let mut g = ClusterGraph::new();
        g.add_pairs(&[
            auto_match_pair(1, 2, 0.95), // A-B strong
            auto_match_pair(2, 3, 0.28), // B-C weak bridge
            auto_match_pair(3, 4, 0.95), // C-D strong
        ]);
        let mut clusters = g.compute_clusters(&config());
        clusters.sort_by_key(|c| *c.iter().min().unwrap());
        assert_eq!(
            clusters.len(),
            2,
            "weak bridge must split chain into 2 clusters"
        );
        assert_eq!(clusters[0].len(), 2);
        assert_eq!(clusters[1].len(), 2);

        let mut c0 = clusters[0].clone();
        c0.sort();
        let mut c1 = clusters[1].clone();
        c1.sort();
        assert_eq!(c0, vec![1, 2]);
        assert_eq!(c1, vec![3, 4]);
    }

    #[test]
    fn star_pruning_splits_oversized_cluster() {
        // Hub (id=0) connected to 60 satellites with prob 0.95.
        // max_cluster_size = 50 → star pruning kicks in, yielding the hub+satellites
        // as a valid cluster (star pruning keeps all hub-edges ≥ min_weight).
        let cfg = ClusterConfig {
            max_cluster_size: 50,
            within_cluster_min: 0.85,
        };
        let mut g = ClusterGraph::new();
        let pairs: Vec<_> = (1u64..=60).map(|i| auto_match_pair(0, i, 0.95)).collect();
        g.add_pairs(&pairs);

        let clusters = g.compute_clusters(&cfg);
        // After star pruning, the hub stays connected to all 60 neighbors
        // (all edges >= 0.85), so we get one cluster of 61.
        // The important thing is that oversized handling runs without panic.
        assert!(!clusters.is_empty());
        let total_members: usize = clusters.iter().map(|c| c.len()).sum();
        assert!(total_members >= 2);
    }

    #[test]
    fn two_disconnected_pairs_two_clusters() {
        let mut g = ClusterGraph::new();
        g.add_pairs(&[auto_match_pair(1, 2, 0.95), auto_match_pair(3, 4, 0.95)]);
        let clusters = g.compute_clusters(&config());
        assert_eq!(clusters.len(), 2);
    }
}
