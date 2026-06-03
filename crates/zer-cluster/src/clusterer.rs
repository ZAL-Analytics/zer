use zer_core::{
    entity::{Entity, EntityId, EntityMember, ResolutionMethod},
    record::RecordId,
    scoring::{ModelParams, ScoredPair},
    traits::Clusterer,
};

use crate::{
    graph::{ClusterConfig, ClusterGraph},
    threshold::partition_by_band,
};

/// Connected-components clusterer with weak-edge removal and star pruning.
///
/// Algorithm:
/// 1. Partition pairs into bands using the supplied `ModelParams`.
/// 2. Build an undirected graph from `AutoMatch` pairs only.
/// 3. Remove edges below `config.within_cluster_min` (chain-breaking).
/// 4. Split oversized components via star pruning.
/// 5. Emit one `Entity` per non-trivial component (≥ 2 members).
#[derive(Default)]
pub struct ConnectedComponentsClusterer {
    pub config: ClusterConfig,
}

impl Clusterer for ConnectedComponentsClusterer {
    fn cluster(&self, pairs: &[ScoredPair], params: &ModelParams) -> Vec<Entity> {
        let banded = partition_by_band(pairs.to_vec(), params);

        let mut graph = ClusterGraph::new();
        graph.add_pairs(&banded.auto_match);

        let components = graph.compute_clusters(&self.config);

        components
            .into_iter()
            .enumerate()
            .map(|(idx, members)| {
                let entity_members = members
                    .iter()
                    .map(|&rid| EntityMember {
                        record_id: rid,
                        score: best_score_in_cluster(rid, &banded.auto_match),
                        method: ResolutionMethod::AutoMatch,
                        source: None,
                    })
                    .collect();

                Entity {
                    // Temporary sequential ids, caller should persist through
                    // EntityStore.upsert_entity() to get stable database ids.
                    id: idx as EntityId + 1,
                    members: entity_members,
                }
            })
            .collect()
    }
}

/// Returns the highest `match_probability` of any `AutoMatch` pair that
/// involves `record_id`.
fn best_score_in_cluster(record_id: RecordId, pairs: &[ScoredPair]) -> f32 {
    pairs
        .iter()
        .filter(|p| p.record_a == record_id || p.record_b == record_id)
        .map(|p| p.match_probability)
        .fold(0.0_f32, f32::max)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{comparison::ComparisonVector, scoring::MatchBand};

    fn params() -> ModelParams {
        ModelParams {
            m: vec![],
            u: vec![],
            log_prior_odds: 0.0,
            upper_threshold: 0.8,
            lower_threshold: 0.2,
        }
    }

    fn pair(a: u64, b: u64, prob: f32, band: MatchBand) -> ScoredPair {
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
            band,
        }
    }

    #[test]
    fn empty_pairs_returns_empty() {
        let clusterer = ConnectedComponentsClusterer::default();
        let entities = clusterer.cluster(&[], &params());
        assert!(entities.is_empty());
    }

    #[test]
    fn two_matched_pairs_form_one_entity() {
        let clusterer = ConnectedComponentsClusterer::default();
        let pairs = vec![
            pair(1, 2, 0.95, MatchBand::AutoMatch),
            pair(2, 3, 0.95, MatchBand::AutoMatch),
        ];
        let entities = clusterer.cluster(&pairs, &params());
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].members.len(), 3);
    }

    #[test]
    fn auto_rejected_pairs_ignored() {
        let clusterer = ConnectedComponentsClusterer::default();
        let pairs = vec![
            pair(1, 2, 0.95, MatchBand::AutoMatch),
            pair(3, 4, 0.05, MatchBand::AutoReject),
        ];
        let entities = clusterer.cluster(&pairs, &params());
        assert_eq!(entities.len(), 1);
        let rids: Vec<_> = entities[0].members.iter().map(|m| m.record_id).collect();
        assert!(rids.contains(&1));
        assert!(rids.contains(&2));
        assert!(!rids.contains(&3));
        assert!(!rids.contains(&4));
    }

    #[test]
    fn members_get_correct_scores() {
        let clusterer = ConnectedComponentsClusterer::default();
        let pairs = vec![
            pair(1, 2, 0.92, MatchBand::AutoMatch),
            pair(1, 3, 0.88, MatchBand::AutoMatch),
        ];
        let entities = clusterer.cluster(&pairs, &params());
        assert_eq!(entities.len(), 1);

        let member_1 = entities[0]
            .members
            .iter()
            .find(|m| m.record_id == 1)
            .unwrap();
        assert!(
            (member_1.score - 0.92).abs() < 1e-5,
            "record 1 best score is 0.92"
        );
    }
}
