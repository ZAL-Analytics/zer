/// Integration tests for `ConnectedComponentsClusterer` using the synthetic
/// CDR and FIU ground-truth datasets from `data/tests/`.
///
/// Test strategy:
/// 1. Load ground-truth CSV and map identifiers to sequential `RecordId`s.
/// 2. For each ground-truth cluster: create `AutoMatch` pairs (prob = 0.95).
/// 3. Add a small number of inter-cluster noise pairs (prob = 0.05, AutoReject).
/// 4. Run the clusterer.
/// 5. Assert recall ≥ 0.90: for every ground-truth cluster of size ≥ 2, all
///    members must appear in the same output entity.
use std::collections::{HashMap, HashSet};

use zer_cluster::ConnectedComponentsClusterer;
use zer_core::{
    comparison::ComparisonVector,
    record::RecordId,
    scoring::{MatchBand, ModelParams, ScoredPair},
    traits::Clusterer,
};

const CDR_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/tests/cdr/ground_truth_clusters.csv"
);

const FIU_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/tests/fiu/ground_truth_clusters.csv"
);

// ── Helpers ───────────────────────────────────────────────────────────────────

fn params() -> ModelParams {
    ModelParams {
        m: vec![],
        u: vec![],
        log_prior_odds: 0.0,
        upper_threshold: 0.80,
        lower_threshold: 0.20,
    }
}

fn scored_pair(a: RecordId, b: RecordId, prob: f32, band: MatchBand) -> ScoredPair {
    ScoredPair {
        record_a:          a,
        record_b:          b,
        match_weight:      0.0,
        match_probability: prob,
        vector:            ComparisonVector { record_a: a, record_b: b, levels: vec![] },
        band,
    }
}

/// Load a two-column key for clustering.  Returns `(groups, id_map)` where
/// `groups` maps cluster_id to a sorted list of RecordIds.
fn load_clusters(
    csv_path: &str,
    key_col: &str,
    cluster_col: &str,
) -> (HashMap<String, Vec<RecordId>>, HashMap<String, RecordId>) {
    let mut rdr = csv::Reader::from_path(csv_path)
        .unwrap_or_else(|_| panic!("CSV not found: {csv_path}"));
    let headers = rdr.headers().unwrap().clone();

    let key_idx = headers.iter().position(|h| h == key_col)
        .unwrap_or_else(|| panic!("column '{key_col}' not found"));
    let clu_idx = headers.iter().position(|h| h == cluster_col)
        .unwrap_or_else(|| panic!("column '{cluster_col}' not found"));

    let mut id_map: HashMap<String, RecordId> = HashMap::new();
    let mut groups: HashMap<String, Vec<RecordId>> = HashMap::new();
    let mut next_id: RecordId = 1;

    for result in rdr.records() {
        let row = result.unwrap();
        let key = row.get(key_idx).unwrap().trim().to_string();
        let clu = row.get(clu_idx).unwrap().trim().to_string();

        let rid = *id_map.entry(key).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });

        groups.entry(clu).or_default().push(rid);
    }

    (groups, id_map)
}

/// Build `AutoMatch` pairs for all members within each cluster.
fn intra_cluster_pairs(groups: &HashMap<String, Vec<RecordId>>) -> Vec<ScoredPair> {
    let mut pairs = Vec::new();
    for members in groups.values() {
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                pairs.push(scored_pair(members[i], members[j], 0.95, MatchBand::AutoMatch));
            }
        }
    }
    pairs
}

/// Compute recall: fraction of ground-truth clusters (size ≥ 2) whose members
/// all appear in the same output entity.
fn compute_recall(
    groups: &HashMap<String, Vec<RecordId>>,
    entities: &[zer_core::entity::Entity],
) -> f64 {
    // Build lookup: record_id → set of co-members in the same output entity.
    let mut entity_of: HashMap<RecordId, usize> = HashMap::new();
    for (idx, entity) in entities.iter().enumerate() {
        for m in &entity.members {
            entity_of.insert(m.record_id, idx);
        }
    }

    let mut total = 0usize;
    let mut correct = 0usize;

    for members in groups.values() {
        if members.len() < 2 {
            continue;
        }
        total += 1;

        // All members must map to the same entity index.
        let entity_ids: HashSet<usize> = members
            .iter()
            .filter_map(|rid| entity_of.get(rid).copied())
            .collect();

        if entity_ids.len() == 1 {
            correct += 1;
        }
    }

    if total == 0 { 1.0 } else { correct as f64 / total as f64 }
}

// ── CDR integration test ───────────────────────────────────────────────────────

#[test]
fn cdr_ground_truth_recall_at_least_90_percent() {
    let (groups, _) = load_clusters(CDR_CSV, "msisdn", "cluster_id");

    let pairs = intra_cluster_pairs(&groups);
    let clusterer = ConnectedComponentsClusterer::default();
    let entities = clusterer.cluster(&pairs, &params());

    let recall = compute_recall(&groups, &entities);
    assert!(
        recall >= 0.90,
        "CDR recall {recall:.4} is below the 0.90 threshold"
    );
}

// ── FIU integration test ───────────────────────────────────────────────────────

#[test]
fn fiu_ground_truth_recall_at_least_90_percent() {
    let (groups, _) = load_clusters(FIU_CSV, "iban", "cluster_id");

    let pairs = intra_cluster_pairs(&groups);
    let clusterer = ConnectedComponentsClusterer::default();
    let entities = clusterer.cluster(&pairs, &params());

    let recall = compute_recall(&groups, &entities);
    assert!(
        recall >= 0.90,
        "FIU recall {recall:.4} is below the 0.90 threshold"
    );
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[test]
fn synthetic_three_clusters_recovered_exactly() {
    // 3 known clusters: {1,2,3}, {4,5}, {6,7,8}
    let pairs = vec![
        scored_pair(1, 2, 0.95, MatchBand::AutoMatch),
        scored_pair(2, 3, 0.95, MatchBand::AutoMatch),
        scored_pair(4, 5, 0.95, MatchBand::AutoMatch),
        scored_pair(6, 7, 0.95, MatchBand::AutoMatch),
        scored_pair(7, 8, 0.95, MatchBand::AutoMatch),
        // Inter-cluster noise (AutoReject, ignored by clusterer)
        scored_pair(1, 4, 0.05, MatchBand::AutoReject),
        scored_pair(3, 6, 0.05, MatchBand::AutoReject),
    ];

    let clusterer = ConnectedComponentsClusterer::default();
    let entities = clusterer.cluster(&pairs, &params());

    assert_eq!(entities.len(), 3, "should recover exactly 3 clusters");

    let sizes: Vec<usize> = {
        let mut s: Vec<_> = entities.iter().map(|e| e.members.len()).collect();
        s.sort();
        s
    };
    assert_eq!(sizes, vec![2, 3, 3]);
}

#[test]
fn weak_bridge_breaks_chain_in_clusterer() {
    // A-[0.95]-B-[0.28]-C-[0.95]-D: B-C is below within_cluster_min=0.85
    let pairs = vec![
        scored_pair(1, 2, 0.95, MatchBand::AutoMatch),
        scored_pair(2, 3, 0.28, MatchBand::AutoMatch), // weak bridge
        scored_pair(3, 4, 0.95, MatchBand::AutoMatch),
    ];

    let clusterer = ConnectedComponentsClusterer::default();
    let entities = clusterer.cluster(&pairs, &params());

    assert_eq!(entities.len(), 2, "weak bridge should split into 2 entities");
}
