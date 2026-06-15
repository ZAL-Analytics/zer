/// Example: cluster synthetic entity groups and persist to a `.zes` file.
///
/// Generates 5 known entity groups from synthetic scored pairs, runs
/// `ConnectedComponentsClusterer`, and saves the resulting entities to
/// `data/v1.1/examples/demo_entities.zes`.
///
/// Run order:
///   cargo run --example cluster_save -p zer-cluster
///   cargo run --example cluster_load -p zer-cluster
use std::path::Path;

use zer_cluster::{ConnectedComponentsClusterer, ZalEntityStore};
use zer_core::{
    comparison::ComparisonVector,
    scoring::{MatchBand, ModelParams, ScoredPair},
    traits::{Clusterer, EntityStore},
};

const STORE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/v1.1/examples/demo_entities.zes"
);

// Known entity groups, record IDs are meaningful across both examples.
const GROUPS: &[&[u64]] = &[
    &[101, 102, 103],
    &[201, 202, 203, 204],
    &[301, 302],
    &[401, 402, 403],
    &[501, 502, 503, 504, 505],
];

fn make_pair(a: u64, b: u64, prob: f32, band: MatchBand) -> ScoredPair {
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

fn main() {
    println!("=== cluster_save: clustering and writing .zes file ===\n");

    let store_path = Path::new(STORE_PATH);
    println!("Output: {}", store_path.display());

    // ── Build synthetic AutoMatch pairs within each group ─────────────────────

    let mut pairs: Vec<ScoredPair> = Vec::new();
    for group in GROUPS {
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                pairs.push(make_pair(group[i], group[j], 0.95, MatchBand::AutoMatch));
            }
        }
    }

    // Inter-group noise pairs that should be filtered out (AutoReject).
    pairs.push(make_pair(103, 201, 0.04, MatchBand::AutoReject));
    pairs.push(make_pair(204, 301, 0.03, MatchBand::AutoReject));
    pairs.push(make_pair(302, 401, 0.06, MatchBand::AutoReject));
    pairs.push(make_pair(403, 501, 0.02, MatchBand::AutoReject));

    println!(
        "Generated {} pairs ({} AutoMatch, {} AutoReject).",
        pairs.len(),
        pairs
            .iter()
            .filter(|p| p.band == MatchBand::AutoMatch)
            .count(),
        pairs
            .iter()
            .filter(|p| p.band == MatchBand::AutoReject)
            .count(),
    );

    // ── Cluster ───────────────────────────────────────────────────────────────

    let clusterer = ConnectedComponentsClusterer::default();
    let params = ModelParams {
        m: vec![],
        u: vec![],
        log_prior_odds: 0.0,
        upper_threshold: 0.80,
        lower_threshold: 0.20,
    };

    let entities = clusterer.cluster(&pairs, &params);
    println!("\nClustering complete: {} entities found.", entities.len());
    assert_eq!(
        entities.len(),
        GROUPS.len(),
        "expected exactly {} groups",
        GROUPS.len()
    );

    // ── Persist to .zes store ─────────────────────────────────────────────────

    // Remove any existing store so this example always starts clean.
    if store_path.exists() {
        std::fs::remove_file(store_path).expect("failed to remove old .zes file");
    }

    let store = ZalEntityStore::open(store_path).expect("failed to open .zes store");

    println!("\nSaving entities:");
    for entity in &entities {
        let eid = store.upsert_entity(entity).expect("upsert failed");
        let mut member_ids: Vec<_> = entity.members.iter().map(|m| m.record_id).collect();
        member_ids.sort();
        let best = entity
            .members
            .iter()
            .map(|m| m.score)
            .fold(0.0_f32, f32::max);
        println!(
            "  Entity #{eid}: {} members {:?}  (best score: {best:.3})",
            entity.members.len(),
            member_ids
        );
    }

    // Verify round-trip in the same session.
    let all = store.all_entities().unwrap();
    assert_eq!(all.len(), GROUPS.len());

    let file_size = std::fs::metadata(store_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "\nFile size on disk: {} bytes ({:.1} KB)",
        file_size,
        file_size as f64 / 1024.0
    );

    println!("\nDone. Run `cargo run --example cluster_load -p zer-cluster` to verify the file.");
}
