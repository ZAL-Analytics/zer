/// Example: full cluster-and-store workflow.
///
/// Demonstrates the end-to-end path using synthetic data:
/// 1. Construct 20 `ScoredPair`s representing 4 known entity groups.
/// 2. Run `ConnectedComponentsClusterer` to recover the groups.
/// 3. Persist entities into an in-memory `ZalEntityStore`.
/// 4. Query entities and individual record mappings.
/// 5. Show how to persist to a real `.zes` file instead.
use zer_cluster::{ConnectedComponentsClusterer, ZalEntityStore};
use zer_core::{
    comparison::ComparisonVector,
    scoring::{MatchBand, ModelParams, ScoredPair},
    traits::{Clusterer, EntityStore},
};

fn main() {
    println!("=== zer-cluster: cluster and store example ===\n");

    // ── Step 1: Build synthetic scored pairs ──────────────────────────────────
    //
    // Four known entity groups:
    //   Group A: records 1,2,3
    //   Group B: records 4,5,6,7
    //   Group C: records 8,9
    //   Group D: records 10,11,12

    let groups: &[&[u64]] = &[
        &[1, 2, 3],
        &[4, 5, 6, 7],
        &[8, 9],
        &[10, 11, 12],
    ];

    let mut pairs: Vec<ScoredPair> = Vec::new();
    for group in groups {
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                pairs.push(make_pair(group[i], group[j], 0.95, MatchBand::AutoMatch));
            }
        }
    }

    // A few inter-group noise pairs that should be ignored (AutoReject).
    pairs.push(make_pair(3, 4, 0.05, MatchBand::AutoReject));
    pairs.push(make_pair(7, 8, 0.05, MatchBand::AutoReject));
    pairs.push(make_pair(9, 10, 0.05, MatchBand::AutoReject));

    println!("Generated {} scored pairs ({} AutoMatch, {} AutoReject).",
        pairs.len(),
        pairs.iter().filter(|p| p.band == MatchBand::AutoMatch).count(),
        pairs.iter().filter(|p| p.band == MatchBand::AutoReject).count(),
    );

    // ── Step 2: Cluster ───────────────────────────────────────────────────────

    let clusterer = ConnectedComponentsClusterer::default();
    let params = ModelParams {
        m: vec![],
        u: vec![],
        log_prior_odds: 0.0,
        upper_threshold: 0.80,
        lower_threshold: 0.20,
    };

    let entities = clusterer.cluster(&pairs, &params);
    println!("\nStep 2, Clustering: found {} entities.", entities.len());
    assert_eq!(entities.len(), groups.len(), "should recover all 4 groups");

    // ── Step 3: Persist to in-memory ZalEntityStore ───────────────────────────

    let store = ZalEntityStore::open_in_memory().expect("failed to open in-memory store");

    for entity in &entities {
        let eid = store.upsert_entity(entity).expect("upsert failed");
        let member_ids: Vec<_> = entity.members.iter().map(|m| m.record_id).collect();
        println!(
            "  Entity #{eid}: {} members {:?}  (best score: {:.3})",
            entity.members.len(),
            member_ids,
            entity.members.iter().map(|m| m.score).fold(0.0_f32, f32::max),
        );
    }

    // ── Step 4: Query the store ───────────────────────────────────────────────

    println!("\nStep 4, Querying store:");

    let all = store.all_entities().expect("all_entities failed");
    println!("  all_entities() returned {} entities.", all.len());

    // Spot-check record_to_entity lookups.
    for &rid in &[1u64, 4, 8, 10] {
        let eid = store.record_to_entity(rid).expect("lookup failed");
        println!("  record_to_entity({rid}) → entity {:?}", eid);
        assert!(eid.is_some(), "record {rid} should be in some entity");
    }

    // Verify records from the same group map to the same entity.
    let eid1 = store.record_to_entity(1).unwrap().unwrap();
    let eid2 = store.record_to_entity(2).unwrap().unwrap();
    let eid3 = store.record_to_entity(3).unwrap().unwrap();
    assert_eq!(eid1, eid2, "records 1 and 2 must be in the same entity");
    assert_eq!(eid2, eid3, "records 2 and 3 must be in the same entity");

    // ── Step 5: Show file-based persistence ───────────────────────────────────

    println!("\nStep 5, File persistence:");
    println!("  To persist to a real .zes file, replace `open_in_memory()` with:");
    println!("    ZalEntityStore::open(std::path::Path::new(\"output.zes\"))");
    println!("  The file can be reopened across process restarts.");

    println!("\nExample completed successfully. ✓");
}

fn make_pair(a: u64, b: u64, prob: f32, band: MatchBand) -> ScoredPair {
    ScoredPair {
        record_a:          a,
        record_b:          b,
        match_weight:      0.0,
        match_probability: prob,
        vector:            ComparisonVector { record_a: a, record_b: b, levels: vec![] },
        band,
    }
}
