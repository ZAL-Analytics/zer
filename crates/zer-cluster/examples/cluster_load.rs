/// Example: reload entities from a persisted `.zes` file.
///
/// Opens `data/examples/demo_entities.zes` written by `cluster_save` and
/// verifies that all entities, member counts, and record-to-entity lookups
/// are intact after a process restart.
///
/// Run order:
///   cargo run --example cluster_save -p zer-cluster
///   cargo run --example cluster_load -p zer-cluster
use std::path::Path;

use zer_cluster::ZalEntityStore;
use zer_core::traits::EntityStore;

const STORE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/examples/demo_entities.zes"
);

// Same groups as cluster_save, used to verify lookups.
const GROUPS: &[&[u64]] = &[
    &[101, 102, 103],
    &[201, 202, 203, 204],
    &[301, 302],
    &[401, 402, 403],
    &[501, 502, 503, 504, 505],
];

fn main() {
    println!("=== cluster_load: reading .zes file ===\n");

    let store_path = Path::new(STORE_PATH);
    if !store_path.exists() {
        println!(
            "Store file not found: {}\nRun `cargo run --example cluster_save -p zer-cluster` first.",
            store_path.display()
        );
        std::process::exit(1);
    }

    println!("Reading: {}", store_path.display());
    let store = ZalEntityStore::open(store_path).expect("failed to open .zes store");

    // ── Step 1: list all entities ─────────────────────────────────────────────

    let all = store.all_entities().expect("all_entities failed");
    println!("\nLoaded {} entities:", all.len());
    assert_eq!(
        all.len(),
        GROUPS.len(),
        "expected {} entities, found {}",
        GROUPS.len(),
        all.len()
    );

    for entity in &all {
        let mut member_ids: Vec<_> = entity.members.iter().map(|m| m.record_id).collect();
        member_ids.sort();
        println!("  Entity #{}: {} members {:?}", entity.id, entity.members.len(), member_ids);
    }

    // ── Step 2: record-to-entity spot checks ─────────────────────────────────

    println!("\nRecord-to-entity lookups:");
    for group in GROUPS {
        // All records in a group must map to the same entity.
        let eids: Vec<_> = group
            .iter()
            .map(|&rid| {
                let eid = store
                    .record_to_entity(rid)
                    .expect("lookup failed")
                    .unwrap_or_else(|| panic!("record {rid} not found in store"));
                eid
            })
            .collect();

        let first = eids[0];
        let all_same = eids.iter().all(|&e| e == first);
        let status = if all_same { "✓" } else { "✗ MISMATCH" };
        println!(
            "  group {:?} → entity #{first}  {status}",
            group
        );
        assert!(all_same, "records in group {:?} must all map to the same entity", group);
    }

    // ── Step 3: verify no cross-group contamination ───────────────────────────

    println!("\nCross-group isolation check:");
    // One representative from each group.
    let representatives: Vec<u64> = GROUPS.iter().map(|g| g[0]).collect();
    let entity_ids: Vec<_> = representatives
        .iter()
        .map(|&rid| store.record_to_entity(rid).unwrap().unwrap())
        .collect();

    let unique_count = {
        let mut ids = entity_ids.clone();
        ids.sort();
        ids.dedup();
        ids.len()
    };
    assert_eq!(
        unique_count,
        GROUPS.len(),
        "each group must map to a distinct entity"
    );
    println!(
        "  {} group representatives map to {} distinct entities  ✓",
        GROUPS.len(),
        unique_count
    );

    println!("\nAll assertions passed. Entities survived process restart. ✓");
}
