use std::collections::{HashMap, HashSet};

use crate::fingerprint::{FieldStats, SchemaFingerprint};

/// Distance 0.0, schemas are structurally identical; saved params can be
/// loaded directly without any EM iterations.
pub const EXACT_MATCH_THRESHOLD: f32 = 0.0;

/// Distance threshold for warm-start eligibility.
/// distance ≤ WARM_START_THRESHOLD → load saved params and run 2–3 EM iterations.
/// distance  > WARM_START_THRESHOLD → cold start (initialize from priors, full EM).
pub const WARM_START_THRESHOLD: f32 = 0.25;

// ── Field-set helpers ─────────────────────────────────────────────────────────

/// Represent each field as a canonical string `"name:DebugKind"` for set ops.
fn field_set(stats: &[FieldStats]) -> HashSet<String> {
    stats
        .iter()
        .map(|f| format!("{}:{:?}", f.name, f.kind))
        .collect()
}

/// Jaccard similarity J(A,B) = |A ∩ B| / |A ∪ B| over (name, kind) pairs.
/// Returns 1.0 when both sets are empty (zero penalty for missing data).
fn jaccard_field_sets(a: &SchemaFingerprint, b: &SchemaFingerprint) -> f32 {
    let a_set = field_set(&a.field_stats);
    let b_set = field_set(&b.field_stats);

    if a_set.is_empty() && b_set.is_empty() {
        return 1.0;
    }

    let intersection = a_set.intersection(&b_set).count();
    let union = a_set.union(&b_set).count();

    if union == 0 {
        return 1.0;
    }

    intersection as f32 / union as f32
}

/// Per-field stat similarity for fields that appear in both fingerprints.
///
/// Combines null_rate proximity and cardinality proximity. Returns 1.0 when
/// there is no sample data to compare (record_count == 0 on either side) so
/// schema-only fingerprints don't receive a spurious stat penalty.
fn matching_field_stat_similarity(a: &SchemaFingerprint, b: &SchemaFingerprint) -> f32 {
    // No sample data on either side → neutral, no penalty.
    if a.record_count == 0 || b.record_count == 0 {
        return 1.0;
    }

    let a_map: HashMap<&str, &FieldStats> =
        a.field_stats.iter().map(|f| (f.name.as_str(), f)).collect();
    let b_map: HashMap<&str, &FieldStats> =
        b.field_stats.iter().map(|f| (f.name.as_str(), f)).collect();

    let matching: Vec<&str> = a_map
        .keys()
        .copied()
        .filter(|name| b_map.contains_key(name))
        .collect();

    if matching.is_empty() {
        return 0.0;
    }

    let total_sim: f32 = matching
        .iter()
        .map(|name| {
            let fa = a_map[name];
            let fb = b_map[name];

            // Null-rate proximity: 1.0 when equal, down to 0.0 when |Δ| = 1.
            let null_sim = 1.0 - (fa.null_rate - fb.null_rate).abs().min(1.0);

            // Cardinality proximity: normalized by the larger value.
            let card_sim = if fa.cardinality == 0 && fb.cardinality == 0 {
                1.0
            } else {
                let max_c = fa.cardinality.max(fb.cardinality) as f32;
                1.0 - (fa.cardinality as f32 - fb.cardinality as f32).abs() / max_c
            };

            (null_sim + card_sim) / 2.0
        })
        .sum();

    total_sim / matching.len() as f32
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the distance between two schema fingerprints.
///
/// Returns a value in `[0.0, 1.0]`:
/// - `0.0`, structurally identical (same hash)
/// - `≤ WARM_START_THRESHOLD`, similar enough for a warm-start
/// - `> WARM_START_THRESHOLD`, too different; use cold start
///
/// The distance is a weighted combination of structural similarity (Jaccard on
/// field name+kind pairs, weight 0.7) and distributional similarity (null-rate
/// and cardinality proximity, weight 0.3).
pub fn fingerprint_distance(a: &SchemaFingerprint, b: &SchemaFingerprint) -> f32 {
    // Fast path: structural identity.
    if a.schema_hash == b.schema_hash {
        return EXACT_MATCH_THRESHOLD;
    }

    let jaccard = jaccard_field_sets(a, b);
    let stat_sim = matching_field_stat_similarity(a, b);

    (1.0 - jaccard) * 0.7 + (1.0 - stat_sim) * 0.3
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::schema::{FieldKind, SchemaBuilder};

    use crate::fingerprint::SchemaFingerprint;

    fn brp_schema() -> zer_core::schema::Schema {
        SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("tussenvoegsel", FieldKind::Categorical)
            .field("geboortedatum", FieldKind::Date)
            .field("geboorteland", FieldKind::Categorical)
            .field("nationaliteit", FieldKind::Categorical)
            .field("straatnaam", FieldKind::Address)
            .field("huisnummer", FieldKind::Address)
            .field("postcode", FieldKind::Id)
            .field("woonplaats", FieldKind::Address)
            .build()
            .unwrap()
    }

    #[test]
    fn identical_fingerprints_zero_distance() {
        let schema = brp_schema();
        let fp1 = SchemaFingerprint::from_schema(&schema);
        let fp2 = SchemaFingerprint::from_schema(&schema);
        assert_eq!(
            fingerprint_distance(&fp1, &fp2),
            0.0,
            "identical fingerprints must have distance 0.0"
        );
    }

    #[test]
    fn one_extra_field_is_warm_start_range() {
        let base = brp_schema();

        // Extended schema: one extra field added.
        let extended = SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("tussenvoegsel", FieldKind::Categorical)
            .field("geboortedatum", FieldKind::Date)
            .field("geboorteland", FieldKind::Categorical)
            .field("nationaliteit", FieldKind::Categorical)
            .field("straatnaam", FieldKind::Address)
            .field("huisnummer", FieldKind::Address)
            .field("postcode", FieldKind::Id)
            .field("woonplaats", FieldKind::Address)
            .field("verblijfstitel", FieldKind::Categorical) // new field
            .build()
            .unwrap();

        let fp_base = SchemaFingerprint::from_schema(&base);
        let fp_ext = SchemaFingerprint::from_schema(&extended);
        let dist = fingerprint_distance(&fp_base, &fp_ext);

        assert!(
            dist > EXACT_MATCH_THRESHOLD,
            "schemas differ, distance must be > 0"
        );
        assert!(
            dist <= WARM_START_THRESHOLD,
            "one extra field out of 11 should be warm-start eligible, got dist={dist:.4}"
        );
    }

    #[test]
    fn completely_different_schema_is_cold_start() {
        // SIM subscriber schema shares only a few fields with BRP.
        let sim = SchemaBuilder::new()
            .field("sim_id", FieldKind::Id)
            .field("msisdn", FieldKind::Phone)
            .field("imsi", FieldKind::Id)
            .field("iccid", FieldKind::Id)
            .field("carrier", FieldKind::Categorical)
            .field("contract_type", FieldKind::Categorical)
            .field("activatiedatum", FieldKind::Date)
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .field("nationaliteit", FieldKind::Categorical)
            .field("document_type", FieldKind::Categorical)
            .field("document_nummer", FieldKind::Id)
            .field("bsn", FieldKind::Id)
            .build()
            .unwrap();

        let brp = brp_schema();
        let fp_brp = SchemaFingerprint::from_schema(&brp);
        let fp_sim = SchemaFingerprint::from_schema(&sim);
        let dist = fingerprint_distance(&fp_brp, &fp_sim);

        assert!(
            dist > WARM_START_THRESHOLD,
            "BRP vs SIM should exceed warm-start threshold, got dist={dist:.4}"
        );
    }

    #[test]
    fn reordered_fields_same_schema_zero_distance() {
        let s1 = SchemaBuilder::new()
            .field("alpha", FieldKind::Name)
            .field("beta", FieldKind::Date)
            .build()
            .unwrap();
        let s2 = SchemaBuilder::new()
            .field("beta", FieldKind::Date)
            .field("alpha", FieldKind::Name)
            .build()
            .unwrap();

        let fp1 = SchemaFingerprint::from_schema(&s1);
        let fp2 = SchemaFingerprint::from_schema(&s2);
        assert_eq!(
            fingerprint_distance(&fp1, &fp2),
            0.0,
            "reordered fields must produce identical fingerprints (distance = 0)"
        );
    }

    #[test]
    fn distance_is_symmetric() {
        let brp = brp_schema();
        let sim = SchemaBuilder::new()
            .field("msisdn", FieldKind::Phone)
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .build()
            .unwrap();

        let fp_brp = SchemaFingerprint::from_schema(&brp);
        let fp_sim = SchemaFingerprint::from_schema(&sim);

        let d_ab = fingerprint_distance(&fp_brp, &fp_sim);
        let d_ba = fingerprint_distance(&fp_sim, &fp_brp);

        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "distance must be symmetric: d(a,b)={d_ab} d(b,a)={d_ba}"
        );
    }

    #[test]
    fn distance_bounded_zero_to_one() {
        let s1 = SchemaBuilder::new()
            .field("x", FieldKind::Name)
            .build()
            .unwrap();
        let s2 = SchemaBuilder::new()
            .field("y", FieldKind::Date)
            .build()
            .unwrap();

        let fp1 = SchemaFingerprint::from_schema(&s1);
        let fp2 = SchemaFingerprint::from_schema(&s2);
        let d = fingerprint_distance(&fp1, &fp2);
        assert!(d >= 0.0 && d <= 1.0, "distance must be in [0, 1], got {d}");
    }
}
