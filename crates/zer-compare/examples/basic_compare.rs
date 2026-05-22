/// Example: pairwise comparison and Fellegi-Sunter scoring with EM parameter estimation.
///
/// Demonstrates:
/// 1. Defining a schema with `SchemaBuilder`
/// 2. Building a `FieldComparator` from the schema (auto-selects similarity functions)
/// 3. Comparing record pairs to produce `ComparisonVector`s
/// 4. Running EM to estimate `ModelParams` from the comparison vectors
/// 5. Scoring pairs with `FellegiSunterScorer`
/// 6. Reading back match bands and per-field evidence
use zer_compare::{FieldComparator, FellegiSunterScorer};
use zer_core::{
    comparison::ComparisonLevel,
    record::{FieldValue, Record},
    record_pool::RecordPool,
    schema::{FieldKind, SchemaBuilder},
    scoring::MatchBand,
    traits::{Comparator, Scorer},
};

fn main() {
    println!("=== zer-compare: Fellegi-Sunter comparison example ===\n");

    let schema = SchemaBuilder::new()
        .field("voornamen",     FieldKind::Name)
        .field("achternaam",    FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("postcode",      FieldKind::Id)
        .field("nationaliteit", FieldKind::Categorical)
        .build()
        .expect("schema must not be empty");

    // ── Records ───────────────────────────────────────────────────────────────

    // Canonical registration
    let alice_canonical = Record::new(1)
        .with_source("brp")
        .insert("voornamen",     FieldValue::Text("Alice".into()))
        .insert("achternaam",    FieldValue::Text("van den Berg".into()))
        .insert("geboortedatum", FieldValue::Text("1990-03-15".into()))
        .insert("postcode",      FieldValue::Text("1011AB".into()))
        .insert("nationaliteit", FieldValue::Text("Nederland".into()));

    // Same person, name variant (tussenvoegsel dropped, date off by one day)
    let alice_variant = Record::new(2)
        .with_source("hks")
        .insert("voornamen",     FieldValue::Text("A.".into()))
        .insert("achternaam",    FieldValue::Text("Berg".into()))
        .insert("geboortedatum", FieldValue::Text("1990-03-14".into()))
        .insert("postcode",      FieldValue::Text("1011AB".into()))
        .insert("nationaliteit", FieldValue::Text("Nederland".into()));

    // Completely different person
    let bob = Record::new(3)
        .with_source("brp")
        .insert("voornamen",     FieldValue::Text("Mohammed".into()))
        .insert("achternaam",    FieldValue::Text("El Amrani".into()))
        .insert("geboortedatum", FieldValue::Text("1975-11-22".into()))
        .insert("postcode",      FieldValue::Text("3001XY".into()))
        .insert("nationaliteit", FieldValue::Text("Marokko".into()));

    // ── Compare ───────────────────────────────────────────────────────────────

    let cmp = FieldComparator::from_schema(&schema);

    let cv_match    = cmp.compare(&alice_canonical, &alice_variant, &schema);
    let cv_nonmatch = cmp.compare(&alice_canonical, &bob, &schema);

    println!("Comparison: Alice canonical vs Alice variant");
    for (field, &level) in schema.fields.iter().zip(&cv_match.levels) {
        let indicator = match level {
            ComparisonLevel::Exact   => "✓✓",
            ComparisonLevel::Close   => "✓ ",
            ComparisonLevel::Partial => "~ ",
            ComparisonLevel::None    => "✗ ",
            ComparisonLevel::Null    => "✗ ",
        };
        println!("  [{indicator}] {:<20} {:?}", field.name, level);
    }

    println!("\nComparison: Alice canonical vs Bob");
    for (field, &level) in schema.fields.iter().zip(&cv_nonmatch.levels) {
        let indicator = match level {
            ComparisonLevel::Exact   => "✓✓",
            ComparisonLevel::Close   => "✓ ",
            ComparisonLevel::Partial => "~ ",
            ComparisonLevel::None    => "✗ ",
            ComparisonLevel::Null    => "✗ ",
        };
        println!("  [{indicator}] {:<20} {:?}", field.name, level);
    }

    // ── EM parameter estimation ───────────────────────────────────────────────

    // Build a small synthetic training set:
    // - 3 true match pairs (Alice variants)
    // - 5 non-match pairs (Alice vs Bob and combinations)
    let training_pairs = vec![
        (alice_canonical.clone(), alice_variant.clone()),
        (alice_canonical.clone(), alice_variant.clone()),
        (alice_canonical.clone(), alice_variant.clone()),
        (alice_canonical.clone(), bob.clone()),
        (alice_variant.clone(),   bob.clone()),
        (alice_canonical.clone(), bob.clone()),
        (alice_variant.clone(),   bob.clone()),
        (alice_canonical.clone(), bob.clone()),
    ];

    let pool    = RecordPool::from_pairs(&training_pairs, &schema);
    let indices: Vec<(usize, usize)> = (0..training_pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();
    let batch = cmp.compare_batch_from_pool(&pool, &indices, &schema);

    let scorer = FellegiSunterScorer;
    let params = scorer.estimate_params(&batch, None, 100)
        .expect("EM should converge on this small dataset");

    println!("\nEM-estimated parameters:");
    for (i, field) in schema.fields.iter().enumerate() {
        println!("  {:20} m=[{:.3}, {:.3}, {:.3}, {:.3}]  u=[{:.3}, {:.3}, {:.3}, {:.3}]",
            field.name,
            params.m[i][0], params.m[i][1], params.m[i][2], params.m[i][3],
            params.u[i][0], params.u[i][1], params.u[i][2], params.u[i][3],
        );
    }
    println!("  log_prior_odds={:.3}  upper={:.3}  lower={:.3}",
        params.log_prior_odds, params.upper_threshold, params.lower_threshold);

    // ── Score ─────────────────────────────────────────────────────────────────

    let scored_match    = scorer.score(&cv_match,    &params);
    let scored_nonmatch = scorer.score(&cv_nonmatch, &params);

    println!("\nScored pairs:");
    println!("  Alice canonical vs Alice variant: probability={:.4}  band={:?}",
        scored_match.match_probability, scored_match.band);
    println!("  Alice canonical vs Bob:           probability={:.4}  band={:?}",
        scored_nonmatch.match_probability, scored_nonmatch.band);

    // Sanity assertions
    assert!(
        scored_match.match_probability > scored_nonmatch.match_probability,
        "true match should score higher than non-match"
    );
    assert!(
        scored_nonmatch.band != MatchBand::AutoMatch,
        "clear non-match should not be AutoMatch"
    );

    println!("\nExample completed successfully.");
}
