/// Scoring walkthrough demo.
///
/// Walks through the two scoring stages in zer step by step, using hand-crafted
/// Dutch person record pairs that represent realistic data quality scenarios:
///
///   1. Field comparison, produces a `ComparisonVector` (None/Partial/Close/Exact)
///   2. Fellegi-Sunter scoring, converts the vector to a match probability
///
/// No external dataset needed; all records are defined inline.
use demo_common::{
    init_tracing, print_comparison_vectors, print_score_histogram, section, viz::FieldComparison,
};
use zer_compare::FieldComparator;
use zer_core::{
    comparison::ComparisonLevel,
    record::Record,
    schema::{FieldKind, SchemaBuilder},
    scoring::{MatchBand, ModelParams},
    traits::{Comparator, Scorer},
};

fn make_schema() -> zer_core::schema::Schema {
    SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("geslacht", FieldKind::FreeText)
        .field("postcode", FieldKind::FreeText)
        .build()
        .expect("build schema")
}

fn person(id: u64, first: &str, last: &str, dob: &str, gender: &str, pc: &str) -> Record {
    Record::new(id)
        .insert("voornamen", first)
        .insert("achternaam", last)
        .insert("geboortedatum", dob)
        .insert("geslacht", gender)
        .insert("postcode", pc)
}

/// Reasonable EM defaults for a 5-field person schema.
fn default_params(n_fields: usize) -> ModelParams {
    ModelParams {
        m: vec![vec![0.02, 0.06, 0.12, 0.80]; n_fields],
        u: vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
        log_prior_odds: (0.1_f32 / 0.9_f32).ln(),
        upper_threshold: 0.85,
        lower_threshold: 0.15,
    }
}

fn level_to_similarity(level: ComparisonLevel) -> f32 {
    match level {
        ComparisonLevel::Exact => 1.00,
        ComparisonLevel::Close => 0.75,
        ComparisonLevel::Partial => 0.40,
        ComparisonLevel::None => 0.00,
        ComparisonLevel::Null => f32::NAN,
    }
}

fn show_pair(
    label: &str,
    a: &Record,
    b: &Record,
    comparator: &FieldComparator,
    scorer: &zer_compare::FellegiSunterScorer,
    params: &ModelParams,
    schema: &zer_core::schema::Schema,
) -> f32 {
    let vector = comparator.compare(a, b, schema);
    let scored = scorer.score(&vector, params);

    println!("  pair: {} (record #{} ↔ #{})", label, a.id, b.id);

    let field_comparisons: Vec<FieldComparison> = schema
        .fields
        .iter()
        .zip(vector.levels.iter())
        .map(|(field, &level)| FieldComparison {
            field: field.name.clone(),
            similarity: level_to_similarity(level),
        })
        .collect();

    print_comparison_vectors(label, &field_comparisons);

    let band_label = match scored.band {
        MatchBand::AutoMatch => "AUTO-MATCH",
        MatchBand::Borderline => "BORDERLINE",
        MatchBand::AutoReject => "auto-reject",
    };
    println!(
        "    → match weight {:.2}  p = {:.3}  [{}]",
        scored.match_weight, scored.match_probability, band_label
    );

    scored.match_probability
}

fn main() {
    init_tracing();

    let schema = make_schema();
    let comparator = FieldComparator::from_schema(&schema);
    let scorer = zer_compare::FellegiSunterScorer;
    let params = default_params(schema.fields.len());

    // ── Scenario pairs ────────────────────────────────────────────────────────
    section("Record pairs");
    println!("5-field schema: voornamen, achternaam, geboortedatum, geslacht, postcode");
    println!(
        "Thresholds: match ≥ {:.2}, reject < {:.2}",
        params.upper_threshold, params.lower_threshold
    );

    let mut all_scores: Vec<f32> = Vec::new();

    section("Pair 1, exact duplicate");
    let a1 = person(1, "Jan", "de Vries", "1985-03-22", "M", "1234AB");
    let b1 = person(2, "Jan", "de Vries", "1985-03-22", "M", "1234AB");
    all_scores.push(show_pair(
        "Exact duplicate",
        &a1,
        &b1,
        &comparator,
        &scorer,
        &params,
        &schema,
    ));

    section("Pair 2, nickname + address move");
    let a2 = person(3, "Johannes", "de Vries", "1985-03-22", "M", "1234AB");
    let b2 = person(4, "Jan", "de Vries", "1985-03-22", "M", "5678CD");
    all_scores.push(show_pair(
        "Nickname + address move",
        &a2,
        &b2,
        &comparator,
        &scorer,
        &params,
        &schema,
    ));

    section("Pair 3, maiden name change");
    let a3 = person(5, "Anna", "Bakker", "1990-07-15", "V", "2345BC");
    let b3 = person(6, "Anna", "de Groot", "1990-07-15", "V", "2345BC");
    all_scores.push(show_pair(
        "Maiden name change",
        &a3,
        &b3,
        &comparator,
        &scorer,
        &params,
        &schema,
    ));

    section("Pair 4, DOB typo (day/month transposition)");
    let a4 = person(7, "Pieter", "Smit", "1978-04-12", "M", "3456DE");
    let b4 = person(8, "Pieter", "Smit", "1978-12-04", "M", "3456DE");
    all_scores.push(show_pair(
        "DOB transposition",
        &a4,
        &b4,
        &comparator,
        &scorer,
        &params,
        &schema,
    ));

    section("Pair 5, completely different persons");
    let a5 = person(9, "Maria", "Janssen", "1962-09-01", "V", "4567EF");
    let b5 = person(10, "Robert", "Visser", "1975-11-20", "M", "8901GH");
    all_scores.push(show_pair(
        "Different persons",
        &a5,
        &b5,
        &comparator,
        &scorer,
        &params,
        &schema,
    ));

    section("Pair 6, phonetic spelling variant (Marloes vs Marlous)");
    let a6 = person(11, "Marloes", "van Dam", "2001-02-28", "V", "7890FG");
    let b6 = person(12, "Marlous", "van Dam", "2001-02-28", "V", "7890FG");
    all_scores.push(show_pair(
        "Phonetic name variant",
        &a6,
        &b6,
        &comparator,
        &scorer,
        &params,
        &schema,
    ));

    // ── Score distribution ────────────────────────────────────────────────────
    section("Score distribution across all pairs");
    print_score_histogram(&all_scores, params.upper_threshold, params.lower_threshold);

    section("Done");
    println!("The two-stage pipeline (compare → score) converts noisy field values");
    println!("into calibrated match probabilities that drive automated entity resolution.");
}
