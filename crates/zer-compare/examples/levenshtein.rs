/// Demonstrates `LevenshteinSimilarity`, normalised edit-distance similarity
/// for string fields, as a splink-equivalent `LevenshteinAtThresholds`.
///
/// `LevenshteinSimilarity` returns a score in [0.0, 1.0]:
/// - edit distance 0      : 1.0 (exact match)
/// - edit distance <= N   : 1.0 - dist/max_distance
/// - edit distance > N    : 0.0 (too different; treated as no evidence)
///
/// The `max_distance` parameter acts as a threshold: distances beyond it collapse
/// to 0.0 rather than producing misleading fractional scores for unrelated strings.
///
/// Run with:  cargo run --example levenshtein -p zer-compare

use zer_compare::similarity::name::LevenshteinSimilarity;
use zer_compare::similarity::SimilarityFn;
use zer_core::record::FieldValue;

fn main() {
    println!("=== LevenshteinSimilarity demo ===\n");

    // ── Basic usage ───────────────────────────────────────────────────────────

    let sim = LevenshteinSimilarity { max_distance: 3 };

    let pairs = vec![
        ("Jansen",  "Jansen",  "exact match (dist=0)"),
        ("Jansen",  "Jansem",  "one typo  (dist=1)"),
        ("Jansen",  "Jansen",  "same again"),
        ("Janssen", "Jansen",  "double-s → single-s (dist=1)"),
        ("Peterson","Petersen","Americanised spelling (dist=1)"),
        ("Smith",   "Smyth",   "variant spelling (dist=1)"),
        ("hello",   "world",   "unrelated (dist=4 > max)"),
        ("Alice",   "Alicia",  "suffix difference (dist=2)"),
    ];

    println!("{:<15} {:<15} {:<6}  {}", "a", "b", "score", "notes");
    println!("{}", "-".repeat(60));

    for (a, b, notes) in &pairs {
        let va = FieldValue::Text(a.to_string());
        let vb = FieldValue::Text(b.to_string());
        let score = sim.similarity(&va, &vb);
        println!("{:<15} {:<15} {:.4}  {}", a, b, score, notes);
    }

    // ── Null handling ─────────────────────────────────────────────────────────

    println!("\nNull-field behaviour:");
    let null_score = sim.similarity(&FieldValue::Null, &FieldValue::Text("Jansen".into()));
    println!("  Null vs 'Jansen' → {null_score:.4}  (must be 0.0)");
    assert_eq!(null_score, 0.0, "null field must return 0.0");

    // ── Effect of max_distance ────────────────────────────────────────────────

    println!("\nEffect of max_distance on 'Jansen' vs 'Jansem' (dist=1):");
    for max in [1, 2, 3, 4] {
        let s = LevenshteinSimilarity { max_distance: max };
        let score = s.similarity(
            &FieldValue::Text("Jansen".into()),
            &FieldValue::Text("Jansem".into()),
        );
        println!("  max_distance={max}  → score={score:.4}");
    }

    // ── Use inside a FieldComparator schema ───────────────────────────────────

    println!("\nIntegrating with FieldComparator:");
    println!("  The comparator uses similarity functions to produce ComparisonLevels.");
    println!("  LevenshteinSimilarity can replace or complement JaroWinklerSimilarity");
    println!("  for fields where edit distance is more meaningful than positional distance.");
    println!("  Configure via FieldComparator::with_fn() or custom SchemaBuilder extensions.");

    // ── Assertions ───────────────────────────────────────────────────────────

    let sim3 = LevenshteinSimilarity { max_distance: 3 };
    assert_eq!(
        sim3.similarity(&FieldValue::Text("abc".into()), &FieldValue::Text("abc".into())),
        1.0, "exact match must yield 1.0"
    );
    assert_eq!(
        sim3.similarity(&FieldValue::Text("hello".into()), &FieldValue::Text("world".into())),
        0.0, "dist > max must yield 0.0"
    );

    let partial = sim3.similarity(
        &FieldValue::Text("Jansen".into()),
        &FieldValue::Text("Jansem".into()),
    );
    assert!(partial > 0.0 && partial < 1.0, "dist=1 within max must yield (0,1)");

    println!("\nAll assertions passed.");
}
