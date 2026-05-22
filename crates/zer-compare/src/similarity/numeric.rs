use zer_core::{record::FieldValue, schema::FieldKind};

use crate::similarity::SimilarityFn;

// ── NumericBucketedSimilarity ─────────────────────────────────────────────────

/// Similarity for numeric fields based on relative difference, bucketed into
/// four bands.
///
/// Extracts f64 values from Text (parsed), Int, or Float field values and
/// computes `relative_diff = |a - b| / max(|a|, |b|, 1.0)`:
///   relative_diff == 0.0      : 1.0  (exact)
///   relative_diff <= 0.05     : 0.85 (<= 5% difference)
///   relative_diff <= 0.20     : 0.6  (<= 20% difference)
///   relative_diff <= 0.50     : 0.3  (<= 50% difference)
///   otherwise                 : 0.0
pub struct NumericBucketedSimilarity;

fn extract_numeric(v: &FieldValue) -> Option<f64> {
    match v {
        FieldValue::Float(f) => Some(*f),
        FieldValue::Int(i)   => Some(*i as f64),
        FieldValue::Text(s)  => s.trim().parse::<f64>().ok(),
        _                    => None,
    }
}

fn numeric_score(va: f64, vb: f64) -> f32 {
    let diff = (va - vb).abs();
    if diff == 0.0 { return 1.0; }
    let denom    = va.abs().max(vb.abs()).max(1.0);
    let rel_diff = diff / denom;
    if rel_diff <= 0.05 { 0.85 }
    else if rel_diff <= 0.20 { 0.6 }
    else if rel_diff <= 0.50 { 0.3 }
    else { 0.0 }
}

impl SimilarityFn for NumericBucketedSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (extract_numeric(a), extract_numeric(b)) {
            (Some(va), Some(vb)) => numeric_score(va, vb),
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        match (a.trim().parse::<f64>(), b.trim().parse::<f64>()) {
            (Ok(va), Ok(vb)) => numeric_score(va, vb),
            _ => 0.0,
        }
    }
    fn field_kind(&self) -> FieldKind { FieldKind::Numeric }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ti(n: i64)  -> FieldValue { FieldValue::Int(n) }
    fn tf(f: f64)  -> FieldValue { FieldValue::Float(f) }
    fn tv(s: &str) -> FieldValue { FieldValue::Text(s.into()) }

    #[test]
    fn exact_int_match() {
        let sim = NumericBucketedSimilarity;
        assert_eq!(sim.similarity(&ti(180), &ti(180)), 1.0);
    }

    #[test]
    fn close_within_5_percent() {
        let sim = NumericBucketedSimilarity;
        // 180 vs 183 → diff 3, denom 183 → 1.6% → bucket 0.85
        assert_eq!(sim.similarity(&ti(180), &ti(183)), 0.85);
    }

    #[test]
    fn medium_within_20_percent() {
        let sim = NumericBucketedSimilarity;
        // 100 vs 115 → diff 15, denom 115 → 13% → bucket 0.6
        assert_eq!(sim.similarity(&ti(100), &ti(115)), 0.6);
    }

    #[test]
    fn large_within_50_percent() {
        let sim = NumericBucketedSimilarity;
        // 100 vs 140 → diff 40, denom 140 → 28.6% → bucket 0.3
        assert_eq!(sim.similarity(&ti(100), &ti(140)), 0.3);
    }

    #[test]
    fn very_different() {
        let sim = NumericBucketedSimilarity;
        assert_eq!(sim.similarity(&ti(100), &ti(300)), 0.0);
    }

    #[test]
    fn float_parsing_from_text() {
        let sim = NumericBucketedSimilarity;
        // GPS-style: "52.345" vs "52.346", nearly identical
        assert_eq!(sim.similarity(&tv("52.345"), &tv("52.346")), 0.85);
    }

    #[test]
    fn mixed_int_float() {
        let sim = NumericBucketedSimilarity;
        assert_eq!(sim.similarity(&ti(100), &tf(100.0)), 1.0);
    }

    #[test]
    fn null_returns_zero() {
        let sim = NumericBucketedSimilarity;
        assert_eq!(sim.similarity(&FieldValue::Null, &ti(100)), 0.0);
    }
}
