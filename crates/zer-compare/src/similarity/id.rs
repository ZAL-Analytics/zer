use zer_core::{record::FieldValue, schema::FieldKind};

use crate::similarity::SimilarityFn;

// ── ExactIdSimilarity ─────────────────────────────────────────────────────────

/// Returns 1.0 if the string representations are identical, 0.0 otherwise.
/// Works on Text, Int, Float, and Bool field values by comparing their
/// string/numeric representations.
pub struct ExactIdSimilarity;

impl SimilarityFn for ExactIdSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a),  FieldValue::Text(b))  => if a == b { 1.0 } else { 0.0 },
            (FieldValue::Int(a),   FieldValue::Int(b))   => if a == b { 1.0 } else { 0.0 },
            (FieldValue::Float(a), FieldValue::Float(b)) => if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 },
            (FieldValue::Bool(a),  FieldValue::Bool(b))  => if a == b { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 { if a == b { 1.0 } else { 0.0 } }
    fn field_kind(&self) -> FieldKind { FieldKind::Id }
}

// ── HammingSimilarity ─────────────────────────────────────────────────────────

/// Hamming-distance similarity for equal-length strings.
///
/// Only applies when both values have the same character length. Returns:
///   distance 0              : 1.0
///   distance <= max_distance: 0.8
///   otherwise               : 0.0
pub struct HammingSimilarity {
    pub max_distance: usize,
}

impl SimilarityFn for HammingSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) if a.len() == b.len() => {
                match strsim::hamming(a, b) {
                    Ok(0)    => 1.0,
                    Ok(dist) if dist <= self.max_distance => 0.8,
                    _ => 0.0,
                }
            }
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        if a.len() != b.len() { return 0.0; }
        match strsim::hamming(a, b) {
            Ok(0)    => 1.0,
            Ok(dist) if dist <= self.max_distance => 0.8,
            _ => 0.0,
        }
    }
    fn field_kind(&self) -> FieldKind { FieldKind::Id }
}

// ── SuffixMatchSimilarity ─────────────────────────────────────────────────────

/// Returns 1.0 when the last `n` characters of both values are identical.
///
/// Useful for partial ID matching (e.g. BSN last 4 digits, phone suffix).
/// Returns 0.0 if either string is shorter than `n`.
pub struct SuffixMatchSimilarity {
    pub n: usize,
}

impl SimilarityFn for SuffixMatchSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => {
                if a.len() < self.n || b.len() < self.n { return 0.0; }
                let sa = &a[a.len() - self.n..];
                let sb = &b[b.len() - self.n..];
                if sa == sb { 1.0 } else { 0.0 }
            }
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        if a.len() < self.n || b.len() < self.n { return 0.0; }
        if &a[a.len() - self.n..] == &b[b.len() - self.n..] { 1.0 } else { 0.0 }
    }
    fn field_kind(&self) -> FieldKind { FieldKind::Id }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tv(s: &str) -> FieldValue { FieldValue::Text(s.into()) }

    #[test]
    fn exact_id_match() {
        let sim = ExactIdSimilarity;
        assert_eq!(sim.similarity(&tv("IR7406812"), &tv("IR7406812")), 1.0);
    }

    #[test]
    fn exact_id_mismatch() {
        let sim = ExactIdSimilarity;
        assert_eq!(sim.similarity(&tv("IR7406812"), &tv("IR7406813")), 0.0);
    }

    #[test]
    fn exact_id_integer() {
        let sim = ExactIdSimilarity;
        assert_eq!(sim.similarity(&FieldValue::Int(12345), &FieldValue::Int(12345)), 1.0);
        assert_eq!(sim.similarity(&FieldValue::Int(12345), &FieldValue::Int(99999)), 0.0);
    }

    #[test]
    fn hamming_exact() {
        let sim = HammingSimilarity { max_distance: 1 };
        assert_eq!(sim.similarity(&tv("IR7406812"), &tv("IR7406812")), 1.0);
    }

    #[test]
    fn hamming_within_distance() {
        let sim = HammingSimilarity { max_distance: 1 };
        // One character different
        assert_eq!(sim.similarity(&tv("IR7406812"), &tv("IR7406813")), 0.8);
    }

    #[test]
    fn hamming_exceeds_distance() {
        let sim = HammingSimilarity { max_distance: 1 };
        // Two characters different
        assert_eq!(sim.similarity(&tv("IR7406812"), &tv("IR7406899")), 0.0);
    }

    #[test]
    fn hamming_different_lengths() {
        let sim = HammingSimilarity { max_distance: 1 };
        // Hamming undefined for unequal lengths → 0.0
        assert_eq!(sim.similarity(&tv("IR7406812"), &tv("IR740681")), 0.0);
    }

    #[test]
    fn suffix_match_exact_suffix() {
        let sim = SuffixMatchSimilarity { n: 4 };
        // Different last-4: "6789" vs "0001" → 0.0
        assert_eq!(sim.similarity(&tv("123456789"), &tv("987650001")), 0.0);
        // Different last-4: "6789" vs "1234" → 0.0
        assert_eq!(sim.similarity(&tv("123456789"), &tv("111111234")), 0.0);
        // Same last-4: "6789" == "6789" → 1.0
        assert_eq!(sim.similarity(&tv("123456789"), &tv("111116789")), 1.0);
    }

    #[test]
    fn suffix_match_too_short() {
        let sim = SuffixMatchSimilarity { n: 6 };
        assert_eq!(sim.similarity(&tv("123"), &tv("123")), 0.0);
    }
}
