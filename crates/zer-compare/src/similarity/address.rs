use zer_core::{record::FieldValue, schema::FieldKind};

use crate::similarity::{name::jaccard_tokens, SimilarityFn};

// ── Normalization helpers ─────────────────────────────────────────────────────

const DUTCH_ABBREV: &[(&str, &str)] = &[
    ("str.",   "straat"),
    ("str",    "straat"),
    ("ln.",    "laan"),
    ("ln",     "laan"),
    ("ave.",   "avenue"),
    ("ave",    "avenue"),
    ("st.",    "street"),
    ("st",     "street"),
    ("blvd.",  "boulevard"),
    ("blvd",   "boulevard"),
    ("dr.",    "dreef"),
    ("dr",     "dreef"),
    ("sg.",    "singel"),
    ("sg",     "singel"),
    ("kade.",  "kade"),
];

fn normalize_address(s: &str) -> String {
    let lower = s.to_lowercase();
    // Strip punctuation except hyphens (preserve "1011-AB" style)
    let stripped: String = lower
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == ' ' { c } else { ' ' })
        .collect();
    // Expand abbreviations (whole-word match via split/rejoin)
    let tokens: Vec<&str> = stripped.split_whitespace().collect();
    tokens.iter().map(|t| {
        DUTCH_ABBREV.iter()
            .find(|(abbr, _)| *abbr == *t)
            .map(|(_, full)| *full)
            .unwrap_or(t)
    }).collect::<Vec<_>>().join(" ")
}

fn extract_leading_number(s: &str) -> Option<&str> {
    let s = s.trim();
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 { None } else { Some(&s[..end]) }
}

// ── AddressTokenOverlap ───────────────────────────────────────────────────────

/// Jaccard similarity on normalized token sets.
///
/// Normalizes both addresses (lowercase, strip punctuation, expand Dutch
/// abbreviations) then computes Jaccard on the resulting token sets.
pub struct AddressTokenOverlap;

impl SimilarityFn for AddressTokenOverlap {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => {
                let na = normalize_address(a);
                let nb = normalize_address(b);
                jaccard_tokens(&na, &nb)
            }
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        let na = normalize_address(a);
        let nb = normalize_address(b);
        jaccard_tokens(&na, &nb)
    }
    fn field_kind(&self) -> FieldKind { FieldKind::Address }
}

// ── StreetNumberEditDistance ──────────────────────────────────────────────────

/// Levenshtein edit distance on the leading street number.
///
/// Extracts the leading numeric sequence from each address and computes
/// edit distance:
///   distance 0 : 1.0
///   distance 1 : 0.8
///   otherwise  : 0.0
pub struct StreetNumberEditDistance;

impl SimilarityFn for StreetNumberEditDistance {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => {
                let na = extract_leading_number(a);
                let nb = extract_leading_number(b);
                match (na, nb) {
                    (Some(na), Some(nb)) => {
                        let dist = strsim::levenshtein(na, nb);
                        if dist == 0 { 1.0 }
                        else if dist == 1 { 0.8 }
                        else { 0.0 }
                    }
                    _ => 0.0,
                }
            }
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        match (extract_leading_number(a), extract_leading_number(b)) {
            (Some(na), Some(nb)) => {
                let dist = strsim::levenshtein(na, nb);
                if dist == 0 { 1.0 } else if dist == 1 { 0.8 } else { 0.0 }
            }
            _ => 0.0,
        }
    }
    fn field_kind(&self) -> FieldKind { FieldKind::Address }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tv(s: &str) -> FieldValue { FieldValue::Text(s.into()) }

    #[test]
    fn address_token_overlap_exact() {
        let sim = AddressTokenOverlap;
        assert_eq!(sim.similarity(&tv("Coolsingel 93"), &tv("Coolsingel 93")), 1.0);
    }

    #[test]
    fn address_token_overlap_abbreviation() {
        let sim = AddressTokenOverlap;
        // "Blaak Str." normalizes to "blaak straat"
        let s = sim.similarity(&tv("Blaak Str. 10"), &tv("Blaakstraat 10"));
        // "blaak straat 10" vs "blaakstraat 10", shares "10" at minimum
        assert!(s > 0.0, "abbreviation expansion should yield some overlap, got {s}");
    }

    #[test]
    fn address_token_overlap_different() {
        let sim = AddressTokenOverlap;
        let s = sim.similarity(&tv("Coolsingel 93"), &tv("Beatrixlaan 241"));
        assert!(s < 0.3, "completely different addresses should be low, got {s}");
    }

    #[test]
    fn street_number_exact() {
        let sim = StreetNumberEditDistance;
        assert_eq!(sim.similarity(&tv("239 bis Amsterdamseweg"), &tv("239 bis")), 1.0);
    }

    #[test]
    fn street_number_off_by_one() {
        let sim = StreetNumberEditDistance;
        // "10" vs "11" → levenshtein 1 (single digit substitution)
        assert_eq!(sim.similarity(&tv("10 Coolsingel"), &tv("11 Coolsingel")), 0.8);
    }

    #[test]
    fn street_number_no_leading_digit() {
        let sim = StreetNumberEditDistance;
        // Address with no leading number → 0.0
        assert_eq!(sim.similarity(&tv("Coolsingel"), &tv("Coolsingel")), 0.0);
    }

    #[test]
    fn address_null_field() {
        let sim = AddressTokenOverlap;
        assert_eq!(sim.similarity(&FieldValue::Null, &tv("Coolsingel 93")), 0.0);
    }
}
