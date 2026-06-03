use std::collections::HashSet;

use rphonetic::{DoubleMetaphone, Encoder};
use unicode_normalization::UnicodeNormalization;
use zer_core::{record::FieldValue, schema::FieldKind};

use crate::similarity::SimilarityFn;

/// Strip diacritics via NFKD decomposition and keep only ASCII characters,
/// then uppercase. Prevents rphonetic from panicking on multi-byte chars.
fn to_ascii_upper(s: &str) -> String {
    s.nfkd()
        .filter(|c| c.is_ascii())
        .collect::<String>()
        .to_ascii_uppercase()
}

// ── Jaro-Winkler ─────────────────────────────────────────────────────────────

pub struct JaroWinklerSimilarity;

impl SimilarityFn for JaroWinklerSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => strsim::jaro_winkler(a, b) as f32,
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        strsim::jaro_winkler(a, b) as f32
    }
    fn field_kind(&self) -> FieldKind {
        FieldKind::Name
    }
}

// ── Phonetic equality (Double Metaphone) ─────────────────────────────────────

pub struct PhoneticEqualitySimilarity;

impl SimilarityFn for PhoneticEqualitySimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => {
                let dm = DoubleMetaphone::default();
                let norm_a = to_ascii_upper(a);
                let norm_b = to_ascii_upper(b);
                if norm_a.is_empty() || norm_b.is_empty() {
                    return 0.0;
                }
                let code_a = dm.encode(&norm_a);
                let code_b = dm.encode(&norm_b);
                if code_a.is_empty() || code_b.is_empty() {
                    return 0.0;
                }
                if code_a == code_b {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        let dm = DoubleMetaphone::default();
        let norm_a = to_ascii_upper(a);
        let norm_b = to_ascii_upper(b);
        if norm_a.is_empty() || norm_b.is_empty() {
            return 0.0;
        }
        let code_a = dm.encode(&norm_a);
        let code_b = dm.encode(&norm_b);
        if code_a.is_empty() || code_b.is_empty() {
            return 0.0;
        }
        if code_a == code_b {
            1.0
        } else {
            0.0
        }
    }
    fn field_kind(&self) -> FieldKind {
        FieldKind::Name
    }
}

// ── Token overlap (Jaccard) ───────────────────────────────────────────────────

pub struct TokenOverlapSimilarity;

impl SimilarityFn for TokenOverlapSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => jaccard_tokens(a, b),
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        jaccard_tokens(a, b)
    }
    fn field_kind(&self) -> FieldKind {
        FieldKind::Name
    }
}

/// Jaccard coefficient of whitespace-separated token sets.
pub(crate) fn jaccard_tokens(a: &str, b: &str) -> f32 {
    let set_a: HashSet<&str> = a.split_whitespace().collect();
    let set_b: HashSet<&str> = b.split_whitespace().collect();
    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.len() + set_b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

// ── Levenshtein edit distance ─────────────────────────────────────────────────

/// Normalised Levenshtein similarity in [0.0, 1.0].
///
/// The raw edit distance is clipped at `max_distance`; distances above that
/// return 0.0.  Within the allowed range similarity is
/// `1.0 - dist / max_distance`.
///
/// ```
/// use zer_compare::similarity::name::LevenshteinSimilarity;
/// use zer_compare::similarity::SimilarityFn;
/// use zer_core::record::FieldValue;
///
/// let sim = LevenshteinSimilarity { max_distance: 3 };
/// let a = FieldValue::Text("Jansen".into());
/// let b = FieldValue::Text("Jansen".into());
/// assert_eq!(sim.similarity(&a, &b), 1.0); // exact match
/// ```
pub struct LevenshteinSimilarity {
    /// Maximum edit distance above which similarity is 0.0.
    pub max_distance: usize,
}

impl SimilarityFn for LevenshteinSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        let (sa, sb) = match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => (a.as_str(), b.as_str()),
            _ => return 0.0,
        };
        let dist = edit_distance::edit_distance(sa, sb);
        if dist > self.max_distance {
            0.0
        } else {
            1.0 - (dist as f32 / self.max_distance.max(1) as f32)
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        let dist = edit_distance::edit_distance(a, b);
        if dist > self.max_distance {
            0.0
        } else {
            1.0 - (dist as f32 / self.max_distance.max(1) as f32)
        }
    }
    fn field_kind(&self) -> zer_core::schema::FieldKind {
        zer_core::schema::FieldKind::Name
    }
}

// ── Alias token overlap (pipe-delimited multi-value field) ────────────────────

/// Compares pipe-delimited alias lists by taking the maximum Jaccard score
/// across all cross-product pairs of individual alias values.
pub struct AliasTokenOverlapSimilarity;

impl SimilarityFn for AliasTokenOverlapSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => {
                let aliases_a: Vec<&str> = a
                    .split('|')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                let aliases_b: Vec<&str> = b
                    .split('|')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                if aliases_a.is_empty() || aliases_b.is_empty() {
                    return 0.0;
                }
                aliases_a
                    .iter()
                    .flat_map(|aa| aliases_b.iter().map(move |ab| jaccard_tokens(aa, ab)))
                    .fold(0.0_f32, f32::max)
            }
            _ => 0.0,
        }
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        let aliases_a: Vec<&str> = a
            .split('|')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let aliases_b: Vec<&str> = b
            .split('|')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if aliases_a.is_empty() || aliases_b.is_empty() {
            return 0.0;
        }
        aliases_a
            .iter()
            .flat_map(|aa| aliases_b.iter().map(move |ab| jaccard_tokens(aa, ab)))
            .fold(0.0_f32, f32::max)
    }
    fn field_kind(&self) -> FieldKind {
        FieldKind::Alias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaro_winkler_similar_names() {
        let sim = JaroWinklerSimilarity;
        let a = FieldValue::Text("JOHN SMITH".into());
        let b = FieldValue::Text("JON SMYTH".into());
        let s = sim.similarity(&a, &b);
        assert!(s > 0.8, "similar names should score > 0.8, got {s}");
    }

    #[test]
    fn jaro_winkler_different_names() {
        let sim = JaroWinklerSimilarity;
        let a = FieldValue::Text("JOHN SMITH".into());
        let b = FieldValue::Text("JANE DOE".into());
        let s = sim.similarity(&a, &b);
        assert!(s < 0.6, "very different names should score < 0.6, got {s}");
    }

    #[test]
    fn phonetic_equality_sound_alikes() {
        let sim = PhoneticEqualitySimilarity;
        let a = FieldValue::Text("Smith".into());
        let b = FieldValue::Text("Smyth".into());
        assert_eq!(
            sim.similarity(&a, &b),
            1.0,
            "Smith and Smyth should be phonetically equal"
        );
    }

    #[test]
    fn phonetic_equality_different() {
        let sim = PhoneticEqualitySimilarity;
        let a = FieldValue::Text("Jansen".into());
        let b = FieldValue::Text("Bakker".into());
        assert_eq!(
            sim.similarity(&a, &b),
            0.0,
            "Jansen and Bakker should not be phonetically equal"
        );
    }

    #[test]
    fn token_overlap_swapped_name() {
        let sim = TokenOverlapSimilarity;
        let a = FieldValue::Text("John Smith".into());
        let b = FieldValue::Text("Smith John".into());
        assert_eq!(
            sim.similarity(&a, &b),
            1.0,
            "token overlap should be 1.0 for swapped tokens"
        );
    }

    #[test]
    fn token_overlap_partial() {
        let sim = TokenOverlapSimilarity;
        // "Alice van Berg" ∩ "Alice Berg" = {"Alice","Berg"}, union=3 → 0.67 > 0.3
        let a = FieldValue::Text("Alice van Berg".into());
        let b = FieldValue::Text("Alice Berg".into());
        let s = sim.similarity(&a, &b);
        assert!(
            s > 0.3,
            "partial name overlap should produce > 0.3, got {s}"
        );
    }

    #[test]
    fn alias_overlap_cross_product() {
        let sim = AliasTokenOverlapSimilarity;
        // "Benabdallah Fatima" matches in alias_b
        let a = FieldValue::Text("Benabdallah Fatima|F. Benabdallah".into());
        let b = FieldValue::Text("Fatima Benabdallah".into());
        let s = sim.similarity(&a, &b);
        assert!(s > 0.5, "alias cross-product should find overlap, got {s}");
    }

    #[test]
    fn alias_overlap_empty_field() {
        let sim = AliasTokenOverlapSimilarity;
        let a = FieldValue::Text("".into());
        let b = FieldValue::Text("Jansen".into());
        assert_eq!(sim.similarity(&a, &b), 0.0);
    }

    #[test]
    fn similarity_null_fields_return_zero() {
        let sim = JaroWinklerSimilarity;
        assert_eq!(
            sim.similarity(&FieldValue::Null, &FieldValue::Text("test".into())),
            0.0
        );
        assert_eq!(
            sim.similarity(&FieldValue::Text("test".into()), &FieldValue::Null),
            0.0
        );
    }

    #[test]
    fn levenshtein_exact_match() {
        let sim = LevenshteinSimilarity { max_distance: 3 };
        let a = FieldValue::Text("Jansen".into());
        let b = FieldValue::Text("Jansen".into());
        assert_eq!(
            sim.similarity(&a, &b),
            1.0,
            "edit distance 0 must yield 1.0"
        );
    }

    #[test]
    fn levenshtein_over_max() {
        let sim = LevenshteinSimilarity { max_distance: 2 };
        let a = FieldValue::Text("hello".into());
        let b = FieldValue::Text("world".into()); // edit distance 4 > max_distance 2
        assert_eq!(
            sim.similarity(&a, &b),
            0.0,
            "edit distance > max_distance must yield 0.0"
        );
    }

    #[test]
    fn levenshtein_partial() {
        let sim = LevenshteinSimilarity { max_distance: 3 };
        let a = FieldValue::Text("Jansen".into());
        let b = FieldValue::Text("Jansem".into()); // edit distance 1
        let s = sim.similarity(&a, &b);
        assert!(
            s > 0.0 && s < 1.0,
            "partial distance must yield value in (0.0, 1.0), got {s}"
        );
    }

    #[test]
    fn levenshtein_null_fields_return_zero() {
        let sim = LevenshteinSimilarity { max_distance: 3 };
        assert_eq!(
            sim.similarity(&FieldValue::Null, &FieldValue::Text("test".into())),
            0.0
        );
        assert_eq!(
            sim.similarity(&FieldValue::Text("test".into()), &FieldValue::Null),
            0.0
        );
    }
}
