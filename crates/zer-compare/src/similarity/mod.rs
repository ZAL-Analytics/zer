use zer_core::{record::FieldValue, schema::FieldKind};

pub mod address;
pub mod date;
pub mod id;
pub mod name;
pub mod numeric;

/// Returns a similarity in [0.0, 1.0].
/// 0.0 = completely different, 1.0 = identical.
pub trait SimilarityFn: Send + Sync {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32;
    fn field_kind(&self) -> FieldKind;

    /// Zero-alloc hot path for pool-native comparison.
    ///
    /// Called by `compare_pool_field` to avoid wrapping `&str` in `FieldValue::Text`
    /// on every comparison. Override in concrete types to eliminate the allocation.
    #[inline]
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        let va = FieldValue::Text(a.to_owned());
        let vb = FieldValue::Text(b.to_owned());
        self.similarity(&va, &vb)
    }
}

/// Returns 0.0 when either field is `FieldValue::Null`, and 1.0 otherwise.
pub struct NullSimilarity;

impl SimilarityFn for NullSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        match (a, b) {
            (FieldValue::Null, _) | (_, FieldValue::Null) => 0.0,
            _ => 1.0,
        }
    }
    // compare_pool_field already guards empty strings → ComparisonLevel::None;
    // any non-empty strings reaching here mean neither side is null.
    fn similarity_str(&self, _a: &str, _b: &str) -> f32 { 1.0 }
    fn field_kind(&self) -> FieldKind { FieldKind::Name }
}

#[cfg(test)]
mod null_tests {
    use super::*;

    #[test]
    fn null_similarity_either_null() {
        let sim = NullSimilarity;
        assert_eq!(sim.similarity(&FieldValue::Null, &FieldValue::Text("x".into())), 0.0);
        assert_eq!(sim.similarity(&FieldValue::Text("x".into()), &FieldValue::Null), 0.0);
        assert_eq!(sim.similarity(&FieldValue::Null, &FieldValue::Null), 0.0);
    }

    #[test]
    fn null_similarity_both_present_returns_one() {
        let sim = NullSimilarity;
        let a = FieldValue::Text("Alice".into());
        let b = FieldValue::Text("Bob".into());
        assert_eq!(sim.similarity(&a, &b), 1.0, "non-null values pass through as 1.0");
    }
}

/// Look up the default similarity function(s) for a `FieldKind`.
///
/// Multiple functions per field kind allow complementary signals, e.g.
/// Jaro-Winkler for gradual string proximity AND phonetic equality for
/// sound-alike variants. The `FieldComparator` takes the maximum across all
/// functions for a given field.
pub fn default_fns_for(kind: FieldKind) -> Vec<Box<dyn SimilarityFn>> {
    use address::AddressTokenOverlap;
    use date::DateSimilarity;
    use id::{ExactIdSimilarity, HammingSimilarity};
    use name::{AliasTokenOverlapSimilarity, JaroWinklerSimilarity, TokenOverlapSimilarity};
    use numeric::NumericBucketedSimilarity;

    match kind {
        FieldKind::Name => vec![
            Box::new(JaroWinklerSimilarity),
            Box::new(TokenOverlapSimilarity),
        ],
        FieldKind::Date | FieldKind::Timestamp => vec![
            Box::new(DateSimilarity),
        ],
        FieldKind::Address => vec![
            Box::new(AddressTokenOverlap),
        ],
        FieldKind::Id => vec![
            Box::new(ExactIdSimilarity),
            Box::new(HammingSimilarity { max_distance: 1 }),
        ],
        FieldKind::Phone => vec![
            Box::new(ExactIdSimilarity),
            Box::new(HammingSimilarity { max_distance: 1 }),
        ],
        FieldKind::LicensePlate => vec![
            Box::new(ExactIdSimilarity),
            Box::new(HammingSimilarity { max_distance: 1 }),
        ],
        FieldKind::Numeric | FieldKind::GpsCoordinate => vec![
            Box::new(NumericBucketedSimilarity),
        ],
        FieldKind::Categorical => vec![
            Box::new(ExactIdSimilarity),
        ],
        FieldKind::FreeText => vec![
            Box::new(TokenOverlapSimilarity),
        ],
        FieldKind::Alias => vec![
            Box::new(AliasTokenOverlapSimilarity),
        ],
    }
}
