//! Accuracy strategy for `brp/dedupe`, `micro_brp/dedupe`, and `kvk/dedupe`.
//!
//! Restores the phonetic (double-metaphone) similarity function for Name fields
//! and the street-number edit-distance function for Address fields, which were
//! removed from the defaults to match Splink's throughput benchmark field set.
//! These functions improve accuracy on full-name BRP/KVK records where
//! sound-alike variants and transposed street numbers are common error patterns.

use zer::prelude::{
    AddressTokenOverlap, FieldComparator, JaroWinklerSimilarity, PhoneticEqualitySimilarity,
    SimilarityFn, StreetNumberEditDistance, TokenOverlapSimilarity,
};
use zer_core::schema::{FieldKind, Schema};

use super::ScenarioStrategy;

pub fn strategy() -> ScenarioStrategy {
    ScenarioStrategy {
        comparator_fn: Some(build_comparator),
        ..ScenarioStrategy::default()
    }
}

fn build_comparator(schema: &Schema) -> FieldComparator {
    let mut cmp = FieldComparator::from_schema(schema);
    for (i, field) in schema.fields.iter().enumerate() {
        match field.kind {
            FieldKind::Name => {
                cmp = cmp.with_fns(
                    i,
                    vec![
                        Box::new(JaroWinklerSimilarity) as Box<dyn SimilarityFn>,
                        Box::new(PhoneticEqualitySimilarity) as Box<dyn SimilarityFn>,
                        Box::new(TokenOverlapSimilarity) as Box<dyn SimilarityFn>,
                    ],
                );
            }
            FieldKind::Address => {
                cmp = cmp.with_fns(
                    i,
                    vec![
                        Box::new(AddressTokenOverlap) as Box<dyn SimilarityFn>,
                        Box::new(StreetNumberEditDistance) as Box<dyn SimilarityFn>,
                    ],
                );
            }
            _ => {}
        }
    }
    cmp
}
