use zer_core::{record::Record, schema::Schema};

pub mod alias;
pub mod date;
pub mod document;
pub mod exact;
pub mod phonetic;
pub mod phonetic_initial;
pub mod suffix;
pub mod token;
pub mod transliterated;
pub mod vehicle;

pub use alias::{AliasPhoneticKey, FuzzyYearKey};
pub use date::{DateFragmentKey, DateGranularity};
pub use document::{DocumentDigitSuffixKey, DocumentSuffixKey};
pub use exact::ExactFieldKey;
pub use phonetic::{PhoneticAlgo, PhoneticNameDobKey};
pub use phonetic_initial::PhoneticNameDobInitialKey;
pub use suffix::SuffixKey;
pub use token::AddressInitialKey;
pub use transliterated::TransliteratedPhoneticKey;
pub use vehicle::{CameraTimeWindowKey, GeoGridKey, LicensePlateNormKey, PlateOCRFuzzyKey};

/// Maps a record to zero or more opaque blocking key strings.
/// Records sharing a key become candidate pairs for comparison.
pub trait BlockingKey: Send + Sync {
    fn name(&self) -> &str;
    fn extract(&self, record: &Record, schema: &Schema) -> Vec<String>;
}
