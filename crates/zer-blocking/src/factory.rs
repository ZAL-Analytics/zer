use zer_core::schema::{FieldKind, Schema};

use crate::{
    blocker::CompositeBlocker,
    keys::{
        AddressInitialKey, AliasPhoneticKey, BlockingKey, CameraTimeWindowKey, DateFragmentKey,
        DateGranularity, DocumentSuffixKey, ExactFieldKey, FuzzyYearKey, GeoGridKey,
        LicensePlateNormKey, PhoneticNameDobInitialKey, PhoneticNameDobKey, PlateOCRFuzzyKey,
        SuffixKey, TransliteratedPhoneticKey,
    },
};

/// High-level domain category for a dataset.
///
/// Pass to `BlockerFactory::from_schema_category` to get a `CompositeBlocker`
/// whose keys are pre-tuned for that category, rather than relying solely on
/// generic `FieldKind` heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaCategory {
    /// General person registry (KvK directors, population registers).
    PersonRegistry,
    /// SIS II wanted/missing persons, aliases, name transpositions, estimated DOBs.
    WantedPersons,
    /// ANPR vehicle passage logs, license plate OCR, camera/time windows, GPS grid.
    ANPRPassages,
    /// Call detail records, phone number suffix, categorical (cell tower / IMEI).
    CallDetailRecords,
    /// SIM subscriber snapshots, phone suffix, IMSI/ICCID suffix, categorical.
    SIMSubscribers,
    /// FIU financial intelligence, account/transaction ID suffix, date fragments.
    FinancialIntelligence,
}

/// A single blocking rule used inside [`CustomSchemaCategory`].
///
/// Each variant describes how to derive one set of blocking keys from a
/// schema's `FieldKind` annotations. The `Custom` variant is an escape
/// hatch for any key the built-in rules don't cover.
enum CategoryRule {
    /// `SuffixKey(n)` on every `FieldKind::Phone` field.
    PhoneSuffix(usize),
    /// `SuffixKey(n)` on every `FieldKind::Id` field.
    IdSuffix(usize),
    /// `DocumentSuffixKey(n)` on every `FieldKind::Id` field (strips non-alphanumeric, uppercases).
    DocumentSuffix(usize),
    /// `ExactFieldKey` on every `FieldKind::Categorical` field.
    ExactCategorical,
    /// `DateFragmentKey` with the given granularity on the first `FieldKind::Date` field.
    DateFragment(DateGranularity),
    /// `PhoneticNameDobKey` using the last `Name` field as surname and first `Date` field as DOB.
    PhoneticNameDob,
    /// `PhoneticNameDobInitialKey`: surname phonetic + first-name initial + DOB year.
    /// Requires at least two `Name` fields (first = given name, last = surname).
    PhoneticNameDobInitial,
    /// `AddressInitialKey` using the first `Address` field and first `Name` field as initial.
    AddressInitial,
    /// A fully custom key, the escape hatch for anything the built-in rules don't cover.
    Custom(Box<dyn BlockingKey>),
}

/// A user-defined blocking category assembled from individual rules.
///
/// Build one with the fluent `with_*` methods and pass it to
/// [`BlockerFactory::from_custom_category`].
///
/// # Example
/// ```
/// use zer_blocking::{BlockerFactory, CustomSchemaCategory};
/// use zer_blocking::keys::DateGranularity;
///
/// let category = CustomSchemaCategory::new()
///     .with_phonetic_name_dob()
///     .with_id_suffix(4)
///     .with_exact_categorical();
///
/// // let blocker = BlockerFactory::from_custom_category(&schema, category);
/// ```
pub struct CustomSchemaCategory {
    rules: Vec<CategoryRule>,
}

impl CustomSchemaCategory {
    pub fn new() -> Self {
        Self { rules: vec![] }
    }

    /// Add a `SuffixKey(n)` on every `Phone` field.
    pub fn with_phone_suffix(mut self, n: usize) -> Self {
        self.rules.push(CategoryRule::PhoneSuffix(n));
        self
    }

    /// Add a `SuffixKey(n)` on every `Id` field (digits-only suffix).
    pub fn with_id_suffix(mut self, n: usize) -> Self {
        self.rules.push(CategoryRule::IdSuffix(n));
        self
    }

    /// Add a `DocumentSuffixKey(n)` on every `Id` field (alphanumeric suffix, uppercased).
    pub fn with_document_suffix(mut self, n: usize) -> Self {
        self.rules.push(CategoryRule::DocumentSuffix(n));
        self
    }

    /// Add an `ExactFieldKey` on every `Categorical` field.
    pub fn with_exact_categorical(mut self) -> Self {
        self.rules.push(CategoryRule::ExactCategorical);
        self
    }

    /// Add a `DateFragmentKey` with the given granularity on the first `Date` field.
    pub fn with_date_fragment(mut self, granularity: DateGranularity) -> Self {
        self.rules.push(CategoryRule::DateFragment(granularity));
        self
    }

    /// Add a `PhoneticNameDobKey` using the last `Name` field and the first `Date` field.
    pub fn with_phonetic_name_dob(mut self) -> Self {
        self.rules.push(CategoryRule::PhoneticNameDob);
        self
    }

    /// Add a `PhoneticNameDobInitialKey` (surname phonetic + first-name initial + DOB year).
    /// Falls back to `PhoneticNameDobKey` when only one `Name` field is present.
    pub fn with_phonetic_name_dob_initial(mut self) -> Self {
        self.rules.push(CategoryRule::PhoneticNameDobInitial);
        self
    }

    /// Add an `AddressInitialKey` using the first `Address` field and first `Name` field as initial.
    pub fn with_address_initial(mut self) -> Self {
        self.rules.push(CategoryRule::AddressInitial);
        self
    }

    /// Add an arbitrary blocking key, escape hatch for keys the built-in rules don't cover.
    pub fn with_key(mut self, key: impl BlockingKey + 'static) -> Self {
        self.rules.push(CategoryRule::Custom(Box::new(key)));
        self
    }
}

impl Default for CustomSchemaCategory {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BlockerFactory;

impl BlockerFactory {
    /// Build a `CompositeBlocker` whose blocking keys are chosen automatically
    /// from the `Schema`'s `FieldKind` annotations.
    ///
    /// Priority rules (applied in order):
    /// - 2+ Name fields + Date: uses `PhoneticNameDobInitialKey` (surname phonetic + first-name initial + DOB year)
    /// - 1 Name field + Date: uses `PhoneticNameDobKey` (surname phonetic + DOB year)
    /// - Name + Address: uses `AddressInitialKey` (first address token + first name initial)
    /// - Phone: adds `SuffixKey(7)` on the first Phone field
    /// - Id: adds `SuffixKey(4)` on each Id field
    /// - Date only (no Name): adds `DateFragmentKey(YearMonth)` on the first Date field
    /// - Categorical: adds `ExactFieldKey` on each Categorical field
    pub fn from_schema(schema: &Schema) -> CompositeBlocker {
        let mut blocker = CompositeBlocker::new();

        let name_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Name).collect();
        let date_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Date).collect();
        let addr_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Address).collect();
        let phone_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Phone).collect();
        let id_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Id).collect();
        let cat_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Categorical).collect();

        if let (Some(&surname), Some(&dob)) = (name_fields.last(), date_fields.first()) {
            if name_fields.len() >= 2 {
                let first_name = name_fields[0];
                blocker = blocker.add(PhoneticNameDobInitialKey::new(surname, first_name, dob));
            } else {
                blocker = blocker.add(PhoneticNameDobKey::new(surname, dob));
            }
            // Secondary key: catches pairs whose surnames differ phonetically but share birth year-month.
            blocker = blocker.add(DateFragmentKey::new(dob, DateGranularity::YearMonth));
        }

        if let (Some(&first_name), Some(&addr)) = (name_fields.first(), addr_fields.first()) {
            blocker = blocker.add(AddressInitialKey::new(addr, first_name));
        }

        if let Some(&phone) = phone_fields.first() {
            blocker = blocker.add(SuffixKey::new(phone, 7));
        }

        for &id in &id_fields {
            blocker = blocker.add(SuffixKey::new(id, 4));
        }

        if name_fields.is_empty() {
            if let Some(&dob) = date_fields.first() {
                blocker = blocker.add(DateFragmentKey::new(dob, DateGranularity::YearMonth));
            }
        }

        for &cat in &cat_fields {
            blocker = blocker.add(ExactFieldKey::new(cat));
        }

        blocker
    }

    /// Build a `CompositeBlocker` from a user-defined [`CustomSchemaCategory`].
    ///
    /// Each rule in the category is applied in order against `schema`'s
    /// `FieldKind` annotations. Rules that reference field kinds not present
    /// in the schema are silently skipped, no panic, no empty blocker.
    pub fn from_custom_category(schema: &Schema, cat: CustomSchemaCategory) -> CompositeBlocker {
        let mut blocker = CompositeBlocker::new();

        for rule in cat.rules {
            match rule {
                CategoryRule::PhoneSuffix(n) => {
                    for field in schema.fields_of_kind(FieldKind::Phone) {
                        blocker = blocker.add(SuffixKey::new(field, n));
                    }
                }
                CategoryRule::IdSuffix(n) => {
                    for field in schema.fields_of_kind(FieldKind::Id) {
                        blocker = blocker.add(SuffixKey::new(field, n));
                    }
                }
                CategoryRule::DocumentSuffix(n) => {
                    for field in schema.fields_of_kind(FieldKind::Id) {
                        blocker = blocker.add(DocumentSuffixKey::new(field, n));
                    }
                }
                CategoryRule::ExactCategorical => {
                    for field in schema.fields_of_kind(FieldKind::Categorical) {
                        blocker = blocker.add(ExactFieldKey::new(field));
                    }
                }
                CategoryRule::DateFragment(granularity) => {
                    if let Some(field) = schema.fields_of_kind(FieldKind::Date).next() {
                        blocker = blocker.add(DateFragmentKey::new(field, granularity));
                    }
                }
                CategoryRule::PhoneticNameDob => {
                    let names: Vec<&str> = schema.fields_of_kind(FieldKind::Name).collect();
                    let dates: Vec<&str> = schema.fields_of_kind(FieldKind::Date).collect();
                    if let (Some(&surname), Some(&dob)) = (names.last(), dates.first()) {
                        blocker = blocker.add(PhoneticNameDobKey::new(surname, dob));
                    }
                }
                CategoryRule::PhoneticNameDobInitial => {
                    let names: Vec<&str> = schema.fields_of_kind(FieldKind::Name).collect();
                    let dates: Vec<&str> = schema.fields_of_kind(FieldKind::Date).collect();
                    if let Some(&dob) = dates.first() {
                        if names.len() >= 2 {
                            let first_name = names[0];
                            let surname = names[names.len() - 1];
                            blocker = blocker
                                .add(PhoneticNameDobInitialKey::new(surname, first_name, dob));
                        } else if let Some(&surname) = names.last() {
                            blocker = blocker.add(PhoneticNameDobKey::new(surname, dob));
                        }
                    }
                }
                CategoryRule::AddressInitial => {
                    let names: Vec<&str> = schema.fields_of_kind(FieldKind::Name).collect();
                    let addrs: Vec<&str> = schema.fields_of_kind(FieldKind::Address).collect();
                    if let (Some(&first_name), Some(&addr)) = (names.first(), addrs.first()) {
                        blocker = blocker.add(AddressInitialKey::new(addr, first_name));
                    }
                }
                CategoryRule::Custom(key) => {
                    blocker = blocker.add_boxed(key);
                }
            }
        }

        blocker
    }

    fn telecom_blocker(schema: &Schema) -> CompositeBlocker {
        let mut blocker = CompositeBlocker::new();
        for f in schema.fields_of_kind(FieldKind::Phone) {
            blocker = blocker.add(SuffixKey::new(f, 7));
        }
        for f in schema.fields_of_kind(FieldKind::Id) {
            blocker = blocker.add(SuffixKey::new(f, 6));
        }
        for f in schema.fields_of_kind(FieldKind::Categorical) {
            blocker = blocker.add(ExactFieldKey::new(f));
        }
        blocker
    }

    /// Build a `CompositeBlocker` tuned for a specific domain category.
    ///
    /// Keys are chosen based on both the category semantics and the
    /// `FieldKind` annotations present in `schema`.
    pub fn from_schema_category(schema: &Schema, category: SchemaCategory) -> CompositeBlocker {
        match category {
            SchemaCategory::PersonRegistry => Self::from_schema(schema),

            SchemaCategory::WantedPersons => {
                let mut blocker = CompositeBlocker::new();

                let name_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Name).collect();
                let date_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Date).collect();
                let alias_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Alias).collect();
                let id_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Id).collect();

                if let (Some(&surname), Some(&dob)) = (name_fields.last(), date_fields.first()) {
                    blocker = blocker.add(PhoneticNameDobKey::new(surname, dob));
                    blocker = blocker.add(TransliteratedPhoneticKey::new(surname, dob));
                    blocker = blocker.add(FuzzyYearKey::new(surname, dob, 1));
                }

                if let Some(&dob) = date_fields.first() {
                    for &alias in &alias_fields {
                        blocker = blocker.add(AliasPhoneticKey::new(alias, dob));
                    }
                }

                for &id in &id_fields {
                    blocker = blocker.add(DocumentSuffixKey::new(id, 6));
                }

                blocker
            }

            SchemaCategory::ANPRPassages => {
                let mut blocker = CompositeBlocker::new();

                let plate_fields: Vec<&str> =
                    schema.fields_of_kind(FieldKind::LicensePlate).collect();
                let ts_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Timestamp).collect();
                let cat_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Categorical).collect();
                let lat_fields: Vec<&str> =
                    schema.fields_of_kind(FieldKind::GpsCoordinate).collect();

                for &plate in &plate_fields {
                    blocker = blocker.add(LicensePlateNormKey::new(plate));
                    blocker = blocker.add(PlateOCRFuzzyKey::new(plate));
                }

                // camera_id + timestamp → 10-minute window key
                if let (Some(&cam), Some(&ts)) = (cat_fields.first(), ts_fields.first()) {
                    blocker = blocker.add(CameraTimeWindowKey::new(cam, ts, 10));
                }

                // lat + lon → 0.01° grid (~1 km)
                if lat_fields.len() >= 2 {
                    blocker = blocker.add(GeoGridKey::new(lat_fields[0], lat_fields[1], 0.01));
                }

                blocker
            }

            SchemaCategory::CallDetailRecords | SchemaCategory::SIMSubscribers => {
                Self::telecom_blocker(schema)
            }

            SchemaCategory::FinancialIntelligence => {
                let mut blocker = CompositeBlocker::new();

                let id_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Id).collect();
                let date_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Date).collect();
                let cat_fields: Vec<&str> = schema.fields_of_kind(FieldKind::Categorical).collect();

                for &id in &id_fields {
                    blocker = blocker.add(SuffixKey::new(id, 6));
                }

                if let Some(&dob) = date_fields.first() {
                    blocker = blocker.add(DateFragmentKey::new(dob, DateGranularity::YearMonth));
                }

                for &cat in &cat_fields {
                    blocker = blocker.add(ExactFieldKey::new(cat));
                }

                blocker
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::InvertedIndex;
    use zer_core::{
        record::FieldValue,
        schema::{FieldKind, SchemaBuilder},
        traits::Blocker,
    };

    fn person_schema() -> Schema {
        SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .field("woonplaats", FieldKind::Address)
            .field("postcode", FieldKind::Id)
            .build()
            .unwrap()
    }

    #[test]
    fn factory_name_date_schema_adds_secondary_year_month_key() {
        // When Name + Date fields are present, from_schema() must add both
        // PhoneticNameDobKey AND DateFragmentKey(YearMonth).  Two records with
        // identical DOB but different surnames must still be candidates via the
        // secondary key even if their phonetic codes diverge.
        let schema = SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap();
        let blocker = BlockerFactory::from_schema(&schema);

        let mut idx = InvertedIndex::new();
        let r1 = zer_core::record::Record::new(1)
            .insert("achternaam", FieldValue::Text("Jansen".into()))
            .insert("geboortedatum", FieldValue::Text("1985-06-15".into()));
        // Completely different surname but same birth year-month.
        let r2 = zer_core::record::Record::new(2)
            .insert("achternaam", FieldValue::Text("Pietersen".into()))
            .insert("geboortedatum", FieldValue::Text("1985-06-22".into()));

        blocker.index_record(&r1, &schema, &mut idx);
        blocker.index_record(&r2, &schema, &mut idx);

        let cands = blocker.candidates(&r1, &schema, &idx);
        assert!(
            cands.contains(&2),
            "secondary YearMonth key must surface r2 (same birth year-month, different surname)"
        );
    }

    #[test]
    fn factory_produces_non_empty_blocker() {
        let schema = person_schema();
        let blocker = BlockerFactory::from_schema(&schema);
        let record = zer_core::record::Record::new(1)
            .insert("achternaam", FieldValue::Text("Jansen".into()))
            .insert("geboortedatum", FieldValue::Text("1980-01-15".into()));

        let keys = blocker.blocking_keys(&record, &schema);
        assert!(
            !keys.is_empty(),
            "BlockerFactory should produce at least one key"
        );
    }

    #[test]
    fn factory_date_only_schema_uses_date_fragment() {
        let schema = SchemaBuilder::new()
            .field("dob", FieldKind::Date)
            .build()
            .unwrap();
        let blocker = BlockerFactory::from_schema(&schema);
        let r =
            zer_core::record::Record::new(1).insert("dob", FieldValue::Text("1990-06-01".into()));

        let mut idx = InvertedIndex::new();
        blocker.index_record(&r, &schema, &mut idx);
        assert!(!idx.is_empty());
    }

    #[test]
    fn category_wanted_persons_produces_keys() {
        let schema = SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("alias_namen", FieldKind::Alias)
            .field("geboortedatum", FieldKind::Date)
            .field("document_nummer", FieldKind::Id)
            .build()
            .unwrap();
        let blocker = BlockerFactory::from_schema_category(&schema, SchemaCategory::WantedPersons);
        let r = zer_core::record::Record::new(1)
            .insert("achternaam", FieldValue::Text("Benabdallah".into()))
            .insert("geboortedatum", FieldValue::Text("1999-06-14".into()));

        let keys = blocker.blocking_keys(&r, &schema);
        assert!(!keys.is_empty());
    }

    #[test]
    fn category_anpr_produces_plate_keys() {
        let schema = SchemaBuilder::new()
            .field("kenteken", FieldKind::LicensePlate)
            .field("camera_id", FieldKind::Categorical)
            .field("tijdstip", FieldKind::Timestamp)
            .build()
            .unwrap();
        let blocker = BlockerFactory::from_schema_category(&schema, SchemaCategory::ANPRPassages);
        let r = zer_core::record::Record::new(1)
            .insert("kenteken", FieldValue::Text("25-XKL-9".into()))
            .insert("camera_id", FieldValue::Text("CAM-A12-001".into()))
            .insert("tijdstip", FieldValue::Text("2025-06-01T10:00:00".into()));

        let keys = blocker.blocking_keys(&r, &schema);
        assert!(!keys.is_empty());
        assert!(keys.iter().any(|k| k.contains("25XKL9")));
    }

    // ── CustomSchemaCategory tests ────────────────────────────────────────────

    #[test]
    fn custom_phone_suffix_extracts_key() {
        let schema = SchemaBuilder::new()
            .field("telefoon", FieldKind::Phone)
            .build()
            .unwrap();
        let cat = CustomSchemaCategory::new().with_phone_suffix(7);
        let blocker = BlockerFactory::from_custom_category(&schema, cat);
        let r = zer_core::record::Record::new(1)
            .insert("telefoon", FieldValue::Text("0612345678".into()));

        let keys = blocker.blocking_keys(&r, &schema);
        assert!(!keys.is_empty(), "phone suffix rule must produce a key");
        assert!(
            keys.iter().any(|k| k.ends_with("2345678")),
            "key must end with last 7 digits"
        );
    }

    #[test]
    fn custom_id_suffix_correct_length() {
        let schema = SchemaBuilder::new()
            .field("postcode", FieldKind::Id)
            .build()
            .unwrap();
        let cat = CustomSchemaCategory::new().with_id_suffix(4);
        let blocker = BlockerFactory::from_custom_category(&schema, cat);
        let r =
            zer_core::record::Record::new(1).insert("postcode", FieldValue::Text("1011AB".into()));

        let keys = blocker.blocking_keys(&r, &schema);
        // postcode "1011AB" → digits only = "1011" → last 4 = "1011"
        assert!(
            keys.iter().any(|k| k.ends_with("1011")),
            "id suffix must be 4 digits: {keys:?}"
        );
    }

    #[test]
    fn custom_exact_categorical_matches_only_same_value() {
        let schema = SchemaBuilder::new()
            .field("tussenvoegsel", FieldKind::Categorical)
            .build()
            .unwrap();
        let cat = CustomSchemaCategory::new().with_exact_categorical();
        let blocker = BlockerFactory::from_custom_category(&schema, cat);

        let mut idx = InvertedIndex::new();
        let r1 = zer_core::record::Record::new(1)
            .insert("tussenvoegsel", FieldValue::Text("van".into()));
        let r2 = zer_core::record::Record::new(2)
            .insert("tussenvoegsel", FieldValue::Text("van".into()));
        let r3 =
            zer_core::record::Record::new(3).insert("tussenvoegsel", FieldValue::Text("de".into()));

        blocker.index_record(&r1, &schema, &mut idx);
        blocker.index_record(&r2, &schema, &mut idx);
        blocker.index_record(&r3, &schema, &mut idx);

        let cands = blocker.candidates(&r1, &schema, &idx);
        assert!(
            cands.contains(&2),
            "r2 (same tussenvoegsel) must be a candidate"
        );
        assert!(
            !cands.contains(&3),
            "r3 (different tussenvoegsel) must NOT be a candidate"
        );
    }

    #[test]
    fn custom_date_fragment_produces_year_month_key() {
        let schema = SchemaBuilder::new()
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap();
        let cat = CustomSchemaCategory::new().with_date_fragment(DateGranularity::YearMonth);
        let blocker = BlockerFactory::from_custom_category(&schema, cat);
        let r = zer_core::record::Record::new(1)
            .insert("geboortedatum", FieldValue::Text("1990-06-15".into()));

        let keys = blocker.blocking_keys(&r, &schema);
        assert!(
            keys.iter().any(|k| k.contains("1990-06")),
            "key must contain YYYY-MM: {keys:?}"
        );
    }

    #[test]
    fn custom_phonetic_name_dob_links_same_person() {
        let schema = SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap();
        let cat = CustomSchemaCategory::new().with_phonetic_name_dob();
        let blocker = BlockerFactory::from_custom_category(&schema, cat);

        let mut idx = InvertedIndex::new();
        let r1 = zer_core::record::Record::new(1)
            .insert("achternaam", FieldValue::Text("Jansen".into()))
            .insert("geboortedatum", FieldValue::Text("1978-03-15".into()));
        let r2 = zer_core::record::Record::new(2)
            .insert("achternaam", FieldValue::Text("Jansen".into()))
            .insert("geboortedatum", FieldValue::Text("1978-03-15".into()));
        let r3 = zer_core::record::Record::new(3)
            .insert("achternaam", FieldValue::Text("de Wit".into()))
            .insert("geboortedatum", FieldValue::Text("1990-07-22".into()));

        blocker.index_record(&r1, &schema, &mut idx);
        blocker.index_record(&r2, &schema, &mut idx);
        blocker.index_record(&r3, &schema, &mut idx);

        let cands = blocker.candidates(&r1, &schema, &idx);
        assert!(cands.contains(&2), "same surname+DOB must be a candidate");
        assert!(
            !cands.contains(&3),
            "different surname+DOB must NOT be a candidate"
        );
    }

    #[test]
    fn custom_missing_field_kind_produces_no_panic() {
        // Schema has no Phone fields; with_phone_suffix should silently produce nothing.
        let schema = SchemaBuilder::new()
            .field("achternaam", FieldKind::Name)
            .build()
            .unwrap();
        let cat = CustomSchemaCategory::new().with_phone_suffix(7);
        let blocker = BlockerFactory::from_custom_category(&schema, cat);
        let r = zer_core::record::Record::new(1)
            .insert("achternaam", FieldValue::Text("Jansen".into()));

        let keys = blocker.blocking_keys(&r, &schema);
        assert!(keys.is_empty(), "no Phone fields → no keys, no panic");
    }

    #[test]
    fn custom_escape_hatch_with_key_works() {
        let schema = SchemaBuilder::new()
            .field("postcode", FieldKind::Id)
            .build()
            .unwrap();
        // Provide a SuffixKey(4) via the escape hatch instead of with_id_suffix.
        let cat = CustomSchemaCategory::new().with_key(SuffixKey::new("postcode", 4));
        let blocker = BlockerFactory::from_custom_category(&schema, cat);
        let r =
            zer_core::record::Record::new(1).insert("postcode", FieldValue::Text("1011AB".into()));

        let keys = blocker.blocking_keys(&r, &schema);
        assert!(
            !keys.is_empty(),
            "escape-hatch key must produce at least one key"
        );
    }

    #[test]
    fn custom_combined_rules_produce_multiple_key_types() {
        let schema = SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .field("postcode", FieldKind::Id)
            .field("tussenvoegsel", FieldKind::Categorical)
            .build()
            .unwrap();
        let cat = CustomSchemaCategory::new()
            .with_phonetic_name_dob()
            .with_id_suffix(4)
            .with_exact_categorical();
        let blocker = BlockerFactory::from_custom_category(&schema, cat);
        let r = zer_core::record::Record::new(1)
            .insert("achternaam", FieldValue::Text("van den Berg".into()))
            .insert("geboortedatum", FieldValue::Text("1978-03-15".into()))
            .insert("postcode", FieldValue::Text("1011AB".into()))
            .insert("tussenvoegsel", FieldValue::Text("van den".into()));

        let keys = blocker.blocking_keys(&r, &schema);
        // Expect at least a phonetic key and a suffix key and a categorical key.
        assert!(
            keys.len() >= 3,
            "combined rules must produce at least 3 keys: {keys:?}"
        );
    }
}
