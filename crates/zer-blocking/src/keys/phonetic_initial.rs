use rphonetic::{DoubleMetaphone, Encoder, Soundex};
use zer_core::{record::Record, schema::Schema};

use crate::normalize::{extract_surname_token, normalize_text};
use super::{BlockingKey, phonetic::PhoneticAlgo};

/// Composite blocking key: `"{phonetic(surname)}:{initial(firstname)}:{year}"`.
///
/// Compared to `PhoneticNameDobKey`, splitting by first-name initial reduces
/// bucket size ~20 times  for high-frequency surnames (e.g. Dutch "De Jong", "Jansen"),
/// dramatically cutting false candidate pairs while preserving recall for true
/// matches that share surname phonetic code, first-name initial, and birth year.
pub struct PhoneticNameDobInitialKey {
    algo:          PhoneticAlgo,
    name_field:    String,
    initial_field: String,
    dob_field:     String,
}

impl PhoneticNameDobInitialKey {
    pub fn new(name_field: &str, initial_field: &str, dob_field: &str) -> Self {
        Self {
            algo:          PhoneticAlgo::DoubleMetaphone,
            name_field:    name_field.into(),
            initial_field: initial_field.into(),
            dob_field:     dob_field.into(),
        }
    }

    pub fn with_algo(mut self, algo: PhoneticAlgo) -> Self {
        self.algo = algo;
        self
    }

    fn encode(&self, s: &str) -> String {
        match self.algo {
            PhoneticAlgo::DoubleMetaphone => DoubleMetaphone::default().encode(s),
            PhoneticAlgo::Soundex        => Soundex::default().encode(s),
        }
    }
}

impl BlockingKey for PhoneticNameDobInitialKey {
    fn name(&self) -> &str {
        "phonetic_initial_dob"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let surname_cow = record.field_as_str(&self.name_field);
        let surname_raw = match surname_cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };
        let initial_cow = record.field_as_str(&self.initial_field);
        let initial_raw = match initial_cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };
        let dob_cow = record.field_as_str(&self.dob_field);
        let dob_raw = match dob_cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };

        let normalized = normalize_text(surname_raw);
        let surname    = extract_surname_token(&normalized);
        if surname.is_empty() {
            return vec![];
        }

        let code = self.encode(surname);
        if code.is_empty() {
            return vec![];
        }

        let initial = initial_raw
            .trim()
            .chars()
            .next()
            .map(|c| c.to_uppercase().to_string())
            .unwrap_or_default();
        if initial.is_empty() {
            return vec![];
        }

        let year = dob_raw.trim().get(..4).unwrap_or("").to_string();
        if year.len() < 4 || !year.chars().all(|c| c.is_ascii_digit()) {
            return vec![];
        }

        vec![format!("{}:{}:{}", code, initial, year)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{record::FieldValue, schema::SchemaBuilder, schema::FieldKind};

    fn make_schema() -> Schema {
        SchemaBuilder::new()
            .field("last_name",  FieldKind::Name)
            .field("first_name", FieldKind::Name)
            .field("dob",        FieldKind::Date)
            .build()
            .unwrap()
    }

    fn make_record(id: u64, last: &str, first: &str, dob: &str) -> Record {
        Record::new(id)
            .insert("last_name",  FieldValue::Text(last.into()))
            .insert("first_name", FieldValue::Text(first.into()))
            .insert("dob",        FieldValue::Text(dob.into()))
    }

    #[test]
    fn same_surname_same_initial_same_year_collide() {
        let schema = make_schema();
        let key    = PhoneticNameDobInitialKey::new("last_name", "first_name", "dob");

        let r1 = make_record(1, "Jong", "Anna",    "1990-03-01");
        let r2 = make_record(2, "Jong", "Annelies", "1990-07-15");

        let k1 = key.extract(&r1, &schema);
        let k2 = key.extract(&r2, &schema);
        assert!(!k1.is_empty());
        assert_eq!(k1, k2, "Same surname/initial/year should collide");
    }

    #[test]
    fn same_surname_different_initial_no_collision() {
        let schema = make_schema();
        let key    = PhoneticNameDobInitialKey::new("last_name", "first_name", "dob");

        let r1 = make_record(1, "Jong", "Anna",  "1990-03-01");
        let r2 = make_record(2, "Jong", "Pieter", "1990-03-01");

        let k1 = key.extract(&r1, &schema);
        let k2 = key.extract(&r2, &schema);
        assert_ne!(k1, k2, "Different initials should not collide");
    }

    #[test]
    fn different_dob_year_no_collision() {
        let schema = make_schema();
        let key    = PhoneticNameDobInitialKey::new("last_name", "first_name", "dob");

        let r1 = make_record(1, "Berg", "Anna", "1970-01-01");
        let r2 = make_record(2, "Berg", "Anna", "1985-01-01");

        let k1 = key.extract(&r1, &schema);
        let k2 = key.extract(&r2, &schema);
        assert_ne!(k1, k2, "Different DOB years should not collide");
    }

    #[test]
    fn missing_initial_field_returns_empty() {
        let schema = make_schema();
        let key    = PhoneticNameDobInitialKey::new("last_name", "first_name", "dob");

        let r = Record::new(1)
            .insert("last_name", FieldValue::Text("Berg".into()))
            .insert("dob",       FieldValue::Text("1990-01-01".into()));
        assert!(key.extract(&r, &schema).is_empty());
    }

    #[test]
    fn tussenvoegsel_stripped_from_surname() {
        let schema = make_schema();
        let key    = PhoneticNameDobInitialKey::new("last_name", "first_name", "dob");

        let r1 = make_record(1, "van den Berg", "Anna", "1990-06-15");
        let r2 = make_record(2, "Berg",          "Anna", "1990-06-15");

        let k1 = key.extract(&r1, &schema);
        let k2 = key.extract(&r2, &schema);
        assert!(!k1.is_empty());
        assert_eq!(k1, k2, "van den Berg and Berg should collide after prefix stripping");
    }
}
