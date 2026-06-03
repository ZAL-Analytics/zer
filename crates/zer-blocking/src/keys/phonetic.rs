use rphonetic::{DoubleMetaphone, Encoder, Soundex};
use zer_core::{record::Record, schema::Schema};

use super::BlockingKey;
use crate::normalize::{extract_surname_token, normalize_text};

/// Phonetic encoding algorithm.
#[derive(Debug, Clone, Copy)]
pub enum PhoneticAlgo {
    DoubleMetaphone,
    Soundex,
}

/// Blocking key that encodes the surname phonetically combined with the birth year.
pub struct PhoneticNameDobKey {
    algo: PhoneticAlgo,
    name_field: String,
    dob_field: String,
}

impl PhoneticNameDobKey {
    pub fn new(name_field: &str, dob_field: &str) -> Self {
        Self {
            algo: PhoneticAlgo::DoubleMetaphone,
            name_field: name_field.into(),
            dob_field: dob_field.into(),
        }
    }

    pub fn with_algo(mut self, algo: PhoneticAlgo) -> Self {
        self.algo = algo;
        self
    }

    fn encode(&self, s: &str) -> String {
        match self.algo {
            PhoneticAlgo::DoubleMetaphone => DoubleMetaphone::default().encode(s),
            PhoneticAlgo::Soundex => Soundex::default().encode(s),
        }
    }
}

impl BlockingKey for PhoneticNameDobKey {
    fn name(&self) -> &str {
        "phonetic_dob"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let surname_cow = record.field_as_str(&self.name_field);
        let surname_raw = match surname_cow.as_deref() {
            Some(s) => s,
            None => return vec![],
        };
        let dob_cow = record.field_as_str(&self.dob_field);
        let dob_raw = match dob_cow.as_deref() {
            Some(s) => s,
            None => return vec![],
        };

        let normalized = normalize_text(surname_raw);
        let surname = extract_surname_token(&normalized);
        if surname.is_empty() {
            return vec![];
        }

        let code = self.encode(surname);
        if code.is_empty() {
            return vec![];
        }

        let year = dob_raw.trim().get(..4).unwrap_or("").to_string();
        if year.len() < 4 || !year.chars().all(|c| c.is_ascii_digit()) {
            return vec![];
        }

        vec![format!("{}:{}", code, year)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{record::FieldValue, schema::FieldKind, schema::SchemaBuilder};

    fn make_schema() -> Schema {
        SchemaBuilder::new()
            .field("last_name", FieldKind::Name)
            .field("dob", FieldKind::Date)
            .build()
            .unwrap()
    }

    fn make_record(id: u64, last_name: &str, dob: &str) -> Record {
        Record::new(id)
            .insert("last_name", FieldValue::Text(last_name.into()))
            .insert("dob", FieldValue::Text(dob.into()))
    }

    #[test]
    fn phonetic_key_same_surname_variants_collide() {
        let schema = make_schema();
        let key = PhoneticNameDobKey::new("last_name", "dob");

        let r1 = make_record(1, "Smith", "1985-03-01");
        let r2 = make_record(2, "Smyth", "1985-03-01");
        let r3 = make_record(3, "Smythe", "1985-03-01");

        let k1 = key.extract(&r1, &schema);
        let k2 = key.extract(&r2, &schema);
        let k3 = key.extract(&r3, &schema);

        assert!(!k1.is_empty(), "SMITH should produce a key");
        assert_eq!(k1, k2, "SMITH and SMYTH should share a phonetic key");
        assert_eq!(k1, k3, "SMITH and SMYTHE should share a phonetic key");
    }

    #[test]
    fn phonetic_key_different_dob_year_no_collision() {
        let schema = make_schema();
        let key = PhoneticNameDobKey::new("last_name", "dob");

        let r1 = make_record(1, "Berg", "1970-01-01");
        let r2 = make_record(2, "Berg", "1985-01-01");

        let k1 = key.extract(&r1, &schema);
        let k2 = key.extract(&r2, &schema);
        assert_ne!(k1, k2, "Different DOB years should produce different keys");
    }

    #[test]
    fn phonetic_key_missing_field_returns_empty() {
        let schema = make_schema();
        let key = PhoneticNameDobKey::new("last_name", "dob");

        let r = Record::new(1).insert("last_name", FieldValue::Text("Berg".into()));
        assert!(key.extract(&r, &schema).is_empty());
    }

    #[test]
    fn phonetic_key_tussenvoegsel_stripped() {
        let schema = make_schema();
        let key = PhoneticNameDobKey::new("last_name", "dob");

        let r1 = make_record(1, "van den Berg", "1990-06-15");
        let r2 = make_record(2, "Berg", "1990-06-15");

        let k1 = key.extract(&r1, &schema);
        let k2 = key.extract(&r2, &schema);
        assert!(!k1.is_empty());
        assert_eq!(
            k1, k2,
            "van den Berg and Berg should collide after prefix stripping"
        );
    }
}
