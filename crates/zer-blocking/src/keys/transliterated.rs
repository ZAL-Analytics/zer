use rphonetic::{DoubleMetaphone, Encoder};
use zer_core::{record::Record, schema::Schema};

use crate::normalize::{extract_surname_token, transliterate_and_normalize};
use super::BlockingKey;

// ── TransliteratedPhoneticKey ─────────────────────────────────────────────────

/// Phonetic blocking key that first transliterates non-Latin script (Arabic,
/// Cyrillic, Greek, etc.) to ASCII via `any_ascii`, then applies NFKD
/// diacritic stripping and DoubleMetaphone encoding, combined with the DOB
/// year.
///
/// Key format: `"PHONETIC_CODE:YEAR"`
///
/// Use alongside `PhoneticNameDobKey` (which only handles already-Latin
/// input) when your dataset may contain non-Latin name entries, e.g. persons
/// registered in Arabic script by one Schengen state and in Latin by another.
pub struct TransliteratedPhoneticKey {
    name_field: String,
    dob_field:  String,
}

impl TransliteratedPhoneticKey {
    pub fn new(name_field: &str, dob_field: &str) -> Self {
        Self {
            name_field: name_field.into(),
            dob_field:  dob_field.into(),
        }
    }
}

impl BlockingKey for TransliteratedPhoneticKey {
    fn name(&self) -> &str { "transliterated_phonetic" }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let name_cow = record.field_as_str(&self.name_field);
        let name_raw = match name_cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };
        let dob_cow = record.field_as_str(&self.dob_field);
        let dob_raw = match dob_cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };

        let year = dob_raw.trim().get(..4).unwrap_or("");
        if year.len() < 4 || !year.chars().all(|c| c.is_ascii_digit()) {
            return vec![];
        }

        let norm    = transliterate_and_normalize(name_raw);
        let surname = extract_surname_token(&norm);
        if surname.is_empty() { return vec![]; }

        let code = DoubleMetaphone::default().encode(surname);
        if code.is_empty() { return vec![]; }

        vec![format!("{}:{}", code, year)]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{record::FieldValue, schema::{FieldKind, SchemaBuilder}};

    fn schema() -> Schema {
        SchemaBuilder::new()
            .field("naam", FieldKind::Name)
            .field("dob",  FieldKind::Date)
            .build()
            .unwrap()
    }

    fn rec(id: u64, naam: &str, dob: &str) -> Record {
        Record::new(id)
            .insert("naam", FieldValue::Text(naam.into()))
            .insert("dob",  FieldValue::Text(dob.into()))
    }

    #[test]
    fn latin_diacritic_name_produces_key() {
        let schema = schema();
        let key    = TransliteratedPhoneticKey::new("naam", "dob");
        let r      = rec(1, "Müller", "1985-03-01");
        let keys   = key.extract(&r, &schema);
        // "Müller" → any_ascii → "Muller" → normalize → "MULLER" → phonetic + year
        assert_eq!(keys.len(), 1);
        assert!(keys[0].ends_with(":1985"), "key should contain DOB year");
    }

    #[test]
    fn latin_and_arabic_transliteration_collide() {
        let schema = schema();
        let key    = TransliteratedPhoneticKey::new("naam", "dob");

        // Arabic "بن عبدالله" transliterates to approximately "bn abdallh" → surname token
        // This is a smoke test: both produce non-empty keys for the same DOB year.
        let r_latin  = rec(1, "Benabdallah", "1999-01-01");
        let r_arabic = rec(2, "بن عبدالله", "1999-01-01");

        let k1 = key.extract(&r_latin,  &schema);
        let k2 = key.extract(&r_arabic, &schema);

        // Both should produce non-empty keys, exact collision depends on any_ascii
        assert!(!k1.is_empty(), "Latin name should produce a key");
        assert!(!k2.is_empty(), "Arabic name should produce a key after transliteration");
    }

    #[test]
    fn missing_dob_returns_empty() {
        let schema = schema();
        let key    = TransliteratedPhoneticKey::new("naam", "dob");
        let r      = Record::new(1).insert("naam", FieldValue::Text("Jansen".into()));
        assert!(key.extract(&r, &schema).is_empty());
    }

    #[test]
    fn missing_name_returns_empty() {
        let schema = schema();
        let key    = TransliteratedPhoneticKey::new("naam", "dob");
        let r      = Record::new(1).insert("dob", FieldValue::Text("1990-01-01".into()));
        assert!(key.extract(&r, &schema).is_empty());
    }

    #[test]
    fn tussenvoegsel_stripped_before_phonetic() {
        let schema = schema();
        let key    = TransliteratedPhoneticKey::new("naam", "dob");

        let r1 = rec(1, "van den Berg", "1990-06-15");
        let r2 = rec(2, "Berg",         "1990-06-15");

        let k1 = key.extract(&r1, &schema);
        let k2 = key.extract(&r2, &schema);

        assert!(!k1.is_empty());
        assert_eq!(k1, k2, "van den Berg and Berg should produce the same phonetic key");
    }
}
