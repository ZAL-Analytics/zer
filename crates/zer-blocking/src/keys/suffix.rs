use zer_core::{record::Record, schema::Schema};

use crate::normalize::normalize_digits_only;
use super::BlockingKey;

/// Blocking key that extracts the last N digits from a field value.
pub struct SuffixKey {
    field: String,
    n:     usize,
}

impl SuffixKey {
    pub fn new(field: &str, n: usize) -> Self {
        Self { field: field.into(), n }
    }
}

impl BlockingKey for SuffixKey {
    fn name(&self) -> &str {
        "suffix"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let cow = record.field_as_str(&self.field);
        let raw = match cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };

        let digits = normalize_digits_only(raw);
        if digits.len() < self.n {
            return vec![];
        }

        let suffix = digits[digits.len() - self.n..].to_string();
        vec![suffix]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{record::FieldValue, schema::{SchemaBuilder, FieldKind}};

    fn schema() -> Schema {
        SchemaBuilder::new().field("phone", FieldKind::Phone).build().unwrap()
    }

    #[test]
    fn suffix_strips_punctuation_and_takes_last_n() {
        let k = SuffixKey::new("phone", 7);
        let r = Record::new(1).insert("phone", FieldValue::Text("555-123-4567".into()));
        assert_eq!(k.extract(&r, &schema()), vec!["1234567"]);
    }

    #[test]
    fn suffix_too_short_returns_empty() {
        let k = SuffixKey::new("phone", 7);
        let r = Record::new(1).insert("phone", FieldValue::Text("12345".into()));
        assert!(k.extract(&r, &schema()).is_empty());
    }

    #[test]
    fn same_last_digits_collide() {
        let k  = SuffixKey::new("phone", 4);
        let s  = schema();
        let r1 = Record::new(1).insert("phone", FieldValue::Text("06-1234".into()));
        let r2 = Record::new(2).insert("phone", FieldValue::Text("+31-20-001234".into()));
        assert_eq!(k.extract(&r1, &s), k.extract(&r2, &s));
    }
}
