use zer_core::{record::Record, schema::Schema};

use super::BlockingKey;
use crate::normalize::normalize_text;

pub struct ExactFieldKey {
    field: String,
}

impl ExactFieldKey {
    pub fn new(field: &str) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl BlockingKey for ExactFieldKey {
    fn name(&self) -> &str {
        "exact"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let cow = record.field_as_str(&self.field);
        let raw = match cow.as_deref() {
            Some(s) => s,
            None => return vec![],
        };

        let normalized = normalize_text(raw);
        if normalized.is_empty() {
            return vec![];
        }

        vec![normalized]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{
        record::FieldValue,
        schema::{FieldKind, SchemaBuilder},
    };

    fn schema() -> Schema {
        SchemaBuilder::new()
            .field("category", FieldKind::Categorical)
            .build()
            .unwrap()
    }

    #[test]
    fn exact_normalizes_and_matches() {
        let k = ExactFieldKey::new("category");
        let s = schema();
        let r1 = Record::new(1).insert("category", FieldValue::Text("Eenmanszaak".into()));
        let r2 = Record::new(2).insert("category", FieldValue::Text("EENMANSZAAK".into()));
        assert_eq!(k.extract(&r1, &s), k.extract(&r2, &s));
    }

    #[test]
    fn exact_empty_field_returns_empty() {
        let k = ExactFieldKey::new("category");
        let r = Record::new(1).insert("category", FieldValue::Text("".into()));
        assert!(k.extract(&r, &schema()).is_empty());
    }

    #[test]
    fn exact_missing_field_returns_empty() {
        let k = ExactFieldKey::new("category");
        let r = Record::new(1);
        assert!(k.extract(&r, &schema()).is_empty());
    }
}
