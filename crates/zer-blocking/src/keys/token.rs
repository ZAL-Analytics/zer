use zer_core::{record::Record, schema::Schema};

use crate::normalize::normalize_text;
use super::BlockingKey;

/// Blocks on: (first token of address field) + ":" + (first char of first-name field).
/// Handles surname transpositions, two records at the same address with the same initial
/// should end up in the same bucket even if the surname differs.
pub struct AddressInitialKey {
    address_field:    String,
    first_name_field: String,
}

impl AddressInitialKey {
    pub fn new(address_field: &str, first_name_field: &str) -> Self {
        Self {
            address_field:    address_field.into(),
            first_name_field: first_name_field.into(),
        }
    }
}

impl BlockingKey for AddressInitialKey {
    fn name(&self) -> &str {
        "addr_initial"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let addr_cow = record.field_as_str(&self.address_field);
        let addr_raw = match addr_cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };
        let name_cow = record.field_as_str(&self.first_name_field);
        let name_raw = match name_cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };

        let addr_norm = normalize_text(addr_raw);
        let name_norm = normalize_text(name_raw);

        let addr_token = addr_norm.split_whitespace().next().unwrap_or("").to_string();
        let initial    = name_norm.chars().next().unwrap_or(' ');

        if addr_token.is_empty() || !initial.is_ascii_alphabetic() {
            return vec![];
        }

        vec![format!("{}:{}", addr_token, initial)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{record::FieldValue, schema::{SchemaBuilder, FieldKind}};

    fn schema() -> Schema {
        SchemaBuilder::new()
            .field("address",    FieldKind::Address)
            .field("first_name", FieldKind::Name)
            .build()
            .unwrap()
    }

    #[test]
    fn extracts_first_token_and_initial() {
        let k = AddressInitialKey::new("address", "first_name");
        let r = Record::new(1)
            .insert("address",    FieldValue::Text("123 Main Street".into()))
            .insert("first_name", FieldValue::Text("John".into()));
        assert_eq!(k.extract(&r, &schema()), vec!["123:J"]);
    }

    #[test]
    fn same_address_different_first_name_no_collision() {
        let k  = AddressInitialKey::new("address", "first_name");
        let s  = schema();
        let r1 = Record::new(1)
            .insert("address",    FieldValue::Text("Singel 191".into()))
            .insert("first_name", FieldValue::Text("Alice".into()));
        let r2 = Record::new(2)
            .insert("address",    FieldValue::Text("Singel 191".into()))
            .insert("first_name", FieldValue::Text("Bob".into()));
        assert_ne!(k.extract(&r1, &s), k.extract(&r2, &s));
    }

    #[test]
    fn missing_field_returns_empty() {
        let k = AddressInitialKey::new("address", "first_name");
        let r = Record::new(1).insert("address", FieldValue::Text("Singel 191".into()));
        assert!(k.extract(&r, &schema()).is_empty());
    }
}
