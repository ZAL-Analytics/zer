use crate::{error::ZerError, record::FieldName};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FieldKind {
    Name,
    /// ISO 8601 date (YYYY-MM-DD), no time component.
    Date,
    Address,
    Phone,
    Id,
    FreeText,
    Numeric,
    Categorical,
    /// Pipe-delimited list of name aliases (e.g. SIS II `alias_namen` field).
    Alias,
    /// Vehicle registration plate (e.g. Dutch kenteken). Enables OCR-fuzzy blocking.
    LicensePlate,
    /// Geographic coordinate stored as a decimal float string (lat or lon).
    GpsCoordinate,
    /// ISO 8601 datetime including time component (YYYY-MM-DDTHH:MM:SS).
    Timestamp,
}

/// Name and kind for a single schema field.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FieldDef {
    pub name: FieldName,
    pub kind: FieldKind,
}

/// Ordered list of field definitions for a dataset.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Schema {
    pub fields: Vec<FieldDef>,
}

impl Schema {
    /// Iterate over field names that match a given kind.
    pub fn fields_of_kind(&self, kind: FieldKind) -> impl Iterator<Item = &str> {
        self.fields.iter()
            .filter(move |f| f.kind == kind)
            .map(|f| f.name.as_str())
    }

    /// Return the position of a field by name, or `None` if absent.
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Fluent builder for constructing a `Schema`.
#[derive(Default)]
pub struct SchemaBuilder {
    fields: Vec<FieldDef>,
}

impl SchemaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn field(mut self, name: &str, kind: FieldKind) -> Self {
        self.fields.push(FieldDef { name: name.into(), kind });
        self
    }

    pub fn build(self) -> Result<Schema, ZerError> {
        if self.fields.is_empty() {
            return Err(ZerError::EmptySchema);
        }
        Ok(Schema { fields: self.fields })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_builder_rejects_empty() {
        assert!(SchemaBuilder::new().build().is_err());
    }

    #[test]
    fn schema_builder_produces_correct_field_count() {
        let s = SchemaBuilder::new()
            .field("first_name", FieldKind::Name)
            .field("last_name", FieldKind::Name)
            .field("dob", FieldKind::Date)
            .build()
            .unwrap();
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn fields_of_kind_filters_correctly() {
        let s = SchemaBuilder::new()
            .field("first_name", FieldKind::Name)
            .field("last_name", FieldKind::Name)
            .field("dob", FieldKind::Date)
            .build()
            .unwrap();

        let names: Vec<&str> = s.fields_of_kind(FieldKind::Name).collect();
        assert_eq!(names, vec!["first_name", "last_name"]);

        let dates: Vec<&str> = s.fields_of_kind(FieldKind::Date).collect();
        assert_eq!(dates, vec!["dob"]);

        let phones: Vec<&str> = s.fields_of_kind(FieldKind::Phone).collect();
        assert!(phones.is_empty());
    }

    #[test]
    fn field_index_lookup() {
        let s = SchemaBuilder::new()
            .field("name", FieldKind::Name)
            .field("dob", FieldKind::Date)
            .build()
            .unwrap();
        assert_eq!(s.field_index("name"), Some(0));
        assert_eq!(s.field_index("dob"), Some(1));
        assert_eq!(s.field_index("missing"), None);
    }
}
