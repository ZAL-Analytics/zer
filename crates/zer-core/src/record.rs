use std::borrow::Cow;

use ahash::AHashMap;

pub type RecordId = u64;
pub type FieldName = String;

/// Typed value stored in a record field.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FieldValue {
    Text(String),
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    Null,
}

impl From<String> for FieldValue {
    fn from(s: String) -> Self {
        FieldValue::Text(s)
    }
}
impl From<&str> for FieldValue {
    fn from(s: &str) -> Self {
        FieldValue::Text(s.to_owned())
    }
}
impl From<i64> for FieldValue {
    fn from(i: i64) -> Self {
        FieldValue::Int(i)
    }
}
impl From<i32> for FieldValue {
    fn from(i: i32) -> Self {
        FieldValue::Int(i as i64)
    }
}
impl From<u64> for FieldValue {
    fn from(u: u64) -> Self {
        FieldValue::UInt(u)
    }
}
impl From<Vec<u8>> for FieldValue {
    fn from(b: Vec<u8>) -> Self {
        FieldValue::Bytes(b)
    }
}
impl From<u32> for FieldValue {
    fn from(u: u32) -> Self {
        FieldValue::UInt(u as u64)
    }
}
impl From<f64> for FieldValue {
    fn from(f: f64) -> Self {
        FieldValue::Float(f)
    }
}
impl From<f32> for FieldValue {
    fn from(f: f32) -> Self {
        FieldValue::Float(f as f64)
    }
}
impl From<bool> for FieldValue {
    fn from(b: bool) -> Self {
        FieldValue::Bool(b)
    }
}
impl<T: Into<FieldValue>> From<Option<T>> for FieldValue {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => FieldValue::Null,
        }
    }
}

/// A single data record with a unique ID and a map of field values.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Record {
    pub id: RecordId,
    pub fields: AHashMap<FieldName, FieldValue>,
    pub source: Option<String>,
}

impl Record {
    pub fn new(id: RecordId) -> Self {
        Self {
            id,
            fields: AHashMap::new(),
            source: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn insert(mut self, name: impl Into<String>, value: impl Into<FieldValue>) -> Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    pub fn get(&self, name: &str) -> Option<&FieldValue> {
        self.fields.get(name)
    }

    pub fn text(&self, name: &str) -> Option<&str> {
        match self.fields.get(name) {
            Some(FieldValue::Text(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the field value as a string, coercing non-text scalars to their string representation.
    pub fn field_as_str(&self, name: &str) -> Option<Cow<'_, str>> {
        match self.fields.get(name)? {
            FieldValue::Text(s) => Some(Cow::Borrowed(s.as_str())),
            FieldValue::Int(i) => Some(Cow::Owned(i.to_string())),
            FieldValue::UInt(u) => Some(Cow::Owned(u.to_string())),
            FieldValue::Float(f) => Some(Cow::Owned(f.to_string())),
            FieldValue::Bool(b) => Some(Cow::Owned(b.to_string())),
            FieldValue::Bytes(_) => None,
            FieldValue::Null => None,
        }
    }

    /// Extract a typed value from a named field using the [`FromFieldValue`] trait.
    ///
    /// ```rust
    /// use zer_core::record::{Record, FieldValue};
    /// let r = Record::new(1).insert("lat", 52.37f64);
    /// let lat: Option<f64> = r.field_as::<f64>("lat");
    /// assert_eq!(lat, Some(52.37));
    /// ```
    pub fn field_as<T: FromFieldValue>(&self, name: &str) -> Option<T> {
        self.fields.get(name).and_then(T::from_field_value)
    }
}

/// Typed extraction from a [`FieldValue`].
pub trait FromFieldValue: Sized {
    fn from_field_value(v: &FieldValue) -> Option<Self>;
}

impl FromFieldValue for f64 {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::Float(f) => Some(*f),
            FieldValue::Int(i) => Some(*i as f64),
            FieldValue::UInt(u) => Some(*u as f64),
            // Text fallback: typed data avoids the parse; string data still works.
            FieldValue::Text(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }
}

impl FromFieldValue for f32 {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::Float(f) => Some(*f as f32),
            FieldValue::Int(i) => Some(*i as f32),
            FieldValue::UInt(u) => Some(*u as f32),
            FieldValue::Text(s) => s.parse::<f32>().ok(),
            _ => None,
        }
    }
}

impl FromFieldValue for i64 {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::Int(i) => Some(*i),
            FieldValue::UInt(u) => i64::try_from(*u).ok(),
            FieldValue::Text(s) => s.parse::<i64>().ok(),
            _ => None,
        }
    }
}

impl FromFieldValue for i32 {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::Int(i) => i32::try_from(*i).ok(),
            FieldValue::UInt(u) => i32::try_from(*u).ok(),
            FieldValue::Text(s) => s.parse::<i32>().ok(),
            _ => None,
        }
    }
}

impl FromFieldValue for u64 {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::UInt(u) => Some(*u),
            FieldValue::Int(i) => u64::try_from(*i).ok(),
            FieldValue::Text(s) => s.parse::<u64>().ok(),
            _ => None,
        }
    }
}

impl FromFieldValue for u32 {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::UInt(u) => u32::try_from(*u).ok(),
            FieldValue::Int(i) => u32::try_from(*i).ok(),
            FieldValue::Text(s) => s.parse::<u32>().ok(),
            _ => None,
        }
    }
}

impl FromFieldValue for bool {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl FromFieldValue for String {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::Text(s) => Some(s.clone()),
            FieldValue::Int(i) => Some(i.to_string()),
            FieldValue::UInt(u) => Some(u.to_string()),
            FieldValue::Float(f) => Some(f.to_string()),
            FieldValue::Bool(b) => Some(b.to_string()),
            FieldValue::Bytes(_) | FieldValue::Null => None,
        }
    }
}

impl FromFieldValue for Vec<u8> {
    fn from_field_value(v: &FieldValue) -> Option<Self> {
        match v {
            FieldValue::Bytes(b) => Some(b.clone()),
            FieldValue::Text(s) => Some(s.as_bytes().to_vec()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_value_equality() {
        assert_eq!(
            FieldValue::Text("hello".into()),
            FieldValue::Text("hello".into())
        );
        assert_ne!(FieldValue::Int(1), FieldValue::Int(2));
        assert_eq!(FieldValue::Null, FieldValue::Null);
    }

    #[test]
    fn record_builder_chain() {
        let r = Record::new(42)
            .with_source("kvk")
            .insert("name", "Alice")
            .insert("age", 30i64);

        assert_eq!(r.id, 42);
        assert_eq!(r.source.as_deref(), Some("kvk"));
        assert_eq!(r.text("name"), Some("Alice"));
        assert_eq!(r.get("age"), Some(&FieldValue::Int(30)));
        assert_eq!(r.get("missing"), None);
    }

    #[test]
    fn field_as_str_coerces_scalars() {
        let r = Record::new(1)
            .insert("phone", 5551234567i64)
            .insert("lat", 52.345f64)
            .insert("active", true)
            .insert("name", "Alice")
            .insert("empty", FieldValue::Null);

        assert_eq!(r.field_as_str("phone").as_deref(), Some("5551234567"));
        assert_eq!(r.field_as_str("lat").as_deref(), Some("52.345"));
        assert_eq!(r.field_as_str("active").as_deref(), Some("true"));
        assert_eq!(r.field_as_str("name").as_deref(), Some("Alice"));
        assert_eq!(r.field_as_str("empty"), None);
        assert_eq!(r.field_as_str("missing"), None);
    }

    #[test]
    fn from_impls_roundtrip() {
        assert_eq!(FieldValue::from("hello"), FieldValue::Text("hello".into()));
        assert_eq!(FieldValue::from(42i64), FieldValue::Int(42));
        assert_eq!(FieldValue::from(3.14f64), FieldValue::Float(3.14));
        assert_eq!(FieldValue::from(true), FieldValue::Bool(true));
        assert_eq!(FieldValue::from(Some("hi")), FieldValue::Text("hi".into()));
        assert_eq!(FieldValue::from(None::<&str>), FieldValue::Null);
        // u64 now produces UInt, not Int
        assert_eq!(FieldValue::from(u64::MAX), FieldValue::UInt(u64::MAX));
        // bytes roundtrip
        assert_eq!(
            FieldValue::from(vec![1u8, 2, 3]),
            FieldValue::Bytes(vec![1, 2, 3])
        );
    }

    #[test]
    fn field_as_str_new_variants() {
        let r = Record::new(1)
            .insert("count", 42u64)
            .insert("data", FieldValue::Bytes(vec![0xff]));
        assert_eq!(r.field_as_str("count").as_deref(), Some("42"));
        assert_eq!(r.field_as_str("data"), None);
    }

    #[test]
    fn field_as_typed() {
        let r = Record::new(1)
            .insert("lat", 52.37f64)
            .insert("count", 10u64)
            .insert("age", 30i64)
            .insert("active", true)
            .insert("name", "Alice")
            .insert("blob", FieldValue::Bytes(vec![1, 2, 3]));

        assert_eq!(r.field_as::<f64>("lat"), Some(52.37));
        assert_eq!(r.field_as::<f32>("lat"), Some(52.37f32));
        assert_eq!(r.field_as::<u64>("count"), Some(10u64));
        assert_eq!(r.field_as::<i64>("count"), Some(10i64));
        assert_eq!(r.field_as::<i64>("age"), Some(30i64));
        assert_eq!(r.field_as::<bool>("active"), Some(true));
        assert_eq!(r.field_as::<String>("name"), Some("Alice".to_string()));
        assert_eq!(r.field_as::<Vec<u8>>("blob"), Some(vec![1u8, 2, 3]));
        assert_eq!(r.field_as::<f64>("missing"), None);
    }

    #[test]
    fn field_as_cross_variant_coercions() {
        let r = Record::new(1)
            .insert("int_val", 100i64)
            .insert("uint_val", 200u64);

        // Int → f64
        assert_eq!(r.field_as::<f64>("int_val"), Some(100.0));
        // UInt → f64
        assert_eq!(r.field_as::<f64>("uint_val"), Some(200.0));
        // UInt → i64 (in range)
        assert_eq!(r.field_as::<i64>("uint_val"), Some(200i64));
        // Int → u64 (non-negative)
        assert_eq!(r.field_as::<u64>("int_val"), Some(100u64));

        // negative Int → u64 fails
        let r2 = Record::new(2).insert("neg", -1i64);
        assert_eq!(r2.field_as::<u64>("neg"), None);
    }
}
