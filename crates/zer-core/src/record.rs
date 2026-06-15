use std::borrow::Cow;

use ahash::AHashMap;

pub type RecordId = u64;
pub type FieldName = String;

/// Derive a stable `RecordId` from a `(source, key)` pair using FNV-1a.
///
/// The hash is deterministic across runs. same source and key always produce
/// the same u64.  Use this when loading records from external datasets so that
/// each record's identity is anchored to its natural key rather than a
/// caller-managed sequential integer.
pub fn derive_record_id(source: &str, key: &str) -> RecordId {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut h = OFFSET;
    for &b in source
        .as_bytes()
        .iter()
        .chain(b":".iter())
        .chain(key.as_bytes())
    {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

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
///
/// `id` is an internal u64 used for fast indexing and joins. treat it as
/// opaque.  `key` is the user-visible natural key: the value of whichever
/// column was nominated as the identity column when loading the dataset (e.g.
/// BSN, UUID, or primary-key value).  The `.zes` output references records by
/// `(source, key)`, not by `id`.
///
/// # Construction
///
/// * [`Record::from_key`]. preferred when loading real data via a
///   `zer_adapters::DatasetConfig`.  Derives `id` from `hash(source:key)`.
/// * [`Record::new`]. for synthetic/test records; sets `key = id.to_string()`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Record {
    pub id: RecordId,
    pub key: String,
    pub fields: AHashMap<FieldName, FieldValue>,
    pub source: Option<String>,
}

impl Record {
    /// Create a record with an explicit numeric ID.
    ///
    /// `key` is set to `id.to_string()`.  Use this only for synthetic or test
    /// records.  For real data use [`Record::from_key`] so that the natural
    /// key is preserved in the `.zes` output.
    pub fn new(id: RecordId) -> Self {
        Self {
            id,
            key: id.to_string(),
            fields: AHashMap::new(),
            source: None,
        }
    }

    /// Create a record whose identity comes from a natural key column.
    ///
    /// `id` is derived deterministically via `FNV-1a(source:key)` so that the
    /// same `(source, key)` pair always produces the same internal ID.  The
    /// `source` label is stored on the record, so calling [`Record::with_source`]
    /// afterwards is not required (but is a no-op).
    pub fn from_key(source: impl Into<String>, key: impl Into<String>) -> Self {
        let source = source.into();
        let key = key.into();
        let id = derive_record_id(&source, &key);
        Self {
            id,
            key,
            fields: AHashMap::new(),
            source: Some(source),
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
    /// assert_eq!(lat, Some(52.37f64));
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
        assert_eq!(r.key, "42");
        assert_eq!(r.source.as_deref(), Some("kvk"));
        assert_eq!(r.text("name"), Some("Alice"));
        assert_eq!(r.get("age"), Some(&FieldValue::Int(30)));
        assert_eq!(r.get("missing"), None);
    }

    #[test]
    fn from_key_derives_id_deterministically() {
        use super::derive_record_id;
        let r = Record::from_key("brp", "893479421");
        assert_eq!(r.key, "893479421");
        assert_eq!(r.source.as_deref(), Some("brp"));
        assert_eq!(r.id, derive_record_id("brp", "893479421"));

        // Same source+key always gives the same id.
        let r2 = Record::from_key("brp", "893479421");
        assert_eq!(r.id, r2.id);

        // Different key gives different id.
        let r3 = Record::from_key("brp", "999999999");
        assert_ne!(r.id, r3.id);

        // Different source gives different id even for the same key.
        let r4 = Record::from_key("kvk", "893479421");
        assert_ne!(r.id, r4.id);
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
