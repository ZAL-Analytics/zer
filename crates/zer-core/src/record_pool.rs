use crate::{
    record::{FieldValue, Record, RecordId},
    schema::Schema,
    traits::RecordStore,
};

/// Column-major record store: `columns[field_idx][record_idx]`.
///
/// All field values are stored as UTF-8 strings in schema field order.
/// Missing or non-text values become empty strings (treated as
/// `ComparisonLevel::None` by every comparator).
///
/// The `RecordId` for record `r` is `ids[r]`.
#[derive(Debug, Clone)]
pub struct RecordPool {
    pub ids: Vec<RecordId>,
    /// `columns[field_idx][record_idx]` = UTF-8 text value.
    pub columns: Vec<Vec<String>>,
    pub n_fields: usize,
}

impl RecordPool {
    pub fn new(n_fields: usize) -> Self {
        Self {
            ids: Vec::new(),
            columns: vec![Vec::new(); n_fields],
            n_fields,
        }
    }

    pub fn from_records(records: &[Record], schema: &Schema) -> Self {
        let mut pool = Self::with_capacity(records.len(), schema.fields.len());
        for r in records {
            pool.push(r, schema);
        }
        pool
    }

    /// Build a pool from a [`RecordStore`], loading only the records with IDs
    /// listed in `ids`.  Records are inserted in `ids` order; pool position `i`
    /// corresponds to `ids[i]`.
    pub fn from_store(store: &dyn RecordStore, ids: &[RecordId], schema: &Schema) -> Self {
        let records: Vec<Record> = ids
            .iter()
            .filter_map(|id| store.get(*id).map(|c| c.into_owned()))
            .collect();
        Self::from_records(&records, schema)
    }

    /// Build a pool from `(Record, Record)` pairs: record `2*i` is side A of
    /// pair `i`, record `2*i+1` is side B.  Allows `compare_batch(&pairs)` to
    /// build a pool once and delegate to `compare_batch_from_pool`.
    pub fn from_pairs(pairs: &[(Record, Record)], schema: &Schema) -> Self {
        let mut pool = Self::with_capacity(pairs.len() * 2, schema.fields.len());
        for (a, b) in pairs {
            pool.push(a, schema);
            pool.push(b, schema);
        }
        pool
    }

    pub fn with_capacity(cap: usize, n_fields: usize) -> Self {
        Self {
            ids: Vec::with_capacity(cap),
            columns: vec![Vec::with_capacity(cap); n_fields],
            n_fields,
        }
    }

    /// Append one record.  Fields are stored in schema order.
    pub fn push(&mut self, record: &Record, schema: &Schema) {
        self.ids.push(record.id);
        for (fi, field) in schema.fields.iter().enumerate() {
            self.columns[fi].push(field_value_to_string(record.fields.get(&field.name)));
        }
    }

    /// Direct column access: bytes of field `f` for record `r`.
    #[inline]
    pub fn get(&self, field_idx: usize, record_idx: usize) -> &str {
        &self.columns[field_idx][record_idx]
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

fn field_value_to_string(v: Option<&FieldValue>) -> String {
    match v {
        Some(FieldValue::Text(s)) => s.clone(),
        Some(FieldValue::Int(i)) => i.to_string(),
        Some(FieldValue::UInt(u)) => u.to_string(),
        Some(FieldValue::Float(f)) => f.to_string(),
        Some(FieldValue::Bool(b)) => b.to_string(),
        Some(FieldValue::Bytes(_)) => String::new(),
        Some(FieldValue::Null) | None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        record::FieldValue,
        schema::{FieldKind, SchemaBuilder},
    };

    use super::*;

    fn person_schema() -> Schema {
        SchemaBuilder::new()
            .field("naam", FieldKind::Name)
            .field("dob", FieldKind::Date)
            .build()
            .unwrap()
    }

    #[test]
    fn pool_from_records_stores_in_column_order() {
        let schema = person_schema();
        let records = vec![
            Record::new(1)
                .insert("naam", FieldValue::Text("Alice".into()))
                .insert("dob", FieldValue::Text("1990-01-01".into())),
            Record::new(2)
                .insert("naam", FieldValue::Text("Bob".into()))
                .insert("dob", FieldValue::Text("1985-06-15".into())),
        ];
        let pool = RecordPool::from_records(&records, &schema);

        assert_eq!(pool.len(), 2);
        assert_eq!(pool.ids, vec![1, 2]);
        assert_eq!(pool.get(0, 0), "Alice");
        assert_eq!(pool.get(0, 1), "Bob");
        assert_eq!(pool.get(1, 0), "1990-01-01");
        assert_eq!(pool.get(1, 1), "1985-06-15");
    }

    #[test]
    fn pool_missing_field_is_empty_string() {
        let schema = person_schema();
        // Record has naam but no dob
        let r = Record::new(1).insert("naam", FieldValue::Text("Alice".into()));
        let pool = RecordPool::from_records(&[r], &schema);
        assert_eq!(pool.get(0, 0), "Alice");
        assert_eq!(pool.get(1, 0), "");
    }

    #[test]
    fn pool_null_field_is_empty_string() {
        let schema = person_schema();
        let r = Record::new(1).insert("naam", FieldValue::Null);
        let pool = RecordPool::from_records(&[r], &schema);
        assert_eq!(pool.get(0, 0), "");
    }

    #[test]
    fn pool_push_incremental() {
        let schema = person_schema();
        let mut pool = RecordPool::new(schema.fields.len());

        pool.push(
            &Record::new(10).insert("naam", FieldValue::Text("X".into())),
            &schema,
        );
        pool.push(
            &Record::new(20).insert("naam", FieldValue::Text("Y".into())),
            &schema,
        );

        assert_eq!(pool.len(), 2);
        assert_eq!(pool.ids, vec![10, 20]);
        assert_eq!(pool.get(0, 0), "X");
        assert_eq!(pool.get(0, 1), "Y");
    }
}
