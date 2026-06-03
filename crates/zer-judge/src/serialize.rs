/// Serialize a record pair into the NLI cross-encoder input format.
///
/// Format matches the Phase 9 training data exactly:
/// ```text
/// [CLS] COL:naam VAL:jan smits COL:geboortedatum VAL:1981-04-02 [SEP] COL:naam VAL:jan smyts COL:geboortedatum VAL:1981-04-03 [SEP]
/// ```
///
/// The tokenizer adds [CLS]/[SEP] tokens itself when `add_special_tokens=true`,
/// but we include them explicitly in the text so that fine-tuned models that
/// were trained with this raw-text format also work correctly.
use zer_core::{
    record::{FieldValue, Record},
    schema::Schema,
};

/// Produce the serialized string for a pair of records.
///
/// Fields are emitted in schema order so the output is stable regardless of
/// the `AHashMap` iteration order inside each `Record`.
pub fn serialize_pair(a: &Record, b: &Record, schema: &Schema) -> String {
    let mut out = String::with_capacity(256);

    out.push_str("[CLS]");
    append_record(&mut out, a, schema);
    out.push_str(" [SEP]");
    append_record(&mut out, b, schema);
    out.push_str(" [SEP]");

    out
}

fn append_record(buf: &mut String, record: &Record, schema: &Schema) {
    for field in schema.fields.iter() {
        let value = record.get(&field.name);
        let display = display_value(value);

        buf.push(' ');
        buf.push_str("COL:");
        buf.push_str(&field.name);
        buf.push_str(" VAL:");
        buf.push_str(&display);
    }
}

fn display_value(value: Option<&FieldValue>) -> String {
    match value {
        None => String::new(),
        Some(v) => match v {
            FieldValue::Text(s) => s.clone(),
            FieldValue::Int(n) => n.to_string(),
            FieldValue::UInt(n) => n.to_string(),
            FieldValue::Float(f) => format!("{f:.6}"),
            FieldValue::Bool(b) => {
                if *b {
                    "true".into()
                } else {
                    "false".into()
                }
            }
            FieldValue::Bytes(_) => String::new(), // binary blobs not serializable as text
            FieldValue::Null => String::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{
        record::{FieldValue, Record},
        schema::{FieldKind, SchemaBuilder},
    };

    fn schema() -> Schema {
        SchemaBuilder::new()
            .field("naam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap()
    }

    #[test]
    fn serialize_produces_expected_format() {
        let schema = schema();
        let a = Record::new(1)
            .insert("naam", FieldValue::Text("jan smits".into()))
            .insert("geboortedatum", FieldValue::Text("1981-04-02".into()));
        let b = Record::new(2)
            .insert("naam", FieldValue::Text("jan smyts".into()))
            .insert("geboortedatum", FieldValue::Text("1981-04-03".into()));

        let s = serialize_pair(&a, &b, &schema);
        assert!(
            s.contains("COL:naam VAL:jan smits"),
            "left record not serialized: {s}"
        );
        assert!(
            s.contains("COL:naam VAL:jan smyts"),
            "right record not serialized: {s}"
        );
        assert!(s.starts_with("[CLS]"), "must start with [CLS]: {s}");
        assert!(s.ends_with("[SEP]"), "must end with [SEP]: {s}");
        // Both [SEP] tokens must appear
        assert_eq!(
            s.matches("[SEP]").count(),
            2,
            "must have exactly two [SEP]: {s}"
        );
    }

    #[test]
    fn serialize_null_field_is_empty_val() {
        let schema = schema();
        let a = Record::new(1).insert("naam", FieldValue::Text("jan".into()));
        // geboortedatum intentionally absent
        let b = Record::new(2).insert("naam", FieldValue::Text("jan".into()));

        let s = serialize_pair(&a, &b, &schema);
        // Should contain "COL:geboortedatum VAL:" with nothing after VAL: before the next token
        assert!(s.contains("COL:geboortedatum VAL:"), "{s}");
    }

    #[test]
    fn serialize_fields_in_schema_order() {
        let schema = schema();
        let r = Record::new(1)
            .insert("geboortedatum", FieldValue::Text("1981-04-02".into()))
            .insert("naam", FieldValue::Text("jan".into()));

        let s = serialize_pair(&r, &r, &schema);
        let naam_pos = s.find("COL:naam").unwrap();
        let dob_pos = s.find("COL:geboortedatum").unwrap();
        assert!(
            naam_pos < dob_pos,
            "naam should come before geboortedatum (schema order): {s}"
        );
    }
}
