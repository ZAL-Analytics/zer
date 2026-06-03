use super::BlockingKey;
use zer_core::{record::Record, schema::Schema};

/// Controls how much of an ISO 8601 date is used as a blocking key.
#[derive(Debug, Clone, Copy)]
pub enum DateGranularity {
    Year,
    YearMonth,
    YearMonthDay,
}

/// Blocking key that extracts the leading date fragment at a given granularity.
pub struct DateFragmentKey {
    dob_field: String,
    granularity: DateGranularity,
}

impl DateFragmentKey {
    pub fn new(dob_field: &str, granularity: DateGranularity) -> Self {
        Self {
            dob_field: dob_field.into(),
            granularity,
        }
    }

    fn fragment_len(granularity: DateGranularity) -> usize {
        match granularity {
            DateGranularity::Year => 4,
            DateGranularity::YearMonth => 7,
            DateGranularity::YearMonthDay => 10,
        }
    }
}

impl BlockingKey for DateFragmentKey {
    fn name(&self) -> &str {
        "date_fragment"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let cow = record.field_as_str(&self.dob_field);
        let raw = match cow.as_deref() {
            Some(s) => s.trim(),
            None => return vec![],
        };

        let len = Self::fragment_len(self.granularity);
        if raw.len() < len {
            return vec![];
        }

        let fragment = &raw[..len];
        if !fragment
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            return vec![];
        }

        vec![fragment.to_string()]
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
            .field("dob", FieldKind::Date)
            .build()
            .unwrap()
    }

    #[test]
    fn year_fragment() {
        let k = DateFragmentKey::new("dob", DateGranularity::Year);
        let r = Record::new(1).insert("dob", FieldValue::Text("1985-06-23".into()));
        assert_eq!(k.extract(&r, &schema()), vec!["1985"]);
    }

    #[test]
    fn year_month_fragment() {
        let k = DateFragmentKey::new("dob", DateGranularity::YearMonth);
        let r = Record::new(1).insert("dob", FieldValue::Text("1985-06-23".into()));
        assert_eq!(k.extract(&r, &schema()), vec!["1985-06"]);
    }

    #[test]
    fn year_month_day_fragment() {
        let k = DateFragmentKey::new("dob", DateGranularity::YearMonthDay);
        let r = Record::new(1).insert("dob", FieldValue::Text("1985-06-23".into()));
        assert_eq!(k.extract(&r, &schema()), vec!["1985-06-23"]);
    }

    #[test]
    fn missing_field_returns_empty() {
        let k = DateFragmentKey::new("dob", DateGranularity::Year);
        let r = Record::new(1);
        assert!(k.extract(&r, &schema()).is_empty());
    }

    #[test]
    fn short_value_returns_empty() {
        let k = DateFragmentKey::new("dob", DateGranularity::YearMonth);
        let r = Record::new(1).insert("dob", FieldValue::Text("1985".into()));
        assert!(k.extract(&r, &schema()).is_empty());
    }
}
