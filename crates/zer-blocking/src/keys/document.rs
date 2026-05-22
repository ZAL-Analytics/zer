use zer_core::{record::Record, schema::Schema};

use crate::normalize::normalize_digits_only;
use super::BlockingKey;

// ── DocumentSuffixKey ─────────────────────────────────────────────────────────

/// Blocking key that strips non-alphanumeric characters from a document number
/// and emits the last `suffix_len` characters as a key.
///
/// Useful for matching passport or ID numbers that may be entered with
/// different prefix conventions or formatting (e.g. "P-NL-AB123456" vs
/// "AB123456"), while the suffix (serial part) stays stable.
///
/// Key format: `"SUFFIX"` (uppercase, alphanumeric only)
pub struct DocumentSuffixKey {
    field:      String,
    suffix_len: usize,
}

impl DocumentSuffixKey {
    /// `suffix_len = 6` is a reasonable default for European ID numbers.
    pub fn new(field: &str, suffix_len: usize) -> Self {
        Self { field: field.into(), suffix_len }
    }
}

impl BlockingKey for DocumentSuffixKey {
    fn name(&self) -> &str { "document_suffix" }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let cow = record.field_as_str(&self.field);
        let raw = match cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };
        let clean: String = raw
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_uppercase();
        if clean.len() < self.suffix_len {
            return vec![];
        }
        let suffix = &clean[clean.len() - self.suffix_len..];
        vec![suffix.to_string()]
    }
}

// ── DocumentDigitSuffixKey ────────────────────────────────────────────────────

/// Variant that strips ALL non-digit characters before taking the suffix.
///
/// Intended for purely numeric document identifiers (BSN, fiscal numbers)
/// where alphabetic characters are noise or country-code prefixes.
pub struct DocumentDigitSuffixKey {
    field:      String,
    suffix_len: usize,
}

impl DocumentDigitSuffixKey {
    pub fn new(field: &str, suffix_len: usize) -> Self {
        Self { field: field.into(), suffix_len }
    }
}

impl BlockingKey for DocumentDigitSuffixKey {
    fn name(&self) -> &str { "document_digit_suffix" }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let cow = record.field_as_str(&self.field);
        let raw = match cow.as_deref() {
            Some(s) => s,
            None    => return vec![],
        };
        let digits = normalize_digits_only(raw);
        if digits.len() < self.suffix_len {
            return vec![];
        }
        let suffix = &digits[digits.len() - self.suffix_len..];
        vec![suffix.to_string()]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{record::FieldValue, schema::{FieldKind, SchemaBuilder}};

    fn schema() -> Schema {
        SchemaBuilder::new()
            .field("document_nummer", FieldKind::Id)
            .build()
            .unwrap()
    }

    fn rec(id: u64, doc: &str) -> Record {
        Record::new(id).insert("document_nummer", FieldValue::Text(doc.into()))
    }

    // ── DocumentSuffixKey

    #[test]
    fn suffix_key_strips_non_alphanum_and_uppercases() {
        let schema = schema();
        let key    = DocumentSuffixKey::new("document_nummer", 6);
        let r      = rec(1, "P-NL-AB123456");
        let keys   = key.extract(&r, &schema);
        assert_eq!(keys, vec!["123456"]);
    }

    #[test]
    fn suffix_key_same_serial_different_prefix_collide() {
        let schema = schema();
        let key    = DocumentSuffixKey::new("document_nummer", 6);

        let r1 = rec(1, "P-NL-AB123456");
        let r2 = rec(2, "AB123456");
        assert_eq!(key.extract(&r1, &schema), key.extract(&r2, &schema));
    }

    #[test]
    fn suffix_key_too_short_returns_empty() {
        let schema = schema();
        let key    = DocumentSuffixKey::new("document_nummer", 6);
        let r      = rec(1, "AB12"); // only 4 chars after stripping
        assert!(key.extract(&r, &schema).is_empty());
    }

    #[test]
    fn suffix_key_missing_field_returns_empty() {
        let schema = schema();
        let key    = DocumentSuffixKey::new("document_nummer", 6);
        assert!(key.extract(&Record::new(1), &schema).is_empty());
    }

    // ── DocumentDigitSuffixKey

    #[test]
    fn digit_suffix_strips_all_letters() {
        let schema = schema();
        let key    = DocumentDigitSuffixKey::new("document_nummer", 4);
        let r      = rec(1, "BSN-12345678");
        let keys   = key.extract(&r, &schema);
        assert_eq!(keys, vec!["5678"]);
    }

    #[test]
    fn digit_suffix_same_number_different_format_collide() {
        let schema = schema();
        let key    = DocumentDigitSuffixKey::new("document_nummer", 6);

        let r1 = rec(1, "123-45-6789");
        let r2 = rec(2, "123456789");
        assert_eq!(key.extract(&r1, &schema), key.extract(&r2, &schema));
    }
}
