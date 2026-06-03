use zer_core::{record::FieldValue, schema::FieldKind};

use crate::similarity::SimilarityFn;

/// Similarity function for Date and Timestamp fields.
///
/// Parses ISO-8601 dates (YYYY-MM-DD) and computes similarity based on
/// calendar distance. Levels map to:
///   1.0 , exact
///   0.9 , off by ≤ 1 day (transposition / single-digit error)
///   0.75, same year + month, different day
///   0.5 , same year, different month
///   0.3 , year ± 1 (age-compatible / estimated DOB range)
///   0.0 , otherwise
pub struct DateSimilarity;

/// Parse a date string (ISO-8601 YYYY-MM-DD or Unix timestamp) into (year, month, day).
fn parse_date(s: &str) -> Option<(i32, u32, u32)> {
    let s = s.trim();
    // ISO-8601: YYYY-MM-DD or YYYY-MM-DDThh:mm:ss (take date part only)
    let date_part = s.split('T').next().unwrap_or(s);
    let parts: Vec<&str> = date_part.split('-').collect();
    if parts.len() >= 3 {
        if let (Ok(y), Ok(m), Ok(d)) = (
            parts[0].parse::<i32>(),
            parts[1].parse::<u32>(),
            parts[2].parse::<u32>(),
        ) {
            if (1..=12).contains(&m) && (1..=31).contains(&d) {
                return Some((y, m, d));
            }
        }
    }
    // Unix timestamp (integer seconds)
    if let Ok(ts) = s.parse::<i64>() {
        let days_since_epoch = ts / 86400;
        // Approximate conversion; sufficient for year-level comparisons
        let y = 1970 + (days_since_epoch / 365) as i32;
        return Some((y, 1, 1));
    }
    None
}

/// Convert a calendar date to a Julian Day Number for computing day differences.
fn to_julian(y: i32, m: u32, d: u32) -> i32 {
    let a = (14_i32 - m as i32) / 12;
    let y2 = y + 4800 - a;
    let m2 = m as i32 + 12 * a - 3;
    d as i32 + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045
}

fn days_between(a: (i32, u32, u32), b: (i32, u32, u32)) -> i32 {
    (to_julian(a.0, a.1, a.2) - to_julian(b.0, b.1, b.2)).abs()
}

fn date_score(sa: &str, sb: &str) -> f32 {
    if sa == sb {
        return 1.0;
    }
    let (da, db) = match (parse_date(sa), parse_date(sb)) {
        (Some(a), Some(b)) => (a, b),
        _ => return 0.0,
    };
    let diff = days_between(da, db);
    if diff == 0 {
        1.0
    } else if diff <= 1 {
        0.9
    } else if da.0 == db.0 && da.1 == db.1 {
        0.75
    } else if da.0 == db.0 {
        0.5
    } else if (da.0 - db.0).abs() <= 1 {
        0.3
    } else {
        0.0
    }
}

impl SimilarityFn for DateSimilarity {
    fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
        let (sa, sb) = match (a, b) {
            (FieldValue::Text(a), FieldValue::Text(b)) => (a.as_str(), b.as_str()),
            _ => return 0.0,
        };
        date_score(sa, sb)
    }
    fn similarity_str(&self, a: &str, b: &str) -> f32 {
        date_score(a, b)
    }
    fn field_kind(&self) -> FieldKind {
        FieldKind::Date
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tv(s: &str) -> FieldValue {
        FieldValue::Text(s.into())
    }

    #[test]
    fn exact_date_match() {
        let sim = DateSimilarity;
        assert_eq!(sim.similarity(&tv("1990-06-15"), &tv("1990-06-15")), 1.0);
    }

    #[test]
    fn off_by_one_day() {
        let sim = DateSimilarity;
        assert_eq!(sim.similarity(&tv("1990-06-15"), &tv("1990-06-16")), 0.9);
        assert_eq!(sim.similarity(&tv("1990-06-15"), &tv("1990-06-14")), 0.9);
    }

    #[test]
    fn same_year_month_different_day() {
        let sim = DateSimilarity;
        assert_eq!(sim.similarity(&tv("1990-06-15"), &tv("1990-06-20")), 0.75);
    }

    #[test]
    fn same_year_different_month() {
        let sim = DateSimilarity;
        assert_eq!(sim.similarity(&tv("1990-06-15"), &tv("1990-09-01")), 0.5);
    }

    #[test]
    fn age_compatible_within_one_year() {
        let sim = DateSimilarity;
        assert_eq!(sim.similarity(&tv("1990-06-15"), &tv("1991-01-01")), 0.3);
        assert_eq!(sim.similarity(&tv("1990-01-01"), &tv("1989-07-20")), 0.3);
    }

    #[test]
    fn completely_different_dates() {
        let sim = DateSimilarity;
        assert_eq!(sim.similarity(&tv("1990-06-15"), &tv("1975-03-22")), 0.0);
    }

    #[test]
    fn missing_field_returns_zero() {
        let sim = DateSimilarity;
        assert_eq!(sim.similarity(&FieldValue::Null, &tv("1990-06-15")), 0.0);
        assert_eq!(sim.similarity(&tv("1990-06-15"), &FieldValue::Null), 0.0);
    }

    #[test]
    fn timestamp_date_part_comparison() {
        let sim = DateSimilarity;
        // T-prefixed ISO-8601 datetime, date parts should match
        assert_eq!(
            sim.similarity(&tv("1990-06-15T08:30:00"), &tv("1990-06-15T14:00:00")),
            1.0
        );
    }
}
