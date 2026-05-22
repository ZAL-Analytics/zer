/// Shared time utilities for benchmark output.
///
/// Avoids pulling in `chrono` while still producing sortable ISO-8601 timestamps.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current UTC time as an ISO-8601 string: `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_timestamp_iso() -> String {
    fmt_unix_secs(unix_secs_now())
}

/// Returns the current UNIX timestamp in seconds.
pub fn unix_secs_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Format a UNIX timestamp (seconds since epoch) as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn fmt_unix_secs(s: u64) -> String {
    let sec  = s % 60;
    let min  = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    let year_400 = days / 146097;
    let rem      = days % 146097;
    let year_100 = rem / 36524;
    let rem      = rem % 36524;
    let year_4   = rem / 1461;
    let rem      = rem % 1461;
    let year_1   = rem / 365;
    let year     = 1970 + year_400 * 400 + year_100 * 100 + year_4 * 4 + year_1;
    let doy      = rem % 365;
    let (month, day) = doy_to_md(doy, is_leap(year));
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn doy_to_md(doy: u64, leap: bool) -> (u64, u64) {
    let days_in_month: [u64; 12] = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut rem = doy;
    for (i, &dim) in days_in_month.iter().enumerate() {
        if rem < dim {
            return (i as u64 + 1, rem + 1);
        }
        rem -= dim;
    }
    (12, 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_formats_correctly() {
        assert_eq!(fmt_unix_secs(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn known_date_formats_correctly() {
        // 2024-01-01T00:00:00Z = 1704067200
        assert_eq!(fmt_unix_secs(1704067200), "2024-01-01T00:00:00Z");
    }

    #[test]
    fn unix_secs_now_is_reasonable() {
        let t = unix_secs_now();
        assert!(t > 1_700_000_000, "timestamp must be after 2023");
    }

    #[test]
    fn utc_timestamp_iso_parses() {
        let s = utc_timestamp_iso();
        assert_eq!(s.len(), 20);
        assert!(s.ends_with('Z'));
    }
}
