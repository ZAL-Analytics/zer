use zer_core::{record::Record, schema::Schema};

use super::BlockingKey;
use crate::normalize::normalize_plate;

// ── LicensePlateNormKey ───────────────────────────────────────────────────────

/// Normalizes a license plate (strips hyphens/spaces, uppercases) and emits
/// the result as a single exact blocking key.
///
/// Use together with `PlateOCRFuzzyKey` for full OCR-resilient plate matching.
pub struct LicensePlateNormKey {
    plate_field: String,
}

impl LicensePlateNormKey {
    pub fn new(plate_field: &str) -> Self {
        Self {
            plate_field: plate_field.into(),
        }
    }
}

impl BlockingKey for LicensePlateNormKey {
    fn name(&self) -> &str {
        "plate_norm"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let cow = record.field_as_str(&self.plate_field);
        let plate = match cow.as_deref() {
            Some(s) => s,
            None => return vec![],
        };
        let norm = normalize_plate(plate);
        if norm.is_empty() {
            return vec![];
        }
        vec![norm]
    }
}

// ── PlateOCRFuzzyKey ──────────────────────────────────────────────────────────

/// Emits the normalized plate **plus** a deletion-neighbourhood key for each
/// character position.
///
/// The deletion-neighbourhood approach handles any single-character OCR
/// confusion (0/O, 1/I, M/W, G/C, etc.) without an explicit confusion table:
/// two plates that differ by exactly one character at position `i` will both
/// produce the same key when character `i` is removed, so they land in the
/// same candidate bucket.
///
/// Example: "CX180W" vs "CXI80W" (1/I confusion at position 2) both become "CX80W" after
/// deleting position 2, so they share a bucket key.
pub struct PlateOCRFuzzyKey {
    plate_field: String,
}

impl PlateOCRFuzzyKey {
    pub fn new(plate_field: &str) -> Self {
        Self {
            plate_field: plate_field.into(),
        }
    }
}

impl BlockingKey for PlateOCRFuzzyKey {
    fn name(&self) -> &str {
        "plate_ocr"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let cow = record.field_as_str(&self.plate_field);
        let plate = match cow.as_deref() {
            Some(s) => s,
            None => return vec![],
        };
        let norm = normalize_plate(plate);
        if norm.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = norm.chars().collect();
        let n = chars.len();
        let mut keys = Vec::with_capacity(n + 1);
        keys.push(norm.clone());

        // For each position, emit the plate with that character removed.
        // Two plates differing by one substitution at position i share the
        // deletion key produced by removing position i from both.
        for i in 0..n {
            let variant: String = chars
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &c)| c)
                .collect();
            keys.push(variant);
        }

        keys.sort();
        keys.dedup();
        keys
    }
}

// ── CameraTimeWindowKey ───────────────────────────────────────────────────────

/// Groups passages by camera identifier and a fixed-width time window.
///
/// Key format: `"{camera_id}:{date}:{slot}"` where `slot = (hour*60 + min) / window_mins`.
/// Useful for detecting duplicate sensor reads of the same vehicle at the same
/// camera location within a short interval.
pub struct CameraTimeWindowKey {
    camera_field: String,
    time_field: String,
    window_mins: u32,
}

impl CameraTimeWindowKey {
    pub fn new(camera_field: &str, time_field: &str, window_mins: u32) -> Self {
        Self {
            camera_field: camera_field.into(),
            time_field: time_field.into(),
            window_mins,
        }
    }
}

fn time_to_slot(datetime: &str, window: u32) -> Option<u32> {
    let t_idx = datetime.find('T')?;
    let time_part = &datetime[t_idx + 1..];
    let mut parts = time_part.splitn(3, ':');
    let hour: u32 = parts.next()?.parse().ok()?;
    let minute: u32 = parts.next()?.parse().ok()?;
    Some((hour * 60 + minute) / window)
}

impl BlockingKey for CameraTimeWindowKey {
    fn name(&self) -> &str {
        "cam_time_window"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let cam_cow = record.field_as_str(&self.camera_field);
        let cam = match cam_cow.as_deref() {
            Some(s) => s,
            None => return vec![],
        };
        let ts_cow = record.field_as_str(&self.time_field);
        let ts = match ts_cow.as_deref() {
            Some(s) => s,
            None => return vec![],
        };
        let date = ts.get(..10).unwrap_or("");
        let slot = match time_to_slot(ts, self.window_mins) {
            Some(s) => s,
            None => return vec![],
        };
        vec![format!("{}:{}:{}", cam, date, slot)]
    }
}

// ── GeoGridKey ────────────────────────────────────────────────────────────────

/// Groups records by rounding geographic coordinates to a fixed grid cell.
///
/// `grid_size = 0.01` (degrees) ≈ 1 km. Useful for clustering passages near
/// the same highway camera position or street intersection.
pub struct GeoGridKey {
    lat_field: String,
    lon_field: String,
    grid_size: f64,
}

impl GeoGridKey {
    pub fn new(lat_field: &str, lon_field: &str, grid_size: f64) -> Self {
        Self {
            lat_field: lat_field.into(),
            lon_field: lon_field.into(),
            grid_size,
        }
    }
}

impl BlockingKey for GeoGridKey {
    fn name(&self) -> &str {
        "geo_grid"
    }

    fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
        let lat = match record.field_as::<f64>(&self.lat_field) {
            Some(v) => v,
            None => return vec![],
        };
        let lon = match record.field_as::<f64>(&self.lon_field) {
            Some(v) => v,
            None => return vec![],
        };
        let lat_cell = (lat / self.grid_size).floor() as i64;
        let lon_cell = (lon / self.grid_size).floor() as i64;
        vec![format!("{}:{}", lat_cell, lon_cell)]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{
        record::FieldValue,
        schema::{FieldKind, SchemaBuilder},
    };

    fn schema() -> Schema {
        SchemaBuilder::new()
            .field("kenteken", FieldKind::LicensePlate)
            .field("camera_id", FieldKind::Categorical)
            .field("tijdstip", FieldKind::Timestamp)
            .field("lat", FieldKind::GpsCoordinate)
            .field("lon", FieldKind::GpsCoordinate)
            .build()
            .unwrap()
    }

    fn rec(id: u64, kenteken: &str, camera: &str, ts: &str, lat: &str, lon: &str) -> Record {
        Record::new(id)
            .insert("kenteken", FieldValue::Text(kenteken.into()))
            .insert("camera_id", FieldValue::Text(camera.into()))
            .insert("tijdstip", FieldValue::Text(ts.into()))
            .insert("lat", FieldValue::Text(lat.into()))
            .insert("lon", FieldValue::Text(lon.into()))
    }

    // ── LicensePlateNormKey

    #[test]
    fn plate_norm_strips_hyphens() {
        let schema = schema();
        let key = LicensePlateNormKey::new("kenteken");
        let r = rec(
            1,
            "25-XKL-9",
            "CAM-A1-001",
            "2025-01-01T10:00:00",
            "52.3",
            "4.9",
        );
        let keys = key.extract(&r, &schema);
        assert_eq!(keys, vec!["25XKL9"]);
    }

    #[test]
    fn plate_norm_empty_field_returns_empty() {
        let schema = schema();
        let key = LicensePlateNormKey::new("kenteken");
        let r = Record::new(1);
        assert!(key.extract(&r, &schema).is_empty());
    }

    // ── PlateOCRFuzzyKey

    #[test]
    fn ocr_fuzzy_original_and_confused_share_key() {
        let schema = schema();
        let key = PlateOCRFuzzyKey::new("kenteken");

        // True plate "CX-180-W" → normalized "CX180W"
        let true_r = rec(
            1,
            "CX-180-W",
            "CAM-A1-001",
            "2025-01-01T10:00:00",
            "52.3",
            "4.9",
        );
        // OCR plate  "CX-I80-W" (1→I confusion) → normalized "CXI80W"
        let ocr_r = rec(
            2,
            "CX-I80-W",
            "CAM-A1-001",
            "2025-01-01T10:00:00",
            "52.3",
            "4.9",
        );

        let true_keys: std::collections::HashSet<String> =
            key.extract(&true_r, &schema).into_iter().collect();
        let ocr_keys: std::collections::HashSet<String> =
            key.extract(&ocr_r, &schema).into_iter().collect();

        let shared: Vec<_> = true_keys.intersection(&ocr_keys).collect();
        assert!(
            !shared.is_empty(),
            "true plate and OCR plate must share at least one fuzzy key; true={true_keys:?}, ocr={ocr_keys:?}"
        );
    }

    #[test]
    fn ocr_fuzzy_emits_multiple_variants() {
        let schema = schema();
        let key = PlateOCRFuzzyKey::new("kenteken");
        // "L01A4" has 5 chars → original + 5 deletion keys = 6 distinct keys
        let r = rec(1, "L01A4", "CAM", "2025-01-01T08:00:00", "52.0", "4.0");
        let keys = key.extract(&r, &schema);
        assert!(
            keys.len() >= 4,
            "should emit original + deletion variants; got {keys:?}"
        );
        assert!(
            keys.contains(&"L01A4".to_string()),
            "original key must be present"
        );
        // Deletion-neighbourhood keys: each char removed once
        assert!(
            keys.contains(&"01A4".to_string()),
            "deletion at pos 0 (L) expected"
        );
        assert!(
            keys.contains(&"L0A4".to_string()),
            "deletion at pos 2 (1) expected"
        );
    }

    // ── CameraTimeWindowKey

    #[test]
    fn camera_time_window_same_slot() {
        let schema = schema();
        let key = CameraTimeWindowKey::new("camera_id", "tijdstip", 10);

        let r1 = rec(1, "X", "CAM-A1-001", "2025-06-01T14:02:00", "52.0", "4.0");
        let r2 = rec(2, "Y", "CAM-A1-001", "2025-06-01T14:08:00", "52.0", "4.0");
        // 14:02 → slot 84  (84*10=840 min = 14h00m); 14:08 → slot 84  (same)
        assert_eq!(key.extract(&r1, &schema), key.extract(&r2, &schema));
    }

    #[test]
    fn camera_time_window_different_slot() {
        let schema = schema();
        let key = CameraTimeWindowKey::new("camera_id", "tijdstip", 10);

        let r1 = rec(1, "X", "CAM-A1-001", "2025-06-01T14:02:00", "52.0", "4.0");
        let r2 = rec(2, "Y", "CAM-A1-001", "2025-06-01T14:12:00", "52.0", "4.0");
        // 14:02 → slot 84; 14:12 → slot 85
        assert_ne!(key.extract(&r1, &schema), key.extract(&r2, &schema));
    }

    // ── GeoGridKey

    #[test]
    fn geo_grid_nearby_records_share_key() {
        let schema = schema();
        let key = GeoGridKey::new("lat", "lon", 0.01);

        let r1 = rec(1, "X", "CAM", "2025-01-01T10:00:00", "52.345", "4.901");
        let r2 = rec(2, "Y", "CAM", "2025-01-01T10:00:00", "52.349", "4.907");
        // Both land in the same 0.01° cell (lat 5234, lon 490)
        assert_eq!(key.extract(&r1, &schema), key.extract(&r2, &schema));
    }

    #[test]
    fn geo_grid_distant_records_differ() {
        let schema = schema();
        let key = GeoGridKey::new("lat", "lon", 0.01);

        let r1 = rec(1, "X", "CAM", "2025-01-01T10:00:00", "52.345", "4.901");
        let r2 = rec(2, "Y", "CAM", "2025-01-01T10:00:00", "51.922", "4.479");
        assert_ne!(key.extract(&r1, &schema), key.extract(&r2, &schema));
    }
}
