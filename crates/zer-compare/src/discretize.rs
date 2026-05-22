use zer_core::{comparison::ComparisonLevel, schema::FieldKind};

/// Configurable per-field thresholds for mapping a float similarity score to a `ComparisonLevel`.
///
/// The defaults are tuned per `FieldKind` based on the expected noise distribution
/// for that field type in Dutch law enforcement data.
#[derive(Debug, Clone)]
pub struct LevelThresholds {
    /// Similarity >= this gives `ComparisonLevel::Exact`.
    pub exact: f32,
    /// Similarity >= this (and < exact) gives `ComparisonLevel::Close`.
    pub close: f32,
    /// Similarity >= this (and < close) gives `ComparisonLevel::Partial`.
    pub partial: f32,
    // similarity < partial gives `ComparisonLevel::None`
}

impl LevelThresholds {
    /// Default thresholds tuned per `FieldKind`.
    pub fn for_kind(kind: FieldKind) -> Self {
        match kind {
            FieldKind::Name        => Self { exact: 0.92, close: 0.75, partial: 0.50 },
            FieldKind::Date
            | FieldKind::Timestamp => Self { exact: 0.99, close: 0.85, partial: 0.60 },
            FieldKind::Phone       => Self { exact: 0.98, close: 0.90, partial: 0.70 },
            FieldKind::Address     => Self { exact: 0.90, close: 0.70, partial: 0.40 },
            FieldKind::Id          => Self { exact: 0.99, close: 0.90, partial: 0.75 },
            FieldKind::LicensePlate => Self { exact: 0.99, close: 0.75, partial: 0.50 },
            FieldKind::Numeric
            | FieldKind::GpsCoordinate => Self { exact: 0.95, close: 0.80, partial: 0.50 },
            FieldKind::Categorical => Self { exact: 1.00, close: 0.95, partial: 0.70 },
            FieldKind::FreeText    => Self { exact: 0.90, close: 0.65, partial: 0.35 },
            FieldKind::Alias       => Self { exact: 0.90, close: 0.65, partial: 0.35 },
        }
    }

    /// Map a raw similarity score to a `ComparisonLevel`.
    pub fn apply(&self, sim: f32) -> ComparisonLevel {
        if sim >= self.exact        { ComparisonLevel::Exact   }
        else if sim >= self.close   { ComparisonLevel::Close   }
        else if sim >= self.partial { ComparisonLevel::Partial }
        else                        { ComparisonLevel::None    }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_thresholds_produce_correct_levels() {
        let t = LevelThresholds::for_kind(FieldKind::Name);
        assert_eq!(t.apply(0.95), ComparisonLevel::Exact);
        assert_eq!(t.apply(0.80), ComparisonLevel::Close);
        assert_eq!(t.apply(0.60), ComparisonLevel::Partial);
        assert_eq!(t.apply(0.30), ComparisonLevel::None);
    }

    #[test]
    fn categorical_is_binary() {
        let t = LevelThresholds::for_kind(FieldKind::Categorical);
        assert_eq!(t.apply(1.00), ComparisonLevel::Exact);
        assert_eq!(t.apply(0.99), ComparisonLevel::Close);
        assert_eq!(t.apply(0.00), ComparisonLevel::None);
    }

    #[test]
    fn date_thresholds_tight_bands() {
        let t = LevelThresholds::for_kind(FieldKind::Date);
        // 1.0 → Exact (same day)
        assert_eq!(t.apply(1.0),  ComparisonLevel::Exact);
        // 0.9 → Close (off by 1 day)
        assert_eq!(t.apply(0.9),  ComparisonLevel::Close);
        // 0.75 → Partial (same month)
        assert_eq!(t.apply(0.75), ComparisonLevel::Partial);
        // 0.3 → None (age-compatible only)
        assert_eq!(t.apply(0.3),  ComparisonLevel::None);
    }

    #[test]
    fn boundary_values_are_exclusive_on_lower_bound() {
        let t = LevelThresholds::for_kind(FieldKind::Name);
        // Exactly at the exact threshold → Exact
        assert_eq!(t.apply(t.exact),       ComparisonLevel::Exact);
        // One epsilon below exact → Close
        assert_eq!(t.apply(t.exact - 0.01), ComparisonLevel::Close);
    }
}
