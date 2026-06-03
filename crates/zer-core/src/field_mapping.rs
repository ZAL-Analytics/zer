use crate::record::FieldName;

/// Declares how one field from source A maps to one field from source B.
///
/// Used when the two sources have structurally different schemas (e.g. BRP
/// `voornamen` maps to SIS `name`) and the comparator needs explicit guidance about
/// which field on each side should be compared.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FieldMapping {
    pub a_field: FieldName,
    pub b_field: FieldName,
    pub null_policy: NullPolicy,
}

impl FieldMapping {
    pub fn new(a_field: impl Into<FieldName>, b_field: impl Into<FieldName>) -> Self {
        Self {
            a_field: a_field.into(),
            b_field: b_field.into(),
            null_policy: NullPolicy::Skip,
        }
    }

    pub fn with_null_policy(mut self, policy: NullPolicy) -> Self {
        self.null_policy = policy;
        self
    }
}

/// How the comparator treats a field pair where one or both values are absent.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullPolicy {
    /// Return `ComparisonLevel::Null` (255), EM skips this field for the pair.
    #[default]
    Skip,
    /// Return `ComparisonLevel::None` (0), treated as a hard non-match signal.
    PenaliseAbsence,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_null_policy_is_skip() {
        let m = FieldMapping::new("voornamen", "name");
        assert_eq!(m.null_policy, NullPolicy::Skip);
    }

    #[test]
    fn null_policy_round_trips_json() {
        let m =
            FieldMapping::new("geboortedatum", "dob").with_null_policy(NullPolicy::PenaliseAbsence);
        let json = serde_json::to_string(&m).unwrap();
        let back: FieldMapping = serde_json::from_str(&json).unwrap();
        assert_eq!(back.null_policy, NullPolicy::PenaliseAbsence);
        assert_eq!(back.a_field, "geboortedatum");
        assert_eq!(back.b_field, "dob");
    }
}
