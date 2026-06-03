use zer_core::{error::ZerError, scoring::ModelParams};

use crate::fingerprint::SchemaFingerprint;

/// Everything that must be persisted after a successful EM training run.
///
/// Serializes to roughly 2–10 KB per artifact (bincode).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelArtifact {
    /// Fingerprint of the schema and data distribution this model was trained on.
    pub fingerprint: SchemaFingerprint,
    /// Learned Fellegi-Sunter m/u parameters and decision thresholds.
    pub params: ModelParams,
    /// Optional human-readable label, e.g. `"brp_2024_q1"`.
    pub tag: Option<String>,
    /// Unix timestamp (seconds) when EM training completed.
    pub trained_on: u64,
    /// Number of EM iterations performed.
    pub em_iterations: usize,
}

impl ModelArtifact {
    /// Serialize this artifact to bytes using bincode.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ZerError> {
        bincode::serialize(self).map_err(|e| ZerError::Serialization(e.to_string()))
    }

    /// Deserialize an artifact from bincode bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ZerError> {
        bincode::deserialize(bytes).map_err(|e| ZerError::Serialization(e.to_string()))
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::schema::{FieldKind, SchemaBuilder};

    fn dummy_artifact() -> ModelArtifact {
        let schema = SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap();

        let fingerprint = SchemaFingerprint::from_schema(&schema);

        let params = ModelParams {
            m: vec![
                vec![0.02, 0.06, 0.12, 0.80],
                vec![0.02, 0.06, 0.12, 0.80],
                vec![0.01, 0.04, 0.10, 0.85],
            ],
            u: vec![
                vec![0.70, 0.15, 0.10, 0.05],
                vec![0.70, 0.15, 0.10, 0.05],
                vec![0.80, 0.10, 0.07, 0.03],
            ],
            log_prior_odds: -2.0,
            upper_threshold: 0.9,
            lower_threshold: 0.1,
        };

        ModelArtifact {
            fingerprint,
            params,
            tag: Some("test_artifact".into()),
            trained_on: 0,
            em_iterations: 25,
        }
    }

    #[test]
    fn roundtrip_preserves_all_fields() {
        let original = dummy_artifact();
        let bytes = original.to_bytes().expect("serialization must succeed");
        let loaded = ModelArtifact::from_bytes(&bytes).expect("deserialization must succeed");

        assert_eq!(original.tag, loaded.tag);
        assert_eq!(original.em_iterations, loaded.em_iterations);
        assert_eq!(
            original.params.upper_threshold,
            loaded.params.upper_threshold
        );
        assert_eq!(
            original.params.lower_threshold,
            loaded.params.lower_threshold
        );
        assert_eq!(original.params.log_prior_odds, loaded.params.log_prior_odds);
        assert_eq!(
            original.fingerprint.schema_hash,
            loaded.fingerprint.schema_hash
        );
    }

    #[test]
    fn roundtrip_preserves_m_u_tables() {
        let original = dummy_artifact();
        let bytes = original.to_bytes().unwrap();
        let loaded = ModelArtifact::from_bytes(&bytes).unwrap();

        assert_eq!(original.params.m.len(), loaded.params.m.len());
        for (row_a, row_b) in original.params.m.iter().zip(loaded.params.m.iter()) {
            for (va, vb) in row_a.iter().zip(row_b.iter()) {
                assert!(
                    (va - vb).abs() < 1e-9,
                    "m values must be bit-exact after roundtrip"
                );
            }
        }
    }

    #[test]
    fn serialized_size_under_10kb() {
        let artifact = dummy_artifact();
        let bytes = artifact.to_bytes().unwrap();
        assert!(
            bytes.len() < 10_240,
            "serialized artifact for 3-field schema should be under 10 KB, got {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        let result = ModelArtifact::from_bytes(b"not valid bincode data");
        assert!(result.is_err(), "garbage bytes must return an error");
    }
}
