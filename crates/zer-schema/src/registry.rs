use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Mutex,
};

use zer_core::error::ZerError;

use crate::{
    artifact::ModelArtifact,
    fingerprint::SchemaFingerprint,
    similarity::{fingerprint_distance, WARM_START_THRESHOLD},
};

const MAGIC: &[u8] = b"ZSM\x01";

/// Decides how the pipeline should initialize when a new dataset arrives.
#[derive(Debug)]
pub enum StartupMode {
    /// Schema hash matches exactly, skip EM and use the saved params directly.
    WarmLoad(ModelArtifact),
    /// Schema is similar (distance ≤ threshold), use saved params as the EM
    /// warm-start initializer and run 2–3 iterations to fine-tune.
    WarmStart {
        artifact: ModelArtifact,
        distance: f32,
    },
    /// Schema is new or too different, initialize from priors and run full EM.
    ColdStart,
}

struct RegistryInner {
    path: Option<PathBuf>,
    artifacts: HashMap<[u8; 32], ModelArtifact>,
}

/// Persistent store for trained [`ModelArtifact`]s.
///
/// Backed by a single portable `.zsm` binary file (`b"ZSM\x01"` magic +
/// bincode-serialized `HashMap`). The file is written atomically on every
/// mutation, a `.zsm.tmp` file is written first then renamed into place, so a
/// crash during flush can never leave a partially-written registry.
///
/// The registry is small in practice (< 1 000 entries), so nearest-neighbor
/// lookup performs a full linear scan without an index.
pub struct SchemaRegistry {
    inner: Mutex<RegistryInner>,
}

impl SchemaRegistry {
    /// Open (or create) a registry at the given `.zsm` file path.
    ///
    /// If the file does not exist yet it is created on the first [`Self::save`] call.
    pub fn open(path: &Path) -> Result<Self, ZerError> {
        let artifacts = load(path)?;
        Ok(Self {
            inner: Mutex::new(RegistryInner {
                path: Some(path.to_path_buf()),
                artifacts,
            }),
        })
    }

    /// Create an in-memory registry. No file I/O; data is lost on drop.
    #[cfg(test)]
    pub(crate) fn open_temporary() -> Result<Self, ZerError> {
        Ok(Self {
            inner: Mutex::new(RegistryInner {
                path: None,
                artifacts: HashMap::new(),
            }),
        })
    }

    // ── Write ────────────────────────────────────────────────────────────────

    /// Persist a trained model artifact. Overwrites any existing artifact with
    /// the same schema hash and atomically flushes to disk.
    pub fn save(&self, artifact: &ModelArtifact) -> Result<(), ZerError> {
        let mut inner = self.inner.lock().unwrap();
        inner
            .artifacts
            .insert(artifact.fingerprint.schema_hash, artifact.clone());
        flush(&inner)?;
        tracing::debug!(tag = artifact.tag.as_deref(), "saved model artifact");
        Ok(())
    }

    // ── Read ─────────────────────────────────────────────────────────────────

    /// Exact lookup by schema hash. Returns `None` if no matching artifact exists.
    pub fn get_exact(
        &self,
        fingerprint: &SchemaFingerprint,
    ) -> Result<Option<ModelArtifact>, ZerError> {
        let inner = self.inner.lock().unwrap();
        Ok(inner.artifacts.get(&fingerprint.schema_hash).cloned())
    }

    /// Nearest-neighbor lookup: returns the closest artifact and its distance.
    ///
    /// Performs a full linear scan, acceptable because the registry is expected
    /// to hold far fewer than 1 000 entries.
    ///
    /// Returns `None` when the registry is empty.
    pub fn get_nearest(
        &self,
        fingerprint: &SchemaFingerprint,
    ) -> Result<Option<(ModelArtifact, f32)>, ZerError> {
        let inner = self.inner.lock().unwrap();
        let best = inner
            .artifacts
            .values()
            .map(|a| {
                let dist = fingerprint_distance(fingerprint, &a.fingerprint);
                (a.clone(), dist)
            })
            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal));
        Ok(best)
    }

    /// Determine the startup mode for an incoming dataset given its fingerprint.
    ///
    /// ```text
    /// exact hash match         → WarmLoad   (skip EM entirely)
    /// distance ≤ 0.25          → WarmStart  (2–3 EM iterations from saved init)
    /// distance  > 0.25 / empty → ColdStart  (full EM from priors)
    /// ```
    pub fn lookup_startup_mode(
        &self,
        fingerprint: &SchemaFingerprint,
    ) -> Result<StartupMode, ZerError> {
        if let Some(exact) = self.get_exact(fingerprint)? {
            tracing::info!("exact schema match, warm load");
            return Ok(StartupMode::WarmLoad(exact));
        }

        match self.get_nearest(fingerprint)? {
            Some((artifact, dist)) if dist <= WARM_START_THRESHOLD => {
                tracing::info!(dist, "similar schema, warm start");
                Ok(StartupMode::WarmStart {
                    artifact,
                    distance: dist,
                })
            }
            _ => {
                tracing::info!("no suitable prior, cold start");
                Ok(StartupMode::ColdStart)
            }
        }
    }

    // ── Enumeration / deletion ────────────────────────────────────────────────

    /// Return all stored artifacts in arbitrary order.
    pub fn list_all(&self) -> Result<Vec<ModelArtifact>, ZerError> {
        let inner = self.inner.lock().unwrap();
        Ok(inner.artifacts.values().cloned().collect())
    }

    /// Delete the artifact for the given schema hash.
    ///
    /// Returns `true` if an artifact was found and removed, `false` otherwise.
    pub fn delete(&self, schema_hash: &[u8; 32]) -> Result<bool, ZerError> {
        let mut inner = self.inner.lock().unwrap();
        let removed = inner.artifacts.remove(schema_hash).is_some();
        if removed {
            flush(&inner)?;
        }
        Ok(removed)
    }
}

// ── File I/O ──────────────────────────────────────────────────────────────────

fn flush(inner: &RegistryInner) -> Result<(), ZerError> {
    let Some(path) = &inner.path else {
        return Ok(());
    };
    let payload =
        bincode::serialize(&inner.artifacts).map_err(|e| ZerError::Serialization(e.to_string()))?;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(MAGIC);
    buf.extend(payload);
    let tmp = path.with_extension("zsm.tmp");
    std::fs::write(&tmp, &buf).map_err(|e| ZerError::Store(e.to_string()))?;
    std::fs::rename(&tmp, path).map_err(|e| ZerError::Store(e.to_string()))?;
    Ok(())
}

fn load(path: &Path) -> Result<HashMap<[u8; 32], ModelArtifact>, ZerError> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let bytes = std::fs::read(path).map_err(|e| ZerError::Store(e.to_string()))?;
    if bytes.get(..4) != Some(MAGIC) {
        return Err(ZerError::Store("invalid .zsm magic".into()));
    }
    bincode::deserialize(&bytes[4..]).map_err(|e| ZerError::Serialization(e.to_string()))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zer_core::{
        schema::{FieldKind, SchemaBuilder},
        scoring::ModelParams,
    };

    use crate::{artifact::ModelArtifact, fingerprint::SchemaFingerprint};

    fn dummy_params(n_fields: usize) -> ModelParams {
        ModelParams {
            m: vec![vec![0.02, 0.06, 0.12, 0.80]; n_fields],
            u: vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
            log_prior_odds: -2.0,
            upper_threshold: 0.9,
            lower_threshold: 0.1,
        }
    }

    fn make_artifact(schema: &zer_core::schema::Schema, tag: &str) -> ModelArtifact {
        ModelArtifact {
            fingerprint: SchemaFingerprint::from_schema(schema),
            params: dummy_params(schema.len()),
            tag: Some(tag.into()),
            trained_on: 0,
            em_iterations: 25,
        }
    }

    fn brp_schema() -> zer_core::schema::Schema {
        SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .field("nationaliteit", FieldKind::Categorical)
            .field("postcode", FieldKind::Id)
            .build()
            .unwrap()
    }

    fn sim_schema() -> zer_core::schema::Schema {
        SchemaBuilder::new()
            .field("sim_id", FieldKind::Id)
            .field("msisdn", FieldKind::Phone)
            .field("imsi", FieldKind::Id)
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .field("nationaliteit", FieldKind::Categorical)
            .build()
            .unwrap()
    }

    #[test]
    fn roundtrip_save_and_get_exact() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        let schema = brp_schema();
        let artifact = make_artifact(&schema, "brp_test");

        registry.save(&artifact).unwrap();

        let fp = SchemaFingerprint::from_schema(&schema);
        let loaded = registry.get_exact(&fp).unwrap().unwrap();

        assert_eq!(loaded.tag.as_deref(), Some("brp_test"));
        assert_eq!(
            loaded.fingerprint.schema_hash,
            artifact.fingerprint.schema_hash
        );
        assert_eq!(
            loaded.params.upper_threshold,
            artifact.params.upper_threshold
        );
    }

    #[test]
    fn get_exact_returns_none_for_unknown_schema() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        let fp = SchemaFingerprint::from_schema(&brp_schema());
        let result = registry.get_exact(&fp).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn list_all_returns_all_artifacts() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        let brp = brp_schema();
        let sim = sim_schema();

        registry.save(&make_artifact(&brp, "brp")).unwrap();
        registry.save(&make_artifact(&sim, "sim")).unwrap();

        let all = registry.list_all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn delete_removes_artifact_and_returns_true() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        let schema = brp_schema();
        let artifact = make_artifact(&schema, "brp");
        registry.save(&artifact).unwrap();

        let removed = registry.delete(&artifact.fingerprint.schema_hash).unwrap();
        assert!(removed, "delete should return true when the key existed");

        let fp = SchemaFingerprint::from_schema(&schema);
        assert!(registry.get_exact(&fp).unwrap().is_none());
    }

    #[test]
    fn delete_returns_false_for_missing_key() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        let hash = [0u8; 32];
        assert!(!registry.delete(&hash).unwrap());
    }

    #[test]
    fn startup_mode_exact_match_is_warm_load() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        let schema = brp_schema();
        registry.save(&make_artifact(&schema, "brp")).unwrap();

        let fp = SchemaFingerprint::from_schema(&schema);
        let mode = registry.lookup_startup_mode(&fp).unwrap();

        assert!(
            matches!(mode, StartupMode::WarmLoad(_)),
            "exact schema match must return WarmLoad"
        );
    }

    #[test]
    fn startup_mode_added_field_is_warm_start() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        registry.save(&make_artifact(&brp_schema(), "brp")).unwrap();

        let extended = SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .field("nationaliteit", FieldKind::Categorical)
            .field("postcode", FieldKind::Id)
            .field("verblijfstitel", FieldKind::Categorical)
            .build()
            .unwrap();

        let fp = SchemaFingerprint::from_schema(&extended);
        let mode = registry.lookup_startup_mode(&fp).unwrap();

        assert!(
            matches!(mode, StartupMode::WarmStart { .. }),
            "one added field should return WarmStart"
        );
    }

    #[test]
    fn startup_mode_incompatible_schema_is_cold_start() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        registry.save(&make_artifact(&brp_schema(), "brp")).unwrap();

        let fp = SchemaFingerprint::from_schema(&sim_schema());
        let mode = registry.lookup_startup_mode(&fp).unwrap();

        assert!(
            matches!(mode, StartupMode::ColdStart),
            "BRP artifact vs SIM schema should return ColdStart"
        );
    }

    #[test]
    fn startup_mode_empty_registry_is_cold_start() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        let fp = SchemaFingerprint::from_schema(&brp_schema());
        assert!(matches!(
            registry.lookup_startup_mode(&fp).unwrap(),
            StartupMode::ColdStart
        ));
    }

    #[test]
    fn nearest_prefers_closer_artifact() {
        let registry = SchemaRegistry::open_temporary().unwrap();
        registry.save(&make_artifact(&brp_schema(), "brp")).unwrap();
        registry.save(&make_artifact(&sim_schema(), "sim")).unwrap();

        let brp_like = SchemaBuilder::new()
            .field("voornamen", FieldKind::Name)
            .field("achternaam", FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .field("nationaliteit", FieldKind::Categorical)
            .field("postcode", FieldKind::Id)
            .field("verblijfstitel", FieldKind::Categorical)
            .build()
            .unwrap();

        let (nearest, _dist) = registry
            .get_nearest(&SchemaFingerprint::from_schema(&brp_like))
            .unwrap()
            .expect("registry is not empty");

        assert_eq!(
            nearest.tag.as_deref(),
            Some("brp"),
            "BRP-like schema should match the BRP artifact, not SIM"
        );
    }
}
