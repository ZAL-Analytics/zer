use std::{
    collections::{HashMap, HashSet},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use zer_blocking::InvertedIndex;
use zer_core::{
    entity::{EntityId},
    error::ZerError,
    record::{Record, RecordId},
    record_pool::RecordPool,
    scoring::{MatchBand, ModelParams},
    traits::Judge,
};
use zer_schema::{ModelArtifact, SchemaFingerprint, StartupMode as RegistryStartupMode};

use crate::{config::{BatchStartupMode, LinkMode}, pipeline::Pipeline, progress::PipelineEvent};

// Construct and send the event only when a receiver is actually attached.
// This prevents event allocation in the common case where `progress` is None.
macro_rules! emit {
    ($pipeline:expr, $event:expr) => {
        if let Some(tx) = &$pipeline.progress { let _ = tx.send($event); }
    };
}

/// Summary statistics produced by [`Pipeline::run_batch`].
#[derive(Debug, Clone)]
pub struct BatchReport {
    pub total_records:    usize,
    pub candidate_pairs:  usize,
    pub auto_matched:     usize,
    pub borderline:       usize,
    pub auto_rejected:    usize,
    pub judge_promoted:   usize,
    pub judge_demoted:    usize,
    pub entities_created: usize,
    pub entities_updated: usize,
    pub em_iterations:    usize,
    pub startup_mode:     BatchStartupMode,
    /// Wall-clock milliseconds from the start of `run_batch` to the final persist.
    pub elapsed_ms:       u64,
    /// Wall-clock milliseconds spent inside the judge's `adjudicate()` call only.
    /// Zero when no judge is configured or when the borderline set is empty.
    pub judge_elapsed_ms: u64,
    /// The linking mode used for this run.
    pub link_mode:            LinkMode,
    /// Number of candidate pairs where the two records have different source labels.
    pub cross_source_pairs:   usize,
    /// Number of candidate pairs where both records share the same source label
    /// (or neither carries a source label).
    pub within_source_pairs:  usize,
    /// All scored pairs `(record_a, record_b, match_probability)`, populated
    /// only when the `SCORED_PAIR_COLLECT=1` env var is set.  Empty otherwise.
    pub scored_pairs: Vec<(RecordId, RecordId, f32)>,
}

impl Pipeline {
    /// Process a batch of records: block, compare, EM-estimate, score, cluster,
    /// and persist.  Returns a [`BatchReport`] with counts for each stage.
    pub async fn run_batch(&self, records: Vec<Record>) -> Result<BatchReport, ZerError> {
        let t0 = Instant::now();

        if records.is_empty() {
            return Ok(BatchReport {
                total_records:    0,
                candidate_pairs:  0,
                auto_matched:     0,
                borderline:       0,
                auto_rejected:    0,
                judge_promoted:   0,
                judge_demoted:    0,
                entities_created: 0,
                entities_updated: 0,
                em_iterations:    0,
                startup_mode:     BatchStartupMode::ColdStart,
                elapsed_ms:       0,
                judge_elapsed_ms: 0,
                link_mode:           self.config.link_mode,
                cross_source_pairs:  0,
                within_source_pairs: 0,
                scored_pairs:        Vec::new(),
            });
        }

        // 1. Persist all records to the record store before processing
        for record in &records {
            self.record_store.insert(record.clone());
        }

        // 2. Fingerprint schema against a sample of records
        let sample_end  = records.len().min(1_000);
        let fingerprint = SchemaFingerprint::from_sample(&self.schema, &records[..sample_end]);
        let startup_mode = self.registry.lookup_startup_mode(&fingerprint)?;
        let startup_kind = match &startup_mode {
            RegistryStartupMode::WarmLoad(_)      => BatchStartupMode::WarmLoad,
            RegistryStartupMode::WarmStart { .. } => BatchStartupMode::WarmStart,
            RegistryStartupMode::ColdStart        => BatchStartupMode::ColdStart,
        };
        tracing::info!(startup_mode = ?startup_kind, records = records.len(), "run_batch started");

        // 3. Build blocking index and record-id → pool-index map
        let _span_blocking = tracing::info_span!("blocking", records = records.len()).entered();
        emit!(self, PipelineEvent::BlockingStarted { total_records: records.len() });
        let mut index:     InvertedIndex          = InvertedIndex::new();
        let mut id_to_idx: HashMap<RecordId, usize> = HashMap::with_capacity(records.len());
        for (pos, record) in records.iter().enumerate() {
            id_to_idx.insert(record.id, pos);
            self.blocker.index_record(record, &self.schema, &mut index);
        }

        // Log how many buckets exceeded the cap so users can tune if needed.
        let cap = self.config.max_bucket_size;
        if cap > 0 {
            let skipped = index.oversized_buckets(cap);
            if skipped > 0 {
                tracing::warn!(
                    max_bucket_size = cap,
                    skipped_buckets = skipped,
                    "blocking buckets exceeded cap and will be skipped (too broad to be useful)"
                );
            }
        }

        // 4. Generate canonical (i < j) candidate pairs, deduplicated.
        // all_pairs visits each bucket once and sort+deduplicates, faster than
        // a per-record lookup loop that allocates a HashSet per call.
        let mut pair_indices = index.all_pairs(&id_to_idx, cap);
        if self.config.link_mode == LinkMode::LinkOnly {
            pair_indices.retain(|&(a, b)| {
                records[a].source.as_deref() != records[b].source.as_deref()
            });
        }
        let candidate_pairs = pair_indices.len();

        // Count cross-source vs within-source pairs for the report.
        let mut cross_source_pairs  = 0usize;
        let mut within_source_pairs = 0usize;
        for &(a, b) in &pair_indices {
            if records[a].source.as_deref() != records[b].source.as_deref() {
                cross_source_pairs += 1;
            } else {
                within_source_pairs += 1;
            }
        }
        emit!(self, PipelineEvent::CandidatesReady {
            candidate_pairs,
            cross_source:  cross_source_pairs,
            within_source: within_source_pairs,
        });

        // 4. Batch comparison: mapped path for cross-schema, pool path otherwise.
        drop(_span_blocking);
        let _span_compare = tracing::info_span!("compare", pairs = candidate_pairs).entered();
        emit!(self, PipelineEvent::ComparingPairs { candidate_pairs });
        let batch = match &self.mapped_comparator {
            Some(mapped_cmp) => mapped_cmp.compare_batch_mapped(
                &records,
                &pair_indices,
                &self.config.field_mappings,
            ),
            None => {
                let pool = RecordPool::from_records(&records, &self.schema);
                self.comparator.compare_batch_from_pool(&pool, &pair_indices, &self.schema)
            }
        };

        // 5. EM: WarmLoad = skip, WarmStart = few iterations, ColdStart = full
        drop(_span_compare);
        let _span_em = tracing::info_span!("em").entered();
        let (params, em_iterations) = match startup_mode {
            RegistryStartupMode::WarmLoad(artifact) => {
                tracing::debug!("warm load, using saved params, skipping EM");
                emit!(self, PipelineEvent::EmStarted {
                    startup_mode:   "WarmLoad".into(),
                    max_iterations: 0,
                });
                emit!(self, PipelineEvent::EmComplete { iterations: 0 });
                (artifact.params, 0)
            }
            RegistryStartupMode::WarmStart { artifact, distance } => {
                tracing::debug!(distance, "warm start, fine-tuning saved params");
                emit!(self, PipelineEvent::EmStarted {
                    startup_mode:   "WarmStart".into(),
                    max_iterations: self.config.em_max_iter_warm,
                });
                if batch.n_pairs == 0 {
                    emit!(self, PipelineEvent::EmComplete { iterations: 0 });
                    (artifact.params, 0)
                } else {
                    let p = self.scorer.estimate_params(
                        &batch,
                        Some(artifact.params),
                        self.config.em_max_iter_warm,
                    )?;
                    emit!(self, PipelineEvent::EmComplete { iterations: self.config.em_max_iter_warm });
                    (p, self.config.em_max_iter_warm)
                }
            }
            RegistryStartupMode::ColdStart => {
                tracing::debug!("cold start, initializing from priors");
                emit!(self, PipelineEvent::EmStarted {
                    startup_mode:   "ColdStart".into(),
                    max_iterations: self.config.em_max_iter_cold,
                });
                let n_fields = if self.mapped_comparator.is_some() {
                    self.config.field_mappings.len()
                } else {
                    self.schema.fields.len()
                };
                if batch.n_pairs == 0 {
                    emit!(self, PipelineEvent::EmComplete { iterations: 0 });
                    (default_params(n_fields), 0)
                } else {
                    let p = self.scorer.estimate_params(
                        &batch,
                        None,
                        self.config.em_max_iter_cold,
                    )?;
                    emit!(self, PipelineEvent::EmComplete { iterations: self.config.em_max_iter_cold });
                    (p, self.config.em_max_iter_cold)
                }
            }
        };

        // 6. Apply optional threshold overrides before scoring.
        let params = {
            let mut p = params;
            if let Some(upper) = self.config.upper_threshold {
                p.upper_threshold = upper;
            }
            if let Some(lower) = self.config.lower_threshold {
                p.lower_threshold = lower;
            }
            p
        };

        // 6. Score all candidate pairs
        drop(_span_em);
        let _span_score = tracing::info_span!("score", pairs = candidate_pairs).entered();
        let mut scored = self.scorer.score_batch(&batch, &params);

        // 7. Count bands before applying the judge
        let mut auto_matched  = 0usize;
        let mut borderline    = 0usize;
        let mut auto_rejected = 0usize;
        for sp in &scored {
            match sp.band {
                MatchBand::AutoMatch  => auto_matched  += 1,
                MatchBand::Borderline => borderline    += 1,
                MatchBand::AutoReject => auto_rejected += 1,
            }
        }
        emit!(self, PipelineEvent::ScoringComplete { auto_matched, borderline, auto_rejected });

        // 8. Optional judge pass on borderlines
        drop(_span_score);
        let _span_judge = tracing::info_span!("judge", borderline).entered();
        if self.judge.is_some() {
            emit!(self, PipelineEvent::JudgeStarted { borderline });
        }
        let t_judge = Instant::now();
        let (judge_promoted, judge_demoted) = if let Some(judge) = &self.judge {
            let result = apply_judge(&mut scored, judge.as_ref())?;
            emit!(self, PipelineEvent::JudgeComplete { promoted: result.0, demoted: result.1 });
            result
        } else {
            (0, 0)
        };
        let judge_elapsed_ms = t_judge.elapsed().as_millis() as u64;

        // Collect scored pairs AFTER the judge so that judge-promoted/demoted borderlines
        // are reflected in the effective probability used for PR-AUC / optimal-threshold
        // computation in zer-bench.  For non-judge runs the bands are unchanged so
        // `max`/`min` are no-ops for auto_match/auto_reject pairs.
        let scored_pairs: Vec<(RecordId, RecordId, f32)> =
            if cfg!(feature = "collect-pairs") {
                scored.iter().map(|sp| {
                    let eff_prob = match sp.band {
                        MatchBand::AutoMatch  => sp.match_probability.max(params.upper_threshold),
                        MatchBand::AutoReject => sp.match_probability.min(params.lower_threshold),
                        MatchBand::Borderline => sp.match_probability,
                    };
                    (sp.record_a, sp.record_b, eff_prob)
                }).collect()
            } else {
                Vec::new()
            };

        // 9. Cluster using connected components
        drop(_span_judge);
        let _span_cluster = tracing::info_span!("cluster_and_persist").entered();
        emit!(self, PipelineEvent::PersistingEntities);
        let mut entities = self.clusterer.cluster(&scored, &params);

        // Enrich entity members with source labels, the clusterer doesn't carry
        // per-record metadata, so we fill it in here from the input records.
        let id_to_source: HashMap<RecordId, Option<String>> =
            records.iter().map(|r| (r.id, r.source.clone())).collect();
        for entity in &mut entities {
            for member in &mut entity.members {
                if let Some(src) = id_to_source.get(&member.record_id) {
                    member.source = src.clone();
                }
            }
        }

        // 10. Persist entities, counting new vs merged
        let mut entities_created = 0usize;
        let mut entities_updated = 0usize;
        let mut seen_entity_ids:  HashSet<EntityId> = HashSet::new();
        for entity in &entities {
            let id = self.store.upsert_entity(entity)?;
            if seen_entity_ids.insert(id) {
                entities_created += 1;
            } else {
                entities_updated += 1;
            }
        }

        // 11. Persist trained artifact so subsequent runs can warm-start
        let artifact = ModelArtifact {
            fingerprint,
            params,
            tag:           None,
            trained_on:    unix_secs(),
            em_iterations,
        };
        self.registry.save(&artifact)?;

        let elapsed_ms = t0.elapsed().as_millis() as u64;

        tracing::info!(
            entities_created,
            entities_updated,
            auto_matched,
            borderline,
            elapsed_ms,
            "run_batch complete"
        );
        emit!(self, PipelineEvent::Done { elapsed_ms });

        Ok(BatchReport {
            total_records: records.len(),
            candidate_pairs,
            auto_matched,
            borderline,
            auto_rejected,
            judge_promoted,
            judge_demoted,
            entities_created,
            entities_updated,
            em_iterations,
            startup_mode: startup_kind,
            elapsed_ms,
            judge_elapsed_ms,
            link_mode:           self.config.link_mode,
            cross_source_pairs,
            within_source_pairs,
            scored_pairs,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub(crate) fn default_params(n_fields: usize) -> ModelParams {
    // m/u priors from Fellegi-Sunter (1969): m≈0.9 for match-typical fields,
    // u≈0.05–0.10 for rare agreement under non-match.  Skewed toward high
    // specificity so cold-start EM converges without labelled data.
    ModelParams {
        m:               vec![vec![0.01, 0.04, 0.10, 0.85]; n_fields],
        u:               vec![vec![0.70, 0.15, 0.10, 0.05]; n_fields],
        log_prior_odds:  -2.0,
        upper_threshold: 0.85,
        lower_threshold: 0.15,
    }
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn apply_judge(
    scored: &mut Vec<zer_core::scoring::ScoredPair>,
    judge:  &dyn Judge,
) -> Result<(usize, usize), ZerError> {
    use zer_core::traits::JudgeVerdict;

    let borderline_indices: Vec<usize> = scored.iter()
        .enumerate()
        .filter(|(_, p)| p.band == MatchBand::Borderline)
        .map(|(i, _)| i)
        .collect();

    if borderline_indices.is_empty() {
        return Ok((0, 0));
    }

    let borderlines: Vec<_> = borderline_indices.iter().map(|&i| scored[i].clone()).collect();
    let verdicts = judge.adjudicate(&borderlines)?;
    let mut promoted = 0usize;
    let mut demoted  = 0usize;

    for (&idx, verdict) in borderline_indices.iter().zip(verdicts) {
        match verdict {
            JudgeVerdict::IncreaseConfidence => { scored[idx].band = MatchBand::AutoMatch;   promoted += 1; }
            JudgeVerdict::DecreaseConfidence => { scored[idx].band = MatchBand::AutoReject;  demoted  += 1; }
            _                                => {}
        }
    }

    Ok((promoted, demoted))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use zer_cluster::ZalEntityStore;
    use zer_core::{
        record::FieldValue,
        schema::{FieldKind, SchemaBuilder},
    };

    use crate::{config::PipelineConfig, pipeline::Pipeline};

    fn person_schema() -> zer_core::schema::Schema {
        SchemaBuilder::new()
            .field("voornamen",     FieldKind::Name)
            .field("achternaam",    FieldKind::Name)
            .field("geboortedatum", FieldKind::Date)
            .build()
            .unwrap()
    }

    fn make_pipeline(dir: &TempDir) -> std::sync::Arc<Pipeline> {
        Pipeline::builder()
            .schema(person_schema())
            .store(ZalEntityStore::open_in_memory().unwrap())
            .config(PipelineConfig {
                registry_path: dir.path().join("test.zsm"),
                ..PipelineConfig::default()
            })
            .build()
            .unwrap()
    }

    fn make_record(id: u64, name: &str, last: &str, dob: &str) -> Record {
        Record::new(id)
            .insert("voornamen",     FieldValue::Text(name.into()))
            .insert("achternaam",    FieldValue::Text(last.into()))
            .insert("geboortedatum", FieldValue::Text(dob.into()))
    }

    #[tokio::test]
    async fn empty_batch_returns_zero_report() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let report   = pipeline.run_batch(vec![]).await.unwrap();
        assert_eq!(report.total_records,   0);
        assert_eq!(report.candidate_pairs, 0);
        assert_eq!(report.auto_matched,    0);
        assert_eq!(report.em_iterations,   0);
    }

    #[tokio::test]
    async fn single_record_no_candidates() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let record   = make_record(1, "Alice", "Smith", "1990-01-01");
        let report   = pipeline.run_batch(vec![record]).await.unwrap();
        assert_eq!(report.total_records,   1);
        assert_eq!(report.candidate_pairs, 0);
        assert_eq!(report.auto_matched,    0);
    }

    #[tokio::test]
    async fn duplicate_records_produce_candidates() {
        let dir     = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let records: Vec<Record> = (1..=5)
            .map(|i| make_record(i, "Jan", "de Vries", "1985-03-15"))
            .collect();
        let report = pipeline.run_batch(records).await.unwrap();
        assert_eq!(report.total_records, 5);
        assert!(report.candidate_pairs > 0, "identical records should block together");
    }

    #[tokio::test]
    async fn cold_start_label_on_fresh_registry() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let record   = make_record(1, "Alice", "Smith", "1990-01-01");
        let report   = pipeline.run_batch(vec![record]).await.unwrap();
        assert_eq!(report.startup_mode, BatchStartupMode::ColdStart);
    }

    #[tokio::test]
    async fn second_run_with_same_schema_warm_loads() {
        let dir = TempDir::new().unwrap();

        // First run, trains and saves params
        let pipeline1 = make_pipeline(&dir);
        let records1: Vec<Record> = (1..=10)
            .map(|i| make_record(i, "Test", "User", "1980-01-01"))
            .collect();
        let r1 = pipeline1.run_batch(records1).await.unwrap();
        assert_eq!(r1.startup_mode, BatchStartupMode::ColdStart);

        // Second run, same schema, same path → WarmLoad
        let pipeline2 = Pipeline::builder()
            .schema(person_schema())
            .store(ZalEntityStore::open_in_memory().unwrap())
            .config(PipelineConfig {
                registry_path: dir.path().join("test.zsm"),
                ..PipelineConfig::default()
            })
            .build()
            .unwrap();
        let records2: Vec<Record> = (100..=110)
            .map(|i| make_record(i, "Test", "User", "1980-01-01"))
            .collect();
        let r2 = pipeline2.run_batch(records2).await.unwrap();
        assert_eq!(r2.startup_mode,  BatchStartupMode::WarmLoad);
        assert_eq!(r2.em_iterations, 0);
    }

    #[tokio::test]
    async fn default_params_has_correct_shape() {
        let n     = 3;
        let p     = default_params(n);
        assert_eq!(p.m.len(), n);
        assert_eq!(p.u.len(), n);
        assert!(p.upper_threshold > p.lower_threshold);
    }

    fn make_pipeline_with_mode(dir: &TempDir, link_mode: crate::config::LinkMode) -> std::sync::Arc<Pipeline> {
        Pipeline::builder()
            .schema(person_schema())
            .store(ZalEntityStore::open_in_memory().unwrap())
            .config(PipelineConfig {
                registry_path: dir.path().join("test.zsm"),
                link_mode,
                ..PipelineConfig::default()
            })
            .build()
            .unwrap()
    }

    fn make_record_with_source(id: u64, name: &str, last: &str, dob: &str, source: &str) -> Record {
        make_record(id, name, last, dob).with_source(source)
    }

    #[tokio::test]
    async fn link_only_filters_within_source_pairs() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline_with_mode(&dir, crate::config::LinkMode::LinkOnly);
        // All records from the same source, LinkOnly must produce zero within-source pairs.
        let records: Vec<Record> = (1..=5)
            .map(|i| make_record_with_source(i, "Jan", "de Vries", "1985-03-15", "brp"))
            .collect();
        let report = pipeline.run_batch(records).await.unwrap();
        assert_eq!(report.within_source_pairs, 0, "LinkOnly must not produce within-source pairs");
    }

    #[tokio::test]
    async fn link_only_allows_cross_source_pairs() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline_with_mode(&dir, crate::config::LinkMode::LinkOnly);
        // Records from two different sources with same name, should produce cross-source pairs.
        let mut records = Vec::new();
        for i in 1..=3 {
            records.push(make_record_with_source(i, "Jan", "de Vries", "1985-03-15", "brp"));
        }
        for i in 4..=6 {
            records.push(make_record_with_source(i, "Jan", "de Vries", "1985-03-15", "kvk"));
        }
        let report = pipeline.run_batch(records).await.unwrap();
        assert!(report.cross_source_pairs > 0, "LinkOnly must produce cross-source pairs when sources differ");
        assert_eq!(report.within_source_pairs, 0, "LinkOnly must not produce within-source pairs");
    }

    #[tokio::test]
    async fn deduplicate_default_unchanged() {
        // Deduplicate (default) on records with no source labels, behaviour identical to pre-07c.
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline(&dir);
        let records: Vec<Record> = (1..=5)
            .map(|i| make_record(i, "Jan", "de Vries", "1985-03-15"))
            .collect();
        let report = pipeline.run_batch(records).await.unwrap();
        assert_eq!(report.link_mode, LinkMode::Deduplicate);
        // All pairs are within-source (no source labels → treated as same source)
        assert_eq!(report.cross_source_pairs, 0);
        assert!(report.candidate_pairs > 0);
        assert_eq!(report.within_source_pairs, report.candidate_pairs);
    }

    #[tokio::test]
    async fn link_and_dedupe_includes_all_pairs() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline_with_mode(&dir, crate::config::LinkMode::LinkAndDedupe);
        let mut records = Vec::new();
        for i in 1..=3 {
            records.push(make_record_with_source(i, "Jan", "de Vries", "1985-03-15", "brp"));
        }
        for i in 4..=6 {
            records.push(make_record_with_source(i, "Jan", "de Vries", "1985-03-15", "kvk"));
        }
        let report = pipeline.run_batch(records).await.unwrap();
        assert_eq!(report.link_mode, LinkMode::LinkAndDedupe);
        // Must include both within- and cross-source pairs
        assert!(report.within_source_pairs > 0, "LinkAndDedupe must include within-source pairs");
        assert!(report.cross_source_pairs  > 0, "LinkAndDedupe must include cross-source pairs");
    }

    #[tokio::test]
    async fn batch_report_pair_counts_sum_correctly() {
        let dir      = TempDir::new().unwrap();
        let pipeline = make_pipeline_with_mode(&dir, crate::config::LinkMode::LinkAndDedupe);
        let mut records = Vec::new();
        for i in 1..=4 {
            records.push(make_record_with_source(i, "Jan", "de Vries", "1985-03-15", "brp"));
        }
        for i in 5..=8 {
            records.push(make_record_with_source(i, "Jan", "de Vries", "1985-03-15", "kvk"));
        }
        let report = pipeline.run_batch(records).await.unwrap();
        assert_eq!(
            report.cross_source_pairs + report.within_source_pairs,
            report.candidate_pairs,
            "cross_source_pairs + within_source_pairs must equal candidate_pairs"
        );
    }
}
