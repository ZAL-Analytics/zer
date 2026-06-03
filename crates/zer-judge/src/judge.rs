/// Channel-based DeBERTa judge.
///
/// `DebertaJudge` owns a background `std::thread` that holds the ORT
/// `OnnxSession`.  `adjudicate()` is synchronous and works from both sync
/// and async contexts because the ORT session never touches the tokio runtime.
///
/// # Deadlock avoidance
///
/// Running a blocking ORT session inside `tokio::spawn` or directly inside
/// `async fn Pipeline::run_batch()` blocks all tokio worker threads and
/// deadlocks.  The solution is a dedicated `std` thread that owns the session
/// and receives work through a `std::sync::mpsc` channel.  `adjudicate()` is
/// entirely synchronous and can be called from any context.
use std::sync::{mpsc, Arc};

use zer_core::{
    schema::Schema,
    scoring::ScoredPair,
    traits::{Judge, JudgeVerdict, RecordStore},
};

use crate::{
    audit::{AuditEntry, AuditLog},
    backend::JudgeBackend,
    calibration::CalibrationTable,
    error::JudgeError,
    serialize::serialize_pair,
    session::OnnxSession,
    spec::JudgeModelSpec,
    tokenize::JudgeTokenizer,
};

// ── Wire types ────────────────────────────────────────────────────────────────

struct InferRequest {
    texts: Vec<String>,
    reply_tx: mpsc::SyncSender<Result<Vec<f32>, JudgeError>>,
}

// ── Worker guard ───────────────────────────────────────────────────────────────

/// Owns the worker `JoinHandle` and joins it on drop.
///
/// Wrapped in `Arc` inside `DebertaJudge` so that clones share it; the last
/// clone to be dropped is the one that actually joins.  Field ordering in
/// `DebertaJudge` ensures `work_tx` (and therefore the mpsc channel) is closed
/// before this guard is dropped, so the worker thread has already exited its
/// receive loop before `join()` is called.
struct WorkerGuard(Option<std::thread::JoinHandle<()>>);

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            // Ignore panics in the worker, we just need it to finish so that
            // the ORT/TRT session is fully dropped before process teardown.
            let _ = handle.join();
        }
    }
}

// ── Worker thread ──────────────────────────────────────────────────────────────

fn worker_loop(
    mut session: OnnxSession,
    tokenizer: JudgeTokenizer,
    rx: mpsc::Receiver<InferRequest>,
) {
    for req in rx {
        let result = (|| {
            let (ids, mask, types) = tokenizer.encode_batch(&req.texts)?;
            session.run_batch(&ids, &mask, &types)
        })();
        let _ = req.reply_tx.send(result);
    }
}

// ── DebertaJudge ──────────────────────────────────────────────────────────────

/// Configuration for `DebertaJudge`.
#[derive(Clone)]
pub struct DebertaJudgeConfig {
    /// Entailment probability above which the pair is promoted.
    pub promote_threshold: f32,
    /// Entailment probability below which the pair is demoted.
    pub demote_threshold: f32,
    /// Maximum number of pairs sent to the ORT worker in a single call.
    ///
    /// Chunking keeps memory usage bounded and lets the caller observe tracing
    /// progress between chunks.  Defaults to 64.
    pub batch_size: usize,
    /// Bayesian calibration applied to matched/rejected pairs.
    pub calibration: CalibrationTable,
    /// Optional audit log, if `None`, no audit trail is written.
    pub audit_log: Option<Arc<AuditLog>>,
}

impl Default for DebertaJudgeConfig {
    fn default() -> Self {
        Self {
            promote_threshold: 0.6,
            demote_threshold: 0.35,
            batch_size: 64,
            calibration: CalibrationTable::default(),
            audit_log: None,
        }
    }
}

/// ORT-based neural judge that uses a DeBERTa (or MiniLM) NLI model to
/// adjudicate borderline record pairs.
///
/// Cloning a `DebertaJudge` is cheap, it shares the underlying ORT worker
/// thread via `SyncSender` clone; no second session or thread is created.
#[derive(Clone)]
pub struct DebertaJudge {
    // IMPORTANT: `work_tx` must be declared before `_worker` so it is dropped
    // first.  Dropping `work_tx` closes the mpsc channel, which causes the
    // worker thread to exit its receive loop.  Only then does `_worker`'s Arc
    // decrement; when it reaches zero `WorkerGuard::drop` joins the thread,
    // ensuring the ORT/TRT session is fully released before process teardown.
    work_tx: mpsc::SyncSender<InferRequest>,
    _worker: Arc<WorkerGuard>,
    record_store: Arc<dyn RecordStore>,
    schema: Schema,
    config: DebertaJudgeConfig,
}

impl DebertaJudge {
    /// Build a `DebertaJudge`, spawning the background ORT worker thread.
    pub fn new(
        spec: &dyn JudgeModelSpec,
        backend: &JudgeBackend,
        record_store: Arc<dyn RecordStore>,
        schema: Schema,
        config: DebertaJudgeConfig,
    ) -> Result<Self, JudgeError> {
        let session = OnnxSession::from_spec(spec, backend)?;
        let tokenizer = JudgeTokenizer::from_spec(spec)?;

        let (work_tx, work_rx) = mpsc::sync_channel::<InferRequest>(32);

        let model_name = spec.name().to_owned();
        let handle = std::thread::Builder::new()
            .name(format!("zer-judge[{model_name}]"))
            .spawn(move || worker_loop(session, tokenizer, work_rx))
            .map_err(JudgeError::Io)?;

        let judge = Self {
            work_tx,
            _worker: Arc::new(WorkerGuard(Some(handle))),
            record_store,
            schema,
            config,
        };

        // Trigger CUDA kernel JIT during new() rather than first adjudicate() call.
        // Failure is silently ignored, the judge still works, just with a slower first inference.
        let _ = judge.send_batch(vec![
            "COL voornamen VAL Jan COL achternaam VAL Jansen COL geboortedatum VAL 1985-01-01 \
             [SEP] COL voornamen VAL Janna COL achternaam VAL Jansen COL geboortedatum VAL 1985-06-15"
                .to_string(),
        ]);
        tracing::info!(model = %model_name, "ORT warm-up complete");

        Ok(judge)
    }

    fn send_batch(&self, texts: Vec<String>) -> Result<Vec<f32>, JudgeError> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.work_tx
            .send(InferRequest { texts, reply_tx })
            .map_err(|_| JudgeError::WorkerDisconnected)?;
        reply_rx
            .recv()
            .map_err(|_| JudgeError::WorkerDisconnected)?
    }
}

impl Judge for DebertaJudge {
    fn adjudicate(&self, pairs: &[ScoredPair]) -> zer_core::traits::Result<Vec<JudgeVerdict>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // Build pair texts, look up full records from the store.
        let mut texts = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let a = self
                .record_store
                .get(pair.record_a)
                .ok_or(JudgeError::RecordNotFound(pair.record_a))?;
            let b = self
                .record_store
                .get(pair.record_b)
                .ok_or(JudgeError::RecordNotFound(pair.record_b))?;
            texts.push(serialize_pair(&a, &b, &self.schema));
        }

        // Run inference in chunks to bound memory and allow progress tracing.
        let mut probs = Vec::with_capacity(texts.len());
        let batch_size = self.config.batch_size.max(1);
        for (chunk_idx, chunk) in texts.chunks(batch_size).enumerate() {
            tracing::debug!(
                chunk = chunk_idx,
                chunk_size = chunk.len(),
                total = texts.len(),
                "judge inference chunk"
            );
            let chunk_probs = self
                .send_batch(chunk.to_vec())
                .map_err(zer_core::error::ZerError::from)?;
            probs.extend(chunk_probs);
        }

        let mut verdicts = Vec::with_capacity(pairs.len());
        for (idx, (pair, prob)) in pairs.iter().zip(probs.iter()).enumerate() {
            let verdict = if *prob >= self.config.promote_threshold {
                JudgeVerdict::IncreaseConfidence
            } else if *prob <= self.config.demote_threshold {
                JudgeVerdict::DecreaseConfidence
            } else {
                JudgeVerdict::NoChange
            };

            if let Some(log) = &self.config.audit_log {
                let verdict_str = match &verdict {
                    JudgeVerdict::IncreaseConfidence => "increase",
                    JudgeVerdict::DecreaseConfidence => "decrease",
                    JudgeVerdict::NoChange => "no_change",
                };
                log.append(&AuditEntry {
                    record_a: pair.record_a,
                    record_b: pair.record_b,
                    pair_text: texts[idx].clone(),
                    match_probability: pair.match_probability,
                    entailment_score: *prob,
                    verdict: verdict_str,
                });
            }

            verdicts.push(verdict);
        }

        Ok(verdicts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deberta_judge_config_clone() {
        let cfg = DebertaJudgeConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.promote_threshold, cfg.promote_threshold);
        assert_eq!(cloned.demote_threshold, cfg.demote_threshold);
        assert_eq!(cloned.batch_size, cfg.batch_size);
    }

    #[test]
    fn deberta_judge_config_defaults() {
        let cfg = DebertaJudgeConfig::default();
        assert_eq!(cfg.promote_threshold, 0.6);
        assert_eq!(cfg.demote_threshold, 0.35);
        assert_eq!(cfg.batch_size, 64);
        assert!(cfg.audit_log.is_none());
    }

    #[test]
    fn deberta_judge_config_batch_size_custom() {
        let cfg = DebertaJudgeConfig {
            batch_size: 16,
            ..Default::default()
        };
        assert_eq!(cfg.batch_size, 16);
        assert_eq!(cfg.promote_threshold, 0.6);
    }

    #[test]
    fn deberta_judge_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // Compile-time check, fails to compile if DebertaJudge is not Send+Sync
        assert_send_sync::<DebertaJudge>();
    }

    #[test]
    fn deberta_judge_config_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DebertaJudgeConfig>();
    }

    #[test]
    fn batch_size_one_is_clamped_to_one() {
        let cfg = DebertaJudgeConfig {
            batch_size: 0,
            ..Default::default()
        };
        // The adjudicate() impl uses batch_size.max(1) to prevent zero-chunk loops.
        assert_eq!(cfg.batch_size.max(1), 1);
    }
}
