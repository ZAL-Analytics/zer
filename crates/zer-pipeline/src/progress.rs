/// Structured progress events emitted by [`crate::pipeline::Pipeline::run_batch`].
///
/// Attach a sender via [`crate::pipeline::PipelineBuilder::progress`].  The pipeline
/// fires events at each stage boundary using unbounded, fire-and-forget sends, a
/// slow or disconnected receiver never blocks the pipeline.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// Blocking-index construction has started.
    BlockingStarted { total_records: usize },

    /// Blocking is complete; the candidate-pair count is known.
    CandidatesReady {
        candidate_pairs: usize,
        cross_source:    usize,
        within_source:   usize,
    },

    /// Pairwise field comparison is about to begin.
    ComparingPairs { candidate_pairs: usize },

    /// EM parameter estimation has started.
    EmStarted { startup_mode: String, max_iterations: usize },

    /// EM estimation finished.
    EmComplete { iterations: usize },

    /// Scoring finished; counts reflect the state *before* any judge pass.
    ScoringComplete {
        auto_matched:  usize,
        borderline:    usize,
        auto_rejected: usize,
    },

    /// The neural judge is about to adjudicate borderline pairs.
    JudgeStarted { borderline: usize },

    /// The neural judge has finished adjudicating.
    JudgeComplete { promoted: usize, demoted: usize },

    /// Connected-components clustering and entity persistence are in progress.
    PersistingEntities,

    /// `run_batch` completed successfully.
    Done { elapsed_ms: u64 },
}
