/// Error type for the `zer-judge` crate.
#[derive(Debug, thiserror::Error)]
pub enum JudgeError {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("ORT session error: {0}")]
    Session(String),

    #[error("model inference error: {0}")]
    Inference(String),

    #[error("record not found in store: id={0}")]
    RecordNotFound(u64),

    #[error("judge worker thread disconnected")]
    WorkerDisconnected,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<JudgeError> for zer_core::error::ZerError {
    fn from(e: JudgeError) -> Self {
        zer_core::error::ZerError::Judge(e.to_string())
    }
}

impl From<ort::Error> for JudgeError {
    fn from(e: ort::Error) -> Self {
        JudgeError::Session(e.to_string())
    }
}
