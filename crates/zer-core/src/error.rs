#[derive(Debug, thiserror::Error)]
pub enum ZerError {
    #[error("schema has no fields")]
    EmptySchema,

    #[error("field '{0}' not found in schema")]
    UnknownField(String),

    #[error("schema mismatch: expected {expected} fields, got {got}")]
    SchemaMismatch { expected: usize, got: usize },

    #[error("model params not fitted, run estimate_params() first")]
    NotFitted,

    #[error("store error: {0}")]
    Store(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("gpu error: {0}")]
    Gpu(String),

    #[error("judge error: {0}")]
    Judge(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
