//! Core types and traits for the zer entity-resolution library.

pub mod comparison;
pub mod entity;
pub mod error;
pub mod field_mapping;
pub mod record;
pub mod record_pool;
pub mod schema;
pub mod scoring;
pub mod traits;

pub use field_mapping::{FieldMapping, NullPolicy};
pub use record::FromFieldValue;
pub use record_pool::RecordPool;
pub use traits::{IntoRecord, RecordStore, VecRecordStore};
