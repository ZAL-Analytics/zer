//! Blocking strategies and inverted index for candidate pair generation.

pub mod blocker;
pub mod factory;
pub mod index;
pub mod keys;
pub mod normalize;

pub use blocker::CompositeBlocker;
pub use factory::{BlockerFactory, CustomSchemaCategory, SchemaCategory};
pub use index::InvertedIndex;
