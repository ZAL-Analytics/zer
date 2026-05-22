//! End-to-end entity resolution pipeline: ingestion, blocking, comparison, scoring, and clustering.

pub mod batch;
pub mod cluster_view;
pub mod config;
pub mod ingester;
pub mod pipeline;
pub mod progress;
pub mod rate;

pub use batch::BatchReport;
pub use cluster_view::{ClusterIter, ClusterView, LinkedPair};
pub use config::{LinkMode, PipelineConfig, RateConfig};
pub use ingester::{IngestResult, Ingester};
pub use pipeline::{label_source, Pipeline, PipelineBuilder};
pub use progress::PipelineEvent;
