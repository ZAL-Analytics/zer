//! Connected-components clustering and entity storage for resolved groups.

mod clusterer;
mod graph;
mod provenance;
mod threshold;
mod store;

pub use clusterer::ConnectedComponentsClusterer;
pub use graph::{ClusterConfig, ClusterGraph};
pub use provenance::ResolutionEvent;
pub use store::ZalEntityStore;
pub use threshold::{partition_by_band, BandedPairs};
