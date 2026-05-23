Custom Entity Store
====================

By default zer persists resolved entity clusters in ``ZalEntityStore``, a
SQLite-backed store that writes a ``.zes`` file to disk. That is a good fit for
batch pipelines, but some use cases need something different:

* A **graph database** (SurrealDB, Neo4j) so downstream services can query
  entity membership in real time.
* A **shared cache** (Redis) so multiple pipeline workers can coordinate.
* A **no-op store** for testing, where you only care about the returned
  ``ClusterResult`` and never persist it.

This guide shows how to implement the ``EntityStore`` trait and wire it into
the pipeline.

The ``EntityStore`` trait
--------------------------

The trait lives in ``zer_cluster``. You need to implement three methods:

.. code-block:: rust

   use zer_cluster::{EntityStore, ClusterResult};

   pub trait EntityStore: Send + Sync + 'static {
       type Error: std::error::Error + Send + Sync + 'static;

       /// Called once per resolved cluster. ``result`` contains the cluster
       /// ID, the set of member RecordIds, and the representative RecordId.
       fn upsert_cluster(&self, result: &ClusterResult) -> Result<(), Self::Error>;

       /// Called when two previously separate clusters are found to be the
       /// same entity and must be merged. ``survivor`` is the ID that lives on.
       fn merge_clusters(
           &self,
           absorbed: ClusterResult,
           survivor: ClusterResult,
       ) -> Result<(), Self::Error>;

       /// Called after a run_batch completes. Flush any buffered writes.
       fn flush(&self) -> Result<(), Self::Error>;
   }

``upsert_cluster`` is called for every new or updated cluster after each
``run_batch``. ``merge_clusters`` is called when the clusterer discovers two
clusters that were previously separate now belong to the same entity
(possible when a new batch introduces bridging records).

SurrealDB example
------------------

This example streams entity clusters into a SurrealDB instance so other
services can query entity membership via the SurrealDB REST or WebSocket API.

Add the dependency:

.. code-block:: toml

   [dependencies]
   zer          = { version = "1.0", features = ["pipeline"] }
   surrealdb    = { version = "2" }
   tokio        = { version = "1", features = ["full"] }

Implement the store:

.. code-block:: rust

   use std::sync::Arc;
   use surrealdb::{Surreal, engine::remote::ws::Client};
   use surrealdb::sql::Thing;
   use zer_cluster::{EntityStore, ClusterResult};

   pub struct SurrealEntityStore {
       db: Arc<Surreal<Client>>,
   }

   impl SurrealEntityStore {
       pub async fn connect(url: &str) -> anyhow::Result<Self> {
           let db = Surreal::new::<surrealdb::engine::remote::ws::Ws>(url).await?;
           db.use_ns("zer").use_db("entities").await?;
           Ok(Self { db: Arc::new(db) })
       }
   }

   impl EntityStore for SurrealEntityStore {
       type Error = surrealdb::Error;

       fn upsert_cluster(&self, result: &ClusterResult) -> Result<(), Self::Error> {
           let db = Arc::clone(&self.db);
           let id = result.cluster_id.to_string();
           let members: Vec<u64> = result.member_ids.iter().copied().collect();
           let repr = result.representative_id;

           // Block on the async SurrealDB call. In a fully async store you
           // would return a Future instead; see the streaming guide for that.
           tokio::task::block_in_place(|| {
               tokio::runtime::Handle::current().block_on(async {
                   let _: Option<serde_json::Value> = db
                       .upsert(("entity", id.as_str()))
                       .content(serde_json::json!({
                           "members":         members,
                           "representative":  repr,
                           "updated_at":      chrono::Utc::now().to_rfc3339(),
                       }))
                       .await?;
                   Ok::<_, surrealdb::Error>(())
               })
           })?;
           Ok(())
       }

       fn merge_clusters(
           &self,
           absorbed: ClusterResult,
           survivor: ClusterResult,
       ) -> Result<(), Self::Error> {
           // Delete the absorbed cluster record; the survivor will be upserted
           // via a follow-up upsert_cluster call from the clusterer.
           let db = Arc::clone(&self.db);
           let absorbed_id = absorbed.cluster_id.to_string();
           tokio::task::block_in_place(|| {
               tokio::runtime::Handle::current().block_on(async {
                   let _: Option<serde_json::Value> = db
                       .delete(("entity", absorbed_id.as_str()))
                       .await?;
                   Ok::<_, surrealdb::Error>(())
               })
           })?;
           self.upsert_cluster(&survivor)
       }

       fn flush(&self) -> Result<(), Self::Error> {
           // SurrealDB writes are immediate; nothing to flush.
           Ok(())
       }
   }

Wire it into the pipeline:

.. code-block:: rust

   use zer_pipeline::{config::PipelineConfig, pipeline::Pipeline};

   #[tokio::main]
   async fn main() -> anyhow::Result<()> {
       let store = SurrealEntityStore::connect("ws://localhost:8000").await?;

       let pipeline = Pipeline::builder()
           .schema(schema)
           .store(store)
           .build()?;

       let report = pipeline.run_batch(records).await?;
       println!("entities written to SurrealDB: {}", report.entities_created);
       Ok(())
   }

After the run you can query entity membership in SurrealDB:

.. code-block:: sql

   -- all records belonging to entity "42"
   SELECT members FROM entity:42;

   -- which entity does record 1337 belong to?
   SELECT id FROM entity WHERE members CONTAINS 1337;

No-op store for testing
------------------------

When you only care about the ``PipelineReport`` and the in-memory
``ClusterView``, use a no-op store to skip all persistence:

.. code-block:: rust

   use zer_cluster::{EntityStore, ClusterResult};

   pub struct NoOpStore;

   impl EntityStore for NoOpStore {
       type Error = std::convert::Infallible;
       fn upsert_cluster(&self, _: &ClusterResult) -> Result<(), Self::Error> { Ok(()) }
       fn merge_clusters(&self, _: ClusterResult, _: ClusterResult) -> Result<(), Self::Error> { Ok(()) }
       fn flush(&self) -> Result<(), Self::Error> { Ok(()) }
   }

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(NoOpStore)
       .build()?;

Error handling
---------------

The associated ``Error`` type must implement ``std::error::Error + Send + Sync``.
The pipeline wraps it in ``ZerError::Store`` and surfaces it as the ``Err``
variant of ``run_batch``. Handle it like any other pipeline error:

.. code-block:: rust

   match pipeline.run_batch(records).await {
       Ok(report)  => println!("done: {} entities", report.entities_created),
       Err(e)      => eprintln!("pipeline error: {e}"),
   }

What to explore next
---------------------

* :doc:`custom-record-store`, replace the in-memory record store used by the neural judge.
* :doc:`streaming-pipeline`, keep the pipeline running and continuously write new clusters to your store.
* :doc:`/how-to/neural-judge`, the neural judge also reads from ``VecRecordStore``,swap it here if you need persistence.
