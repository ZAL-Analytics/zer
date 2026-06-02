Custom Entity Store
====================

By default zer persists resolved entities in ``ZalEntityStore``, a
SQLite-backed store that writes a ``.zes`` file to disk. That is a good fit for
batch pipelines, but some use cases need something different:

* A **graph database** (SurrealDB, Neo4j) so downstream services can query
  entity membership in real time.
* A **shared cache** (Redis) so multiple pipeline workers can coordinate.
* A **no-op store** for testing, where you only care about the returned
  ``PipelineReport`` and never persist entities.

This guide shows how to implement the ``EntityStore`` trait and wire it into
the pipeline.

The ``EntityStore`` trait
--------------------------

The trait lives in ``zer_core::traits``. You need to implement four methods:

.. code-block:: rust

   use zer_core::traits::EntityStore;
   use zer_core::entity::{Entity, EntityId};
   use zer_core::record::RecordId;
   // Result<T> is std::result::Result<T, zer_core::error::ZerError>
   use zer_core::traits::Result;

   pub trait EntityStore: Send + Sync {
       /// Create or replace the stored entity. Returns the entity's ID.
       fn upsert_entity(&self, entity: &Entity) -> Result<EntityId>;

       /// Fetch a single entity by its ID.
       fn get_entity(&self, id: EntityId) -> Result<Entity>;

       /// Find which entity contains a given record, or None if not yet resolved.
       fn record_to_entity(&self, record_id: RecordId) -> Result<Option<EntityId>>;

       /// Return every entity in the store (used by cluster_view).
       fn all_entities(&self) -> Result<Vec<Entity>>;
   }

``Entity`` and its members are defined in ``zer_core::entity``:

.. code-block:: rust

   pub struct Entity {
       pub id:      EntityId,          // u64
       pub members: Vec<EntityMember>,
   }

   pub struct EntityMember {
       pub record_id: RecordId,        // u64
       pub score:     f32,
       pub method:    ResolutionMethod,
       pub source:    Option<String>,
   }

Wrap external errors as ``ZerError::Store(message.to_string())`` and return
them as ``Err``. The pipeline surfaces them as the ``Err`` variant of
``run_batch``.

SurrealDB example
------------------

This example streams resolved entities into a SurrealDB instance so other
services can query entity membership via the SurrealDB REST or WebSocket API.

Add the dependencies:

.. code-block:: toml

   [dependencies]
   zer          = { version = "1.0", features = ["pipeline"] }
   surrealdb    = { version = "2" }
   tokio        = { version = "1", features = ["full"] }

Implement the store:

.. code-block:: rust

   use std::sync::Arc;
   use surrealdb::{Surreal, engine::remote::ws::Client};
   use zer_core::entity::{Entity, EntityId};
   use zer_core::error::ZerError;
   use zer_core::record::RecordId;
   use zer_core::traits::{EntityStore, Result};

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
       fn upsert_entity(&self, entity: &Entity) -> Result<EntityId> {
           let db         = Arc::clone(&self.db);
           let id         = entity.id;
           // Use the Entity helper to collect member IDs
           let member_ids: Vec<u64> = entity.member_ids().collect();
           let scores: Vec<f32>     = entity.members.iter().map(|m| m.score).collect();

           // EntityStore is synchronous; bridge to the async SurrealDB client
           // with block_in_place so we do not block the async executor thread.
           tokio::task::block_in_place(|| {
               tokio::runtime::Handle::current().block_on(async {
                   let _: Option<serde_json::Value> = db
                       .upsert(("entity", id.to_string().as_str()))
                       .content(serde_json::json!({
                           "member_ids": member_ids,
                           "scores":     scores,
                       }))
                       .await
                       .map_err(|e| ZerError::Store(e.to_string()))?;
                   Ok::<_, ZerError>(())
               })
           })?;
           Ok(id)
       }

       fn get_entity(&self, id: EntityId) -> Result<Entity> {
           // A full implementation queries SurrealDB and reconstructs the
           // Entity + Vec<EntityMember> from the stored document.
           Err(ZerError::Store(format!("get_entity({id}) not implemented")))
       }

       fn record_to_entity(&self, _record_id: RecordId) -> Result<Option<EntityId>> {
           // Implement by querying for the entity whose member_ids contains
           // record_id.  Return Ok(None) when no entity is found.
           Ok(None)
       }

       fn all_entities(&self) -> Result<Vec<Entity>> {
           // Used by pipeline.cluster_view().  Implement with a SELECT * FROM entity
           // query if you need the ClusterView over a SurrealDB-backed store.
           Ok(vec![])
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

   -- all record IDs belonging to entity 42
   SELECT member_ids FROM entity:42;

   -- which entity contains record 1337?
   SELECT id FROM entity WHERE member_ids CONTAINS 1337;

No-op store for testing
------------------------

When you only care about the ``PipelineReport`` and the in-memory
``ClusterView``, use a no-op store to skip all persistence:

.. code-block:: rust

   use zer_core::entity::{Entity, EntityId};
   use zer_core::error::ZerError;
   use zer_core::record::RecordId;
   use zer_core::traits::{EntityStore, Result};

   pub struct NoOpStore;

   impl EntityStore for NoOpStore {
       fn upsert_entity(&self, entity: &Entity) -> Result<EntityId> { Ok(entity.id) }
       fn get_entity(&self, id: EntityId) -> Result<Entity> {
           Err(ZerError::Store(format!("no-op: get_entity({id})")))
       }
       fn record_to_entity(&self, _: RecordId) -> Result<Option<EntityId>> { Ok(None) }
       fn all_entities(&self) -> Result<Vec<Entity>> { Ok(vec![]) }
   }

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(NoOpStore)
       .build()?;

Error handling
---------------

Store errors are wrapped in ``ZerError::Store`` and surfaced as the ``Err``
variant of ``run_batch``. Handle them like any other pipeline error:

.. code-block:: rust

   match pipeline.run_batch(records).await {
       Ok(report)  => println!("done: {} entities", report.entities_created),
       Err(e)      => eprintln!("pipeline error: {e}"),
   }

What to explore next
---------------------

* :doc:`custom-record-store`, replace the in-memory record store used by the neural judge.
* :doc:`streaming-pipeline`, keep the pipeline running and continuously write new entities to your store.
* :doc:`/how-to/neural-judge`, the neural judge also reads from ``VecRecordStore``; swap it here if you need persistence.
