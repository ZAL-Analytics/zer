zer an Entity Resolution library for Dutch Administrative Data
==============================================================

**zer** is a Rust library for probabilistic entity resolution: finding which records
across different datasets refer to the same real-world person, vehicle, or entity, 
without a shared unique key.

It is designed for Dutch administrative data (BRP, KvK, SIS II, ANPR) but is
fully configurable for any domain.

.. code-block:: rust

   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(ZalEntityStore::open_in_memory()?)
       .config(PipelineConfig {
           link_mode: LinkMode::LinkOnly,
           ..PipelineConfig::default()
       })
       .build()?;

   let report = pipeline.run_batch(records).await?;
   println!("matched {} entities in {} ms", report.entities_created, report.elapsed_ms);

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Introduction
      :link: introduction/index
      :link-type: doc

      What zer is, how to install it, and a five-minute tour of the pipeline.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step walkthroughs: deduplication, cross-source linkage, ANPR
      matching, and multi-source pipelines.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: How-To Guides
      :link: how-to/index
      :link-type: doc

      Task-oriented recipes: define a schema, choose a blocking strategy,
      tune the scorer, use the neural judge, connect Polars, run on GPU.

   .. grid-item-card:: Explanation
      :link: explanation/index
      :link-type: doc

      Deep dives into entity resolution fundamentals, the Fellegi-Sunter
      model, Dutch name normalization, OCR confusion handling, and the
      ONNX judge internals.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Developers
      :link: developers/index
      :link-type: doc

      Extend zer for your own use cases: custom entity stores, record stores,
      blocking keys, similarity functions, and streaming pipelines.

   .. grid-item-card:: Contribution
      :link: contribution/index
      :link-type: doc

      Report bugs, run and share benchmarks, generate test datasets, and
      get in touch with the team.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      FieldKind table, blocking keys catalog, similarity functions,
      SchemaCategory presets, and benchmark results.

   .. grid-item-card:: API Docs
      :link: reference/api
      :link-type: doc

      Full rustdoc API reference for all zer crates.

.. toctree::
   :hidden:
   :maxdepth: 2

   introduction/index
   tutorials/index
   how-to/index
   explanation/index
   developers/index
   contribution/index
   reference/index
