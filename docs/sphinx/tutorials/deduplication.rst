Tutorial: Person Deduplication
===============================

This tutorial deduplicates a synthetic Dutch BRP (Basisregistratie Personen)
dataset containing deliberate duplicates: the same person registered multiple
times with name variants, typos, and address differences. zer finds them without
any labelled training data.

The full runnable demo lives in ``demos/person_deduplication/``.

What you will build
--------------------

A pipeline that:

1. Defines a ``Schema`` for BRP person fields.
2. Reads a CSV of ~500 synthetic Dutch person records.
3. Runs ``LinkMode::Deduplicate`` to find within-source duplicate pairs.
4. Evaluates precision, recall, and F1 against a ground-truth file.
5. Prints a cluster tree showing which records were grouped together.

Prepare the data
-----------------

The demo reads from ``data/v1.1/demos/persons/``. Download the full dataset bundle
(all tutorials share the same download) as described in
:doc:`/introduction/installation`, or regenerate locally:

.. code-block:: bash

   $ python data_generator/generate_demo_persons.py
   # Writes:   data/v1.1/demos/persons/records.csv
   #           data/v1.1/demos/persons/ground_truth.csv

Define the schema
------------------

The schema tells zer how to interpret each field. ``FieldKind::Name`` triggers
phonetic blocking and Jaro-Winkler + token-overlap similarity. ``FieldKind::Date``
triggers year-based phonetic blocking and exact/partial date comparison.

.. code-block:: rust

   use zer_core::schema::{FieldKind, SchemaBuilder};

   let schema = SchemaBuilder::new()
       .field("voornamen",     FieldKind::Name)
       .field("achternaam",    FieldKind::Name)
       .field("geboortedatum", FieldKind::Date)
       .field("geslacht",      FieldKind::FreeText)
       .field("straatnaam",    FieldKind::FreeText)
       .field("postcode",      FieldKind::FreeText)
       .field("woonplaats",    FieldKind::FreeText)
       .build()?;

Load records from CSV
----------------------

zer records are plain structs, load them however suits your project.
Here we use the ``csv`` crate:

.. code-block:: rust

   #[derive(serde::Deserialize)]
   struct PersonRow {
       bsn:           String,
       voornamen:     String,
       tussenvoegsel: String,
       achternaam:    String,
       geboortedatum: String,
       geslacht:      String,
       straatnaam:    String,
       huisnummer:    String,
       postcode:      String,
       woonplaats:    String,
   }

   fn load_records(path: &Path) -> Vec<Record> {
       csv::Reader::from_path(path)
           .unwrap()
           .deserialize::<PersonRow>()
           .map(|r| {
               let row = r.unwrap();
               Record::from_key("brp", &row.bsn)
                   .insert("voornamen",     row.voornamen)
                   .insert("achternaam",    row.achternaam)
                   .insert("geboortedatum", row.geboortedatum)
                   .insert("geslacht",      row.geslacht)
                   .insert("straatnaam",    row.straatnaam)
                   .insert("postcode",      row.postcode)
                   .insert("woonplaats",    row.woonplaats)
           })
           .collect()
   }

Build and run the pipeline
---------------------------

.. code-block:: rust

   use zer_cluster::ZalEntityStore;
   use zer_pipeline::{Pipeline, PipelineConfig};

   let store    = ZalEntityStore::open_in_memory()?;
   let pipeline = Pipeline::builder()
       .schema(schema)
       .store(store)
       .config(PipelineConfig {
           registry_path: "/tmp/demo_persons.zsm".into(),
           ..PipelineConfig::default()
       })
       .build()?;

   let records = load_records(Path::new("data/v1.1/demos/persons/records.csv"));
   let report  = pipeline.run_batch(records).await?;

   println!("total records   : {}", report.total_records);
   println!("candidate pairs : {}", report.candidate_pairs);
   println!("auto-matched    : {}", report.auto_matched);
   println!("borderline      : {}", report.borderline);
   println!("auto-rejected   : {}", report.auto_rejected);
   println!("entities created: {}", report.entities_created);
   println!("elapsed         : {} ms", report.elapsed_ms);

Typical output on the synthetic BRP dataset::

   total records   : 500
   candidate pairs : 1,842
   auto-matched    : 187
   borderline      : 34
   auto-rejected   : 1,621
   entities created: 213
   elapsed         : 148 ms

Evaluate against ground truth
-------------------------------

.. code-block:: rust

   #[derive(serde::Deserialize)]
   struct GroundTruthRow { bsn_a: String, bsn_b: String }

   let ground_truth: HashSet<(String, String)> =
       csv::Reader::from_path("data/v1.1/demos/persons/ground_truth.csv")
           .unwrap()
           .deserialize::<GroundTruthRow>()
           .map(|r| {
               let row = r.unwrap();
               // canonical order so (a,b) == (b,a)
               if row.bsn_a <= row.bsn_b {
                   (row.bsn_a, row.bsn_b)
               } else {
                   (row.bsn_b, row.bsn_a)
               }
           })
           .collect();

   // Collect all within-cluster pairs as predictions
   let view = pipeline.cluster_view();
   let mut predicted: HashSet<(String, String)> = HashSet::new();
   for (entity, _) in &view {
       let keys: Vec<&str> = entity.members.iter().map(|m| m.record_key.as_str()).collect();
       for i in 0..keys.len() {
           for j in (i + 1)..keys.len() {
               let (a, b) = if keys[i] <= keys[j] {
                   (keys[i].to_string(), keys[j].to_string())
               } else {
                   (keys[j].to_string(), keys[i].to_string())
               };
               predicted.insert((a, b));
           }
       }
   }

   let tp  = predicted.intersection(&ground_truth).count();
   let fp  = predicted.difference(&ground_truth).count();
   let fn_ = ground_truth.difference(&predicted).count();

   let precision = tp as f64 / (tp + fp) as f64;
   let recall    = tp as f64 / (tp + fn_) as f64;
   let f1        = 2.0 * precision * recall / (precision + recall);

   println!("precision: {:.3}", precision);
   println!("recall:    {:.3}", recall);
   println!("F1:        {:.3}", f1);

Inspect the cluster tree
-------------------------

Resolved clusters show which records were grouped together and their best
match scores:

.. code-block:: rust

   let clusters: Vec<_> = view.into_iter()
       .filter(|(entity, _)| entity.members.len() > 1)
       .collect();

   for (entity, records) in clusters.iter().take(10) {
       println!("Entity {} ({} members):", entity.id, records.len());
       for (member, record) in entity.members.iter().zip(records) {
           println!(
               "  record {:>5}  score={:.3}  {}",
               member.record_id,
               member.score,
               record.text("achternaam").unwrap_or(""),
           );
       }
   }

Example output::

   Entity 42 (3 members):
     record   101  score=0.971  van den Berg
     record   102  score=0.943  Berg
     record   103  score=0.918  v/d Berg

Run the full demo
------------------

.. code-block:: bash

   $ cargo run -p person_deduplication

What to explore next
---------------------

* :doc:`cross-source-linkage`, link two separate datasets instead of deduplicating one.
* :doc:`/how-to/blocking-strategy`, understand why the pipeline found (or missed) specific pairs.
* :doc:`/explanation/blocking-recall`, the theory behind recall vs. precision in blocking.
