Custom Similarity Functions
============================

The comparator evaluates each candidate pair by computing a similarity score
for every field in the schema. zer ships with a catalog of built-in functions
(Jaro-Winkler, exact match, date proximity, and others; see
:doc:`/reference/similarity-functions`). When none of those fit your domain
you can register your own.

Similarity scores flow directly into the Fellegi-Sunter EM model as match
weights, so getting the right metric for a field matters more than it might
appear. A poor similarity function for a high-weight field degrades EM
accuracy across the whole dataset.

The ``SimilarityFn`` trait
---------------------------

The trait lives in ``zer_compare::similarity``:

.. code-block:: rust

   use zer_compare::similarity::SimilarityFn;
   use zer_core::{record::FieldValue, schema::FieldKind};

   pub trait SimilarityFn: Send + Sync {
       /// Return a score in [0.0, 1.0].
       /// 1.0 = definite match, 0.0 = definite non-match.
       /// Return 0.0 when either value is ``FieldValue::Null`` or an unexpected
       /// variant; the EM model interprets low scores on missing fields correctly.
       fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32;

       /// The ``FieldKind`` this function is designed for.
       /// Used by ``FieldComparator`` to route fields to the right functions.
       fn field_kind(&self) -> FieldKind;
   }

The ``field_kind`` method declares which kind of field the function is intended
for. When you register the function via ``FieldComparator::with_fns``, this
value is informational; the field index you pass to ``with_fns`` determines
which schema field actually uses your function.

Double Metaphone for non-Dutch names
--------------------------------------

The built-in Dutch phonetic key works well for Dutch names but poorly for
Arabic, Slavic, or South Asian names that appear in Dutch administrative data.
`Double Metaphone <https://en.wikipedia.org/wiki/Metaphone#Double_Metaphone>`_
gives reasonable coverage across European and Middle Eastern names.

Add the dependency:

.. code-block:: toml

   [dependencies]
   double-metaphone = { version = "0.3" }

.. code-block:: rust

   use double_metaphone::double_metaphone;
   use zer_compare::similarity::SimilarityFn;
   use zer_core::{record::FieldValue, schema::FieldKind};

   pub struct DoubleMetaphoneSimilarity;

   impl SimilarityFn for DoubleMetaphoneSimilarity {
       fn field_kind(&self) -> FieldKind { FieldKind::Name }

       fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
           let (ta, tb) = match (a, b) {
               (FieldValue::Text(a), FieldValue::Text(b)) => (a.as_str(), b.as_str()),
               _ => return 0.0,
           };

           let (primary_a, alt_a) = double_metaphone(ta);
           let (primary_b, alt_b) = double_metaphone(tb);

           // Full match on either primary or alternate code
           let exact = primary_a == primary_b
               || primary_a == alt_b
               || alt_a == primary_b
               || alt_a == alt_b;

           if exact { 1.0 } else { 0.0 }
       }
   }

This function returns only 0.0 or 1.0. That is fine for Fellegi-Sunter: the
EM model learns separate match and non-match distributions from the data, so
a binary score is valid as long as it discriminates well.

Continuous edit-distance similarity
--------------------------------------

When you need a graded score (e.g. for numeric codes where single-digit
typos are common), implement a normalised edit distance:

.. code-block:: rust

   use zer_compare::similarity::SimilarityFn;
   use zer_core::{record::FieldValue, schema::FieldKind};

   pub struct NormalisedLevenshtein;

   impl SimilarityFn for NormalisedLevenshtein {
       fn field_kind(&self) -> FieldKind { FieldKind::Id }

       fn similarity(&self, a: &FieldValue, b: &FieldValue) -> f32 {
           let (ta, tb) = match (a, b) {
               (FieldValue::Text(a), FieldValue::Text(b)) => (a.as_str(), b.as_str()),
               _ => return 0.0,
           };
           if ta.is_empty() && tb.is_empty() { return 1.0; }

           let dist = levenshtein(ta, tb);           // any edit-distance crate
           let max  = ta.len().max(tb.len()) as f32;
           1.0 - dist as f32 / max
       }
   }

   fn levenshtein(a: &str, b: &str) -> usize {
       // Wagner-Fischer DP; replace with a crate like `edit-distance` in practice
       let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
       let (m, n) = (a.len(), b.len());
       let mut dp = vec![vec![0usize; n + 1]; m + 1];
       for i in 0..=m { dp[i][0] = i; }
       for j in 0..=n { dp[0][j] = j; }
       for i in 1..=m {
           for j in 1..=n {
               dp[i][j] = if a[i-1] == b[j-1] {
                   dp[i-1][j-1]
               } else {
                   1 + dp[i-1][j].min(dp[i][j-1]).min(dp[i-1][j-1])
               };
           }
       }
       dp[m][n]
   }

Registering a custom similarity function
------------------------------------------

Use ``FieldComparator::from_schema`` to build the default comparator, then
override individual fields with ``with_fns``. The index passed to ``with_fns``
is the zero-based field position as declared in the schema.

.. code-block:: rust

   use zer_core::schema::{FieldKind, SchemaBuilder};
   use zer_compare::FieldComparator;
   use zer_pipeline::Pipeline;

   let schema = SchemaBuilder::new()
       .field("voornamen",           FieldKind::Name)     // index 0
       .field("achternaam",          FieldKind::Name)     // index 1
       .field("naam_niet_latinized", FieldKind::Name)     // index 2
       .field("geboortedatum",       FieldKind::Date)     // index 3
       .build()?;

   // Override index 2 ("naam_niet_latinized") with the custom phonetic function.
   // All other fields keep their default functions derived from FieldKind.
   let comparator = FieldComparator::from_schema(&schema)
       .with_fns(2, vec![Box::new(DoubleMetaphoneSimilarity)]);

   let pipeline = Pipeline::builder()
       .schema(schema)
       .comparator(comparator)
       .store(store)
       .build()?;

Only the field at index 2 uses ``DoubleMetaphoneSimilarity``. Fields 0, 1,
and 3 keep the built-in functions selected for ``FieldKind::Name`` and
``FieldKind::Date`` respectively.

Calibrating a new similarity function
---------------------------------------

The EM model estimates match and non-match distributions from unlabelled data.
A new function takes effect immediately on the next ``run_batch``. To verify it
is contributing correctly, compare precision and recall against a labelled
ground truth before and after adding the function:

.. code-block:: rust

   // Without custom function: baseline run
   let report_baseline = pipeline_baseline.run_batch(records.clone()).await?;

   // With custom function at field index 2
   let report_custom = pipeline_custom.run_batch(records).await?;

   println!("baseline entities: {}", report_baseline.entities_created);
   println!("custom   entities: {}", report_custom.entities_created);

If precision and recall don't improve on your labelled sample, the function
likely returns similar values for both matches and non-matches. Consider a
more selective metric or check whether the field has low coverage in your data.

What to explore next
---------------------

* :doc:`/reference/similarity-functions`, the full catalog of built-in functions and which ``FieldKind`` they are assigned to by default.
* :doc:`/explanation/fellegi-sunter`, how EM uses similarity scores to estimate match probabilities.
* :doc:`/how-to/tune-scorer`, adjust EM thresholds after adding a new field.
