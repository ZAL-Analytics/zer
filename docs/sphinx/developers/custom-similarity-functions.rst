Custom Similarity Functions
============================

The comparator evaluates each candidate pair by computing a similarity score
for every field in the schema. zer ships with a catalog of built-in functions
(Jaro-Winkler, exact match, date proximity, and others,see
:doc:`/reference/similarity-functions`). When none of those fit your domain
you can register your own.

Similarity scores flow directly into the Fellegi-Sunter EM model as match
weights, so getting the right metric for a field matters more than it might
appear. A poor similarity function for a high-weight field degrades EM
accuracy across the whole dataset.

The ``SimilarityFn`` trait
---------------------------

The trait lives in ``zer_compare``:

.. code-block:: rust

   use zer_compare::SimilarityFn;
   use zer_core::FieldValue;

   pub trait SimilarityFn: Send + Sync + 'static {
       /// Display name used in audit logs and the EM weight table.
       fn name(&self) -> &str;

       /// Return a score in [0.0, 1.0].
       /// 1.0 = definite match, 0.0 = definite non-match.
       /// Return None if either value is Null/missing,the EM model
       /// treats missing pairs separately from scored pairs.
       fn score(&self, a: &FieldValue, b: &FieldValue) -> Option<f64>;
   }

The ``None`` return path is important: if either field is missing, returning
``None`` tells the EM model to use its missing-value prior rather than
treating the pair as a non-match. Never return ``0.0`` for a missing field.

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
   use zer_compare::SimilarityFn;
   use zer_core::FieldValue;

   pub struct DoubleMetaphoneSimilarity;

   impl SimilarityFn for DoubleMetaphoneSimilarity {
       fn name(&self) -> &str { "double_metaphone" }

       fn score(&self, a: &FieldValue, b: &FieldValue) -> Option<f64> {
           let (ta, tb) = match (a, b) {
               (FieldValue::Text(a), FieldValue::Text(b)) => (a.as_str(), b.as_str()),
               _ => return None,
           };

           let (primary_a, alt_a) = double_metaphone(ta);
           let (primary_b, alt_b) = double_metaphone(tb);

           // Full match on either primary or alternate code
           let exact = primary_a == primary_b
               || primary_a == alt_b
               || alt_a == primary_b
               || alt_a == alt_b;

           if exact { Some(1.0) } else { Some(0.0) }
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

   use zer_compare::SimilarityFn;
   use zer_core::FieldValue;

   pub struct NormalisedLevenshtein;

   impl SimilarityFn for NormalisedLevenshtein {
       fn name(&self) -> &str { "norm_levenshtein" }

       fn score(&self, a: &FieldValue, b: &FieldValue) -> Option<f64> {
           let (ta, tb) = match (a, b) {
               (FieldValue::Text(a), FieldValue::Text(b)) => (a.as_str(), b.as_str()),
               _ => return None,
           };
           if ta.is_empty() && tb.is_empty() { return Some(1.0); }

           let dist = levenshtein(ta, tb);           // any edit-distance crate
           let max  = ta.len().max(tb.len()) as f64;
           Some(1.0 - dist as f64 / max)
       }
   }

   fn levenshtein(a: &str, b: &str) -> usize {
       // Wagner-Fischer DP,replace with a crate like `edit-distance` in practice
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

Attach a custom function to a field in the schema builder:

.. code-block:: rust

   use zer_schema::SchemaBuilder;
   use zer_core::FieldKind;

   let schema = SchemaBuilder::new()
       .field("voornamen",    FieldKind::PersonName)
       .field("achternaam",   FieldKind::PersonName)
       // Override the similarity function for a specific field
       .field_with_similarity(
           "naam_niet_latinized",
           FieldKind::PersonName,
           DoubleMetaphoneSimilarity,
       )
       .field("geboortedatum", FieldKind::Date)
       .build()?;

Only fields registered via ``field_with_similarity`` use your custom metric.
All other fields continue to use the built-in function selected for their
``FieldKind``.

Calibrating a new similarity function
---------------------------------------

The EM model estimates match and non-match distributions from unlabelled data.
A new similarity function starts without calibration data; the first
``run_batch`` call seeds the EM model. To verify the function is contributing
correctly, inspect the learned weight for the field after a run:

.. code-block:: rust

   let view   = pipeline.cluster_view();
   let weights = view.em_weights();

   for (field, w) in &weights {
       println!("{field}: log_ratio = {:.3}", w.log_ratio);
   }

A ``log_ratio`` close to zero means the function is not discriminating,the
field looks the same whether the pair is a match or not. This usually means the
similarity function returns similar values for both matches and non-matches.
Consider a more selective metric or check whether the field itself has low
coverage in your data.

What to explore next
---------------------

* :doc:`/reference/similarity-functions`, the full catalog of built-in functions and which ``FieldKind`` they are assigned to by default.
* :doc:`/explanation/fellegi-sunter`, how EM uses similarity scores to estimate match probabilities.
* :doc:`/how-to/tune-scorer`, adjust EM thresholds after adding a new field.
