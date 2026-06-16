Why Blocking Recall Is the Critical Metric
===========================================

Blocking is the step that determines which record pairs are ever compared.
Pairs that are not candidates after blocking are **permanently invisible** to
the scorer and the neural judge. A missed pair at the blocking stage is a
missed match in the final output, with no recovery path.

This makes **blocking recall** the most important metric in an entity
resolution pipeline, more important than comparator precision, scorer
thresholds, or neural judge accuracy.

Recall vs. precision in blocking
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Definition
     - What goes wrong when it is low
   * - Blocking recall
     - Fraction of true matches that appear as candidate pairs
     - True matches are permanently missed; output F1 degrades with no visible error
   * - Reduction ratio
     - Fraction of all pairs that are NOT candidates
     - Low ratio (too many candidates) means the comparator is slow; cost scales quadratically

The trade-off is asymmetric: **a missed candidate can never be recovered**,
but a false candidate is simply eliminated by the scorer at low cost. This
means it is almost always better to err toward more candidates (lower
reduction ratio) than toward fewer candidates (lower recall).

zer targets blocking recall of at least 0.99 on standard Dutch administrative datasets
by default.

Why keys miss true matches
---------------------------

A blocking key misses a true match when the two records disagree on the
field the key is derived from:

* **Name variant**, "Johannes" and "J." produce different phonetic codes.
  zer mitigates this with the first-name initial in ``PhoneticNameDobInitialKey``.
* **Transliteration**, "Benabdallah" and "Ben Abdallah" differ after ASCII
  normalization. zer adds ``TransliteratedPhoneticKey`` for the
  ``WantedPersons`` category.
* **DOB transcription error**, a year off by one: 1978 vs. 1979. zer adds
  ``FuzzyYearKey(+/-1)`` for ``WantedPersons``.
* **Postcode formatting**, "1011 AB" and "1011AB". zer normalizes both to
  ``1011AB`` before generating the suffix key.
* **OCR confusion**, "CX-180-W" and "CX-I80-W". zer generates all
  single-character OCR variant keys for each plate.

Measuring blocking recall
--------------------------

To measure blocking recall, you need a ground-truth set of true match pairs.
Compare the candidate pairs produced by the blocker against the ground truth:

.. code-block:: rust

   use std::collections::HashSet;
   use zer_blocking::InvertedIndex;
   use zer_core::traits::Blocker;

   let mut index = InvertedIndex::new();
   for r in &all_records {
       blocker.index_record(r, &schema, &mut index);
   }

   // Ground truth: set of (id_a, id_b) pairs with id_a < id_b
   let ground_truth: HashSet<(u64, u64)> = load_ground_truth();

   // Collect all candidate pairs
   let mut candidates: HashSet<(u64, u64)> = HashSet::new();
   for r in &all_records {
       for candidate_id in blocker.candidates(r, &schema, &index) {
           let a = r.id.min(candidate_id);
           let b = r.id.max(candidate_id);
           candidates.insert((a, b));
       }
   }

   let true_positives = ground_truth.intersection(&candidates).count();
   let false_negatives = ground_truth.difference(&candidates).count();
   let recall = true_positives as f64 / ground_truth.len() as f64;

   println!("blocking recall    : {:.4}", recall);
   println!("missed true matches: {}", false_negatives);
   println!("candidate pairs    : {}", candidates.len());

What reduction ratio tells you
--------------------------------

Reduction ratio measures how much work the blocker is saving the comparator:

.. code-block:: text

   reduction_ratio = 1 - (candidate_pairs / total_pairs)
   total_pairs = n * (n - 1) / 2

A dataset of 10,000 records has 49,995,000 possible pairs. If blocking
produces 50,000 candidates, the reduction ratio is 0.999, the comparator
sees only 0.1% of all pairs.

A high reduction ratio with high recall is the goal. If you see a very low
reduction ratio (many candidates), your blocking strategy is probably too
permissive. Add a tighter secondary key or switch to a more specific
``SchemaCategory``.

Secondary keys improve recall without hurting precision
---------------------------------------------------------

zer always adds a secondary key alongside the primary phonetic/DOB key.
For ``PersonRegistry``, this is ``DateFragmentKey(YearMonth)``. A pair
only needs to share **one** key to become a candidate, so the secondary key
catches cases where the primary key fails:

* Records with the same birth year-month but different phonetic codes
  (e.g. a transcription error in the surname) are still candidates via the
  ``YearMonth`` key.
* The comparator then rejects the pair if the names don't match sufficiently,
  at a cost of a few microseconds, far cheaper than missing a true match.

Rule of thumb: when recall is too low
---------------------------------------

If you measure blocking recall below 0.95 on your dataset:

1. Inspect which true-match pairs are missing from the candidates.
2. Check which fields differ between those pairs.
3. Add a key or category rule that covers that field combination.

For person data: common causes are initial-vs-full-name disagreement (add
``with_phonetic_name_dob_initial()``), translit disagreement (add
``TransliteratedPhoneticKey`` via ``with_key()``), and DOB year errors
(add ``FuzzyYearKey`` via ``with_key()``).

What to explore next
---------------------

* :doc:`/how-to/blocking-strategy`, how to configure and customize the blocker.
* :doc:`/reference/blocking-keys`, every key and what error modes it covers.
* :doc:`fellegi-sunter`, how the scorer handles the candidates the blocker passes on.
