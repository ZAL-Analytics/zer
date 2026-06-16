Entity Resolution: What It Is and Why It Is Hard
=================================================

Entity resolution (also called record linkage, deduplication, or identity
resolution) is the task of deciding which records in one or more datasets
refer to the same real-world entity, a person, company, vehicle, or
financial account, without a shared unique identifier.

The core problem
-----------------

Administrative systems rarely agree on how to represent the same fact:

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Source
     - Name
     - Date
   * - BRP (authoritative)
     - Johannes van den Berg
     - 1978-03-15
   * - KvK extract
     - J. Berg
     - 15-03-1978
   * - SIS II alert
     - YOHANNES VAN DEN BERG
     - 1978-03-15
   * - ANPR passage
     - (no name; plate CX-180-W)
     - 2025-06-01

No common key exists across these four records. Exact matching fails. The
same person appears four times, represented differently each time.

Why not just fuzzy-match everything?
--------------------------------------

A naive approach would compare every record against every other and apply a
fuzzy string threshold. This fails in practice for two reasons:

1. **Quadratic cost**, a dataset of 10 million records has 50 trillion record
   pairs. Even at 1 microsecond per comparison, exhaustive comparison takes
   over a year on a single machine.

2. **False positive explosion**, Dutch is a small-vocabulary language. Common
   names like ``Jansen``, ``de Vries``, and ``Bakker`` appear hundreds of
   thousands of times. A low string-similarity threshold produces millions of
   wrong links between unrelated people.

zer's two-stage answer
------------------------

zer solves both problems with a two-stage architecture:

**Stage 1: Blocking.** Generate cheap, exact keys from each record, phonetic
codes, DOB fragments, address initials, postcode suffixes. Two records only
become a candidate pair if they share at least one key. This reduces O(n^2)
comparisons to O(n * k) where *k* is the average number of matches per key,
typically 1-100.

**Stage 2: Probabilistic scoring.** For each candidate pair, compare every
field and produce a ``ComparisonVector`` of ``ComparisonLevel`` values (None /
Partial / Close / Exact). Feed this vector through a Fellegi-Sunter
probabilistic model to get a match probability. The model's parameters are
estimated by EM from the comparison vectors themselves, no labelled data is
needed.

The trade-off between stages
------------------------------

Blocking and scoring trade off against each other:

* **More blocking keys** mean higher recall (fewer missed matches) but more
  candidate pairs, which increases comparator work.
* **Stricter scoring thresholds** mean higher precision (fewer false links) but
  more pairs fall into the borderline band, requiring a human review or a
  neural judge.

In practice, zer targets **blocking recall of at least 0.99**, at most 1% of true
matches are missed at the blocking stage, and lets the scorer handle
precision tuning.

What entity resolution is not
--------------------------------

* **De-identification**, zer links records; it does not pseudonymise or
  remove personal data.
* **Deduplication of identical rows**, if two rows are byte-for-byte equal
  across all fields, use a database DISTINCT. zer is for *noisy* duplicates
  where fields differ due to data entry, OCR errors, or format differences.
* **Named entity recognition**, zer works on structured fields (name,
  date, postcode), not free-running text.

The entity as a cluster
------------------------

zer's output is a set of **clusters**, not a set of linked pairs. A cluster
is a connected component in the graph where edges are auto-matched pairs.
Each cluster is assigned an entity ID and stored in the ``ZalEntityStore``.

Subsequent ingestion batches extend existing clusters when new records match
already-stored entities, creating a growing identity graph over time.

What to explore next
---------------------

* :doc:`blocking-recall`, how to measure and reason about blocking recall.
* :doc:`fellegi-sunter`, the probabilistic scoring model in detail.
* :doc:`/how-to/define-schema`, start building your own pipeline.
