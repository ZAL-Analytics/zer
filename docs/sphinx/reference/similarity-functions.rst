Similarity Functions Reference
================================

Similarity functions map a pair of ``FieldValue``s to a score in ``[0.0, 1.0]``.
The comparator then discretizes that score into a ``ComparisonLevel`` using
per-field thresholds estimated by EM.

ComparisonLevel values
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Level
     - Meaning
   * - ``Exact``
     - Maximum similarity or identical value; strong evidence for a match
   * - ``Close``
     - High similarity; one edit, one day off, or one digit different
   * - ``Partial``
     - Moderate similarity; prefix match, shared tokens, same year
   * - ``None``
     - No detectable similarity; strong evidence against a match
   * - ``Null``
     - One or both values are missing; no evidence either way

Name similarity (``FieldKind::Name``)
----------------------------------------

Three functions are combined; the maximum of the three is used:

**Jaro-Winkler**, positional string distance that gives more weight to
common prefix matches. Score 1.0 for identical strings; ≈ 0.9 for a one-
character edit in a short name.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Pair
     - Score
     - Level
   * - "Jansen" / "Jansen"
     - 1.00
     - Exact
   * - "Jansen" / "Janssen"
     - 0.97
     - Exact or Close (threshold-dependent)
   * - "Jansen" / "Jensen"
     - 0.89
     - Close
   * - "Jansen" / "Smith"
     - 0.00
     - None

**Token overlap**, for multi-token names. Computes Jaccard similarity over
space-delimited tokens. "Alice Marie van den Berg" vs. "Alice Berg" → 2
shared tokens / 4 unique tokens = 0.50 (Partial).

**Phonetic equality**, Double Metaphone codes are compared. Identical codes
score 1.0 regardless of spelling. Useful for "Jansen" / "Jansen" exact-match
bypass.

Date similarity (``FieldKind::Date``)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Comparison
     - Level
     - Condition
   * - Exact date match
     - Exact
     - Both dates identical: "1978-03-15" / "1978-03-15"
   * - Day off by 1
     - Close
     - "1978-03-15" / "1978-03-14" (transposition error)
   * - Same year-month
     - Partial
     - "1978-03-15" / "1978-03-22"
   * - Same year only
     - Partial (lower)
     - "1978-03-15" / "1978-07-01"
   * - Different year
     - None
     - "1978-03-15" / "1979-03-15"

Address similarity (``FieldKind::Address``)
----------------------------------------------

Two functions are combined:

**Jaro-Winkler**, applied to the full address string after normalization.

**Address token overlap**, Jaccard over word tokens, useful for street names
with different word orderings ("Hoofdstraat Noord" vs. "Noord Hoofdstraat").

**Street number edit distance**, extracts numeric tokens and computes edit
distance. "Hoofdstraat 12A" vs. "Hoofdstraat 12" → Close; vs. "Kerkstraat 12"
→ None (street names differ).

Id similarity (``FieldKind::Id``)
------------------------------------

**Exact match**, identical normalized values score 1.0. Any difference scores
0.0. No partial credit for postcodes or BSNs, a one-digit difference is
meaningless for identity.

Phone similarity (``FieldKind::Phone``)
-----------------------------------------

**Exact match on normalized digits**, strip all non-digit characters
(spaces, dashes, country code prefix), then compare digit strings. "06-12 34 56
78" and "+31612345678" both normalize to the same digit string and score Exact.

Categorical similarity (``FieldKind::Categorical``)
------------------------------------------------------

**Exact match only**, identical normalized values score 1.0; any difference
scores 0.0. Categorical fields are either equal or they are not.

Numeric similarity (``FieldKind::Numeric``)
----------------------------------------------

Two functions; the higher score is used:

**Absolute difference**, ``1 - |a - b| / max_expected_range``. Configurable
via the schema.

**Relative difference**, ``1 - |a - b| / max(|a|, |b|)``. Useful for
monetary amounts where a 1% difference is Close and a 50% difference is None.

FreeText similarity (``FieldKind::FreeText``)
-----------------------------------------------

**Jaro-Winkler only**, no blocking key. The field contributes to the
comparison vector but does not drive candidate generation.

LicensePlate similarity (``FieldKind::LicensePlate``)
-------------------------------------------------------

**Normalized plate Jaro-Winkler**, after stripping hyphens and uppercasing.
Plates that differ by one OCR-confused character score Close.

Null handling
--------------

All similarity functions return 0.0 when either input is ``FieldValue::Null``.
The comparator maps this to ``ComparisonLevel::Null`` rather than
``ComparisonLevel::None``. In the Fellegi-Sunter model, ``Null`` contributes
no evidence for or against a match, whereas ``None`` is treated as active
evidence against.

LevenshteinSimilarity (custom use)
------------------------------------

``LevenshteinSimilarity`` is available for fields where edit distance is more
meaningful than Jaro-Winkler's positional weighting. It is not used by default
for any ``FieldKind`` but can be injected via ``FieldComparator::with_fns()``:

.. code-block:: rust

   use zer_compare::similarity::name::LevenshteinSimilarity;

   let sim = LevenshteinSimilarity { max_distance: 3 };
   // Scores 1.0 for distance 0, 0.0 for distance > max_distance,
   // and 1 - dist/max_distance in between.
