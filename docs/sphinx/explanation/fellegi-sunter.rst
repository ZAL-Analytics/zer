The Fellegi-Sunter Model
=========================

zer's scoring step uses a probabilistic model due to Fellegi and Sunter
(1969) that assigns a match probability to every candidate pair. The model's
parameters are estimated by EM (Expectation-Maximization) from the comparison
vectors themselves, **no labelled training data is required**.

Comparison vectors
-------------------

Before scoring, the comparator converts each candidate pair into a
``ComparisonVector``: one ``ComparisonLevel`` per field.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Level
     - Meaning
   * - ``Exact``
     - Fields are identical (or phonetically / numerically equivalent)
   * - ``Close``
     - Fields differ by one edit or one DOB day
   * - ``Partial``
     - Fields share a common prefix, token, or year
   * - ``None``
     - No detectable similarity
   * - ``Null``
     - One or both fields are missing; counted as ``None`` but not penalised

Each field has its own similarity function that determines the mapping from
field values to comparison levels (see :doc:`/reference/similarity-functions`).

The m and u parameters
-----------------------

The Fellegi-Sunter model has two parameter vectors per field:

* **m[field][level]**, probability that a comparison level is observed
  *given that the pair is a true match*.
* **u[field][level]**, probability that the same level is observed *given
  that the pair is a non-match* (i.e. two random records).

For a highly discriminating field like DOB:

* ``m[dob][Exact]`` is close to 1.0, true matches almost always share an exact DOB.
* ``u[dob][Exact]`` is close to 1/365, around 0.003, two random records rarely share a DOB.

For a non-discriminating field like nationality:

* ``m[nat][Exact]`` is around 0.7, true matches often share nationality but not always.
* ``u[nat][Exact]`` is around 0.5, two random Dutch records share "Netherlands" about half the time.

The log Bayes factor (match weight) for one field and one comparison level is:

.. code-block:: text

   weight = log( m[field][level] / u[field][level] )

A positive weight is evidence for a match; a negative weight is evidence
against. The total match score is the sum of weights across all fields.

EM estimation (no labels needed)
----------------------------------

zer uses the Expectation-Maximization algorithm to estimate m and u from the
comparison vectors without requiring any labelled pairs. EM alternates between:

1. **E-step**, given current m, u, and a prior match probability (lambda), compute
   the posterior match probability for every pair.
2. **M-step**, re-estimate m and u as the weighted average comparison level
   distributions, using the posterior probabilities as weights.

After convergence (typically 50-200 iterations on real data), the model
has learned that:

* Records agreeing on name + DOB are likely matches.
* Records differing on every field are likely non-matches.

This is unsupervised, the algorithm infers the structure from the data.

Threshold selection
--------------------

After EM, the scorer derives two probability thresholds from the estimated
parameters:

* **upper_threshold**, match probabilities above this are ``AutoMatch``.
* **lower_threshold**, match probabilities below this are ``AutoReject``.
* Pairs between the thresholds are ``Borderline``.

The default thresholds are chosen to minimise expected classification error
under the estimated prior match rate (lambda). You can override them with
``PipelineConfig::upper_threshold`` and ``lower_threshold``, see
:doc:`/how-to/tune-scorer`.

Why the prior matters
----------------------

Lambda is the estimated fraction of candidate pairs that are true matches.
On a deduplication run over a clean population register, lambda is around 0.001 (one
duplicate per thousand records). On a linkage run between two near-identical
exports of the same register, lambda is around 0.9.

EM estimates lambda jointly with m and u. If EM converges to an unrealistic lambda
(e.g. lambda = 0.5 on a deduplication task), the thresholds will be wrong and
precision will suffer.

The pipeline automatically warm-starts EM from the previous ``.zsm`` registry
file on each run, guiding lambda toward realistic values as the model accumulates
evidence. On the very first batch, EM starts from equal priors (lambda = 0.5) and
typically converges within 50-200 iterations given enough variation in the
comparison vectors. If EM stays stuck at an unrealistic lambda, delete the ``.zsm``
file so the model re-estimates from scratch on the next batch.

The warm-start advantage
-------------------------

After the first batch, zer writes the estimated parameters to a ``.zsm``
registry file. On the next batch, EM starts from the previous parameters
rather than from the uniform prior. This has two benefits:

1. **Faster convergence**, EM typically converges in 5-20 iterations
   instead of 50-200 when starting from a reasonable prior.
2. **Stable thresholds**, incremental ingestion does not re-estimate
   thresholds from scratch; the registry accumulates evidence across batches.

Limitations
------------

* **Conditional independence**, the Fellegi-Sunter model assumes fields are
  independent within the match and non-match populations. This is violated by
  correlated fields (e.g. first name and gender are correlated). In practice,
  the model is robust to mild violations; severe correlation can bias the
  estimated match weights.
* **Small datasets**, EM needs enough comparison vectors to estimate m and u
  reliably. On datasets with fewer than a few hundred candidate pairs, the
  estimates may be unreliable. Adding a prior lambda helps.
* **Changing data distributions**, if the population distribution shifts
  significantly between batches (e.g. a bulk import of a new nationality
  group), the warm-started parameters may lag behind. Delete the ``.zsm``
  file to force a full re-estimation.

What to explore next
---------------------

* :doc:`/how-to/tune-scorer`, override thresholds and inspect EM parameters.
* :doc:`/reference/similarity-functions`, how each field type maps to comparison levels.
* :doc:`judge-internals`, what happens to borderline pairs after scoring.
