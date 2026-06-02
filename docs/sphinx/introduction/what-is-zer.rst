What is zer?
============

**zer** solves one problem: given multiple datasets that contain records about
the same real-world people, vehicles, or entities, find which records belong
together, even when there is no shared unique identifier and the data contains
spelling variations, OCR errors, and missing fields.

This is called **entity resolution** (also known as record linkage or
deduplication depending on the task).

zer is domain-agnostic at its core. Every component, blocking keys, similarity
functions, comparators, and storage backends, is pluggable. It ships with
built-in support for Dutch administrative data (BRP, KvK, SIS II, ANPR), but the
same pipeline structure applies to any domain: healthcare records, corporate
registries, logistics data, or anything else with noisy identifiers.

The problem in concrete terms
------------------------------

The same person can appear in five KvK (Chamber of Commerce) registrations,
each typed by hand in a slightly different way:

.. list-table::
   :header-rows: 1
   :widths: 15 20 15 18 15

   * - kvkNummer
     - voornamen
     - tussenvoegsel
     - achternaam
     - geboortedatum
   * - 57346300
     - Liam Arjan
     - van der
     - Wal
     - 1961-06-23
   * - 66047634
     - L.A.
     - v/d
     - Wal
     - 1961-06-23
   * - 79741090
     - Liam
     - ,
     - v.d. Wal
     - 1961-06-23
   * - 83946644
     - LIAM ARJAN
     - VAN DER
     - WAL
     - 1961-06-23

All four rows are the same person. zer finds them, without a BSN or any shared
unique key, by combining phonetic name matching, Dutch tussenvoegsel
normalization, and probabilistic scoring.

Similarly, a highway camera reads the same license plate twice but the OCR
system makes a single-character error:

.. list-table::
   :header-rows: 1
   :widths: 25 20 22 18

   * - passage_id
     - camera_id
     - tijdstip
     - kenteken
   * - 8DDE6D8D-81A
     - CAM-A12-001
     - 2025-06-01T10:04:00
     - CX-180-W
   * - F3A2B891-C04
     - CAM-A12-001
     - 2025-06-01T10:04:03
     - **CX-I80-W**

``1`` was read as ``I``. zer's ANPR blocking generates all single-character
OCR variants and links both passages to the same vehicle.

What zer does not do
---------------------

* **It is not a database.** zer processes records in memory and persists
  resolved entities to a SQLite-backed ``ZalEntityStore`` (``.zes`` file).
  It does not replace a production database.

* **It is not a rules engine.** zer uses probabilistic scoring (Fellegi-Sunter
  with EM parameter estimation) rather than hand-written business rules.


When to use zer
---------------

zer is a good fit when you need to:

* Deduplicate a registry with no reliable unique identifier (e.g. a person
  directory, a product catalogue, a patient index).
* Link two or more datasets that represent the same population without a shared
  key (e.g. BRP ↔ KvK ↔ HKS, or any cross-system identity matching task).
* Match records despite OCR noise or transcription errors (e.g. ANPR licence
  plate reads, scanned document fields).
* Resolve identities across registries that use different name conventions or
  character sets.
* Build an entity graph that persists and updates incrementally as new records
  arrive over time.

zer is built in Rust, exposes a ``Pipeline`` API that works with plain
``Vec<Record>`` batches, and integrates with Polars DataFrames and Apache Arrow
through the ``zer-adapters`` crate.
