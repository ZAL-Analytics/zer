How to Define a Schema
=======================

A ``Schema`` is the entry point for every zer pipeline. It maps field names to
``FieldKind`` values, which drive three downstream decisions:

* Which **blocking keys** are generated for a field.
* Which **similarity function** is applied when comparing two field values.
* How the field is **weighted** in the Fellegi-Sunter EM model.

Basic schema
------------

.. code-block:: rust

   use zer_core::schema::{FieldKind, SchemaBuilder};

   let schema = SchemaBuilder::new()
       .field("voornamen",     FieldKind::Name)
       .field("achternaam",    FieldKind::Name)
       .field("geboortedatum", FieldKind::Date)
       .field("postcode",      FieldKind::Id)
       .field("nationaliteit", FieldKind::Categorical)
       .build()?;

Fields are processed in declaration order. The order matters for the EM
parameter matrix, keep it stable across pipeline runs.

Adding every supported field kind
-----------------------------------

.. code-block:: rust

   let schema = SchemaBuilder::new()
       // Names: phonetic blocking + Jaro-Winkler / token overlap / phonetic similarity
       .field("voornamen",       FieldKind::Name)
       .field("achternaam",      FieldKind::Name)

       // Date: year-based blocking + exact / partial / year-only matching
       .field("geboortedatum",   FieldKind::Date)

       // Address: address-initial blocking + Jaro-Winkler similarity
       .field("woonplaats",      FieldKind::Address)
       .field("straatnaam",      FieldKind::Address)

       // Id: suffix blocking + exact match
       .field("postcode",        FieldKind::Id)
       .field("bsn",             FieldKind::Id)

       // Phone: digit-suffix blocking + exact match
       .field("telefoonnummer",  FieldKind::Phone)

       // Categorical: exact-field blocking + exact match
       .field("geslacht",        FieldKind::Categorical)
       .field("nationaliteit",   FieldKind::Categorical)

       // Alias: pipe-delimited aliases; alias-phonetic blocking + max alias similarity
       .field("alias_namen",     FieldKind::Alias)

       // LicensePlate: plate-norm + OCR-fuzzy blocking
       .field("kenteken",        FieldKind::LicensePlate)

       // Timestamp: used for camera time-window blocking
       .field("tijdstip",        FieldKind::Timestamp)

       // GPS: used for geo-grid blocking
       .field("lat",             FieldKind::GpsCoordinate)
       .field("lon",             FieldKind::GpsCoordinate)

       // Free text: no blocking; Jaro-Winkler similarity only
       .field("opmerkingen",     FieldKind::FreeText)

       // Numeric: no blocking; absolute / relative difference similarity
       .field("bedrag",          FieldKind::Numeric)
       .build()?;

The schema must have at least one field or ``build()`` returns an error.

FieldKind quick reference
--------------------------

See :doc:`/reference/field-kind` for the complete table. The most common choices:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Kind
     - Use for
     - Blocking generated
   * - ``Name``
     - First names, last names
     - Phonetic + DOB year (combined with ``Date``)
   * - ``Date``
     - Birth dates, registration dates
     - Year fragment; combined with ``Name`` for phonetic-dob key
   * - ``Id``
     - Postcodes, BSN, document numbers
     - 4-character suffix
   * - ``Categorical``
     - Gender, nationality, vehicle type
     - Exact field value
   * - ``Alias``
     - Pipe-delimited alias lists (SIS II)
     - Phonetic code of each alias surname
   * - ``LicensePlate``
     - Dutch and EU license plates
     - Normalized plate + OCR variants
   * - ``Address``
     - Street names, city names
     - First address token + first-name initial

Choosing between ``Id`` and ``Categorical``
---------------------------------------------

* Use ``Id`` for fields with high cardinality where you want suffix blocking
  (postcodes, BSNs, IBANs, document numbers). zer generates ``SuffixKey(4)``,
  the last four characters, so that formatting differences in the prefix
  are absorbed.

* Use ``Categorical`` for low-cardinality discrete values where exact matching
  is sufficient (gender, nationality, rechtsvorm, camera IDs). zer generates
  ``ExactFieldKey``, records that share the exact normalized value are
  candidates.

When fields are absent in a record
------------------------------------

Missing fields produce ``FieldValue::Null`` when inserted as such, or are
simply absent from the record's field map. The comparator treats any ``Null``
value as ``ComparisonLevel::Null``, which is scored as a non-match but does not
count against the record. This matches the Fellegi-Sunter treatment of missing
data.

.. code-block:: rust

   // Explicit null
   let record = Record::new(1)
       .insert("voornamen",     FieldValue::Text("Jan".into()))
       .insert("geboortedatum", FieldValue::Null);  // explicit null

   // Implicit null: field simply absent, same effect
   let record = Record::new(2)
       .insert("voornamen", FieldValue::Text("Jan".into()));
   // geboortedatum is absent, treated as Null by the comparator
