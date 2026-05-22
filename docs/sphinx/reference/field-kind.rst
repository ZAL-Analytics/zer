FieldKind Reference
====================

``FieldKind`` is the enum that annotates each field in a ``Schema``. It drives
three downstream decisions: which blocking keys are generated, which similarity
function is used during comparison, and how the field is treated by the EM model.

Complete table
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - Kind
     - Use for
     - Blocking keys generated
     - Similarity function
   * - ``Name``
     - First names, last names, full names
     - ``PhoneticNameDobKey`` or ``PhoneticNameDobInitialKey`` (combined with ``Date``); ``AddressInitialKey`` (combined with ``Address``)
     - Jaro-Winkler; token overlap for multi-token names; phonetic equality
   * - ``Date``
     - Birth dates, registration dates, event dates
     - ``DateFragmentKey(YearMonth)`` as secondary key; DOB year incorporated into phonetic keys
     - Exact date; partial (year+month); year-only; day-off-by-one (Close)
   * - ``Address``
     - Street names, city names, place names
     - ``AddressInitialKey`` (first token + first-name initial)
     - Jaro-Winkler; address token overlap; street number edit distance
   * - ``Id``
     - Postcodes, BSN, IBANs, document numbers
     - ``SuffixKey(4)``, last four alphanumeric characters
     - Exact match; Jaro-Winkler for partial credit
   * - ``Phone``
     - Phone numbers, mobile numbers
     - ``SuffixKey(7)``, last seven digits
     - Exact match on normalized digits
   * - ``Categorical``
     - Gender, nationality, rechtsvorm, camera IDs
     - ``ExactFieldKey``, exact normalized value
     - Exact match only
   * - ``Alias``
     - Pipe-delimited alias lists (SIS II)
     - ``AliasPhoneticKey``, phonetic code of each alias surname + DOB year
     - Maximum similarity across all alias pairs
   * - ``LicensePlate``
     - Dutch and EU license plates
     - ``LicensePlateNormKey`` + ``PlateOCRFuzzyKey`` (OCR variants)
     - Normalized plate string; Jaro-Winkler for partial credit
   * - ``Timestamp``
     - Camera passage timestamps, event timestamps
     - Used by ``CameraTimeWindowKey`` (combined with ``Categorical`` camera ID)
     - Absolute time difference (seconds/minutes)
   * - ``GpsCoordinate``
     - Latitude, longitude (decimal degrees)
     - Used by ``GeoGridKey`` (combined, lat+lon pair)
     - Euclidean distance in degrees
   * - ``FreeText``
     - Remarks, descriptions, free-form notes
     - No blocking key generated
     - Jaro-Winkler only
   * - ``Numeric``
     - Monetary amounts, ages, counts
     - No blocking key generated
     - Absolute difference; relative (percentage) difference

Choosing between kinds
-----------------------

``Id`` vs. ``Categorical``
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use ``Id`` for **high-cardinality** fields where values are unique or
  near-unique (postcodes, BSNs, IBANs). zer generates ``SuffixKey(4)``, the
  last four alphanumeric characters, so that prefix formatting differences
  are absorbed.
* Use ``Categorical`` for **low-cardinality** discrete values where exact
  matching is appropriate (gender codes, nationality ISO codes, camera IDs).
  zer generates ``ExactFieldKey``, records must share the exact normalized
  value to become candidates through this key.

``Name`` vs. ``FreeText``
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use ``Name`` for structured name fields. zer strips tussenvoegsels, applies
  Double Metaphone, and combines the phonetic code with DOB for blocking.
* Use ``FreeText`` for unstructured text fields where phonetic blocking would
  produce too many false candidates. No blocking key is generated; the field
  still contributes to the Jaro-Winkler similarity score.

``Date`` vs. ``Timestamp``
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use ``Date`` for birth dates and calendar dates. zer generates date fragment
  keys (year, year-month) and applies date-specific comparison logic.
* Use ``Timestamp`` for camera passage times, transaction timestamps, and
  event times. zer uses this field for time-window blocking with
  ``CameraTimeWindowKey``; the field does not participate in date fragment
  blocking.

``GpsCoordinate``
~~~~~~~~~~~~~~~~~

* Assign ``GpsCoordinate`` to both the latitude and longitude fields. zer
  detects a pair of ``GpsCoordinate`` fields and generates a ``GeoGridKey``
  from the combination. If only one coordinate field is present, no geo key
  is generated.

Null and missing fields
------------------------

Any field with ``FieldValue::Null`` or simply absent from the record map is
treated as ``ComparisonLevel::Null`` by the comparator. Null comparisons
contribute no evidence to the Fellegi-Sunter score, they neither increase
nor decrease the match probability. This follows the standard Fellegi-Sunter
treatment of missing data.

.. code-block:: rust

   // Explicit null
   let record = Record::new(1)
       .insert("voornamen",     FieldValue::Text("Jan".into()))
       .insert("geboortedatum", FieldValue::Null);

   // Implicit null: same effect as Null above
   let record = Record::new(2)
       .insert("voornamen", FieldValue::Text("Jan".into()));
   // geboortedatum is absent, treated as Null
