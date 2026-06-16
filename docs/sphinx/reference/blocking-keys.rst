Blocking Keys Reference
========================

Every blocking key in zer, its parameters, the domains it targets, and its
normalization steps.

PhoneticNameDobKey
-------------------

Encodes the surname phonetically with Double Metaphone, then appends the
birth year. Strips tussenvoegsels and diacritics before encoding.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``name_field: &str``, ``dob_field: &str``
   * - **Key format**
     - ``"<phonetic_code>:<year>"``
   * - **Example**
     - "van den Berg" + "1978-03-15" gives ``"PRK:1978"``
   * - **Domains**
     - BRP, KvK, any person registry
   * - **Handles**
     - Tussenvoegsel variants, diacritics, common surname misspellings

PhoneticNameDobInitialKey
--------------------------

Extends ``PhoneticNameDobKey`` by appending the first character of the given
name. Used when two Name fields are present. Falls back to
``PhoneticNameDobKey`` with a single Name field.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``surname_field: &str``, ``first_name_field: &str``, ``dob_field: &str``
   * - **Key format**
     - ``"<phonetic_code>_<initial>_<year>"``
   * - **Example**
     - "Berg" + "Johannes" + "1978" gives ``"PRK_J_1978"``
   * - **Domains**
     - BRP, KvK, multi-field name schemas
   * - **Handles**
     - First-name disambiguation within the same phonetic surname + DOB group

AliasPhoneticKey
-----------------

For pipe-delimited alias lists (``FieldKind::Alias``). Splits the alias string
on ``|``, extracts the surname token from each alias, encodes it phonetically,
and appends the DOB year.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``alias_field: &str``, ``dob_field: &str``
   * - **Key format**
     - One key per alias: ``"alias:<phonetic_code>:<year>"``
   * - **Example**
     - ``"Hassan|Mohamed"`` + "1999" gives ``["alias:HSN:1999", "alias:MHMT:1999"]``
   * - **Domains**
     - SIS II wanted/missing persons
   * - **Handles**
     - Multiple alias surnames, name transpositions in alias lists

TransliteratedPhoneticKey
--------------------------

Transliterates non-Latin script to ASCII (via ``any_ascii``) before applying
Double Metaphone + DOB year combination.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``name_field: &str``, ``dob_field: &str``
   * - **Key format**
     - ``"translit:<phonetic_code>:<year>"``
   * - **Example**
     - Arabic "محمد" transliterates to "mhmd", which encodes to ``"translit:MMT:1978"``
   * - **Domains**
     - SIS II (Arabic, Cyrillic, Greek input)
   * - **Handles**
     - Cross-script name variants; Latin vs. original-script registrations

FuzzyYearKey
-------------

Generates keys for DOB year +/- *n* to catch year transcription errors.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``name_field: &str``, ``dob_field: &str``, ``year_delta: i32``
   * - **Key format**
     - One key per year in range: ``"<phonetic_code>:<year-delta>"`` through ``"<phonetic_code>:<year+delta>"``
   * - **Example**
     - ``FuzzyYearKey(year_delta=1)`` on "Berg" + "1978" gives keys for 1977, 1978, 1979
   * - **Domains**
     - SIS II (estimated DOBs), historical registers
   * - **Handles**
     - Off-by-one year errors, estimated birth years

SuffixKey
----------

Takes the last *n* characters of the field value (digits only, or
alphanumeric depending on field kind) as the blocking key.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``field: &str``, ``n: usize``
   * - **Key format**
     - ``"suffix:<last_n_chars>"``
   * - **Example**
     - ``SuffixKey("postcode", 4)`` on "1011AB" gives ``"suffix:1011"``
   * - **Example**
     - ``SuffixKey("telefoon", 7)`` on "0612345678" gives ``"suffix:2345678"``
   * - **Domains**
     - Postcodes (n=4), phone numbers (n=7), BSN (n=4)
   * - **Handles**
     - Prefix formatting differences, country code variations in phone numbers

DocumentSuffixKey
------------------

Like ``SuffixKey`` but strips all non-alphanumeric characters and uppercases
before extracting the suffix. Suitable for document numbers and IBANs.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``field: &str``, ``n: usize``
   * - **Key format**
     - ``"doc_suffix:<last_n_alphanumeric>"``
   * - **Example**
     - ``DocumentSuffixKey("iban", 6)`` on "NL91 ABNA 0417 1643 00" gives ``"doc_suffix:164300"``
   * - **Domains**
     - IBANs, passport numbers, SIS II document IDs

ExactFieldKey
--------------

Uses the exact normalized value of the field as the blocking key. Only
records that share the exact same normalized value become candidates.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``field: &str``
   * - **Key format**
     - ``"exact:<normalized_value>"``
   * - **Example**
     - ``ExactFieldKey("geslacht")`` on "M" gives ``"exact:M"``
   * - **Domains**
     - Gender, nationality, camera IDs, rechtsvorm
   * - **Handles**
     - Low-cardinality categorical fields where exact matching is intended

DateFragmentKey
----------------

Extracts a fragment of the date field at a given granularity.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``field: &str``, ``granularity: DateGranularity``
   * - **Granularity values**
     - ``Year``, ``YearMonth``, ``YearMonthDay``
   * - **Key format**
     - ``"date_frag:<fragment>"``
   * - **Example**
     - ``DateFragmentKey(YearMonth)`` on "1978-03-15" gives ``"date_frag:1978-03"``
   * - **Domains**
     - Secondary key for all person data; primary key for date-only schemas

AddressInitialKey
------------------

Combines the first token of an address field with the first character of a
name field (the first-name initial).

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``address_field: &str``, ``name_field: &str``
   * - **Key format**
     - ``"addr_initial:<first_address_token>_<first_name_initial>"``
   * - **Example**
     - "Amsterdam" + "Johannes" gives ``"addr_initial:AMSTERDAM_J"``
   * - **Domains**
     - BRP, KvK (secondary blocking path when DOB is missing)

LicensePlateNormKey
--------------------

Normalizes the plate (strip hyphens/spaces, uppercase) and uses it as the
blocking key.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``field: &str``
   * - **Key format**
     - ``"plate_norm:<normalized_plate>"``
   * - **Example**
     - "CX-180-W" gives ``"plate_norm:CX180W"``
   * - **Domains**
     - ANPR passages
   * - **Handles**
     - Formatting differences (hyphenation, spacing)

PlateOCRFuzzyKey
-----------------

Generates all single-character OCR confusion variants of the normalized
plate. See :doc:`/explanation/anpr-ocr` for the full confusion table.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``field: &str``
   * - **Key format**
     - One key per variant: ``"plate_ocr:<variant>"``
   * - **Example**
     - "CX180W" gives ``["plate_ocr:CX180W", "plate_ocr:CXI80W", "plate_ocr:CX1B0W", "plate_ocr:CX18OW"]``
   * - **Domains**
     - ANPR passages
   * - **Handles**
     - Single-character OCR confusion: 0/O, 1/I, 8/B, 5/S, 2/Z

CameraTimeWindowKey
--------------------

Generates a key from camera ID + date + time window (bucketed to *window_minutes*).

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``camera_field: &str``, ``timestamp_field: &str``, ``window_minutes: u32``
   * - **Key format**
     - ``"cam_time_window:<camera_id>:<date>:<window_bucket>"``
   * - **Example**
     - CAM-A12-001 + "2025-06-01T10:04:00" + window=10 gives ``"cam_time_window:CAM-A12-001:2025-06-01:0"`` (bucket 0 = 00:00-09:59)
   * - **Domains**
     - ANPR passages, event logs

GeoGridKey
-----------

Truncates latitude and longitude to the given resolution and uses the
grid cell as the blocking key.

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - ``lat_field: &str``, ``lon_field: &str``, ``resolution: f64``
   * - **Key format**
     - ``"geo_grid:<lat_bucket>:<lon_bucket>"``
   * - **Example**
     - lat 52.345, lon 4.901, resolution 0.01 gives ``"geo_grid:52.34:4.90"``
   * - **Domains**
     - ANPR passages, GPS-tagged events
   * - **Handles**
     - Records within the same ~1 sq km grid cell (resolution=0.01 deg)
