SchemaCategory Reference
=========================

``SchemaCategory`` presets select a curated set of blocking keys tuned for a
specific domain. Pass to ``BlockerFactory::from_schema_category`` to get a
``CompositeBlocker`` without manually specifying individual keys.

PersonRegistry
---------------

**Use for:** BRP (population register), KvK director extracts, and any general
person dataset with name + DOB fields.

Applies the same logic as ``BlockerFactory::from_schema`` (the automatic
fallback). Key selection depends on which ``FieldKind``\s are present:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Schema fields
     - Keys added
   * - 2+ Name + Date
     - ``PhoneticNameDobInitialKey`` (surname phonetic + first-name initial + year)
   * - 1 Name + Date
     - ``PhoneticNameDobKey`` (surname phonetic + year)
   * - Name + Date (any)
     - ``DateFragmentKey(YearMonth)`` as secondary key
   * - Name + Address
     - ``AddressInitialKey`` (first address token + first-name initial)
   * - Phone
     - ``SuffixKey(7)`` on each Phone field
   * - Id
     - ``SuffixKey(4)`` on each Id field
   * - Date only (no Name)
     - ``DateFragmentKey(YearMonth)``
   * - Categorical
     - ``ExactFieldKey`` on each Categorical field

WantedPersons
--------------

**Use for:** SIS II wanted/missing persons alerts. Names may be transliterated,
DOBs may be estimated (year-only or off-by-one), and records often carry
multiple aliases.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Keys added
     - Purpose
   * - ``PhoneticNameDobKey``
     - Primary surname phonetic + DOB year
   * - ``TransliteratedPhoneticKey``
     - Cross-script variant for Arabic/Cyrillic input
   * - ``FuzzyYearKey(±1)``
     - Off-by-one year errors in estimated DOBs
   * - ``AliasPhoneticKey`` (per alias field)
     - Phonetic code of each alias surname + DOB year
   * - ``DocumentSuffixKey(6)`` (per Id field)
     - Last 6 alphanumeric characters of document/passport numbers

ANPRPassages
-------------

**Use for:** ANPR camera passage logs. Plates contain OCR errors; records
may be from different cameras at different times and locations.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Keys added
     - Purpose
   * - ``LicensePlateNormKey`` (per LicensePlate field)
     - Normalized plate string (no hyphens/spaces)
   * - ``PlateOCRFuzzyKey`` (per LicensePlate field)
     - All single-character OCR confusion variants
   * - ``CameraTimeWindowKey(window=10 min)``
     - Camera ID + date + 10-minute time window bucket
   * - ``GeoGridKey(resolution=0.01°)``
     - ~1 km² geo grid cell from lat + lon pair

CallDetailRecords
------------------

**Use for:** CDR (Call Detail Records), phone call and data logs linking
subscriber IDs and cell towers.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Keys added
     - Purpose
   * - ``SuffixKey(7)`` (per Phone field)
     - Last 7 digits of phone numbers
   * - ``SuffixKey(6)`` (per Id field)
     - IMSI, ICCID suffix
   * - ``ExactFieldKey`` (per Categorical field)
     - Cell tower ID, carrier code

SIMSubscribers
---------------

**Use for:** SIM subscriber snapshots. Same keys as ``CallDetailRecords``, phone suffix, IMSI/ICCID suffix, categorical fields.

Identical to ``CallDetailRecords`` key selection; provided as a separate
category name for documentation clarity.

FinancialIntelligence
----------------------

**Use for:** FIU financial intelligence reports. Links bank accounts,
transactions, and entities across multiple reporting sources.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Keys added
     - Purpose
   * - ``SuffixKey(6)`` (per Id field)
     - Account number / IBAN / transaction ID suffix
   * - ``DateFragmentKey(YearMonth)``
     - Transaction date year-month bucket
   * - ``ExactFieldKey`` (per Categorical field)
     - Institution code, currency, transaction type

Custom categories
------------------

When no preset fits, use ``CustomSchemaCategory`` to compose your own key set.
See :doc:`/how-to/blocking-strategy` for the full API.

.. code-block:: rust

   use zer_blocking::{BlockerFactory, CustomSchemaCategory};

   let category = CustomSchemaCategory::new()
       .with_phonetic_name_dob()
       .with_id_suffix(4)
       .with_exact_categorical();

   let blocker = BlockerFactory::from_custom_category(&schema, category);
