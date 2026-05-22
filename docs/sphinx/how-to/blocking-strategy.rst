How to Choose a Blocking Strategy
===================================

Blocking is the step that produces candidate pairs without comparing every
record to every other. Choosing the right strategy is the most important
tuning decision in zer: too few keys miss real matches (low recall); too many
keys flood the comparator with false candidates (slow throughput).

zer offers three layers of control:

1. **Automatic**, ``BlockerFactory::from_schema`` infers keys from ``FieldKind`` annotations.
2. **Category preset**, ``BlockerFactory::from_schema_category`` applies a domain-tuned key set.
3. **Custom**, ``CustomSchemaCategory`` lets you compose exactly the keys you want.

Automatic blocking (``from_schema``)
--------------------------------------

The simplest option. Pass the schema and get a ``CompositeBlocker`` whose keys
are chosen by ``FieldKind`` heuristics:

.. code-block:: rust

   use zer_blocking::BlockerFactory;
   use zer_core::schema::{FieldKind, SchemaBuilder};

   let schema = SchemaBuilder::new()
       .field("voornamen",     FieldKind::Name)
       .field("achternaam",    FieldKind::Name)
       .field("geboortedatum", FieldKind::Date)
       .field("postcode",      FieldKind::Id)
       .build()?;

   let blocker = BlockerFactory::from_schema(&schema);

Priority rules applied (in order):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Schema contains
     - Keys added
   * - 2+ Name fields + Date
     - ``PhoneticNameDobInitialKey`` (surname phonetic + first-name initial + DOB year)
   * - 1 Name field + Date
     - ``PhoneticNameDobKey`` (surname phonetic + DOB year)
   * - Name + Date (any count)
     - Also adds ``DateFragmentKey(YearMonth)`` as a secondary key
   * - Name + Address
     - ``AddressInitialKey`` (first address token + first-name initial)
   * - Phone field
     - ``SuffixKey(7)`` on the first Phone field
   * - Id field(s)
     - ``SuffixKey(4)`` on each Id field
   * - Date only (no Name)
     - ``DateFragmentKey(YearMonth)``
   * - Categorical field(s)
     - ``ExactFieldKey`` on each Categorical field

Domain category presets
------------------------

Use ``from_schema_category`` when your dataset fits a known domain. The preset
applies a curated key combination tuned for that domain's specific error modes.

.. code-block:: rust

   use zer_blocking::{BlockerFactory, SchemaCategory};

   // Dutch population / commercial register (the default for person data)
   let blocker = BlockerFactory::from_schema_category(&schema, SchemaCategory::PersonRegistry);

   // SIS II wanted/missing persons: translit keys, alias phonetics, fuzzy DOB year
   let blocker = BlockerFactory::from_schema_category(&schema, SchemaCategory::WantedPersons);

   // ANPR camera passages: plate norm, OCR fuzzy, camera+time window, geo grid
   let blocker = BlockerFactory::from_schema_category(&schema, SchemaCategory::ANPRPassages);

   // CDR / SIM: phone suffix, IMSI/ICCID suffix, categorical
   let blocker = BlockerFactory::from_schema_category(&schema, SchemaCategory::CallDetailRecords);

   // FIU financial intelligence: account/transaction ID suffix, date fragments
   let blocker = BlockerFactory::from_schema_category(&schema, SchemaCategory::FinancialIntelligence);

Custom category
----------------

When no preset fits exactly, use ``CustomSchemaCategory`` to compose your own
key set from the same building blocks. Each ``with_*`` method appends one rule:

.. code-block:: rust

   use zer_blocking::{BlockerFactory, CustomSchemaCategory};
   use zer_blocking::keys::DateGranularity;

   let category = CustomSchemaCategory::new()
       .with_phonetic_name_dob()        // PhoneticNameDobKey on last Name + first Date
       .with_address_initial()          // AddressInitialKey on first Address + first Name
       .with_id_suffix(4)               // SuffixKey(4) on each Id field
       .with_exact_categorical();       // ExactFieldKey on each Categorical field

   let blocker = BlockerFactory::from_custom_category(&schema, category);

Available rules:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Effect
   * - ``with_phonetic_name_dob()``
     - ``PhoneticNameDobKey`` on last Name field + first Date field
   * - ``with_phonetic_name_dob_initial()``
     - ``PhoneticNameDobInitialKey`` when 2+ Name fields; falls back to ``PhoneticNameDobKey``
   * - ``with_address_initial()``
     - ``AddressInitialKey`` on first Address field + first Name field
   * - ``with_id_suffix(n)``
     - ``SuffixKey(n)`` on each Id field
   * - ``with_document_suffix(n)``
     - ``DocumentSuffixKey(n)`` on each Id field (strips non-alphanumeric before suffix)
   * - ``with_phone_suffix(n)``
     - ``SuffixKey(n)`` on each Phone field
   * - ``with_exact_categorical()``
     - ``ExactFieldKey`` on each Categorical field
   * - ``with_date_fragment(granularity)``
     - ``DateFragmentKey`` with ``Year``, ``YearMonth``, or ``YearMonthDay``
   * - ``with_key(key)``
     - Any type implementing ``BlockingKey``, the escape hatch

Escape hatch: injecting a raw key
-----------------------------------

``with_key()`` accepts any type that implements ``BlockingKey``, so you can
plug in blocking keys that the built-in rules do not expose:

.. code-block:: rust

   use zer_blocking::{BlockerFactory, CustomSchemaCategory};
   use zer_blocking::keys::SuffixKey;

   // Only block on postcode suffix, useful when names are unreliable
   let category = CustomSchemaCategory::new()
       .with_key(SuffixKey::new("postcode", 4));

   let blocker = BlockerFactory::from_custom_category(&schema, category);

Inspecting the generated keys
-------------------------------

``blocker.blocking_keys(record, schema)`` returns the full list of string keys
that would be emitted for a given record. Use it to verify your strategy before
running a batch:

.. code-block:: rust

   use zer_core::traits::Blocker;

   let keys = blocker.blocking_keys(&r1, &schema);
   for key in &keys {
       println!("{key}");
   }

Example output for a person record with ``PersonRegistry``:

.. code-block:: text

   phonetic_dob_initial:ALS_A_1990
   address_initial:AMSTERDAM_A
   suffix:1011AB
   exact_cat:Nederland

Combining multiple candidates
-------------------------------

``blocker.candidates(record, schema, index)`` returns the union of all records
reachable from any of the record's keys. A record only needs to share one key to
become a candidate, you do not need to tune threshold values at this stage.

.. code-block:: rust

   use zer_blocking::InvertedIndex;
   use zer_core::traits::Blocker;

   let mut index = InvertedIndex::new();
   for record in &records {
       blocker.index_record(record, &schema, &mut index);
   }

   let candidates = blocker.candidates(&query_record, &schema, &index);
   println!("candidate count: {}", candidates.len());

.. note::

   Missing fields produce no keys and are silently skipped. A record with a
   ``Null`` postcode simply emits no suffix key, it is neither a blocker for
   others nor blocked by others through that key.

What to explore next
---------------------

* :doc:`/explanation/blocking-recall`, why recall is the primary blocking metric.
* :doc:`/reference/blocking-keys`, every key, its parameters, and the domains it targets.
* :doc:`/reference/schema-categories`, full list of ``SchemaCategory`` presets and their key sets.
