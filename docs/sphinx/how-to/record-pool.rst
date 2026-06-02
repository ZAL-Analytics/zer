How to Use a RecordPool
========================

``RecordPool`` is a flat, column-major in-memory store that holds a snapshot of
``Record`` values for batch comparison. Instead of a ``HashMap<FieldName, FieldValue>``
per record, it flattens all field values into contiguous ``Vec<String>`` columns::

   columns[field_idx][record_idx] = UTF-8 string value

This layout lets comparators access every field of every record with a direct
array index and no hashing or allocation per lookup. All non-text values
(integers, floats, booleans) are converted to their string representation.
``Null``, ``Bytes``, and missing fields become empty strings, which every
comparator treats as ``ComparisonLevel::None``.

``RecordPool`` is the required input for
``Comparator::compare_batch_from_pool``, the hot-path comparison entry point
used by the pipeline and scoring layer.

Import
------

.. code-block:: rust

   use zer_core::RecordPool;
   // or the explicit module path:
   use zer_core::record_pool::RecordPool;

Constructors
------------

``RecordPool`` provides four constructors depending on where your data comes
from.

From a slice of records
~~~~~~~~~~~~~~~~~~~~~~~

Build a pool from records you already hold in memory:

.. code-block:: rust

   use zer_core::{
       record::{FieldValue, Record},
       record_pool::RecordPool,
       schema::{FieldKind, SchemaBuilder},
   };

   let schema = SchemaBuilder::new()
       .field("naam",          FieldKind::Name)
       .field("geboortedatum", FieldKind::Date)
       .build()?;

   let records = vec![
       Record::new(1)
           .insert("naam",          FieldValue::Text("Alice".into()))
           .insert("geboortedatum", FieldValue::Text("1990-01-01".into())),
       Record::new(2)
           .insert("naam",          FieldValue::Text("Bob".into()))
           .insert("geboortedatum", FieldValue::Text("1985-06-15".into())),
   ];

   let pool = RecordPool::from_records(&records, &schema);
   assert_eq!(pool.len(), 2);

From record pairs
~~~~~~~~~~~~~~~~~

When you have a list of ``(Record, Record)`` candidate pairs, ``from_pairs``
interleaves them so that pool position ``2*i`` is side A and ``2*i+1`` is side B
of pair ``i``. Build the pool once and pass it directly to
``compare_batch_from_pool``:

.. code-block:: rust

   use zer_compare::FieldComparator;
   use zer_core::{record_pool::RecordPool, traits::Comparator};

   let pairs: Vec<(Record, Record)> = vec![
       (alice_canonical.clone(), alice_variant.clone()),
       (alice_canonical.clone(), bob.clone()),
   ];

   let pool    = RecordPool::from_pairs(&pairs, &schema);
   let indices: Vec<(usize, usize)> =
       (0..pairs.len()).map(|i| (i * 2, i * 2 + 1)).collect();

   let comparator = FieldComparator::from_schema(&schema);
   let batch      = comparator.compare_batch_from_pool(&pool, &indices, &schema);

From a RecordStore
~~~~~~~~~~~~~~~~~~

Load a subset of records from any ``RecordStore`` by ID. Pool position ``i``
corresponds to ``ids[i]``; records not found in the store are silently skipped:

.. code-block:: rust

   use zer_core::record_pool::RecordPool;

   // ids_to_load typically comes from blocking output.
   let ids_to_load: Vec<u64> = vec![1, 2, 3, 42];
   let pool = RecordPool::from_store(&*record_store, &ids_to_load, &schema);

Incremental construction
~~~~~~~~~~~~~~~~~~~~~~~~

Build a pool record-by-record with ``new`` and ``push``. Use ``with_capacity``
when you know the final size upfront to avoid reallocations:

.. code-block:: rust

   use zer_core::record_pool::RecordPool;

   let mut pool = RecordPool::with_capacity(1000, schema.fields.len());
   for record in &incoming_records {
       pool.push(record, &schema);
   }

Accessing data
--------------

``RecordPool`` exposes three direct accessors:

.. code-block:: rust

   // Number of records in the pool.
   let n = pool.len();

   // Field value as a string: pool.get(field_idx, record_idx).
   // field_idx is the 0-based position of the field in schema.fields.
   let naam = pool.get(0, 0);  // field 0, record 0 -> "Alice"

   // RecordId for the record at position r.
   let id: u64 = pool.ids[0];

Fields are indexed in the same order they were declared in ``SchemaBuilder``.

Missing values
--------------

``RecordPool`` stores every field as a ``String``. When a field is absent,
``Null``, or a ``Bytes`` value, it is stored as an empty string. Every
built-in comparator interprets an empty string as ``ComparisonLevel::None``,
so missing data degrades gracefully without a panic:

.. code-block:: rust

   // Record with no geboortedatum field.
   let r = Record::new(3).insert("naam", FieldValue::Text("Charlie".into()));
   let pool = RecordPool::from_records(&[r], &schema);

   assert_eq!(pool.get(0, 0), "Charlie");
   assert_eq!(pool.get(1, 0), "");  // missing field -> empty string

What to explore next
--------------------

* :doc:`/developers/custom-record-store` -- implement a disk-backed store (e.g. RocksDB).
* :doc:`/developers/custom-similarity-functions` -- write a custom comparator that reads from a ``RecordPool``.
* :doc:`/how-to/tune-scorer` -- use batch comparison output to train a ``FellegiSunterScorer``.
