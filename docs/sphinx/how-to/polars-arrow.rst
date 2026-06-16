How to Ingest from Polars and Arrow
=====================================

zer works with ``Vec<Record>`` at its ingestion boundary. The
``zer-adapters`` crate provides ``PolarsIngest``, an extension trait on
Polars ``DataFrame`` that converts each row into a ``Record`` in one call.
If you are working directly with Apache Arrow ``RecordBatch`` objects, see
the Arrow section below.

Both adapters require a ``DatasetConfig`` that names the source label and
the column to use as the record's natural key. The adapter derives each
record's internal ``RecordId`` via ``FNV-1a(source:key)``, so IDs are
stable across runs and you never need to manage integer offsets manually.

Add the dependency
-------------------

.. code-block:: toml

   [dependencies]
   zer          = { version = "1.1", features = ["pipeline"] }
   zer-adapters = { version = "1.1", features = ["polars"] }
   polars       = { version = "0.46", features = ["lazy", "csv"] }

Ingest from a DataFrame
------------------------

Create a ``DatasetConfig`` with the source label and the name of the
column that holds the record's natural key (e.g. a BSN, UUID, or
primary-key column), then pass it to ``into_records``:

.. code-block:: rust

   use polars::prelude::*;
   use zer_adapters::{DatasetConfig, PolarsIngest};

   let df = CsvReadOptions::default()
       .with_has_header(true)
       .try_into_reader_with_file_path(Some("data/brp.csv".into()))?
       .finish()?;

   let config  = DatasetConfig::new("brp", "bsn");
   let records = df.into_records(&config);

Each record's ``id`` is derived from ``FNV-1a("brp:bsn_value")`` and its
``key`` field holds the raw value from the ``bsn`` column. Both are
stored in the ``.zes`` entity output, so cluster results map directly to
your own identifiers.

Column-to-field mapping
------------------------

Every column in the DataFrame becomes a field in the ``Record`` under the
column's name. The Polars type is converted to ``FieldValue`` as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Polars ``AnyValue``
     - ``FieldValue``
     - Notes
   * - ``Null``
     - ``Null``
     - Treated as missing by the comparator
   * - ``String`` / ``StringOwned``
     - ``Text(String)``
     -
   * - ``Int8`` to ``Int64``
     - ``Int(i64)``
     - Widened to ``i64``
   * - ``UInt8`` to ``UInt32``
     - ``Int(i64)``
     - Widened to ``i64``
   * - ``UInt64``
     - ``UInt(u64)``
     - Preserved as unsigned to avoid precision loss
   * - ``Float32`` / ``Float64``
     - ``Float(f64)``
     -
   * - ``Boolean``
     - ``Bool(bool)``
     -
   * - ``Binary`` / ``BinaryOwned``
     - ``Bytes(Vec<u8>)``
     -
   * - Date, Datetime, Duration, ...
     - ``Text(String)``
     - Polars Display representation (ISO-8601)

.. note::

   Temporal types (``Date``, ``Datetime``) are serialized via Polars'
   ``Display`` implementation. For ``geboortedatum`` fields, ensure the column
   has the format ``YYYY-MM-DD`` before ingesting, or use ``.cast(Utf8)``
   with an explicit format string.

Multi-source ingestion
-----------------------

When linking two DataFrames, give each source its own ``DatasetConfig``.
Because IDs are derived from ``FNV-1a(source:key)``, records from
different sources with the same natural key value will still get distinct
IDs. no manual offset management is needed:

.. code-block:: rust

   use zer_adapters::{DatasetConfig, PolarsIngest};

   let df_a = load_csv("source_a.csv")?;
   let df_b = load_csv("source_b.csv")?;

   let records_a = df_a.into_records(&DatasetConfig::new("A", "person_id"));
   let records_b = df_b.into_records(&DatasetConfig::new("B", "person_id"));

   let all = [records_a, records_b].concat();

Ingesting from Apache Arrow
-----------------------------

``ArrowIngest`` works directly with an Arrow ``RecordBatch`` and has the
same ``DatasetConfig``-based API:

.. code-block:: rust

   use zer_adapters::{ArrowIngest, DatasetConfig};
   use arrow_array::{RecordBatch, Int64Array, StringArray};
   use arrow_schema::{DataType, Field, Schema};
   use std::sync::Arc;

   let schema = Arc::new(Schema::new(vec![
       Field::new("bsn",  DataType::Utf8,  false),
       Field::new("name", DataType::Utf8,  false),
       Field::new("age",  DataType::Int64, false),
   ]));

   // ... build or receive a RecordBatch ...

   let config  = DatasetConfig::new("brp", "bsn");
   let records = batch.into_records(&config);

For streaming Arrow (e.g. large Parquet files), read in chunks and
accumulate a ``Vec<Record>``. no cursor arithmetic needed:

.. code-block:: rust

   use zer_adapters::{DatasetConfig, PolarsIngest};

   let config = DatasetConfig::new("brp", "bsn");
   let mut all_records: Vec<Record> = Vec::new();

   for path in parquet_files {
       let df = LazyFrame::scan_parquet(path, Default::default())?
           .collect()?;
       all_records.extend(df.into_records(&config));
   }

Selecting and renaming columns
--------------------------------

zer matches columns to schema fields by name. If your DataFrame uses
different column names than your schema field names, rename before
converting:

.. code-block:: rust

   use polars::prelude::*;
   use zer_adapters::{DatasetConfig, PolarsIngest};

   let df = CsvReadOptions::default()
       .try_into_reader_with_file_path(Some("kvk_export.csv".into()))?
       .finish()?
       .lazy()
       // Rename CSV columns to match zer schema field names
       .rename(
           ["FirstName",  "LastName",   "DateOfBirth"],
           ["voornamen",  "achternaam", "geboortedatum"],
           true,
       )
       // Drop columns zer doesn't need
       .select([col("kvk_nr"), col("voornamen"), col("achternaam"), col("geboortedatum"), col("postcode")])
       .collect()?;

   let records = df.into_records(&DatasetConfig::new("kvk", "kvk_nummer"));

What to explore next
---------------------

* :doc:`/how-to/define-schema`, define the schema that matches your column names.
* :doc:`/tutorials/cross-source-linkage`, full walkthrough using CSV ingestion.
* :doc:`/reference/field-kind`, which ``FieldKind`` to assign each column type.
