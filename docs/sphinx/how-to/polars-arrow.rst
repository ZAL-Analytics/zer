How to Ingest from Polars and Arrow
=====================================

zer works with ``Vec<Record>`` at its ingestion boundary. The
``zer-adapters`` crate provides ``PolarsIngest``, an extension trait on
Polars ``DataFrame`` that converts each row into a ``Record`` in one call.
If you are working directly with Apache Arrow, see the Arrow section below.

Add the dependency
-------------------

.. code-block:: toml

   [dependencies]
   zer          = { version = "1.0", features = ["pipeline"] }
   zer-adapters = { version = "1.0", features = ["polars"] }
   polars       = { version = "0.46", features = ["lazy", "csv"] }

Ingest from a DataFrame
------------------------

``DataFrame::into_records(id_start)`` converts every row into a
``Record``. The ``id_start`` argument sets the ``RecordId`` of the first
row; each subsequent row increments by one.

.. code-block:: rust

   use polars::prelude::*;
   use zer_adapters::PolarsIngest;

   // Load a CSV via Polars
   let df = CsvReadOptions::default()
       .with_has_header(true)
       .try_into_reader_with_file_path(Some("data/brp.csv".into()))?
       .finish()?;

   // Convert: row 0 → Record(1), row 1 → Record(2), …
   let records = df.into_records(1);

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
   * - ``Int8`` … ``Int64``
     - ``Int(i64)``
     - Widened to ``i64``
   * - ``UInt8`` … ``UInt32``
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
   * - Date, Datetime, Duration, …
     - ``Text(String)``
     - Polars Display representation (ISO-8601)

.. note::

   Temporal types (``Date``, ``Datetime``) are serialized via Polars'
   ``Display`` implementation. For ``geboortedatum`` fields, ensure the column
   has the format ``YYYY-MM-DD`` before ingesting, or use ``.cast(Utf8)``
   with an explicit format string.

Multi-source ingestion
-----------------------

When linking two DataFrames, offset the second frame's IDs to avoid
namespace collisions and apply source labels:

.. code-block:: rust

   use zer_adapters::PolarsIngest;
   use zer_pipeline::label_source;

   let df_a = load_csv("source_a.csv")?;
   let df_b = load_csv("source_b.csv")?;

   let n_a = df_a.height() as u64;

   // Source A IDs: 1 … n_a
   let records_a = df_a.into_records(1);
   // Source B IDs: n_a+1 … n_a+n_b (no overlap)
   let records_b = df_b.into_records(n_a + 1);

   let all = [
       label_source(records_a, "A"),
       label_source(records_b, "B"),
   ].concat();

Ingesting from Apache Arrow
-----------------------------

If you have an Arrow ``RecordBatch`` or ``Schema``, convert it to a Polars
``DataFrame`` first, then use ``into_records``:

.. code-block:: rust

   use polars::prelude::*;
   use zer_adapters::PolarsIngest;

   // From an Arrow RecordBatch (e.g. from Parquet, IPC, or Flight)
   let df = DataFrame::try_from(arrow_record_batch)?;
   let records = df.into_records(1);

For streaming Arrow (e.g. large Parquet files), read in chunks and
accumulate a ``Vec<Record>`` with growing ID offsets:

.. code-block:: rust

   use polars::prelude::*;
   use zer_adapters::PolarsIngest;

   let mut all_records: Vec<Record> = Vec::new();
   let mut id_cursor: u64 = 1;

   for path in parquet_files {
       let df = LazyFrame::scan_parquet(path, Default::default())?
           .collect()?;
       let n = df.height() as u64;
       all_records.extend(df.into_records(id_cursor));
       id_cursor += n;
   }

Selecting and renaming columns
--------------------------------

zer matches columns to schema fields by name. If your DataFrame uses
different column names than your schema field names, rename before
converting:

.. code-block:: rust

   use polars::prelude::*;
   use zer_adapters::PolarsIngest;

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
       .select([col("voornamen"), col("achternaam"), col("geboortedatum"), col("postcode")])
       .collect()?;

   let records = df.into_records(1);

What to explore next
---------------------

* :doc:`/how-to/define-schema`, define the schema that matches your column names.
* :doc:`/tutorials/cross-source-linkage`, full walkthrough using CSV ingestion.
* :doc:`/reference/field-kind`, which ``FieldKind`` to assign each column type.
