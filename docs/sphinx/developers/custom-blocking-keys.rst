Custom Blocking Keys
=====================

A blocking key is a function that maps a ``Record`` to a compact string
token. The blocker groups records by token; only records that share at least
one token become candidate pairs. This keeps the number of comparisons
tractable: two records that agree on no token are never compared.

zer ships with Dutch-specific blocking keys (phonetic name encoding,
tussenvoegsel normalization, license plate OCR variants). For other domains
you need your own.

The ``BlockingKey`` trait
--------------------------

The trait lives in ``zer_blocking``:

.. code-block:: rust

   use zer_blocking::BlockingKey;
   use zer_core::Record;

   pub trait BlockingKey: Send + Sync + 'static {
       /// Name shown in debug output and audit logs.
       fn name(&self) -> &str;

       /// Extract zero or more tokens from a record. Returning multiple tokens
       /// increases recall at the cost of more candidate pairs.
       fn extract(&self, record: &Record) -> Vec<String>;
   }

Returning an empty ``Vec`` from ``extract`` means the record participates in
no blocking group for this key, which is correct for genuinely missing fields.
Returning many tokens per record can cause a candidate explosion,aim for
tokens that are selective enough that a group has at most a few hundred
members.

IBAN prefix blocking
---------------------

For financial record linkage, block on the first eight characters of an IBAN
(country code + check digits + first four bank code characters). Two IBANs
that share this prefix almost certainly belong to accounts at the same bank,
dramatically shrinking the comparison space.

.. code-block:: rust

   use zer_blocking::BlockingKey;
   use zer_core::Record;

   pub struct IbanPrefixKey {
       field: String,
   }

   impl IbanPrefixKey {
       pub fn new(field_name: impl Into<String>) -> Self {
           Self { field: field_name.into() }
       }
   }

   impl BlockingKey for IbanPrefixKey {
       fn name(&self) -> &str { "iban_prefix" }

       fn extract(&self, record: &Record) -> Vec<String> {
           let Some(value) = record.get_text(&self.field) else { return vec![] };
           // Normalise: strip spaces, uppercase
           let iban: String = value.chars()
               .filter(|c| !c.is_whitespace())
               .map(|c| c.to_ascii_uppercase())
               .collect();
           if iban.len() < 8 { return vec![]; }
           vec![iban[..8].to_owned()]
       }
   }

Geo-cell blocking with H3
--------------------------

For location datasets (e.g. address matching, telemetry), block on the H3
hexagonal grid cell at resolution 7 (~5 km² cells). Records in the same cell
are candidates; records in adjacent cells are found via multi-key expansion.

Add the dependency:

.. code-block:: toml

   [dependencies]
   h3o = { version = "0.7" }

.. code-block:: rust

   use h3o::{CellIndex, LatLng, Resolution};
   use zer_blocking::BlockingKey;
   use zer_core::Record;

   pub struct H3BlockingKey {
       lat_field: String,
       lng_field: String,
       resolution: Resolution,
   }

   impl H3BlockingKey {
       /// resolution 7 → ~5 km², resolution 8 → ~0.7 km²
       pub fn new(lat: impl Into<String>, lng: impl Into<String>, resolution: u8) -> Self {
           Self {
               lat_field:  lat.into(),
               lng_field:  lng.into(),
               resolution: Resolution::try_from(resolution).unwrap_or(Resolution::Seven),
           }
       }
   }

   impl BlockingKey for H3BlockingKey {
       fn name(&self) -> &str { "h3_cell" }

       fn extract(&self, record: &Record) -> Vec<String> {
           let lat = record.get_float(&self.lat_field)?;
           let lng = record.get_float(&self.lng_field)?;
           let coord = LatLng::new(lat, lng).ok()?;
           let cell  = coord.to_cell(self.resolution);

           // Also emit the 6 immediate neighbours for edge-crossing pairs
           let mut tokens = vec![cell.to_string()];
           tokens.extend(cell.grid_disk::<Vec<_>>(1).into_iter().map(|c| c.to_string()));
           Some(tokens)
       }
   }

.. note::

   Emitting neighbour cells increases recall for pairs that sit near a cell
   boundary. The trade-off is roughly a 7× increase in candidate pairs for
   this key. Combine with a second, more selective key (e.g. postal code) to
   keep the total candidate count manageable.

Registering a custom key with the schema
------------------------------------------

Pass custom blocking keys to the schema builder alongside or instead of the
built-in presets:

.. code-block:: rust

   use zer_schema::SchemaBuilder;
   use zer_blocking::BlockingKeySet;

   let keys = BlockingKeySet::new()
       .add(IbanPrefixKey::new("iban"))
       .add(H3BlockingKey::new("lat", "lng", 7));

   let schema = SchemaBuilder::new()
       .field("iban",    FieldKind::Text)
       .field("lat",     FieldKind::Float)
       .field("lng",     FieldKind::Float)
       .field("name",    FieldKind::PersonName)
       .blocking_keys(keys)
       .build()?;

Measuring blocking recall
--------------------------

A blocking key that is too aggressive misses true matches,pairs that should
have been compared but were not. Measure recall on a labelled sample before
deploying to production:

.. code-block:: rust

   use zer_blocking::RecallAudit;

   let audit = RecallAudit::run(&schema, &ground_truth_pairs, &records)?;
   println!(
       "blocking recall: {:.1}%  ({} / {} true pairs found)",
       audit.recall * 100.0,
       audit.found,
       audit.total,
   );

A recall below 95 % usually means a blocking key is too narrow or a field has
too many missing values. See :doc:`/explanation/blocking-recall` for the
theory behind this trade-off.

What to explore next
---------------------

* :doc:`/explanation/blocking-recall`, how recall and candidate count trade off and how to reason about that for your domain.
* :doc:`/how-to/blocking-strategy`, choosing between the built-in blocking keys for Dutch administrative data.
* :doc:`custom-similarity-functions`, add a matching metric to complement your new blocking key.
