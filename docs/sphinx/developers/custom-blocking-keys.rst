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

The trait lives in ``zer_blocking::keys``:

.. code-block:: rust

   use zer_blocking::keys::BlockingKey;
   use zer_core::{record::Record, schema::Schema};

   pub trait BlockingKey: Send + Sync + 'static {
       /// Name shown in debug output and audit logs.
       fn name(&self) -> &str;

       /// Extract zero or more tokens from a record. The ``schema`` parameter
       /// gives access to field metadata (index, kind) when needed.
       /// Returning multiple tokens increases recall at the cost of more
       /// candidate pairs.
       fn extract(&self, record: &Record, schema: &Schema) -> Vec<String>;
   }

Returning an empty ``Vec`` from ``extract`` means the record participates in
no blocking group for this key, which is correct for genuinely missing fields.
Returning many tokens per record can cause a candidate explosion; aim for
tokens that are selective enough that a group has at most a few hundred
members.

IBAN prefix blocking
---------------------

For financial record linkage, block on the first eight characters of an IBAN
(country code + check digits + first four bank code characters). Two IBANs
that share this prefix almost certainly belong to accounts at the same bank,
dramatically shrinking the comparison space.

.. code-block:: rust

   use zer_blocking::keys::BlockingKey;
   use zer_core::{record::Record, schema::Schema};

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

       fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
           let Some(value) = record.text(&self.field) else { return vec![] };
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
   use zer_blocking::keys::BlockingKey;
   use zer_core::{record::Record, schema::Schema};

   pub struct H3BlockingKey {
       lat_field:  String,
       lng_field:  String,
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

       fn extract(&self, record: &Record, _schema: &Schema) -> Vec<String> {
           let lat = match record.field_as::<f64>(&self.lat_field) {
               Some(v) => v,
               None    => return vec![],
           };
           let lng = match record.field_as::<f64>(&self.lng_field) {
               Some(v) => v,
               None    => return vec![],
           };
           let coord = match LatLng::new(lat, lng) {
               Ok(c)  => c,
               Err(_) => return vec![],
           };
           let cell = coord.to_cell(self.resolution);

           // Also emit the 6 immediate neighbours for edge-crossing pairs
           let mut tokens = vec![cell.to_string()];
           tokens.extend(cell.grid_disk::<Vec<_>>(1).into_iter().map(|c| c.to_string()));
           tokens
       }
   }

.. note::

   Emitting neighbour cells increases recall for pairs that sit near a cell
   boundary. The trade-off is roughly a 7 times increase in candidate pairs for
   this key. Combine with a second, more selective key (e.g. postal code) to
   keep the total candidate count manageable.

Registering custom keys with the pipeline
------------------------------------------

Pass custom blocking keys to a ``CompositeBlocker`` and supply it to the
pipeline builder. The schema and blocker are separate: the schema describes
field types; the blocker decides which records become candidate pairs.

.. code-block:: rust

   use zer_core::schema::{FieldKind, SchemaBuilder};
   use zer_blocking::CompositeBlocker;
   use zer_pipeline::pipeline::Pipeline;

   let schema = SchemaBuilder::new()
       .field("iban", FieldKind::Id)
       .field("lat",  FieldKind::Numeric)
       .field("lng",  FieldKind::Numeric)
       .field("name", FieldKind::Name)
       .build()?;

   let blocker = CompositeBlocker::new()
       .add(IbanPrefixKey::new("iban"))
       .add(H3BlockingKey::new("lat", "lng", 7));

   let pipeline = Pipeline::builder()
       .schema(schema)
       .blocker(blocker)
       .store(store)
       .build()?;

Fields not used by any blocking key are still compared once a pair is
generated by another key — blocking only controls which pairs are formed,
not which fields are scored.

Measuring blocking recall
--------------------------

A blocking key that is too aggressive misses true matches — pairs that should
have been compared but were not. Measure recall on a labelled sample before
deploying to production by checking how many ground-truth pairs share at
least one blocking token:

.. code-block:: rust

   let mut found = 0usize;
   let total = ground_truth_pairs.len();

   for (id_a, id_b) in &ground_truth_pairs {
       let record_a = &records[*id_a];
       let record_b = &records[*id_b];
       let tokens_a: std::collections::HashSet<String> =
           my_key.extract(record_a, &schema).into_iter().collect();
       let tokens_b: std::collections::HashSet<String> =
           my_key.extract(record_b, &schema).into_iter().collect();
       if !tokens_a.is_disjoint(&tokens_b) {
           found += 1;
       }
   }
   println!("blocking recall: {:.1}%  ({found} / {total} true pairs found)",
            found as f64 / total as f64 * 100.0);

A recall below 95 % usually means a blocking key is too narrow or a field has
too many missing values. See :doc:`/explanation/blocking-recall` for the
theory behind this trade-off.

What to explore next
---------------------

* :doc:`/explanation/blocking-recall`, how recall and candidate count trade off and how to reason about that for your domain.
* :doc:`/how-to/blocking-strategy`, choosing between the built-in blocking keys for Dutch administrative data.
* :doc:`custom-similarity-functions`, add a matching metric to complement your new blocking key.
