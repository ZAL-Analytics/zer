Dutch Name Normalization
=========================

Dutch names have several systematic quirks that trip up standard string
matching. zer applies a normalization pipeline before generating blocking
keys and before computing similarity scores. Understanding this pipeline
helps you predict which records will be grouped together.

The normalization pipeline
---------------------------

Every name field goes through four steps before it is used for blocking or
comparison:

1. **Unicode NFKD decomposition**, decomposes precomposed characters into
   base character + combining diacritical mark (e.g. ``ü`` → ``u`` + combining
   umlaut).
2. **Diacritic stripping**, drops all combining diacritical marks, leaving
   only ASCII characters. ``Müller`` → ``MULLER``, ``Çelik`` → ``CELIK``.
3. **Uppercase conversion**, all characters converted to ASCII uppercase.
4. **Whitespace collapsing**, leading, trailing, and multiple internal spaces
   collapsed to a single space.

.. code-block:: text

   Input:        "  Jörgensen  "
   After NFKD:   "  Jörgensen  "    (ö → o + combining umlaut)
   After strip:  "  Jorgensen  "
   After upper:  "  JORGENSEN  "
   After collapse: "JORGENSEN"

Tussenvoegsel stripping
------------------------

Dutch surnames frequently carry a tussenvoegsel, a particle between first
name and surname, such as *van*, *de*, *van den*, *v.d.*, and others.
The same person may be registered as "van den Berg" in BRP and simply "Berg"
in a KvK extract. zer strips the tussenvoegsel before generating a phonetic
code so both produce the same blocking key.

Recognized prefixes (case-insensitive, normalized):

.. code-block:: text

   VAN DER   VAN DEN   VAN DE   VAN HET   VAN 'T   VAN T   VAN
   DEN       DER       DE       TEN       TER      TE
   IN 'T     IN T      OP DEN   OP DE     OP HET   OP
   V/D       V.D.

.. code-block:: text

   "van den Berg"   → strips "VAN DEN " → surname token "BERG"
   "de Vries"       → strips "DE "      → surname token "VRIES"
   "v.d. Hoeven"    → strips "V.D. "    → surname token "HOEVEN"
   "El Amrani"      → no prefix match   → surname token "AMRANI"

The phonetic algorithm: Double Metaphone
-----------------------------------------

After normalization and tussenvoegsel stripping, zer encodes the surname
token with **Double Metaphone** (not Soundex). Double Metaphone was chosen
because:

* It handles German/Dutch consonant clusters (sch, tsch, kn) better than
  Soundex.
* It produces two codes (primary and secondary) for ambiguous sounds, allowing
  zer to match variant spellings while Soundex collapses too aggressively.
* It handles common Arabic surname sounds (``kh``, ``gh``, ``dj``) more
  gracefully than Soundex.

.. code-block:: text

   "JANSEN"   → Double Metaphone: "JNSN"
   "JANSSEN"  → Double Metaphone: "JNSN"  ← same code ✓
   "HANSEN"   → Double Metaphone: "HNSN"  ← different code (H vs J) ✗
   "BERG"     → Double Metaphone: "PRK"
   "BURK"     → Double Metaphone: "PRK"   ← same code (B/P, R, G/K) ✓

Phonetic code + DOB year key
------------------------------

The blocking key combines the phonetic surname code with the birth year.
This prevents false candidates between people with phonetically similar names
but different birth years:

.. code-block:: text

   Johannes van den Berg, born 1978  →  key: "PRK:1978"
   Joost Berg, born 1978             →  key: "PRK:1978"  ← candidate ✓
   Joost Berg, born 1952             →  key: "PRK:1952"  ← not a candidate ✗

Transliteration for non-Latin scripts
---------------------------------------

For ``WantedPersons`` data (SIS II), names may be entered in Arabic, Cyrillic,
or other scripts. zer uses the ``any_ascii`` crate to transliterate non-Latin
characters to their closest ASCII equivalents before normalization:

.. code-block:: text

   Arabic:   "محمد"     → any_ascii → "mhmd"   → normalize → "MHMD"
   Cyrillic: "Иванов"   → any_ascii → "Ivanov"  → normalize → "IVANOV"
   Greek:    "Παπαδόπουλος" → any_ascii → "Papadopoulos" → "PAPADOPOULOS"

The ``TransliteratedPhoneticKey`` generates a phonetic code from the
transliterated form, in addition to the standard code from the original
input. This allows linking records entered in Latin script to records
entered in the original script.

First-name initials
---------------------

The ``PhoneticNameDobInitialKey`` uses the **first character of the given
name** (after normalization) as an additional discriminator. This addresses
the case where different people share a surname and birth year but have
different given names:

.. code-block:: text

   Johannes van den Berg, born 1978  →  key: "PRK_J_1978"
   Julia van den Berg, born 1978     →  key: "PRK_J_1978"   ← candidate (same initial)
   Maria van den Berg, born 1978     →  key: "PRK_M_1978"   ← not a candidate ✗

Address normalization
----------------------

Address fields use a simpler normalization: the first whitespace-delimited
token is extracted and uppercased. For ``woonplaats`` (city) this is typically
the full city name; for ``straatnaam`` it is the first word of the street name.
Combined with the first-name initial, this produces ``AddressInitialKey``:

.. code-block:: text

   "Amsterdam" + first name "Johannes"  →  key: "AMSTERDAM_J"
   "Rotterdam" + first name "Johannes"  →  key: "ROTTERDAM_J"

What to explore next
---------------------

* :doc:`/reference/blocking-keys`, all blocking keys and their normalization steps.
* :doc:`/explanation/anpr-ocr`, OCR normalization for license plates.
* :doc:`/how-to/blocking-strategy`, choosing keys for your schema.
