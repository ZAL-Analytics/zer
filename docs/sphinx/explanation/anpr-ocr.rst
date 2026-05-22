ANPR OCR Confusion and Bidirectional Keys
==========================================

Automatic Number Plate Recognition (ANPR) cameras read license plates under
real-world conditions: motion blur, rain, low angle, and varying contrast. The
OCR engine makes systematic errors where characters that look alike are
confused. zer's ``PlateOCRFuzzyKey`` is designed specifically to bridge these
pairs at the blocking stage.

The OCR confusion table
------------------------

The following character pairs are visually similar in ANPR camera images.
zer handles all five pairs bidirectionally:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Digit
     - Letter
     - Notes
   * - ``0``
     - ``O``
     - Zero vs letter O; very common on old cameras or cold temperatures
   * - ``1``
     - ``I``
     - One vs letter I; also confused with ``L`` in some fonts
   * - ``8``
     - ``B``
     - Eight vs letter B; especially at low resolution
   * - ``5``
     - ``S``
     - Five vs letter S; common in plates starting with ``SS``
   * - ``2``
     - ``Z``
     - Two vs letter Z; less common but systematic

Only single-character substitutions are generated. Double errors (e.g. both a
1→I and a 0→O in the same plate) are not covered because they produce an
exponential key explosion for a very small coverage gain.

How the keys are generated
---------------------------

``PlateOCRFuzzyKey`` first normalizes the plate: strip hyphens and spaces,
convert to uppercase.

.. code-block:: text

   "CX-180-W"  →  "CX180W"

Then it emits one additional key for each position where a confused character
appears:

.. code-block:: text

   CX180W  →  base key: plate_ocr:CX180W
   position 2 (1): 1→I   →  plate_ocr:CXI80W
   position 3 (8): 8→B   →  plate_ocr:CX1B0W
   position 4 (0): 0→O   →  plate_ocr:CX18OW

The OCR-confused read ``CX-I80-W`` normalizes to ``CXI80W``, which generates:

.. code-block:: text

   CXI80W  →  base key: plate_ocr:CXI80W
   position 2 (I): I→1   →  plate_ocr:CX180W   ← shared with true plate ✓

Both records emit ``plate_ocr:CXI80W`` (the true plate's variant) **and**
``plate_ocr:CX180W`` (the OCR plate's variant). Either key is sufficient to
make them candidates.

Bidirectionality
-----------------

The substitutions are bidirectional. When the true plate contains a digit
that looks like a letter, the key generator emits the letter-variant key.
When the confused plate contains a letter where the true plate has a digit,
the key generator emits the digit-variant key. Both variants end up in the
inverted index. A pair only needs to share one key to become a candidate.

This means zer correctly handles:

* True plate ``CX-180-W`` → OCR read ``CX-I80-W`` (digit → letter confusion)
* True plate ``CX-I80-W`` → OCR read ``CX-180-W`` (letter → digit confusion)
* True plate ``25-XKL-9`` where only ``LicensePlateNormKey`` is needed (no confusion)

LicensePlateNormKey vs. PlateOCRFuzzyKey
------------------------------------------

Both keys are generated for every ``LicensePlate`` field:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Purpose
   * - ``LicensePlateNormKey``
     - Normalized plate without OCR variants. Catches exact matches and
       formatting differences (``CX-180-W`` vs ``CX180W``).
   * - ``PlateOCRFuzzyKey``
     - All single-character OCR variants. Catches OCR confusion errors.

A pair matches if they share either key.

Dutch vs. EU plates
--------------------

Dutch plates follow the format ``NN-XXX-N`` or ``XX-NNN-X`` (two letters,
three digits, one letter in different arrangements). EU plates from other
member states follow different formats. zer normalizes all plates the same
way, strip non-alphanumeric, uppercase, so the OCR fuzzy key works for any
European plate format.

Scaling concern
-----------------

For a plate with *k* confused characters, ``PlateOCRFuzzyKey`` emits *k + 1*
keys (1 base + 1 per confused position). Dutch plates typically have 1–3
confused characters, so 2–4 keys per record. The inverted index lookup for
a 6-character plate is still O(4) lookups, negligible.

What to explore next
---------------------

* :doc:`/tutorials/anpr-matching`, step-by-step ANPR pipeline with OCR example.
* :doc:`/reference/blocking-keys`, parameters and edge cases for all blocking keys.
* :doc:`/how-to/blocking-strategy`, combining ANPR keys with custom categories.
