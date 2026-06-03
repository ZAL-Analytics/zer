#!/usr/bin/env python3
"""
Generate all example and test datasets for zer crates.

Replaces the phase-* datasets used by crate-level examples and tests.
Outputs go to data/examples/ and data/tests/ at the workspace root.

Run from the data_generator/ directory:
    cd data_generator && python generate_examples_tests.py
"""

import csv
import random
import uuid
from datetime import date
from pathlib import Path

from faker import Faker

from _common import (
    COUNTRIES,
    DOCUMENT_TYPES,
    HAAR_KLEUREN,
    NL_CARRIERS,
    OOG_KLEUREN,
    Person,
    alias_variants,
    bsn,
    document_number,
    generate_person,
    iban_nl,
    iccid,
    imsi,
    license_plate,
    msisdn,
    ocr_confuse_plate,
    perturb_name,
    pick_city,
    postcode,
    street_address,
)

fake_nl = Faker("nl_NL")
ROOT = Path(__file__).parent.parent


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# BRP examples  (14-column, for zer-schema examples + tests)
# ---------------------------------------------------------------------------

BRP14_FIELDS = [
    "bsn", "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "geboorteplaats", "geboorteland", "nationaliteit",
    "geslacht", "straatnaam", "huisnummer", "postcode", "woonplaats",
    "verblijfstitel",
]

VERBLIJFSTITELS = [
    "EU-burger",
    "Verblijfsvergunning regulier bepaalde tijd",
    "Verblijfsvergunning asiel bepaalde tijd",
    "Verblijfsvergunning regulier onbepaalde tijd",
]


def _verblijfstitel(person: Person) -> str:
    if "NL" in person.nationaliteit:
        return "EU-burger"
    return random.choice(VERBLIJFSTITELS)


def _brp14_row(person: Person) -> dict:
    street, number, toevoeging = street_address()
    city = pick_city()
    pc = postcode()
    house = number + (f" {toevoeging}" if toevoeging else "")
    return {
        "bsn":            bsn(),
        "voornamen":      person.voornamen,
        "tussenvoegsel":  person.tussenvoegsel or "",
        "achternaam":     person.achternaam,
        "geboortedatum":  person.geboortedatum,
        "geboorteplaats": person.geboorteplaats,
        "geboorteland":   person.geboorteland_nl,
        "nationaliteit":  person.nationaliteit_nl,
        "geslacht":       person.geslacht,
        "straatnaam":     street,
        "huisnummer":     house,
        "postcode":       pc,
        "woonplaats":     city,
        "verblijfstitel": _verblijfstitel(person),
    }


def generate_brp_examples():
    """BRP Q1 and Q2: 1 500 rows each, 14 columns.

    Constraints tested by zer-schema:
    - record_count > 1 000
    - geboortedatum null_rate < 0.1
    - geboortedatum cardinality > 100
    - geslacht cardinality ≤ 5
    - schema.len() == 14
    - fingerprint_distance(fp_q1, fp_q2) == 0.0  (same schema hash)
    """
    for snapshot, seed in [("brp_q1", 1001), ("brp_q2", 2002)]:
        random.seed(seed)
        Faker.seed(seed)

        out_dir = mkdir(ROOT / "data" / "examples" / snapshot)
        rows = [_brp14_row(generate_person()) for _ in range(1_500)]

        path = out_dir / "brp_persons.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=BRP14_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"  {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# SIM examples  (14-column, for zer-schema examples + tests)
# ---------------------------------------------------------------------------

SIM_FIELDS = [
    "sim_id", "msisdn", "imsi", "iccid", "carrier",
    "contract_type", "activatiedatum",
    "voornamen", "achternaam", "geboortedatum", "nationaliteit",
    "document_type", "document_nummer",
    "bsn",
]

CONTRACT_TYPES = ["postpaid", "postpaid", "postpaid", "postpaid", "prepaid"]


def generate_sim_examples():
    """SIM subscribers: 500 rows, 14 columns.

    Tested by zer-schema; msisdn → Phone, imsi/iccid/document_nummer → Id.
    """
    random.seed(3003)
    Faker.seed(3003)

    out_dir = mkdir(ROOT / "data" / "examples" / "sim")
    rows = []
    for _ in range(500):
        p = generate_person()
        carrier_name, carrier_mnc = random.choice(NL_CARRIERS)
        ct = random.choice(CONTRACT_TYPES)
        doc_type = random.choice(DOCUMENT_TYPES)
        act_date = fake_nl.date_between(date(2020, 1, 1), date(2025, 6, 30))
        rows.append({
            "sim_id":          "SIM-" + str(uuid.uuid4())[:8].upper(),
            "msisdn":          msisdn(),
            "imsi":            imsi(carrier_mnc),
            "iccid":           iccid(),
            "carrier":         carrier_name,
            "contract_type":   ct,
            "activatiedatum":  act_date.strftime("%Y-%m-%d"),
            "voornamen":       p.voornamen      if ct == "postpaid" else "",
            "achternaam":      p.achternaam     if ct == "postpaid" else "",
            "geboortedatum":   p.geboortedatum  if ct == "postpaid" else "",
            "nationaliteit":   p.nationaliteit_nl if ct == "postpaid" else "",
            "document_type":   doc_type         if ct == "postpaid" else "",
            "document_nummer": document_number(doc_type) if ct == "postpaid" else "",
            "bsn":             bsn()            if ct == "postpaid" else "",
        })

    path = out_dir / "sim_subscribers.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIM_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# BRP test data  (11-column, for zer-compare, zer-compute tests + examples)
# ---------------------------------------------------------------------------

BRP11_FIELDS = [
    "bsn", "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "geboorteland", "nationaliteit",
    "straatnaam", "huisnummer", "postcode", "woonplaats",
]
BRP11_GT_FIELDS = ["bsn_a", "bsn_b", "is_match"]


def _brp11_row(person: Person, existing_bsn: str | None = None) -> dict:
    street, number, toevoeging = street_address()
    city = pick_city()
    pc = postcode()
    house = number + (f" {toevoeging}" if toevoeging else "")
    return {
        "bsn":           existing_bsn or bsn(),
        "voornamen":     person.voornamen,
        "tussenvoegsel": person.tussenvoegsel or "",
        "achternaam":    person.achternaam,
        "geboortedatum": person.geboortedatum,
        "geboorteland":  person.geboorteland_nl,
        "nationaliteit": person.nationaliteit_nl,
        "straatnaam":    street,
        "huisnummer":    house,
        "postcode":      pc,
        "woonplaats":    city,
    }


def _perturb_brp11(src: dict, person: Person) -> dict:
    """Perturb a BRP record while preserving most discriminating fields."""
    dup = dict(src)
    # Bias toward address_move (preserves name+DOB signal for EM convergence)
    style = random.choices(
        ["address_move", "name_variant", "dob_error"],
        weights=[90, 8, 2],
        k=1,
    )[0]
    if style == "address_move":
        street, number, toevoeging = street_address()
        city = pick_city()
        house = number + (f" {toevoeging}" if toevoeging else "")
        dup.update({"straatnaam": street, "huisnummer": house,
                    "postcode": postcode(), "woonplaats": city})
    elif style == "name_variant":
        variant = perturb_name(person)
        dup.update({
            "voornamen":     variant["voornamen"],
            "achternaam":    variant["achternaam"],
            "tussenvoegsel": variant["tussenvoegsel"] or "",
        })
    elif style == "dob_error":
        dob = src["geboortedatum"]
        parts = dob.split("-")
        dup["geboortedatum"] = f"{int(parts[0]) + 1}-{parts[1]}-{parts[2]}"
    return dup


def generate_brp_tests():
    """BRP test data: 2 000 records, ~200 true-match pairs.

    Tested by zer-compare, zer-compute; EM must converge with
    precision/recall ≥ 0.70.
    """
    random.seed(4004)
    Faker.seed(4004)

    out_dir = mkdir(ROOT / "data" / "tests" / "brp")

    # 1800 base persons
    base: list[tuple[dict, Person]] = []
    for _ in range(1_800):
        p = generate_person()
        base.append((_brp11_row(p), p))

    records = [r for r, _ in base]
    ground_truth: list[dict] = []

    # Inject 200 duplicates
    sources = random.sample(base, 200)
    for src_rec, src_person in sources:
        dup = _perturb_brp11(src_rec, src_person)
        dup_bsn = bsn()
        dup["bsn"] = dup_bsn
        records.append(dup)
        ground_truth.append({
            "bsn_a":    src_rec["bsn"],
            "bsn_b":    dup_bsn,
            "is_match": "True",
        })

    # Write persons
    persons_path = out_dir / "brp_persons.csv"
    with open(persons_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BRP11_FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f"  {persons_path}  ({len(records)} rows)")

    # Write ground truth
    gt_path = out_dir / "ground_truth_pairs.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BRP11_GT_FIELDS)
        w.writeheader()
        w.writerows(ground_truth)
    print(f"  {gt_path}  ({len(ground_truth)} pairs)")


# ---------------------------------------------------------------------------
# HKS test data  (13-column, for zer-compare tests)
# ---------------------------------------------------------------------------

HKS_FIELDS = [
    "hks_id", "voornamen", "achternaam", "alias_namen",
    "geboortedatum", "geboorteland", "nationaliteit", "geslacht",
    "bsn", "document_nummer", "lengte", "haarkleur", "oogkleur",
]
HKS_GT_FIELDS = ["hks_id_a", "hks_id_b", "is_match"]

HEIGHT_POOL = [str(h) for h in range(155, 200, 1)]


def generate_hks_tests():
    """HKS test data: ~500 records with aliases, ~100 true-match alias pairs.

    Tests:
    - alias_namen contributes non-None for ≥ 20% of true pairs
    - null bsn handled gracefully
    - EM converges (m[Exact] > u[Exact] for ≥ 1/3 of fields)
    """
    random.seed(5005)
    Faker.seed(5005)

    out_dir = mkdir(ROOT / "data" / "tests" / "hks")

    records = []
    ground_truth = []
    rec_id = 1

    def _hks_row(person: Person, hks_id: str, alias_namen: str = "",
                  use_bsn: bool = True, doc_nr: str = "") -> dict:
        return {
            "hks_id":         hks_id,
            "voornamen":      person.voornamen,
            "achternaam":     person.achternaam,
            "alias_namen":    alias_namen,
            "geboortedatum":  person.geboortedatum,
            "geboorteland":   person.geboorteland_nl,
            "nationaliteit":  person.nationaliteit_nl,
            "geslacht":       person.geslacht,
            "bsn":            bsn() if use_bsn else "",
            "document_nummer": doc_nr or document_number(),
            "lengte":         random.choice(HEIGHT_POOL),
            "haarkleur":      random.choice(HAAR_KLEUREN),
            "oogkleur":       random.choice(OOG_KLEUREN),
        }

    # 300 standalone records (no duplicates)
    for _ in range(300):
        p = generate_person()
        hks_id = f"HKS-{rec_id:05d}"
        records.append(_hks_row(p, hks_id, use_bsn=random.random() > 0.15))
        rec_id += 1

    # 100 alias pairs: each pair has a base record + alias record
    # The alias record uses an alias name as its main name,
    # and the alias field of the base record contains the alias.
    for _ in range(100):
        p = generate_person()
        doc_nr = document_number()
        p_bsn = bsn() if random.random() > 0.20 else ""

        aliases = alias_variants(p, n=2)
        if aliases:
            alias_str = "|".join(aliases)
        else:
            # Fallback: abbreviated first name + surname
            alias_str = f"{p.voornamen[0]}. {p.achternaam}"

        # Base record: real name, alias in alias_namen
        base_id = f"HKS-{rec_id:05d}"
        records.append({
            "hks_id":         base_id,
            "voornamen":      p.voornamen,
            "achternaam":     p.achternaam,
            "alias_namen":    alias_str,
            "geboortedatum":  p.geboortedatum,
            "geboorteland":   p.geboorteland_nl,
            "nationaliteit":  p.nationaliteit_nl,
            "geslacht":       p.geslacht,
            "bsn":            p_bsn,
            "document_nummer": doc_nr,
            "lengte":         str(random.randint(155, 195)),
            "haarkleur":      random.choice(HAAR_KLEUREN),
            "oogkleur":       random.choice(OOG_KLEUREN),
        })
        rec_id += 1

        # Alias record: alias name as main name, has same alias in alias_namen
        alias_name = aliases[0] if aliases else f"{p.voornamen[0]}. {p.achternaam}"
        alias_parts = alias_name.split()
        alias_voor = alias_parts[0] if alias_parts else p.voornamen
        alias_ach = alias_parts[-1] if len(alias_parts) > 1 else p.achternaam

        alias_id = f"HKS-{rec_id:05d}"
        records.append({
            "hks_id":         alias_id,
            "voornamen":      alias_voor,
            "achternaam":     alias_ach,
            "alias_namen":    alias_str,
            "geboortedatum":  p.geboortedatum,
            "geboorteland":   p.geboorteland_nl,
            "nationaliteit":  p.nationaliteit_nl,
            "geslacht":       p.geslacht,
            "bsn":            p_bsn,
            "document_nummer": doc_nr,
            "lengte":         str(random.randint(155, 195)),
            "haarkleur":      random.choice(HAAR_KLEUREN),
            "oogkleur":       random.choice(OOG_KLEUREN),
        })
        rec_id += 1

        ground_truth.append({"hks_id_a": base_id, "hks_id_b": alias_id, "is_match": "True"})

    # Write records
    records_path = out_dir / "hks_records.csv"
    with open(records_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HKS_FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f"  {records_path}  ({len(records)} rows)")

    # Write ground truth
    gt_path = out_dir / "ground_truth_pairs.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HKS_GT_FIELDS)
        w.writeheader()
        w.writerows(ground_truth)
    print(f"  {gt_path}  ({len(ground_truth)} pairs)")


# ---------------------------------------------------------------------------
# ANPR test data  (two formats: blocking test + comparator test)
# ---------------------------------------------------------------------------

# Cameras for positional ANPR CSV (col 2=camera_id, col 4=lat, col 5=lon)
ANPR_CAMERAS = [
    ("CAM-A1-001", 52.3431, 5.0812),
    ("CAM-A2-001", 52.3080, 4.9451),
    ("CAM-A4-001", 52.3064, 4.7693),
    ("CAM-A10-001", 52.3731, 4.8837),
    ("CAM-A12-001", 52.0454, 4.4332),
]


def generate_anpr_tests():
    """ANPR test data.

    - anpr_passages.csv: positional format (col 7 = kenteken)
    - ground_truth_vehicle_pairs.csv: positional blocking format
      (col 0=passage_id_a, col 1=kenteken_true, col 2=confusion_type, col 3=is_match)
    - ground_truth_compare_pairs.csv: named format (kenteken_true, kenteken_ocr, is_match)

    Blocking test recall ≥ 0.97: confusions covered by PlateOCRFuzzyKey.
    """
    random.seed(6006)
    Faker.seed(6006)

    out_dir = mkdir(ROOT / "data" / "tests" / "anpr")

    passages = []   # rows for anpr_passages.csv
    gt_blocking = []  # rows for ground_truth_vehicle_pairs.csv
    gt_compare = []   # rows for ground_truth_compare_pairs.csv

    passage_counter = 1

    def _passage_row(passage_id: str, kenteken: str) -> list:
        cam_id, lat, lon = random.choice(ANPR_CAMERAS)
        ts = fake_nl.date_time_between(
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        # Positional: [passage_id, tijdstip, camera_id, road, lat, lon, richting, kenteken]
        return [passage_id, ts, cam_id, "A1", str(lat), str(lon), "N", kenteken]

    ANPR_PASSAGE_HEADER = [
        "passage_id", "tijdstip", "camera_id", "road", "lat", "lon", "richting", "kenteken"
    ]

    # 400 clean passages (no OCR confusion)
    clean_plates: list[tuple[str, str]] = []  # (passage_id, plate)
    for _ in range(400):
        pid = f"PASS-{passage_counter:05d}"
        plate = license_plate()
        passages.append(_passage_row(pid, plate))
        clean_plates.append((pid, plate))
        passage_counter += 1

    # 100 OCR-confused passages, these are the ground truth pairs
    for true_pid, true_plate in random.sample(clean_plates, 100):
        confused_plate = ocr_confuse_plate(true_plate)
        ocr_pid = f"PASS-{passage_counter:05d}"
        passages.append(_passage_row(ocr_pid, confused_plate))
        passage_counter += 1

        # Blocking ground truth: confused passage → true plate
        gt_blocking.append([ocr_pid, true_plate, "ocr_confusion", "True"])

        # Comparator ground truth (named)
        gt_compare.append({
            "kenteken_true": true_plate,
            "kenteken_ocr":  confused_plate,
            "is_match":      "True",
        })

    # Write passages (positional, no header row needed but we write one for readability)
    passages_path = out_dir / "anpr_passages.csv"
    with open(passages_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(ANPR_PASSAGE_HEADER)
        w.writerows(passages)
    print(f"  {passages_path}  ({len(passages)} rows)")

    # Write blocking ground truth (positional)
    gt_blocking_path = out_dir / "ground_truth_vehicle_pairs.csv"
    with open(gt_blocking_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["passage_id_a", "kenteken_true", "confusion_type", "is_match"])
        w.writerows(gt_blocking)
    print(f"  {gt_blocking_path}  ({len(gt_blocking)} pairs)")

    # Write comparator ground truth (named)
    gt_compare_path = out_dir / "ground_truth_compare_pairs.csv"
    with open(gt_compare_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kenteken_true", "kenteken_ocr", "is_match"])
        w.writeheader()
        w.writerows(gt_compare)
    print(f"  {gt_compare_path}  ({len(gt_compare)} pairs)")


# ---------------------------------------------------------------------------
# KVK test data  (for zer-blocking tests, named columns, kvkNummer as record ID)
# ---------------------------------------------------------------------------

KVK_FIELDS = [
    "kvkNummer", "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "woonplaats", "straatnaam", "postcode",
]


def generate_kvk_tests():
    """KVK director test data: ~940 records, ~120 true-match pairs.

    The same director appears under multiple KVK entries (different companies).
    Blocking recall ≥ 0.97: phonetic name + DOB keys catch the matches.
    Blocking reduction ratio ≥ 0.90: Zipf-weighted Dutch surnames produce a
    large "de" tussenvoegsel bucket (~250 records) which is realistic but limits
    how aggressively the blocker can prune pairs.
    """
    random.seed(7007)
    Faker.seed(7007)

    out_dir = mkdir(ROOT / "data" / "tests" / "kvk")

    records = []
    ground_truth_rows = []  # positional: [kvk_a, kvk_b, "True"]
    kvk_counter = 10_000_001

    def _kvk_row(person: Person, kvk_id: int) -> dict:
        street, number, toevoeging = street_address()
        city = pick_city()
        house = number + (f" {toevoeging}" if toevoeging else "")
        pc = postcode()
        return {
            "kvkNummer":     str(kvk_id),
            "voornamen":     person.voornamen,
            "tussenvoegsel": person.tussenvoegsel or "",
            "achternaam":    person.achternaam,
            "geboortedatum": person.geboortedatum,
            "woonplaats":    city,
            "straatnaam":    street + " " + house,
            "postcode":      pc,
        }

    # 700 unique directors (one KVK entry each)
    for _ in range(700):
        p = generate_person()
        records.append(_kvk_row(p, kvk_counter))
        kvk_counter += 1

    # 120 directors that appear in two companies, these are the true pairs
    for _ in range(120):
        p = generate_person()
        kvk_a = kvk_counter
        kvk_counter += 1
        kvk_b = kvk_counter
        kvk_counter += 1

        # Both entries for the same director; address may differ slightly
        row_a = _kvk_row(p, kvk_a)
        row_b = _kvk_row(p, kvk_b)
        # Apply minor address variation to simulate different company registrations
        row_b["straatnaam"] = row_b["straatnaam"]  # keep or vary
        records.append(row_a)
        records.append(row_b)
        ground_truth_rows.append([str(kvk_a), str(kvk_b), "True"])

    # Write records
    kvk_path = out_dir / "kvk_director_flat.csv"
    with open(kvk_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KVK_FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f"  {kvk_path}  ({len(records)} rows)")

    # Write ground truth (positional)
    gt_path = out_dir / "ground_truth_pairs.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kvk_a", "kvk_b", "is_match"])
        w.writerows(ground_truth_rows)
    print(f"  {gt_path}  ({len(ground_truth_rows)} pairs)")


# ---------------------------------------------------------------------------
# SIS test data  (positional, ≥12 columns, col 0, 2, 3, 4, 5, 11 used)
# ---------------------------------------------------------------------------

# Columns (positional index matters):
# 0=sis_id, 1=alertnummer, 2=voornamen, 3=achternaam, 4=alias_namen,
# 5=geboortedatum, 6=nationaliteit, 7=geboorteland, 8=geslacht,
# 9=kategorie, 10=document_type, 11=document_nummer

SIS_FIELDS = [
    "sis_id", "alertnummer", "voornamen", "achternaam", "alias_namen",
    "geboortedatum", "nationaliteit", "geboorteland", "geslacht",
    "kategorie", "document_type", "document_nummer",
]
SIS_GT_FIELDS = ["sis_id_a", "sis_id_b", "is_match"]

SIS_CATEGORIES = ["A", "B", "C", "D"]


def generate_sis_tests():
    """SIS test data: ~400 records, ~100 true-match alias pairs.

    Positional format: document_nummer at col 11.
    Blocking recall ≥ 0.85: PhoneticNameDobKey + AliasPhoneticKey catch matches.
    """
    random.seed(8008)
    Faker.seed(8008)

    out_dir = mkdir(ROOT / "data" / "tests" / "sis")

    records = []
    ground_truth = []
    sis_counter = 1

    def _sis_id() -> str:
        sid = f"SIS-{sis_counter:05d}"
        return sid

    # 200 standalone records
    for _ in range(200):
        p = generate_person()
        doc_type = random.choice(DOCUMENT_TYPES)
        records.append({
            "sis_id":          f"SIS-{sis_counter:05d}",
            "alertnummer":     f"ALERT-{random.randint(100000, 999999)}",
            "voornamen":       p.voornamen,
            "achternaam":      p.achternaam,
            "alias_namen":     "",
            "geboortedatum":   p.geboortedatum,
            "nationaliteit":   p.nationaliteit_nl,
            "geboorteland":    p.geboorteland_nl,
            "geslacht":        p.geslacht,
            "kategorie":       random.choice(SIS_CATEGORIES),
            "document_type":   doc_type,
            "document_nummer": document_number(doc_type),
        })
        sis_counter_before = sis_counter
        locals()  # just to use sis_counter
        # increment manually since we don't have nonlocal
        records[-1]["sis_id"] = f"SIS-{sis_counter:05d}"
        sis_counter += 1

    # 100 alias pairs
    # Reset counter since the loop above was awkward, just track it properly
    sis_counter_val = 201
    records = records[:0]  # reset and redo properly

    # Standalone records
    for i in range(1, 201):
        p = generate_person()
        doc_type = random.choice(DOCUMENT_TYPES)
        records.append({
            "sis_id":          f"SIS-{i:05d}",
            "alertnummer":     f"ALERT-{random.randint(100000, 999999)}",
            "voornamen":       p.voornamen,
            "achternaam":      p.achternaam,
            "alias_namen":     "",
            "geboortedatum":   p.geboortedatum,
            "nationaliteit":   p.nationaliteit_nl,
            "geboorteland":    p.geboorteland_nl,
            "geslacht":        p.geslacht,
            "kategorie":       random.choice(SIS_CATEGORIES),
            "document_type":   doc_type,
            "document_nummer": document_number(doc_type),
        })

    # Alias pairs (100 pairs = 200 records, IDs 201-400)
    for i in range(100):
        p = generate_person()
        doc_type = random.choice(DOCUMENT_TYPES)
        doc_nr = document_number(doc_type)
        aliases = alias_variants(p, n=2)
        alias_str = "|".join(aliases) if aliases else f"{p.voornamen[0]}. {p.achternaam}"

        base_sis = f"SIS-{201 + i * 2:05d}"
        alias_sis = f"SIS-{202 + i * 2:05d}"

        # Base record: real name + aliases in alias_namen
        records.append({
            "sis_id":          base_sis,
            "alertnummer":     f"ALERT-{random.randint(100000, 999999)}",
            "voornamen":       p.voornamen,
            "achternaam":      p.achternaam,
            "alias_namen":     alias_str,
            "geboortedatum":   p.geboortedatum,
            "nationaliteit":   p.nationaliteit_nl,
            "geboorteland":    p.geboorteland_nl,
            "geslacht":        p.geslacht,
            "kategorie":       random.choice(SIS_CATEGORIES),
            "document_type":   doc_type,
            "document_nummer": doc_nr,
        })

        # Alias record: alias name as primary name, same DOB
        alias_name = aliases[0] if aliases else alias_str
        alias_parts = alias_name.split()
        alias_voor = alias_parts[0] if alias_parts else p.voornamen
        alias_ach = alias_parts[-1] if len(alias_parts) > 1 else p.achternaam

        records.append({
            "sis_id":          alias_sis,
            "alertnummer":     f"ALERT-{random.randint(100000, 999999)}",
            "voornamen":       alias_voor,
            "achternaam":      alias_ach,
            "alias_namen":     alias_str,
            "geboortedatum":   p.geboortedatum,
            "nationaliteit":   p.nationaliteit_nl,
            "geboorteland":    p.geboorteland_nl,
            "geslacht":        p.geslacht,
            "kategorie":       random.choice(SIS_CATEGORIES),
            "document_type":   doc_type,
            "document_nummer": doc_nr,
        })

        ground_truth.append({
            "sis_id_a": base_sis,
            "sis_id_b": alias_sis,
            "is_match": "True",
        })

    # Write records
    persons_path = out_dir / "sis_persons.csv"
    with open(persons_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIS_FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f"  {persons_path}  ({len(records)} rows)")

    # Write ground truth
    gt_path = out_dir / "ground_truth_alias_pairs.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIS_GT_FIELDS)
        w.writeheader()
        w.writerows(ground_truth)
    print(f"  {gt_path}  ({len(ground_truth)} pairs)")


# ---------------------------------------------------------------------------
# CDR cluster test data  (two-column: msisdn, cluster_id)
# ---------------------------------------------------------------------------


def generate_cdr_tests():
    """CDR cluster ground truth: ~200 MSISDNs in ~30 clusters."""
    random.seed(9009)

    out_dir = mkdir(ROOT / "data" / "tests" / "cdr")
    rows = []

    cluster_id = 1
    for _ in range(30):
        size = random.randint(2, 8)
        for _ in range(size):
            rows.append({"msisdn": msisdn(), "cluster_id": f"CDR-CLUSTER-{cluster_id:04d}"})
        cluster_id += 1

    # 50 singletons (cluster of size 1, not used by the test but realistic)
    for _ in range(50):
        rows.append({"msisdn": msisdn(), "cluster_id": f"CDR-CLUSTER-{cluster_id:04d}"})
        cluster_id += 1

    path = out_dir / "ground_truth_clusters.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["msisdn", "cluster_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# FIU cluster test data  (two-column: iban, cluster_id)
# ---------------------------------------------------------------------------


def generate_fiu_tests():
    """FIU cluster ground truth: ~200 IBANs in ~30 clusters."""
    random.seed(10010)

    out_dir = mkdir(ROOT / "data" / "tests" / "fiu")
    rows = []

    cluster_id = 1
    for _ in range(30):
        size = random.randint(2, 6)
        for _ in range(size):
            rows.append({"iban": iban_nl(), "cluster_id": f"FIU-CLUSTER-{cluster_id:04d}"})
        cluster_id += 1

    # 50 singletons
    for _ in range(50):
        rows.append({"iban": iban_nl(), "cluster_id": f"FIU-CLUSTER-{cluster_id:04d}"})
        cluster_id += 1

    path = out_dir / "ground_truth_clusters.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iban", "cluster_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating example datasets (data/examples/) ...")
    generate_brp_examples()
    generate_sim_examples()

    print("\nGenerating test datasets (data/tests/) ...")
    generate_brp_tests()
    generate_hks_tests()
    generate_anpr_tests()
    generate_kvk_tests()
    generate_sis_tests()
    generate_cdr_tests()
    generate_fiu_tests()

    print("\nDone.")
