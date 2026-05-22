#!/usr/bin/env python3
"""
Demo person-deduplication dataset generator.

Reads from the base pools in data/base/brp/ (surnames.csv, nl_addresses.csv)
and generates a realistic Dutch person registry with controlled duplicates.

Outputs:
  data/demos/persons/records.csv      , one row per person record (originals + dupes)
  data/demos/persons/ground_truth.csv , known duplicate pairs

Usage:
  python data_generator/generate_demo_persons.py [--records 1000] [--seed 42]
"""

import argparse
import csv
import os
import random
from pathlib import Path

from _common import (
    Person, bsn, generate_person, perturb_name, pick_city,
    postcode, street_address,
)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "demos" / "persons"

CSV_FIELDS = [
    "record_id", "bsn", "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "geboorteplaats", "geslacht",
    "straatnaam", "huisnummer", "postcode", "woonplaats",
]


def record_from_person(record_id: int, p: Person, existing_bsn: str | None = None) -> dict:
    street, number, toevoeging = street_address()
    city = pick_city()
    pc   = postcode()
    house = number + (f" {toevoeging}" if toevoeging else "")
    return {
        "record_id":     record_id,
        "bsn":           existing_bsn or bsn(),
        "voornamen":     p.voornamen,
        "tussenvoegsel": p.tussenvoegsel or "",
        "achternaam":    p.achternaam,
        "geboortedatum": p.geboortedatum,
        "geboorteplaats": p.geboorteplaats,
        "geslacht":      p.geslacht,
        "straatnaam":    street,
        "huisnummer":    house,
        "postcode":      pc,
        "woonplaats":    city,
    }


def _perturb_record(src: dict, person: Person, record_id: int) -> dict:
    """Apply a realistic perturbation to produce a near-duplicate record."""
    dup = dict(src)
    dup["record_id"] = record_id

    style = random.choice(["address_move", "name_variant", "dob_error", "partial_update"])

    if style == "address_move":
        street, number, toevoeging = street_address()
        house = number + (f" {toevoeging}" if toevoeging else "")
        dup["straatnaam"] = street
        dup["huisnummer"] = house
        dup["postcode"]   = postcode()
        dup["woonplaats"] = pick_city()

    elif style == "name_variant":
        name_patch = perturb_name(person)
        dup["voornamen"]     = name_patch["voornamen"]
        dup["tussenvoegsel"] = name_patch["tussenvoegsel"] or ""
        dup["achternaam"]    = name_patch["achternaam"]

    elif style == "dob_error":
        dob = dup["geboortedatum"]
        if len(dob) == 10:
            year = dob[:4]
            month = str(random.randint(1, 12)).zfill(2)
            day   = str(random.randint(1, 28)).zfill(2)
            dup["geboortedatum"] = f"{year}-{month}-{day}"
        dup["bsn"] = ""

    elif style == "partial_update":
        name_patch = perturb_name(person)
        dup["voornamen"] = name_patch["voornamen"]
        street, number, toevoeging = street_address()
        house = number + (f" {toevoeging}" if toevoeging else "")
        dup["straatnaam"] = street
        dup["huisnummer"] = house
        dup["postcode"]   = postcode()

    return dup


def generate(n_records: int, duplicate_fraction: float, seed: int) -> None:
    random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    ground_truth: list[dict] = []
    next_id = 1

    n_originals  = round(n_records * (1 - duplicate_fraction))
    n_duplicates = n_records - n_originals

    # Generate originals
    persons: list[Person] = []
    for _ in range(n_originals):
        p   = generate_person()
        rec = record_from_person(next_id, p)
        records.append(rec)
        persons.append(p)
        next_id += 1

    # Inject duplicates
    dup_sources = random.sample(range(n_originals), min(n_duplicates, n_originals))
    for src_idx in dup_sources:
        src_rec    = records[src_idx]
        src_person = persons[src_idx]
        dup_rec    = _perturb_record(src_rec, src_person, next_id)
        records.append(dup_rec)
        ground_truth.append({
            "record_id_a": src_rec["record_id"],
            "record_id_b": next_id,
            "is_match":    True,
            "match_type":  "duplicate",
        })
        next_id += 1

    # Shuffle so duplicates aren't all at the end
    random.shuffle(records)

    records_path = OUTPUT_DIR / "records.csv"
    with open(records_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(records)

    gt_path = OUTPUT_DIR / "ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["record_id_a", "record_id_b", "is_match", "match_type"])
        w.writeheader()
        w.writerows(ground_truth)

    print(f"[generate_demo_persons] {len(records)} records → {records_path}")
    print(f"[generate_demo_persons] {len(ground_truth)} true pairs → {gt_path}")
    print(f"[generate_demo_persons] duplicate fraction: {len(ground_truth) / len(records):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",            type=int,   default=1000)
    parser.add_argument("--duplicate-fraction", type=float, default=0.12)
    parser.add_argument("--seed",               type=int,   default=42)
    args = parser.parse_args()
    generate(args.records, args.duplicate_fraction, args.seed)


if __name__ == "__main__":
    main()
