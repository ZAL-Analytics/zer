#!/usr/bin/env python3
"""
Demo cross-source linkage dataset generator.

Generates two independently-perturbed views of the same population, simulating
two different registries (e.g. a municipal register and a benefits register)
that share a ~40% overlap of persons with different data quality profiles.

Perturbation profile per source:
  Source A , authoritative register: minimal perturbation, address is current
  Source B , downstream system: name variants, address lag, occasional DOB drift

Outputs:
  data/demos/linkage/source_a.csv     , source A records
  data/demos/linkage/source_b.csv     , source B records
  data/demos/linkage/ground_truth.csv , linked pairs (record_id_a from A, record_id_b from B)

Usage:
  python data_generator/generate_demo_linkage.py [--persons 600] [--overlap 0.40] [--seed 7]
"""

import argparse
import csv
import random
from pathlib import Path

from _common import (
    Person, bsn, generate_person, perturb_name, pick_city,
    postcode, street_address,
)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "demos" / "linkage"

CSV_FIELDS_A = [
    "record_id", "bsn", "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "geslacht", "straatnaam", "huisnummer", "postcode", "woonplaats",
]

CSV_FIELDS_B = [
    "record_id", "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "geslacht", "straatnaam", "huisnummer", "postcode", "woonplaats",
]


def _make_address() -> dict:
    street, number, toevoeging = street_address()
    house = number + (f" {toevoeging}" if toevoeging else "")
    return {"straatnaam": street, "huisnummer": house, "postcode": postcode(), "woonplaats": pick_city()}


def record_a(record_id: int, p: Person) -> dict:
    """Authoritative source: minimal perturbation."""
    addr = _make_address()
    return {
        "record_id":      record_id,
        "bsn":            bsn(),
        "voornamen":      p.voornamen,
        "tussenvoegsel":  p.tussenvoegsel or "",
        "achternaam":     p.achternaam,
        "geboortedatum":  p.geboortedatum,
        "geslacht":       p.geslacht,
        **addr,
    }


def record_b_from_a(record_id: int, src: dict, person: Person) -> dict:
    """Downstream source: name variant + address lag (old address or current)."""
    dup = dict(src)
    dup["record_id"] = record_id
    dup.pop("bsn", None)

    style = random.choice(["name_variant", "address_lag", "dob_drift", "name_and_address"])

    if style in ("name_variant", "name_and_address"):
        patch = perturb_name(person)
        dup["voornamen"]     = patch["voornamen"]
        dup["tussenvoegsel"] = patch["tussenvoegsel"] or ""
        dup["achternaam"]    = patch["achternaam"]

    if style in ("address_lag", "name_and_address"):
        addr = _make_address()
        dup.update(addr)

    if style == "dob_drift":
        dob = dup.get("geboortedatum", "")
        if len(dob) == 10:
            year = dob[:4]
            dup["geboortedatum"] = f"{year}-{str(random.randint(1,12)).zfill(2)}-{str(random.randint(1,28)).zfill(2)}"

    return {k: dup[k] for k in CSV_FIELDS_B}


def record_b_unique(record_id: int, p: Person) -> dict:
    """A record in source B with no counterpart in source A."""
    addr = _make_address()
    return {
        "record_id":      record_id,
        "voornamen":      p.voornamen,
        "tussenvoegsel":  p.tussenvoegsel or "",
        "achternaam":     p.achternaam,
        "geboortedatum":  p.geboortedatum,
        "geslacht":       p.geslacht,
        **addr,
    }


def generate(n_persons: int, overlap: float, seed: int) -> None:
    random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    persons: list[Person] = [generate_person() for _ in range(n_persons)]

    n_shared = round(n_persons * overlap)
    shared_indices = set(random.sample(range(n_persons), n_shared))

    a_records:    list[dict] = []
    b_records:    list[dict] = []
    ground_truth: list[dict] = []

    id_a = 1
    id_b = 1

    # Source A, all n_persons appear
    a_by_person: dict[int, dict] = {}
    for i, p in enumerate(persons):
        rec = record_a(id_a, p)
        a_records.append(rec)
        a_by_person[i] = rec
        id_a += 1

    # Source B, only shared persons appear (plus some uniques to pad to n_persons)
    n_b_unique = n_persons - n_shared
    unique_b_persons = [generate_person() for _ in range(n_b_unique)]

    for i in sorted(shared_indices):
        p   = persons[i]
        src = a_by_person[i]
        rec = record_b_from_a(id_b, src, p)
        b_records.append(rec)
        ground_truth.append({
            "record_id_a": src["record_id"],
            "record_id_b": id_b,
            "is_match":    True,
            "match_type":  "cross_source",
        })
        id_b += 1

    for p in unique_b_persons:
        rec = record_b_unique(id_b, p)
        b_records.append(rec)
        id_b += 1

    random.shuffle(a_records)
    random.shuffle(b_records)

    path_a = OUTPUT_DIR / "source_a.csv"
    with open(path_a, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_A)
        w.writeheader()
        w.writerows(a_records)

    path_b = OUTPUT_DIR / "source_b.csv"
    with open(path_b, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_B)
        w.writeheader()
        w.writerows(b_records)

    gt_path = OUTPUT_DIR / "ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["record_id_a", "record_id_b", "is_match", "match_type"])
        w.writeheader()
        w.writerows(ground_truth)

    print(f"[generate_demo_linkage] source A: {len(a_records)} records → {path_a}")
    print(f"[generate_demo_linkage] source B: {len(b_records)} records → {path_b}")
    print(f"[generate_demo_linkage] {len(ground_truth)} linked pairs → {gt_path}")
    print(f"[generate_demo_linkage] overlap: {n_shared}/{n_persons} = {overlap:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persons", type=int,   default=600)
    parser.add_argument("--overlap", type=float, default=0.40)
    parser.add_argument("--seed",    type=int,   default=7)
    args = parser.parse_args()
    generate(args.persons, args.overlap, args.seed)


if __name__ == "__main__":
    main()
