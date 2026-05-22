#!/usr/bin/env python3
"""
Demo multi-source dataset generator (BRP + KvK).

Produces two independently-perturbed views of a shared population, plus
within-source duplicate pairs in each source, simulating a scenario where
you want to both deduplicate each register and link persons across registers.

Overlap profile:
  - ~30 % of persons appear in both BRP and KvK  → cross_source ground truth
  - ~8 % within-source duplicates per source     → within_source ground truth
  - Remainder are source-unique records

Outputs in data/demos/multi_source/:
  source_brp.csv     , municipal register records
  source_kvk.csv     , company director extract records
  ground_truth.csv   , all match pairs (cross_source | within_source)

Usage:
  python data_generator/generate_demo_multi_source.py [--brp 400] [--kvk 300] [--seed 11]
"""

import argparse
import csv
import random
from pathlib import Path

from _common import (
    Person, bsn, generate_person, perturb_name, pick_city,
    postcode, street_address,
)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "demos" / "multi_source"

# Globally-unique KvK ID space so BRP and KvK IDs never collide in the record store.
_KVK_ID_OFFSET = 10_000_000

CSV_FIELDS_BRP = [
    "record_id", "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "geslacht", "straatnaam", "huisnummer", "postcode", "woonplaats",
]

CSV_FIELDS_KVK = [
    "record_id", "kvk_nummer", "handelsnaam", "rechtsvorm",
    "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "woonplaats", "postcode",
]

# Rechtsvorm distribution (Dutch company types)
_RECHTSVORMEN = ["BV", "NV", "VOF", "Eenmanszaak", "Coöperatie", "Stichting"]

_COMPANY_WORDS = [
    "Advies", "Bouw", "Consultancy", "Design", "Engineering",
    "Finance", "Groep", "Handel", "Innovations", "Juridisch",
    "Klant", "Logistiek", "Media", "Noord", "Oplossingen",
    "Partners", "Qualiteit", "Research", "Services", "Techniek",
    "Uitvoering", "Vastgoed", "West", "XL", "Zuid",
]


def _make_address() -> dict:
    street, number, toevoeging = street_address()
    house = number + (f" {toevoeging}" if toevoeging else "")
    return {"straatnaam": street, "huisnummer": house, "postcode": postcode(), "woonplaats": pick_city()}


def _kvk_nummer() -> str:
    return str(random.randint(10_000_000, 99_999_999))


def _handelsnaam(p: Person) -> str:
    word = random.choice(_COMPANY_WORDS)
    return f"{p.achternaam} {word} {random.choice(_RECHTSVORMEN)}"


def record_brp(record_id: int, p: Person) -> dict:
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


def perturb_brp(record_id: int, src: dict, p: Person) -> dict:
    """Create a within-source BRP duplicate with light perturbation."""
    dup = dict(src)
    dup["record_id"] = record_id
    style = random.choice(["name_variant", "address_move", "name_variant"])
    if style == "name_variant":
        patch = perturb_name(p)
        dup["voornamen"]    = patch["voornamen"]
        dup["tussenvoegsel"] = patch["tussenvoegsel"] or ""
        dup["achternaam"]   = patch["achternaam"]
    else:
        addr = _make_address()
        dup.update(addr)
    return {k: dup[k] for k in CSV_FIELDS_BRP}


def record_kvk(record_id: int, p: Person) -> dict:
    city = pick_city()
    pc   = postcode()
    return {
        "record_id":     record_id,
        "kvk_nummer":    _kvk_nummer(),
        "handelsnaam":   _handelsnaam(p),
        "rechtsvorm":    random.choice(_RECHTSVORMEN),
        "voornamen":     p.voornamen,
        "tussenvoegsel": p.tussenvoegsel or "",
        "achternaam":    p.achternaam,
        "geboortedatum": p.geboortedatum,
        "woonplaats":    city,
        "postcode":      pc,
    }


def record_kvk_from_brp(record_id: int, p: Person, src_brp: dict) -> dict:
    """KvK cross-source record matching a BRP person, with name/address variation."""
    style = random.choice(["name_variant", "address_lag", "name_variant"])
    voornamen    = src_brp["voornamen"]
    tussenvoegsel = src_brp["tussenvoegsel"]
    achternaam   = src_brp["achternaam"]
    if style == "name_variant":
        patch = perturb_name(p)
        voornamen     = patch["voornamen"]
        tussenvoegsel = patch["tussenvoegsel"] or ""
        achternaam    = patch["achternaam"]
    city = src_brp.get("woonplaats", pick_city()) if style != "address_lag" else pick_city()
    pc   = src_brp.get("postcode",   postcode())  if style != "address_lag" else postcode()
    return {
        "record_id":     record_id,
        "kvk_nummer":    _kvk_nummer(),
        "handelsnaam":   _handelsnaam(p),
        "rechtsvorm":    random.choice(_RECHTSVORMEN),
        "voornamen":     voornamen,
        "tussenvoegsel": tussenvoegsel,
        "achternaam":    achternaam,
        "geboortedatum": src_brp["geboortedatum"],
        "woonplaats":    city,
        "postcode":      pc,
    }


def perturb_kvk(record_id: int, src: dict, p: Person) -> dict:
    """Create a within-source KvK duplicate with light perturbation."""
    dup = dict(src)
    dup["record_id"]  = record_id
    dup["kvk_nummer"] = _kvk_nummer()
    dup["handelsnaam"] = _handelsnaam(p)
    patch = perturb_name(p)
    dup["voornamen"]     = patch["voornamen"]
    dup["tussenvoegsel"] = patch["tussenvoegsel"] or ""
    dup["achternaam"]    = patch["achternaam"]
    return {k: dup[k] for k in CSV_FIELDS_KVK}


def generate(n_brp: int, n_kvk: int, overlap: float, dup_frac: float, seed: int) -> None:
    random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Total unique persons across both sources (union)
    n_shared = round(min(n_brp, n_kvk) * overlap)
    n_brp_unique = n_brp - n_shared
    n_kvk_unique = n_kvk - n_shared

    all_persons: list[Person] = [generate_person() for _ in range(n_brp_unique + n_shared + n_kvk_unique)]
    brp_only_persons  = all_persons[:n_brp_unique]
    shared_persons    = all_persons[n_brp_unique: n_brp_unique + n_shared]
    kvk_only_persons  = all_persons[n_brp_unique + n_shared:]

    brp_records:  list[dict] = []
    kvk_records:  list[dict] = []
    ground_truth: list[dict] = []

    id_brp = 1
    id_kvk = _KVK_ID_OFFSET + 1

    # ── BRP: source-unique persons ────────────────────────────────────────────
    for p in brp_only_persons:
        brp_records.append(record_brp(id_brp, p))
        id_brp += 1

    # ── BRP + KvK: shared persons (cross-source overlap) ─────────────────────
    brp_shared_recs: list[dict] = []
    for p in shared_persons:
        rec = record_brp(id_brp, p)
        brp_records.append(rec)
        brp_shared_recs.append(rec)
        id_brp += 1

    for p, brp_rec in zip(shared_persons, brp_shared_recs):
        rec = record_kvk_from_brp(id_kvk, p, brp_rec)
        kvk_records.append(rec)
        ground_truth.append({
            "record_id_a": brp_rec["record_id"],
            "record_id_b": rec["record_id"],
            "is_match":    True,
            "match_type":  "cross_source",
        })
        id_kvk += 1

    # ── KvK: source-unique persons ────────────────────────────────────────────
    for p in kvk_only_persons:
        kvk_records.append(record_kvk(id_kvk, p))
        id_kvk += 1

    # ── Within-source BRP duplicates ──────────────────────────────────────────
    n_brp_dup = max(1, round(len(brp_records) * dup_frac))
    brp_dup_sources = random.sample(brp_records, n_brp_dup)
    brp_persons_by_id = {}
    all_brp_persons = brp_only_persons + shared_persons
    for rec, p in zip(brp_records[:len(all_brp_persons)], all_brp_persons):
        brp_persons_by_id[rec["record_id"]] = p

    for src_rec in brp_dup_sources:
        p = brp_persons_by_id.get(src_rec["record_id"])
        if p is None:
            continue
        dup = perturb_brp(id_brp, src_rec, p)
        brp_records.append(dup)
        ground_truth.append({
            "record_id_a": min(src_rec["record_id"], id_brp),
            "record_id_b": max(src_rec["record_id"], id_brp),
            "is_match":    True,
            "match_type":  "within_source",
        })
        id_brp += 1

    # ── Within-source KvK duplicates ──────────────────────────────────────────
    n_kvk_dup = max(1, round(len(kvk_records) * dup_frac))
    kvk_dup_sources = random.sample(kvk_records, n_kvk_dup)
    kvk_persons_by_id: dict[int, Person] = {}
    all_kvk_persons = list(zip(
        [r["record_id"] for r in kvk_records[:n_shared + n_kvk_unique]],
        shared_persons + kvk_only_persons,
    ))
    for rid, p in all_kvk_persons:
        kvk_persons_by_id[rid] = p

    for src_rec in kvk_dup_sources:
        p = kvk_persons_by_id.get(src_rec["record_id"])
        if p is None:
            continue
        dup = perturb_kvk(id_kvk, src_rec, p)
        kvk_records.append(dup)
        ground_truth.append({
            "record_id_a": min(src_rec["record_id"], id_kvk),
            "record_id_b": max(src_rec["record_id"], id_kvk),
            "is_match":    True,
            "match_type":  "within_source",
        })
        id_kvk += 1

    random.shuffle(brp_records)
    random.shuffle(kvk_records)

    path_brp = OUTPUT_DIR / "source_brp.csv"
    with open(path_brp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_BRP)
        w.writeheader()
        w.writerows(brp_records)

    path_kvk = OUTPUT_DIR / "source_kvk.csv"
    with open(path_kvk, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_KVK)
        w.writeheader()
        w.writerows(kvk_records)

    gt_path = OUTPUT_DIR / "ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["record_id_a", "record_id_b", "is_match", "match_type"])
        w.writeheader()
        w.writerows(ground_truth)

    cross_count  = sum(1 for g in ground_truth if g["match_type"] == "cross_source")
    within_count = sum(1 for g in ground_truth if g["match_type"] == "within_source")
    print(f"[generate_demo_multi_source] BRP : {len(brp_records)} records → {path_brp}")
    print(f"[generate_demo_multi_source] KvK : {len(kvk_records)} records → {path_kvk}")
    print(f"[generate_demo_multi_source] GT  : {len(ground_truth)} pairs "
          f"({cross_count} cross_source, {within_count} within_source) → {gt_path}")
    print(f"[generate_demo_multi_source] overlap: {n_shared}/{min(n_brp, n_kvk)} = {overlap:.0%}, "
          f"dup_frac: {dup_frac:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brp",      type=int,   default=400)
    parser.add_argument("--kvk",      type=int,   default=300)
    parser.add_argument("--overlap",  type=float, default=0.30)
    parser.add_argument("--dup-frac", type=float, default=0.08)
    parser.add_argument("--seed",     type=int,   default=11)
    args = parser.parse_args()
    generate(args.brp, args.kvk, args.overlap, args.dup_frac, args.seed)


if __name__ == "__main__":
    main()
