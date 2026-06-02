#!/usr/bin/env python3
"""
Canonical benchmark dataset generator for zer.

Scenarios produced by this script correspond 1-to-1 with the entries in
``crates/zer-bench/src/cmd/scenarios/registry.rs``.

Usage:
  python generate_bench.py --scenario brp/dedupe
  python generate_bench.py --scenario brp_sis/link --records 6000
  python generate_bench.py --scenario micro/brp/dedupe  --scale micro
  python generate_bench.py --list-scenarios

All output lands under data/benchmarks/<scenario>/ by default.
Each scenario writes:
  source.csv / source_a.csv / source_b.csv / source_brp.csv / …
  ground_truth.csv   (record_id_a, record_id_b, is_match, match_type)

Legacy flag --mode {dedupe,link,link-dedupe} is kept for backward compat
and maps to brp/{dedupe,link,link_and_dedupe}.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import uuid
from dataclasses import dataclass
from typing import Optional

from faker import Faker

from _common import (
    Person, bsn, generate_person, perturb_name, pick_city, postcode, street_address,
    _get_confounder_surnames,
)

# ── Scale presets ────────────────────────────────────────────────────────────

SCALE_PRESETS = {
    "micro": 1_000,
    "small": 20_000,
}

# ── Source ID offsets (Option A: globally-unique numeric IDs per source) ─────
# BRP uses BSN-style 9-digit integers directly.
# Non-BRP sources start at these offsets so IDs are unambiguous across files
# and parse directly as u64 without needing a string→u64 mapping layer.

_KVK_ID_OFFSET = 10_000_000_000
_SIS_ID_OFFSET = 20_000_000_000
_HKS_ID_OFFSET = 30_000_000_000


def _kvk_id(i: int) -> str:
    return str(_KVK_ID_OFFSET + i)

def _sis_id(i: int) -> str:
    return str(_SIS_ID_OFFSET + i)

def _hks_id(i: int) -> str:
    return str(_HKS_ID_OFFSET + i)


# ── Ground-truth column names ────────────────────────────────────────────────

GT_FIELDS = ["record_id_a", "record_id_b", "is_match", "match_type"]

# ── Per-schema CSV field lists ───────────────────────────────────────────────

BRP_CSV_FIELDS = [
    "record_id",
    "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "geboorteplaats", "geboorteland", "nationaliteit",
    "geslacht", "straatnaam", "huisnummer", "postcode", "woonplaats",
    "verblijfstitel",
]

# KvK director extract, person fields use the same Dutch names as BRP so
# the mapping.toml identity entries (a_field == b_field) work out-of-the-box.
KVK_CSV_FIELDS = [
    "record_id",
    "kvk_nummer", "handelsnaam", "rechtsvorm",
    "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum",
    "woonplaats", "postcode",
]

# SIS entry, subset of BRP person fields; no address.
SIS_CSV_FIELDS = [
    "record_id",
    "sis_id", "categorie",
    "voornamen", "achternaam",
    "geboortedatum", "geboorteplaats", "geboorteland",
    "nationaliteit", "geslacht",
]

# HKS criminal record, similar to BRP person fields; no address.
HKS_CSV_FIELDS = [
    "record_id",
    "hks_id",
    "voornamen", "tussenvoegsel", "achternaam",
    "geboortedatum", "geboorteplaats", "geboorteland",
    "nationaliteit", "geslacht",
    "delict_types",
]

# ── Verblijfstitel helper ────────────────────────────────────────────────────

def _verblijfstitel(person: Person) -> str:
    if "NL" in person.nationaliteit:
        return "EU-burger"
    return random.choice([
        "Verblijfsvergunning regulier bepaalde tijd",
        "Verblijfsvergunning asiel bepaalde tijd",
        "Verblijfsvergunning regulier onbepaalde tijd",
        "EU-burger",
    ])

# ── BRP record builder ───────────────────────────────────────────────────────

def _brp_record(p: Person, record_id: Optional[str] = None) -> dict:
    street, number, toevoeging = street_address()
    house = number + (f" {toevoeging}" if toevoeging else "")
    return {
        "record_id":      record_id or bsn(),
        "voornamen":      p.voornamen,
        "tussenvoegsel":  p.tussenvoegsel or "",
        "achternaam":     p.achternaam,
        "geboortedatum":  p.geboortedatum,
        "geboorteplaats": p.geboorteplaats,
        "geboorteland":   p.geboorteland_nl,
        "nationaliteit":  p.nationaliteit_nl,
        "geslacht":       p.geslacht,
        "straatnaam":     street,
        "huisnummer":     house,
        "postcode":       postcode(),
        "woonplaats":     pick_city(),
        "verblijfstitel": _verblijfstitel(p),
    }

# ── KvK record builder ───────────────────────────────────────────────────────

_RECHTSVORMEN = ["BV", "NV", "Eenmanszaak", "VOF", "Maatschap"]
_HANDELSNAAM_SUFFIXES = ["Groep", "Partners", "Services", "Consultancy", "Solutions",
                          "Trading", "Beheer", "Holding", "Advies", "Tech"]

def _kvk_record(p: Person, record_id: Optional[str] = None) -> dict:
    suffix = random.choice(_HANDELSNAAM_SUFFIXES)
    handelsnaam = f"{p.achternaam} {suffix}"
    return {
        "record_id":     record_id or str(uuid.uuid4()),
        "kvk_nummer":    f"{random.randint(10000000, 99999999)}",
        "handelsnaam":   handelsnaam,
        "rechtsvorm":    random.choice(_RECHTSVORMEN),
        "voornamen":     p.voornamen,
        "tussenvoegsel": p.tussenvoegsel or "",
        "achternaam":    p.achternaam,
        "geboortedatum": p.geboortedatum,
        "woonplaats":    pick_city(),
        "postcode":      postcode(),
    }

# ── SIS record builder ───────────────────────────────────────────────────────

_SIS_CATEGORIEEN = [
    "wanted_arrest", "wanted_extradition",
    "missing_person", "discreet_check", "specific_check",
]

def _sis_record(p: Person, record_id: Optional[str] = None) -> dict:
    return {
        "record_id":     record_id or str(uuid.uuid4()),
        "sis_id":        f"SIS-{uuid.uuid4().hex[:8].upper()}",
        "categorie":     random.choice(_SIS_CATEGORIEEN),
        "voornamen":     p.voornamen,
        "achternaam":    p.achternaam,
        "geboortedatum": p.geboortedatum,
        "geboorteplaats": p.geboorteplaats,
        "geboorteland":  p.geboorteland_nl,
        "nationaliteit": p.nationaliteit_nl,
        "geslacht":      p.geslacht,
    }

# ── HKS record builder ───────────────────────────────────────────────────────

_HKS_DELICTEN = [
    "Diefstal", "Inbraak", "Fraude", "Witwassen", "Handel in verdovende middelen",
    "Mishandeling", "Beroving", "Heling", "Afpersing", "Bedreiging",
]

def _hks_record(p: Person, record_id: Optional[str] = None) -> dict:
    n_delicten = random.randint(1, 3)
    delicten = "/".join(random.sample(_HKS_DELICTEN, min(n_delicten, len(_HKS_DELICTEN))))
    return {
        "record_id":     record_id or str(uuid.uuid4()),
        "hks_id":        f"HKS-{uuid.uuid4().hex[:8].upper()}",
        "voornamen":     p.voornamen,
        "tussenvoegsel": p.tussenvoegsel or "",
        "achternaam":    p.achternaam,
        "geboortedatum": p.geboortedatum,
        "geboorteplaats": p.geboorteplaats,
        "geboorteland":  p.geboorteland_nl,
        "nationaliteit": p.nationaliteit_nl,
        "geslacht":      p.geslacht,
        "delict_types":  delicten,
    }

# ── Generic field perturbation ───────────────────────────────────────────────

def _perturb_name_fields(rec: dict, person: Person) -> dict:
    """Perturb only the name fields of a record (schema-agnostic)."""
    variant = perturb_name(person)
    rec["voornamen"]    = variant["voornamen"]
    rec["achternaam"]   = variant["achternaam"]
    if "tussenvoegsel" in rec:
        rec["tussenvoegsel"] = variant["tussenvoegsel"] or ""
    return rec

def _perturb_dob_field(rec: dict) -> dict:
    """Perturb the geboortedatum field if present.

    Only shifts day by ±1 so year and month are preserved.  Year ±1 and
    month/day swaps are intentionally excluded: both break blocking keys
    (year_month, soundex_initial_year) and produce near-zero blocking recall.
    """
    dob = rec.get("geboortedatum", "")
    parts = dob.split("-")
    if len(parts) != 3:
        return rec
    year, month, day = parts
    d = max(1, min(28, int(day) + random.choice([-1, 1])))
    rec["geboortedatum"] = f"{year}-{month}-{str(d).zfill(2)}"
    return rec

def _perturb_address_fields(rec: dict) -> dict:
    """Perturb BRP/KvK address fields if present."""
    if "straatnaam" in rec:
        street, number, toevoeging = street_address()
        house = number + (f" {toevoeging}" if toevoeging else "")
        rec["straatnaam"] = street
        rec["huisnummer"] = house
    if "woonplaats" in rec:
        rec["woonplaats"] = pick_city()
    if "postcode" in rec:
        rec["postcode"] = postcode()
    return rec

def _perturb_record(src: dict, person: Person, hard: bool = False) -> dict:
    """
    Perturb a record of any schema.

    hard=True applies two independent perturbations instead of one.
    """
    dup = dict(src)
    styles = ["address_move", "name_variant", "dob_error"]
    # Only use address_move if this schema has address fields.
    if "straatnaam" not in dup and "woonplaats" not in dup:
        styles = ["name_variant", "dob_error"]
    chosen = random.sample(styles, min(2 if hard else 1, len(styles)))
    for style in chosen:
        if style == "address_move":
            dup = _perturb_address_fields(dup)
        elif style == "name_variant":
            dup = _perturb_name_fields(dup, person)
        elif style == "dob_error":
            dup = _perturb_dob_field(dup)
    return dup

# ── Confounder injection ─────────────────────────────────────────────────────

def _inject_confounders(
    persons: list[tuple[dict, Person]],
    n: int,
    fields: list[str],
    id_gen=None,
) -> list[dict]:
    """Generate n records with same name as an existing record but different person.

    id_gen: optional callable returning the next record_id string.
    Defaults to bsn() for BRP records; pass a source-specific callable for
    non-BRP sources (e.g. KvK offset IDs).
    """
    name_index: dict[str, list[dict]] = {}
    for rec, _ in persons:
        name_index.setdefault(rec["achternaam"], []).append(rec)

    eligible: list[dict] = []
    for _, _, ach in _get_confounder_surnames():
        eligible.extend(name_index.get(ach, []))
    if not eligible:
        eligible = [r for r, _ in persons]

    _new_id = id_gen if id_gen is not None else bsn

    added: list[dict] = []
    for src in random.choices(eligible, k=n):
        new_rec = dict(src)
        # Fresh ID
        if "record_id" in fields:
            new_rec["record_id"] = _new_id()
        # Tweak DOB: same year+month, different day
        parts = src.get("geboortedatum", "1980-01-01").split("-")
        day = str(random.randint(1, 28)).zfill(2)
        if day == parts[2]:
            day = str((int(day) % 27) + 1).zfill(2)
        new_rec["geboortedatum"] = f"{parts[0]}-{parts[1]}-{day}"
        # Fresh address if applicable
        if "straatnaam" in fields:
            new_rec = _perturb_address_fields(new_rec)
        elif "woonplaats" in fields:
            new_rec["woonplaats"] = pick_city()
            if "postcode" in fields:
                new_rec["postcode"] = postcode()
        added.append(new_rec)
    return added

# ── CSV writer ───────────────────────────────────────────────────────────────

def _write_csv(path: str, fields: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

# ── Person pool helper ───────────────────────────────────────────────────────

def _gen_persons(n: int) -> list[tuple[dict, Person]]:
    result = []
    for _ in range(n):
        p = generate_person()
        result.append((_brp_record(p), p))
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# BRP-only scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def generate_brp_dedup(
    n_records:       int,
    dup_fraction:    float,
    output_dir:      str,
    hard_frac:       float = 0.20,
    confounder_frac: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"[brp/dedupe] Generating {n_records} base persons...")
    persons = _gen_persons(n_records)
    records = [r for r, _ in persons]
    gt: list[dict] = []

    n_dups = int(n_records * dup_fraction)
    print(f"[brp/dedupe] Injecting {n_dups} intra-source duplicates (~{hard_frac:.0%} hard)...")
    for src_rec, src_person in random.sample(persons, min(n_dups, len(persons))):
        hard = random.random() < hard_frac
        dup = _perturb_record(src_rec, src_person, hard=hard)
        dup_id = bsn()
        dup["record_id"] = dup_id
        records.append(dup)
        gt.append({"record_id_a": src_rec["record_id"], "record_id_b": dup_id,
                   "is_match": True, "match_type": "perturbed_duplicate"})

    n_conf = max(1, int(n_records * confounder_frac))
    print(f"[brp/dedupe] Injecting {n_conf} same-name confounders...")
    records.extend(_inject_confounders(persons, n_conf, BRP_CSV_FIELDS))

    _write_csv(os.path.join(output_dir, "source.csv"), BRP_CSV_FIELDS, records)
    _write_csv(os.path.join(output_dir, "ground_truth.csv"), GT_FIELDS, gt)
    print(f"[brp/dedupe] {len(records)} records, {len(gt)} GT pairs → {output_dir}/")


def generate_brp_link(
    n_records:       int,
    link_fraction:   float,
    output_dir:      str,
    hard_frac:       float = 0.20,
    confounder_frac: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n_shared   = int(n_records * link_fraction)
    n_unique_a = (n_records - n_shared) // 2
    n_unique_b = n_records - n_shared - n_unique_a
    print(f"[brp/link] shared={n_shared}  unique_a={n_unique_a}  unique_b={n_unique_b}")

    unique_a = _gen_persons(n_unique_a)
    unique_b = _gen_persons(n_unique_b)
    shared   = _gen_persons(n_shared)

    source_a: list[dict] = [r for r, _ in unique_a]
    source_b: list[dict] = [r for r, _ in unique_b]
    gt: list[dict] = []

    for src_rec, src_person in shared:
        id_a = bsn()
        id_b = bsn()
        hard = random.random() < hard_frac
        copy_a = _perturb_record(src_rec, src_person, hard=hard)
        copy_b = _perturb_record(src_rec, src_person, hard=hard)
        copy_a["record_id"] = id_a
        copy_b["record_id"] = id_b
        source_a.append(copy_a)
        source_b.append(copy_b)
        gt.append({"record_id_a": id_a, "record_id_b": id_b,
                   "is_match": True, "match_type": "cross_source_match"})

    n_conf = max(1, int(n_records * confounder_frac))
    source_a.extend(_inject_confounders(unique_a + [(r, p) for r, p in shared], n_conf, BRP_CSV_FIELDS))
    source_b.extend(_inject_confounders(unique_b + [(r, p) for r, p in shared], n_conf, BRP_CSV_FIELDS))

    _write_csv(os.path.join(output_dir, "source_a.csv"), BRP_CSV_FIELDS, source_a)
    _write_csv(os.path.join(output_dir, "source_b.csv"), BRP_CSV_FIELDS, source_b)
    _write_csv(os.path.join(output_dir, "ground_truth.csv"), GT_FIELDS, gt)
    print(f"[brp/link] source_a={len(source_a)}  source_b={len(source_b)}  GT={len(gt)} → {output_dir}/")


def generate_brp_link_and_dedup(
    n_records:       int,
    dup_fraction:    float,
    link_fraction:   float,
    output_dir:      str,
    hard_frac:       float = 0.20,
    confounder_frac: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n_shared   = int(n_records * link_fraction)
    n_unique_a = (n_records - n_shared) // 2
    n_unique_b = n_records - n_shared - n_unique_a
    print(f"[brp/link_and_dedup] shared={n_shared}  unique_a={n_unique_a}  unique_b={n_unique_b}")

    unique_a = _gen_persons(n_unique_a)
    unique_b = _gen_persons(n_unique_b)
    shared   = _gen_persons(n_shared)

    source_a: list[dict] = [r for r, _ in unique_a]
    source_b: list[dict] = [r for r, _ in unique_b]
    gt: list[dict] = []

    shared_a_pool: list[tuple[dict, Person]] = []
    shared_b_pool: list[tuple[dict, Person]] = []

    for src_rec, src_person in shared:
        id_a = bsn()
        id_b = bsn()
        hard = random.random() < hard_frac
        copy_a = _perturb_record(src_rec, src_person, hard=hard)
        copy_b = _perturb_record(src_rec, src_person, hard=hard)
        copy_a["record_id"] = id_a
        copy_b["record_id"] = id_b
        source_a.append(copy_a)
        source_b.append(copy_b)
        shared_a_pool.append((copy_a, src_person))
        shared_b_pool.append((copy_b, src_person))
        gt.append({"record_id_a": id_a, "record_id_b": id_b,
                   "is_match": True, "match_type": "cross_source_match"})

    all_a = unique_a + shared_a_pool
    all_b = unique_b + shared_b_pool

    for pool, src_list, label in [(all_a, source_a, "a"), (all_b, source_b, "b")]:
        n_dups = int(len(pool) * dup_fraction)
        print(f"[brp/link_and_dedup] Injecting {n_dups} dups in source_{label}...")
        for src_rec, src_person in random.sample(pool, min(n_dups, len(pool))):
            hard = random.random() < hard_frac
            dup = _perturb_record(src_rec, src_person, hard=hard)
            dup_id = bsn()
            dup["record_id"] = dup_id
            src_list.append(dup)
            gt.append({"record_id_a": src_rec["record_id"], "record_id_b": dup_id,
                       "is_match": True, "match_type": "perturbed_duplicate"})

    n_conf = max(1, int(n_records * confounder_frac))
    source_a.extend(_inject_confounders(all_a, n_conf, BRP_CSV_FIELDS))
    source_b.extend(_inject_confounders(all_b, n_conf, BRP_CSV_FIELDS))

    _write_csv(os.path.join(output_dir, "source_a.csv"), BRP_CSV_FIELDS, source_a)
    _write_csv(os.path.join(output_dir, "source_b.csv"), BRP_CSV_FIELDS, source_b)
    _write_csv(os.path.join(output_dir, "ground_truth.csv"), GT_FIELDS, gt)
    print(f"[brp/link_and_dedup] a={len(source_a)} b={len(source_b)} GT={len(gt)} → {output_dir}/")

# ═══════════════════════════════════════════════════════════════════════════════
# Cross-schema scenarios (anchor population pattern)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnchorPerson:
    """Hidden canonical person shared across domain projections."""
    anchor_id: str
    person:    Person


def _gen_anchors(n: int) -> list[AnchorPerson]:
    return [AnchorPerson(anchor_id=str(uuid.uuid4()), person=generate_person())
            for _ in range(n)]


def _cross_gt(id_a: str, id_b: str, match_type: str) -> dict:
    return {"record_id_a": id_a, "record_id_b": id_b,
            "is_match": True, "match_type": match_type}


# ── BRP  times  KvK link ───────────────────────────────────────────────────────────

def generate_brp_kvk_link(
    n_anchors:       int,
    kvk_fraction:    float,  # fraction of anchors that appear in KvK
    output_dir:      str,
    hard_frac:       float = 0.20,
    confounder_frac: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    anchors = _gen_anchors(n_anchors)
    kvk_anchors = random.sample(anchors, int(n_anchors * kvk_fraction))
    print(f"[brp_kvk/link] anchors={n_anchors}  kvk_persons={len(kvk_anchors)}")

    brp_records: list[dict] = []
    kvk_records: list[dict] = []
    gt: list[dict] = []
    brp_by_anchor: dict[str, str] = {}  # anchor_id → brp record_id

    for a in anchors:
        hard = random.random() < hard_frac
        base = _brp_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = bsn()
        brp_records.append(rec)
        brp_by_anchor[a.anchor_id] = rec["record_id"]

    for i, a in enumerate(kvk_anchors):
        hard = random.random() < hard_frac
        base = _kvk_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = _kvk_id(i + 1)
        kvk_records.append(rec)
        gt.append(_cross_gt(brp_by_anchor[a.anchor_id], rec["record_id"], "cross_source_match"))

    n_conf = max(1, int(n_anchors * confounder_frac))
    brp_pool = [(r, a.person) for r, a in zip(brp_records, anchors)]
    brp_records.extend(_inject_confounders(brp_pool, n_conf, BRP_CSV_FIELDS))

    _write_csv(os.path.join(output_dir, "source_brp.csv"), BRP_CSV_FIELDS, brp_records)
    _write_csv(os.path.join(output_dir, "source_kvk.csv"), KVK_CSV_FIELDS, kvk_records)
    _write_csv(os.path.join(output_dir, "ground_truth.csv"), GT_FIELDS, gt)
    print(f"[brp_kvk/link] brp={len(brp_records)} kvk={len(kvk_records)} GT={len(gt)} → {output_dir}/")


# ── BRP  times  SIS link ───────────────────────────────────────────────────────────

def generate_brp_sis_link(
    n_anchors:       int,
    sis_fraction:    float,
    output_dir:      str,
    hard_frac:       float = 0.20,
    confounder_frac: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    anchors = _gen_anchors(n_anchors)
    sis_anchors = random.sample(anchors, int(n_anchors * sis_fraction))
    print(f"[brp_sis/link] anchors={n_anchors}  sis_persons={len(sis_anchors)}")

    brp_records: list[dict] = []
    sis_records: list[dict] = []
    gt: list[dict] = []
    brp_by_anchor: dict[str, str] = {}

    for a in anchors:
        hard = random.random() < hard_frac
        base = _brp_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = bsn()
        brp_records.append(rec)
        brp_by_anchor[a.anchor_id] = rec["record_id"]

    for i, a in enumerate(sis_anchors):
        hard = random.random() < hard_frac
        base = _sis_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = _sis_id(i + 1)
        sis_records.append(rec)
        gt.append(_cross_gt(brp_by_anchor[a.anchor_id], rec["record_id"], "cross_source_match"))

    n_conf = max(1, int(n_anchors * confounder_frac))
    brp_pool = [(r, a.person) for r, a in zip(brp_records, anchors)]
    brp_records.extend(_inject_confounders(brp_pool, n_conf, BRP_CSV_FIELDS))

    _write_csv(os.path.join(output_dir, "source_brp.csv"), BRP_CSV_FIELDS, brp_records)
    _write_csv(os.path.join(output_dir, "source_sis.csv"), SIS_CSV_FIELDS, sis_records)
    _write_csv(os.path.join(output_dir, "ground_truth.csv"), GT_FIELDS, gt)
    print(f"[brp_sis/link] brp={len(brp_records)} sis={len(sis_records)} GT={len(gt)} → {output_dir}/")


# ── BRP  times  HKS link ───────────────────────────────────────────────────────────

def generate_brp_hks_link(
    n_anchors:       int,
    hks_fraction:    float,
    output_dir:      str,
    hard_frac:       float = 0.20,
    confounder_frac: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    anchors = _gen_anchors(n_anchors)
    hks_anchors = random.sample(anchors, int(n_anchors * hks_fraction))
    print(f"[brp_hks/link] anchors={n_anchors}  hks_persons={len(hks_anchors)}")

    brp_records: list[dict] = []
    hks_records: list[dict] = []
    gt: list[dict] = []
    brp_by_anchor: dict[str, str] = {}

    for a in anchors:
        hard = random.random() < hard_frac
        base = _brp_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = bsn()
        brp_records.append(rec)
        brp_by_anchor[a.anchor_id] = rec["record_id"]

    for i, a in enumerate(hks_anchors):
        hard = random.random() < hard_frac
        base = _hks_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = _hks_id(i + 1)
        hks_records.append(rec)
        gt.append(_cross_gt(brp_by_anchor[a.anchor_id], rec["record_id"], "cross_source_match"))

    n_conf = max(1, int(n_anchors * confounder_frac))
    brp_pool = [(r, a.person) for r, a in zip(brp_records, anchors)]
    brp_records.extend(_inject_confounders(brp_pool, n_conf, BRP_CSV_FIELDS))

    _write_csv(os.path.join(output_dir, "source_brp.csv"), BRP_CSV_FIELDS, brp_records)
    _write_csv(os.path.join(output_dir, "source_hks.csv"), HKS_CSV_FIELDS, hks_records)
    _write_csv(os.path.join(output_dir, "ground_truth.csv"), GT_FIELDS, gt)
    print(f"[brp_hks/link] brp={len(brp_records)} hks={len(hks_records)} GT={len(gt)} → {output_dir}/")


# ── BRP  times  KvK  times  HKS link+dedupe ───────────────────────────────────────────────

def generate_brp_kvk_hks_link_and_dedup(
    n_anchors:       int,
    kvk_fraction:    float,
    hks_fraction:    float,
    dup_fraction:    float,
    output_dir:      str,
    hard_frac:       float = 0.20,
    confounder_frac: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    anchors = _gen_anchors(n_anchors)
    kvk_anchors = random.sample(anchors, int(n_anchors * kvk_fraction))
    hks_anchors = random.sample(anchors, int(n_anchors * hks_fraction))
    print(f"[brp_kvk_hks] anchors={n_anchors}  kvk={len(kvk_anchors)}  hks={len(hks_anchors)}")

    brp_records: list[dict] = []
    kvk_records: list[dict] = []
    hks_records: list[dict] = []
    gt: list[dict] = []
    brp_by_anchor: dict[str, str] = {}

    brp_pool: list[tuple[dict, Person]] = []
    for a in anchors:
        hard = random.random() < hard_frac
        base = _brp_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = bsn()
        brp_records.append(rec)
        brp_by_anchor[a.anchor_id] = rec["record_id"]
        brp_pool.append((rec, a.person))

    for i, a in enumerate(kvk_anchors):
        hard = random.random() < hard_frac
        base = _kvk_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = _kvk_id(i + 1)
        kvk_records.append(rec)
        gt.append(_cross_gt(brp_by_anchor[a.anchor_id], rec["record_id"], "cross_source_match"))

    for i, a in enumerate(hks_anchors):
        hard = random.random() < hard_frac
        base = _hks_record(a.person)
        rec  = _perturb_record(base, a.person, hard=hard)
        rec["record_id"] = _hks_id(i + 1)
        hks_records.append(rec)
        gt.append(_cross_gt(brp_by_anchor[a.anchor_id], rec["record_id"], "cross_source_match"))

    # Intra-source dups in BRP
    n_dups = int(len(brp_pool) * dup_fraction)
    print(f"[brp_kvk_hks] Injecting {n_dups} BRP intra-source dups...")
    for src_rec, src_person in random.sample(brp_pool, min(n_dups, len(brp_pool))):
        hard = random.random() < hard_frac
        dup = _perturb_record(src_rec, src_person, hard=hard)
        dup_id = bsn()
        dup["record_id"] = dup_id
        brp_records.append(dup)
        gt.append({"record_id_a": src_rec["record_id"], "record_id_b": dup_id,
                   "is_match": True, "match_type": "perturbed_duplicate"})

    n_conf = max(1, int(n_anchors * confounder_frac))
    brp_records.extend(_inject_confounders(brp_pool, n_conf, BRP_CSV_FIELDS))

    _write_csv(os.path.join(output_dir, "source_brp.csv"), BRP_CSV_FIELDS, brp_records)
    _write_csv(os.path.join(output_dir, "source_kvk.csv"), KVK_CSV_FIELDS, kvk_records)
    _write_csv(os.path.join(output_dir, "source_hks.csv"), HKS_CSV_FIELDS, hks_records)
    _write_csv(os.path.join(output_dir, "ground_truth.csv"), GT_FIELDS, gt)
    print(
        f"[brp_kvk_hks] brp={len(brp_records)} kvk={len(kvk_records)}"
        f" hks={len(hks_records)} GT={len(gt)} → {output_dir}/"
    )


# ── KvK dedupe ────────────────────────────────────────────────────────────────

def generate_kvk_dedup(
    n_records:       int,
    dup_fraction:    float,
    output_dir:      str,
    hard_frac:       float = 0.20,
    confounder_frac: float = 0.01,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"[kvk/dedupe] Generating {n_records} KvK directors...")
    kvk_recs_pool: list[tuple[dict, Person]] = []
    for i in range(n_records):
        p = generate_person()
        rec = _kvk_record(p, record_id=_kvk_id(i + 1))
        kvk_recs_pool.append((rec, p))

    records = [r for r, _ in kvk_recs_pool]
    gt: list[dict] = []
    kvk_counter = n_records + 1

    n_dups = int(n_records * dup_fraction)
    print(f"[kvk/dedupe] Injecting {n_dups} intra-source duplicates...")
    for src_rec, src_person in random.sample(kvk_recs_pool, min(n_dups, len(kvk_recs_pool))):
        hard = random.random() < hard_frac
        dup = _perturb_record(src_rec, src_person, hard=hard)
        dup_id = _kvk_id(kvk_counter)
        kvk_counter += 1
        dup["record_id"] = dup_id
        records.append(dup)
        gt.append({"record_id_a": src_rec["record_id"], "record_id_b": dup_id,
                   "is_match": True, "match_type": "perturbed_duplicate"})

    n_conf = max(1, int(n_records * confounder_frac))

    def _next_kvk_id() -> str:
        nonlocal kvk_counter
        r = _kvk_id(kvk_counter)
        kvk_counter += 1
        return r

    records.extend(_inject_confounders(kvk_recs_pool, n_conf, KVK_CSV_FIELDS, id_gen=_next_kvk_id))

    _write_csv(os.path.join(output_dir, "source.csv"), KVK_CSV_FIELDS, records)
    _write_csv(os.path.join(output_dir, "ground_truth.csv"), GT_FIELDS, gt)
    print(f"[kvk/dedupe] {len(records)} records, {len(gt)} GT pairs → {output_dir}/")

# ═══════════════════════════════════════════════════════════════════════════════
# Scenario dispatch
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "brp/dedupe":                  "BRP single-source dedupe",
    "brp/link":                    "BRP cross-source linkage",
    "brp/link_and_dedupe":         "BRP linkage + intra-source dedupe",
    "brp_kvk/link":                "BRP  times  KvK cross-schema linkage",
    "brp_sis/link":                "BRP  times  SIS cross-schema linkage",
    "brp_hks/link":                "BRP  times  HKS cross-schema linkage",
    "brp_kvk_hks/link_and_dedupe": "BRP  times  KvK  times  HKS three-source link+dedupe",
    "kvk/dedupe":                  "KvK single-source dedupe",
    "micro/brp/dedupe":            "BRP dedupe micro (CI smoke test)",
    "micro/brp/link":              "BRP link micro (CI smoke test)",
    "micro/brp/link_and_dedupe":   "BRP link+dedupe micro (CI smoke test)",
    "micro/brp_sis/link":          "BRP  times  SIS link micro (CI smoke test)",
}


def dispatch(scenario: str, n: int, output_base: str, args) -> None:
    out = os.path.join(output_base, scenario)
    df  = args.dup_fraction
    lf  = args.link_fraction
    hf  = args.hard_frac
    cf  = args.confounder_frac

    if scenario in ("brp/dedupe", "micro/brp/dedupe"):
        generate_brp_dedup(n, df, out, hf, cf)

    elif scenario in ("brp/link", "micro/brp/link"):
        generate_brp_link(n, lf, out, hf, cf)

    elif scenario in ("brp/link_and_dedupe", "micro/brp/link_and_dedupe"):
        generate_brp_link_and_dedup(n, df, lf, out, hf, cf)

    elif scenario == "brp_kvk/link":
        generate_brp_kvk_link(n, kvk_fraction=0.25, output_dir=out,
                               hard_frac=hf, confounder_frac=cf)

    elif scenario in ("brp_sis/link", "micro/brp_sis/link"):
        generate_brp_sis_link(n, sis_fraction=0.05, output_dir=out,
                               hard_frac=hf, confounder_frac=cf)

    elif scenario == "brp_hks/link":
        generate_brp_hks_link(n, hks_fraction=0.15, output_dir=out,
                               hard_frac=hf, confounder_frac=cf)

    elif scenario == "brp_kvk_hks/link_and_dedupe":
        generate_brp_kvk_hks_link_and_dedup(n, kvk_fraction=0.25, hks_fraction=0.15,
                                             dup_fraction=df, output_dir=out,
                                             hard_frac=hf, confounder_frac=cf)

    elif scenario == "kvk/dedupe":
        generate_kvk_dedup(n, df, out, hf, cf)

    else:
        raise SystemExit(f"Unknown scenario: {scenario!r}. Use --list-scenarios.")

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate canonical benchmark datasets for zer-bench"
    )
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS),
        help="Scenario slug (e.g. brp/dedupe, brp_sis/link).  Use --list-scenarios to see all.",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="List all available scenarios and exit.",
    )
    # Legacy alias: --mode dedupe → --scenario brp/dedupe etc.
    parser.add_argument(
        "--mode", choices=["dedupe", "link", "link-dedupe"],
        help="(deprecated) Use --scenario instead.",
    )
    parser.add_argument("--records", type=int, default=None)
    parser.add_argument("--dup-fraction",     type=float, default=0.10)
    parser.add_argument("--link-fraction",    type=float, default=0.40)
    parser.add_argument("--output-dir",       type=str, default="data/benchmarks")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--scale", choices=["micro", "small"], default=None)
    parser.add_argument("--hard-frac",        type=float, default=0.20)
    parser.add_argument("--confounder-frac",  type=float, default=0.01)
    args = parser.parse_args()

    if args.list_scenarios:
        print(f"{'SCENARIO':<35}  DESCRIPTION")
        print("-" * 72)
        for slug, desc in SCENARIOS.items():
            print(f"{slug:<35}  {desc}")
        raise SystemExit(0)

    # Resolve scenario from --mode if --scenario not given
    if args.scenario is None and args.mode is not None:
        mode_map = {"dedupe": "brp/dedupe", "link": "brp/link", "link-dedupe": "brp/link_and_dedupe"}
        args.scenario = mode_map[args.mode]
        print(f"[generate_bench] --mode {args.mode!r} is deprecated; use --scenario {args.scenario!r}")

    if args.scenario is None:
        parser.error("either --scenario or --mode is required")

    Faker.seed(args.seed)
    random.seed(args.seed)

    n = args.records
    if args.scale:
        n = n or SCALE_PRESETS[args.scale]
    elif args.scenario.startswith("micro/"):
        n = n or SCALE_PRESETS["micro"]
    else:
        n = n or SCALE_PRESETS["small"]

    dispatch(args.scenario, n, args.output_dir, args)
