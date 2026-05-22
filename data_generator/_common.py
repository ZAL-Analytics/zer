#!/usr/bin/env python3
"""
Shared data pools and utility functions for zer synthetic data generators.

Import this module from each individual generator script.  None of the
generator scripts depend on each other, only on this module.

On first import, attempts to download three Kaggle datasets into
``data/base/`` (forenames.csv, surnames.csv, nl_addresses.csv) via
``kagglehub``.  A sentinel file ``.kaggle_downloaded`` prevents re-download.
If ``kagglehub`` is not installed or the download fails, the built-in pools
below are used as fallbacks.
"""

import csv
import math
import os
import random
import string
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from faker import Faker

fake_nl = Faker("nl_NL")
fake_de = Faker("de_DE")

# ---------------------------------------------------------------------------
# Kaggle base-data download and caching
# ---------------------------------------------------------------------------

_BASE_DATA_DIR = Path(__file__).parent.parent / "data" / "base"
_BRP_DATA_DIR  = _BASE_DATA_DIR / "brp"
_CDR_DATA_DIR  = _BASE_DATA_DIR / "cdr"

_SENTINELS: dict[str, Path] = {
    "forenames":         _BRP_DATA_DIR / ".forenames_downloaded",
    "surnames":          _BRP_DATA_DIR / ".surnames_downloaded",
    "addresses":         _BRP_DATA_DIR / ".addresses_downloaded",
    "cdr_marcodena":     _CDR_DATA_DIR / ".cdr_marcodena_downloaded",
    "cdr_jakefurguson":  _CDR_DATA_DIR / ".cdr_jakefurguson_downloaded",
}

_TV_PREFIXES = [
    "van der ", "van den ", "van de ", "van het ", "van 't ",
    "van ",  "in de ",  "in den ",  "op de ",  "op den ",
    "uit de ", "uit den ", "de ", "den ", "der ", "des ",
    "'t ", "het ", "te ", "ten ", "ter ",
]


def _split_tv(surname: str) -> tuple:
    """Split 'van der Berg' → ('van der', 'Berg'); 'Jansen' → (None, 'Jansen')."""
    sl = surname.lower()
    for p in _TV_PREFIXES:
        if sl.startswith(p):
            tv   = surname[: len(p) - 1]
            rest = surname[len(p) :]
            return (tv, rest) if rest else (None, surname)
    return None, surname


def _find_col(lower_map: dict, candidates: list) -> Optional[str]:
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _open_csv(path: Path):
    """Open a CSV file with dialect sniffing; yield DictReader rows."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        # Force fieldnames to be read
        if reader.fieldnames is None:
            return
        yield from reader


def _process_names(base: Path) -> bool:
    NAME_COLS   = ["name", "first_name", "firstname", "naam", "voornaam"]
    GENDER_COLS = ["gender", "sex", "label", "class", "geslacht", "target"]
    records: list[tuple[str, str]] = []

    for csv_file in sorted(base.rglob("*.csv")):
        try:
            nc = gc = None
            for row in _open_csv(csv_file):
                if nc is None:
                    lm = {k.lower().strip(): k for k in row.keys()}
                    nc = _find_col(lm, NAME_COLS)
                    gc = _find_col(lm, GENDER_COLS)
                    if not (nc and gc):
                        break
                name = (row.get(nc) or "").strip().title()
                g    = (row.get(gc) or "").strip().upper()[:1]
                # Accept M/F; also map 1/0 or male/female prefixes
                if g == "1":
                    g = "M"
                elif g == "0":
                    g = "F"
                if len(name) >= 2 and g in ("M", "F"):
                    records.append((name, g))
        except Exception:
            continue

    if not records:
        return False

    seen: set = set()
    deduped = [(n, g) for n, g in records if n not in seen and not seen.add(n)]
    _BRP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _BRP_DATA_DIR / "forenames.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "gender"])
        w.writerows(deduped)
    nm = sum(1 for _, g in deduped if g == "M")
    nf = sum(1 for _, g in deduped if g == "F")
    print(f"[_common]   {nm}M + {nf}F forenames → {dest}")
    return True


def _process_surnames(base: Path) -> bool:
    NAME_COLS    = ["name", "surname", "last_name", "lastname", "achternaam"]
    COUNTRY_COLS = ["nationality", "country", "language", "label", "class",
                    "origin", "ethnicity"]
    # Accept Dutch and closely related European surnames
    ACCEPT_LABELS = {
        "dutch", "netherlands", "nl", "german", "germany", "belgian",
        "belgium", "european", "french", "english", "scandinavian",
        "danish", "swedish", "norwegian", "flemish",
    }
    surnames: list[str] = []

    for csv_file in sorted(base.rglob("*.csv")):
        try:
            nc = cc = None
            cc_found = True
            for row in _open_csv(csv_file):
                if nc is None:
                    lm = {k.lower().strip(): k for k in row.keys()}
                    nc = _find_col(lm, NAME_COLS)
                    cc = _find_col(lm, COUNTRY_COLS)
                    cc_found = cc is not None
                    if not nc:
                        break
                name = (row.get(nc) or "").strip().title()
                if len(name) < 2:
                    continue
                if cc_found and cc:
                    origin = (row.get(cc) or "").strip().lower()
                    if not any(lbl in origin for lbl in ACCEPT_LABELS):
                        continue
                surnames.append(name)
        except Exception:
            continue

    if not surnames:
        return False

    seen: set = set()
    deduped = [s for s in surnames if s not in seen and not seen.add(s)]
    _BRP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _BRP_DATA_DIR / "surnames.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["surname", "tussenvoegsel", "achternaam"])
        for s in deduped:
            tv, ach = _split_tv(s)
            w.writerow([s, tv or "", ach])
    print(f"[_common]   {len(deduped)} surnames → {dest}")
    return True


def _process_addresses(base: Path) -> bool:
    STREET_COLS   = ["straatnaam", "street", "straat", "street_name", "streetname",
                     "openbareruimtenaam", "weg", "laan"]
    POSTCODE_COLS = ["postcode", "postal_code", "zip", "zipcode", "postalcode",
                     "postalnumber"]
    CITY_COLS     = ["woonplaats", "city", "stad", "plaatsnaam", "place",
                     "gemeente", "town", "gemeentenaam"]
    rows: list[tuple[str, str, str]] = []

    for csv_file in sorted(base.rglob("*.csv")):
        try:
            sc = pc = cc = None
            for row in _open_csv(csv_file):
                if sc is None:
                    lm = {k.lower().strip(): k for k in row.keys()}
                    sc = _find_col(lm, STREET_COLS)
                    pc = _find_col(lm, POSTCODE_COLS)
                    cc = _find_col(lm, CITY_COLS)
                    if not (sc and pc and cc):
                        break
                street  = (row.get(sc) or "").strip()
                postal  = (row.get(pc) or "").strip().upper().replace(" ", "")
                city    = (row.get(cc) or "").strip().title()
                if (street and city
                        and len(postal) == 6
                        and postal[:4].isdigit()
                        and postal[4:].isalpha()):
                    rows.append((street, postal, city))
        except Exception:
            continue

    if not rows:
        return False

    seen: set = set()
    deduped = [r for r in rows if (r[0], r[1]) not in seen and not seen.add((r[0], r[1]))]
    _BRP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _BRP_DATA_DIR / "nl_addresses.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["straatnaam", "postcode", "woonplaats"])
        w.writerows(deduped)
    print(f"[_common]   {len(deduped)} NL address rows → {dest}")
    return True


def _ensure_forenames() -> None:
    sentinel = _SENTINELS["forenames"]
    if sentinel.exists():
        return
    try:
        import kagglehub  # type: ignore
        print("[_common]   ↓ shubhampatel231/name-classification")
        p = kagglehub.dataset_download("shubhampatel231/name-classification")
        _process_names(Path(p))
        # Write sentinel regardless, if dataset has no usable format, use built-in pool.
        sentinel.touch()
    except Exception as e:
        print(f"[_common]   forenames skipped ({e})")


def _ensure_surnames() -> None:
    sentinel = _SENTINELS["surnames"]
    if sentinel.exists():
        return
    # Backward compat: file may already exist from a prior run
    if (_BRP_DATA_DIR / "surnames.csv").exists():
        sentinel.touch()
        return
    try:
        import kagglehub  # type: ignore
        print("[_common]   ↓ alenic/surname-dataset-classification")
        p = kagglehub.dataset_download("alenic/surname-dataset-classification")
        if _process_surnames(Path(p)):
            sentinel.touch()
    except Exception as e:
        print(f"[_common]   surnames skipped ({e})")


def _ensure_addresses() -> None:
    sentinel = _SENTINELS["addresses"]
    if sentinel.exists():
        return
    # Backward compat: file may already exist from a prior run
    if (_BRP_DATA_DIR / "nl_addresses.csv").exists():
        sentinel.touch()
        return
    try:
        import kagglehub  # type: ignore
        print("[_common]   ↓ pieter79/postalcodes-street-city-netherlands-holland-dutch")
        p = kagglehub.dataset_download(
            "pieter79/postalcodes-street-city-netherlands-holland-dutch"
        )
        if _process_addresses(Path(p)):
            sentinel.touch()
    except Exception as e:
        print(f"[_common]   addresses skipped ({e})")


def _process_cdr_stats(base: Path) -> bool:
    """Extract call-duration distribution from CDR CSVs and write cdr_topology.json.

    Handles two data shapes:
      1. Per-call records with a direct duration column.
      2. Aggregated CDR profiles with Day Mins + Day Calls columns
         (jakefurgoson/fraud-detection-using-call-detail-records format)
         → derives average per-call duration = Day Mins * 60 / Day Calls.
    """
    import json

    DURATION_COLS = ["dur", "duration", "call_duration", "callduration",
                     "duration_seconds", "sec", "seconds"]
    CALLER_COLS   = ["caller", "caller_id", "msisdn_a", "from", "src",
                     "phone number", "phonenumber", "phone_number"]
    CALLEE_COLS   = ["callee", "callee_id", "msisdn_b", "to", "dst"]
    # Aggregated format: derive duration from minutes / calls columns
    MINS_COLS     = ["day mins", "day_mins", "daymins", "total_mins", "total mins"]
    CALLS_COLS    = ["day calls", "day_calls", "daycalls", "total_calls", "total calls"]

    durations: list[float] = []
    edges: list[tuple[str, str]] = []

    for csv_file in sorted(base.rglob("*.csv")):
        try:
            dc = mc = nc = cc_a = cc_b = None
            for row in _open_csv(csv_file):
                if dc is None and mc is None:
                    lm = {k.lower().strip(): k for k in row.keys()}
                    dc   = _find_col(lm, DURATION_COLS)
                    mc   = _find_col(lm, MINS_COLS)
                    nc   = _find_col(lm, CALLS_COLS)
                    cc_a = _find_col(lm, CALLER_COLS)
                    cc_b = _find_col(lm, CALLEE_COLS)
                    if not dc and not (mc and nc):
                        break
                if dc:
                    try:
                        durations.append(float(row[dc]))
                    except (ValueError, TypeError):
                        pass
                elif mc and nc:
                    try:
                        mins  = float(row[mc])
                        calls = float(row[nc])
                        if calls > 0:
                            durations.append(mins * 60.0 / calls)
                    except (ValueError, TypeError):
                        pass
                if cc_a and cc_b:
                    a = (row.get(cc_a) or "").strip()
                    b = (row.get(cc_b) or "").strip()
                    if a and b:
                        edges.append((a, b))
                elif cc_a:
                    # Single-party record; use phone number as a degree node
                    a = (row.get(cc_a) or "").strip()
                    if a:
                        edges.append((a, ""))
        except Exception:
            continue

    if not durations:
        return False

    durations.sort()
    n = len(durations)
    percentiles = {
        "p10": durations[int(n * 0.10)],
        "p25": durations[int(n * 0.25)],
        "p50": durations[int(n * 0.50)],
        "p75": durations[int(n * 0.75)],
        "p90": durations[int(n * 0.90)],
        "p99": durations[int(n * 0.99)],
    }

    # Degree distribution from edges
    from collections import Counter
    degree: Counter = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    deg_vals = sorted(degree.values())
    nd = len(deg_vals)
    cluster_sizes = {
        "min":  deg_vals[0] if deg_vals else 2,
        "p25":  deg_vals[int(nd * 0.25)] if deg_vals else 5,
        "p50":  deg_vals[int(nd * 0.50)] if deg_vals else 10,
        "p75":  deg_vals[int(nd * 0.75)] if deg_vals else 20,
        "max":  min(deg_vals[-1], 100) if deg_vals else 40,
    }

    _CDR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CDR_DATA_DIR / "cdr_topology.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({
            "duration_percentiles": percentiles,
            "cluster_sizes": cluster_sizes,
            "n_records": n,
        }, f, indent=2)
    print(f"[_common]   CDR topology → {dest}")
    return True


def _ensure_cdr() -> None:
    for slug, key in [
        ("marcodena/mobile-phone-activity",                        "cdr_marcodena"),
        ("jakefurgoson/fraud-detection-using-call-detail-records", "cdr_jakefurguson"),
    ]:
        sentinel = _SENTINELS[key]
        if sentinel.exists():
            continue
        try:
            import kagglehub  # type: ignore
            print(f"[_common]   ↓ {slug}")
            p = kagglehub.dataset_download(slug)
            _process_cdr_stats(Path(p))
            # Always write sentinel after a successful download, if the dataset
            # has no usable duration columns, there is no value in re-downloading it.
            sentinel.touch()
        except Exception as e:
            print(f"[_common]   {key} skipped ({e})")


def ensure_base_data() -> None:
    """Download and cache Kaggle base datasets.  Idempotent, each dataset has its own sentinel."""
    _BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _BRP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CDR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub  # type: ignore  # noqa: F401
    except ImportError:
        print("[_common] kagglehub not installed, using built-in name/address pools.")
        print("[_common]   pip install kagglehub   to enable richer synthetic data.")
        # Still write backward-compat sentinels if files already exist
        _ensure_surnames()
        _ensure_addresses()
        return

    _ensure_forenames()
    _ensure_surnames()
    _ensure_addresses()
    _ensure_cdr()


# Run once at import time; no-op if sentinel exists or kagglehub unavailable
ensure_base_data()

# ---------------------------------------------------------------------------
# Dutch demographic name pools  (CBS distributions, fallback if no Kaggle data)
# ---------------------------------------------------------------------------

DUTCH_MALE_FIRSTNAMES = [
    "Jan", "Pieter", "Willem", "Hendrik", "Johan", "Johannes", "Cornelis",
    "Gerrit", "Dirk", "Jacobus", "Thomas", "Mark", "Bram", "Lars", "Sven",
    "Daan", "Liam", "Noah", "Sem", "Finn", "Luca", "Jesse", "Luuk", "Stijn",
    "Tim", "Robin", "Roel", "Joris", "Arjan", "Michel", "Rick", "Kevin",
    "Stefan", "Jeroen", "Martijn", "Frank", "Paul", "Peter", "Erik", "Hans",
]

DUTCH_FEMALE_FIRSTNAMES = [
    "Maria", "Anna", "Johanna", "Elisabeth", "Cornelia", "Hendrika",
    "Wilhelmina", "Catharina", "Adriana", "Sara", "Emma", "Sophie",
    "Julia", "Laura", "Lotte", "Fleur", "Noor", "Iris", "Mila", "Sanne",
    "Eva", "Lisa", "Anouk", "Chantal", "Marieke", "Ingrid", "Petra",
    "Linda", "Sandra", "Monique", "Nicole", "Anita", "Carla", "Ellen",
]

# (full_name, tussenvoegsel, achternaam)
DUTCH_SURNAMES = [
    ("de Jong", "de", "Jong"), ("Jansen", None, "Jansen"),
    ("de Vries", "de", "Vries"), ("van den Berg", "van den", "Berg"),
    ("van Dijk", "van", "Dijk"), ("Bakker", None, "Bakker"),
    ("Janssen", None, "Janssen"), ("Visser", None, "Visser"),
    ("Smit", None, "Smit"), ("Meijer", None, "Meijer"),
    ("Mulder", None, "Mulder"), ("de Graaf", "de", "Graaf"),
    ("de Groot", "de", "Groot"), ("Bos", None, "Bos"),
    ("Vos", None, "Vos"), ("Peters", None, "Peters"),
    ("Hendriks", None, "Hendriks"), ("van Leeuwen", "van", "Leeuwen"),
    ("Dekker", None, "Dekker"), ("Brouwer", None, "Brouwer"),
    ("van der Linden", "van der", "Linden"), ("Prins", None, "Prins"),
    ("Hoekstra", None, "Hoekstra"), ("Maas", None, "Maas"),
    ("Dijkstra", None, "Dijkstra"), ("van 't Hof", "van 't", "Hof"),
    ("den Boer", "den", "Boer"), ("Lammers", None, "Lammers"),
    ("Verhoeven", None, "Verhoeven"), ("Kuipers", None, "Kuipers"),
    ("van der Wal", "van der", "Wal"), ("Peeters", None, "Peeters"),
    ("Schouten", None, "Schouten"), ("Hermans", None, "Hermans"),
    ("van Vliet", "van", "Vliet"), ("Willems", None, "Willems"),
]

# ---------------------------------------------------------------------------
# Moroccan/Arabic names (~2.4% of Dutch population, CBS 2025)
# ---------------------------------------------------------------------------

MOROCCAN_MALE_FIRSTNAMES = [
    "Mohammed", "Ahmed", "Hassan", "Youssef", "Ibrahim", "Rachid",
    "Abderrahim", "Abdellah", "Khalid", "Hamid", "Karim", "Bilal",
    "Younes", "Jamal", "Omar", "Tarik", "Samir", "Nabil", "Mehdi",
    "Amine", "Ilias", "Reda", "Soufiane", "Zakaria", "Driss",
]

MOROCCAN_FEMALE_FIRSTNAMES = [
    "Fatima", "Samira", "Nadia", "Laila", "Aicha", "Khadija",
    "Meryem", "Zineb", "Imane", "Hasnae", "Naima", "Bouchra",
    "Safae", "Nour", "Asma", "Salma", "Houda", "Malak",
]

MOROCCAN_SURNAMES = [
    "El Idrissi", "Boukili", "Benali", "Chaoui", "Amrani",
    "El Haddad", "Bakkali", "Berrada", "Tahiri", "Kadiri",
    "Lahlou", "Ziani", "Belmekki", "Oufkir", "Ouali",
    "El Mouden", "Benabdallah", "Hajji", "Soussi", "Benomar",
    "El Rachidi", "Bouazza", "Belghiti", "Naciri", "Alami",
]

# Romanization variants for the same Moroccan surname (for SIS/HKS aliases)
MOROCCAN_SURNAME_VARIANTS: dict[str, list[str]] = {
    "El Idrissi": ["Al-Idrissi", "Elidrissi", "El Idrisi", "Al Idressi"],
    "El Rachidi":  ["Al-Rachidi", "Rachidi", "El Rashidi", "Al Rashidi"],
    "El Haddad":   ["Al-Haddad", "Haddad", "El Hadad", "Al Hadad"],
    "Benali":      ["Ben Ali", "Ben-Ali", "Benaali"],
    "Chaoui":      ["Shaoui", "Chaaoui", "Shawi"],
    "Amrani":      ["Amrany", "El Amrani", "Al Amrani"],
    "Tahiri":      ["Tahiry", "El Tahiri", "At-Tahiri"],
}

# ---------------------------------------------------------------------------
# Turkish names (~2.4% of Dutch population, CBS 2025)
# ---------------------------------------------------------------------------

TURKISH_MALE_FIRSTNAMES = [
    "Mehmet", "Ali", "Mustafa", "Hasan", "Huseyin", "Ibrahim",
    "Yusuf", "Murat", "Omer", "Ahmet", "Kemal", "Serkan",
    "Burak", "Emre", "Cem", "Onur", "Baris", "Tolga",
]

TURKISH_FEMALE_FIRSTNAMES = [
    "Fatma", "Ayse", "Emine", "Hatice", "Zeynep", "Merve",
    "Elif", "Gamze", "Selin", "Busra", "Derya", "Pinar",
    "Ozge", "Esra", "Gulcan", "Aysun",
]

TURKISH_SURNAMES = [
    "Yilmaz", "Kaya", "Demir", "Celik", "Sahin", "Yildiz",
    "Ozturk", "Aydin", "Arslan", "Erdogan", "Cakir", "Ozdemir",
    "Simsek", "Polat", "Karakas", "Karaca", "Aktas", "Ates",
    "Demirci", "Kurt", "Yildirim", "Gunes", "Kilic", "Aslan",
]

# Diacritic variants for Turkish names (data entry often drops diacritics)
TURKISH_SURNAME_VARIANTS: dict[str, list[str]] = {
    "Celik":   ["Çelik", "Chelik"],
    "Sahin":   ["Şahin", "Sahin"],
    "Ozturk":  ["Öztürk", "Ozturk", "Oeztuerk"],
    "Yildirim": ["Yıldırım", "Yildirim"],
    "Gunes":   ["Güneş", "Gunes"],
}

# ---------------------------------------------------------------------------
# Surinamese/Caribbean names (~2.0% of Dutch population, CBS 2025)
# ---------------------------------------------------------------------------

SURINAMESE_MALE_FIRSTNAMES = [
    "Ravi", "Radj", "Ashwin", "Anand", "Pradeep", "Sanjay",
    "Glenn", "Clifton", "Winston", "Erwin", "Sherwin", "Dwayne",
]

SURINAMESE_FEMALE_FIRSTNAMES = [
    "Priya", "Anita", "Sunita", "Kavita", "Sandrina", "Marisha",
    "Shardlow", "Chantal", "Lisette", "Vanessa",
]

SURINAMESE_SURNAMES = [
    "Ramkhelawan", "Ramkalawan", "Baldewsingh", "Haakmat",
    "Bouterse", "Krishnadath", "Sewdien", "Bissessar",
    "Alibux", "Misiekaba", "Pengel",
]

# ---------------------------------------------------------------------------
# Indonesian names (~2.0% of Dutch population, CBS 2025)
# Dutch-Indonesian community, often second/third generation
# ---------------------------------------------------------------------------

INDONESIAN_MALE_FIRSTNAMES = [
    "Budi", "Agus", "Eko", "Hendra", "Arief", "Rizky",
    "Dimas", "Bayu", "Andi", "Fajar", "Irfan", "Reza",
    "Wahyu", "Dedi", "Anton", "Surya",
]

INDONESIAN_FEMALE_FIRSTNAMES = [
    "Dewi", "Sari", "Putri", "Rina", "Indah", "Yanti",
    "Sri", "Ayu", "Fitri", "Rahayu", "Nurul", "Siti",
    "Wati", "Rini", "Lestari", "Novi",
]

INDONESIAN_SURNAMES = [
    "Santoso", "Wijaya", "Kusuma", "Pratama", "Setiawan",
    "Hartono", "Gunawan", "Sutrisno", "Hidayat", "Wibowo",
    "Rahayu", "Wahyudi", "Susanto", "Purnomo", "Saputra",
]

# ---------------------------------------------------------------------------
# Dutch cities with population weights  (fallback)
# ---------------------------------------------------------------------------

NL_CITIES = [
    ("Amsterdam", 10), ("Rotterdam", 8), ("Den Haag", 7), ("Utrecht", 6),
    ("Eindhoven", 4), ("Groningen", 3), ("Tilburg", 3), ("Almere", 3),
    ("Breda", 2), ("Nijmegen", 2), ("Enschede", 2), ("Haarlem", 2),
    ("Arnhem", 2), ("Zaanstad", 2), ("Amersfoort", 2), ("Apeldoorn", 2),
    ("Den Bosch", 2), ("Zwolle", 1), ("Leiden", 1), ("Maastricht", 1),
    ("Dordrecht", 1), ("Zoetermeer", 1), ("Ede", 1), ("Westland", 1),
    ("Delft", 1), ("Deventer", 1), ("Venlo", 1), ("Alkmaar", 1),
]

_CITY_NAMES   = [c[0] for c in NL_CITIES]
_CITY_WEIGHTS = [c[1] for c in NL_CITIES]

STREET_NAMES = [
    "Hoofdstraat", "Kerkstraat", "Molenstraat", "Schoolstraat",
    "Stationsweg", "Dorpsstraat", "Stadhuisplein", "Markt",
    "Prins Hendriklaan", "Wilhelminastraat", "Nassaulaan",
    "Nieuwstraat", "Parkweg", "Koninginneweg", "Beatrixlaan",
    "Keizersgracht", "Herengracht", "Prinsengracht", "Singel",
    "Damrak", "Kalverstraat", "Leidsestraat", "Coolsingel",
    "Blaak", "Binnenweg", "Vredenburg", "Catharijnesingel",
    "Stadsring", "Utrechtseweg", "Amsterdamseweg",
    "Haagweg", "Lijnbaan", "Brede Hilledijk", "Westersingel",
]

# ---------------------------------------------------------------------------
# Country / nationality data (ISO 3166-1 alpha-2 → Dutch name)
# ---------------------------------------------------------------------------

COUNTRIES = {
    "NL": "Nederland",   "MA": "Marokko",    "TR": "Turkije",
    "SR": "Suriname",    "ID": "Indonesië",  "DE": "Duitsland",
    "BE": "België",      "PL": "Polen",      "GB": "Groot-Brittannië",
    "FR": "Frankrijk",   "IT": "Italië",     "ES": "Spanje",
    "SY": "Syrië",       "AF": "Afghanistan","IQ": "Irak",
    "SO": "Somalië",     "ER": "Eritrea",    "ET": "Ethiopië",
    "RU": "Rusland",     "CN": "China",      "IN": "India",
    "PK": "Pakistan",    "EG": "Egypte",     "LY": "Libië",
    "TN": "Tunesië",     "DZ": "Algerije",   "NG": "Nigeria",
}

# Per-locale typical birth countries and nationalities
LOCALE_BIRTH_COUNTRY: dict[str, list[str]] = {
    "dutch":       ["NL"],
    "moroccan":    ["MA"],
    "turkish":     ["TR"],
    "surinamese":  ["SR"],
    "indonesian":  ["ID"],
    "other":       ["DE", "BE", "PL", "GB", "FR", "SY", "AF", "IQ", "SO", "ER"],
}

LOCALE_NATIONALITY: dict[str, list[str]] = {
    "dutch":       [["NL"]],
    "moroccan":    [["NL", "MA"], ["MA"]],
    "turkish":     [["NL", "TR"], ["TR"]],
    "surinamese":  [["NL", "SR"], ["NL"]],
    "indonesian":  [["NL", "ID"], ["NL"]],
    "other":       [["NL", "DE"], ["DE"], ["NL", "SY"], ["SY"], ["AF"], ["NL", "AF"]],
}

# Dutch Schengen/non-Schengen notice countries for SIS
EU_SCHENGEN_STATES = [
    "NL", "DE", "FR", "BE", "AT", "CH", "SE", "NO", "DK", "FI",
    "PL", "CZ", "HU", "SK", "SI", "IT", "ES", "PT", "GR", "LU",
    "IS", "LI", "LV", "LT", "EE",
]

# ---------------------------------------------------------------------------
# Document type pools
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = ["Paspoort", "Identiteitskaart", "Verblijfsvergunning", "Rijbewijs"]
DOCUMENT_TYPE_CODES = {
    "Paspoort": "P", "Identiteitskaart": "I",
    "Verblijfsvergunning": "V", "Rijbewijs": "D",
}

# Physical descriptor pools
HAAR_KLEUREN = ["zwart", "donkerbruin", "bruin", "lichtbruin", "blond", "rood", "grijs", "wit"]
OOG_KLEUREN  = ["bruin", "donkerbruin", "blauw", "groen", "grijs", "hazel"]

# ---------------------------------------------------------------------------
# Dutch carriers
# ---------------------------------------------------------------------------

NL_CARRIERS = [
    ("KPN",      "20408"),
    ("T-Mobile", "20416"),
    ("Vodafone", "20404"),
    ("Tele2",    "20420"),
    ("Lebara",   "20416"),  # MVNO on T-Mobile
    ("Lyca",     "20416"),  # MVNO on T-Mobile
    ("Simpel",   "20408"),  # MVNO on KPN
]

NL_BANKS = ["ABNA", "INGB", "RABO", "SNSB", "TRIO", "BUNQ", "KNAB", "ASNB"]

# ---------------------------------------------------------------------------
# Dynamic pool loading  (Kaggle data if present, built-ins otherwise)
# ---------------------------------------------------------------------------


def _load_forenames() -> tuple[list[str], list[str]]:
    p = _BRP_DATA_DIR / "forenames.csv"
    if not p.exists():
        return DUTCH_MALE_FIRSTNAMES[:], DUTCH_FEMALE_FIRSTNAMES[:]
    male: list[str] = []
    female: list[str] = []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                g    = (row.get("gender") or "").strip()
                if name:
                    if g == "M":
                        male.append(name)
                    elif g == "F":
                        female.append(name)
    except Exception:
        pass
    return (male or DUTCH_MALE_FIRSTNAMES[:]), (female or DUTCH_FEMALE_FIRSTNAMES[:])


def _load_surnames() -> list[tuple]:
    p = _BRP_DATA_DIR / "surnames.csv"
    if not p.exists():
        return DUTCH_SURNAMES[:]
    result = []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                full = (row.get("surname") or "").strip()
                tv   = (row.get("tussenvoegsel") or "").strip() or None
                ach  = (row.get("achternaam") or "").strip()
                if full and ach:
                    result.append((full, tv, ach))
    except Exception:
        pass
    return result or DUTCH_SURNAMES[:]


def _load_addresses() -> list[tuple]:
    p = _BRP_DATA_DIR / "nl_addresses.csv"
    if not p.exists():
        return []
    rows = []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                street  = (row.get("straatnaam") or "").strip()
                postal  = (row.get("postcode") or "").strip()
                city    = (row.get("woonplaats") or "").strip()
                if street and postal and city:
                    rows.append((street, postal, city))
    except Exception:
        pass
    return rows


# Module-level dynamic pools, used by all generators
_FORENAMES_M:   list[str]   = []
_FORENAMES_F:   list[str]   = []
_SURNAMES_POOL: list[tuple] = []
_ADDRESSES:     list[tuple] = []

_FORENAMES_M, _FORENAMES_F = _load_forenames()
_SURNAMES_POOL              = _load_surnames()
_ADDRESSES                  = _load_addresses()

# Flat street list for street_address(), Kaggle pool if available, else built-in
_STREET_POOL = [a[0] for a in _ADDRESSES] if _ADDRESSES else STREET_NAMES[:]

# City pool from addresses dataset (unique values); weighted built-in pool as fallback
_CITY_POOL_EXT: Optional[list[str]] = (
    list({a[2] for a in _ADDRESSES if a[2]}) if _ADDRESSES else None
)


def _zipf_weights(n: int) -> list[float]:
    """Return Zipf (1/rank) weights for a pool of size n."""
    return [1.0 / r for r in range(1, n + 1)]


# Zipf weights for Dutch name pools, real Dutch frequency data follows a power law:
# top 10 surnames cover ~15% of the population (CBS).
_SURNAMES_WEIGHTS:    list[float] = _zipf_weights(len(_SURNAMES_POOL))
_FORENAMES_M_WEIGHTS: list[float] = _zipf_weights(len(_FORENAMES_M))
_FORENAMES_F_WEIGHTS: list[float] = _zipf_weights(len(_FORENAMES_F))


_CONFOUNDER_TOP_N = 30


def _get_confounder_surnames(n: int = _CONFOUNDER_TOP_N) -> list[tuple]:
    """Return the top-n surnames by Zipf rank (most common Dutch surnames)."""
    return _SURNAMES_POOL[:n]

# ---------------------------------------------------------------------------
# Person dataclass
# ---------------------------------------------------------------------------


@dataclass
class Person:
    voornamen:     str
    achternaam:    str
    tussenvoegsel: Optional[str]
    geboortedatum: str           # ISO 8601: YYYY-MM-DD
    geboorteplaats: str
    geboorteland:  str           # ISO 3166-1 alpha-2
    nationaliteit: list[str]     # list of ISO codes, e.g. ["NL", "MA"]
    geslacht:      str           # M / V / O
    locale:        str           # dutch / moroccan / turkish / surinamese / other

    @property
    def full_name(self) -> str:
        parts = [self.voornamen]
        if self.tussenvoegsel:
            parts.append(self.tussenvoegsel)
        parts.append(self.achternaam)
        return " ".join(parts)

    @property
    def geboorteland_nl(self) -> str:
        return COUNTRIES.get(self.geboorteland, self.geboorteland)

    @property
    def nationaliteit_nl(self) -> str:
        return "/".join(COUNTRIES.get(n, n) for n in self.nationaliteit)


# ---------------------------------------------------------------------------
# Core generators
# ---------------------------------------------------------------------------


def pick_city() -> str:
    if _CITY_POOL_EXT:
        return random.choice(_CITY_POOL_EXT)
    return random.choices(_CITY_NAMES, weights=_CITY_WEIGHTS, k=1)[0]


def locale_group() -> str:
    # Weights based on CBS 2025 population by origin (cbs.nl)
    # Dutch ~72%, Other European + rest ~17%, Moroccan ~2.4%, Turkish ~2.4%,
    # Surinamese ~2.0%, Indonesian ~2.0%
    return random.choices(
        ["dutch", "moroccan", "turkish", "surinamese", "indonesian", "other"],
        weights=[72, 2, 2, 2, 2, 20],
        k=1,
    )[0]


def generate_person(locale: Optional[str] = None) -> Person:
    if locale is None:
        locale = locale_group()

    dob = fake_nl.date_of_birth(minimum_age=18, maximum_age=75)
    birthplace = pick_city()
    birth_country = random.choice(LOCALE_BIRTH_COUNTRY.get(locale, ["NL"]))
    nationality   = random.choice(LOCALE_NATIONALITY.get(locale, [["NL"]]))

    if locale == "dutch":
        gender = random.choice(["M", "V"])
        if gender == "M":
            pool, weights = _FORENAMES_M, _FORENAMES_M_WEIGHTS
        else:
            pool, weights = _FORENAMES_F, _FORENAMES_F_WEIGHTS
        firstname = random.choices(pool, weights=weights, k=1)[0]
        if random.random() < 0.2:
            firstname = f"{firstname} {random.choices(pool, weights=weights, k=1)[0]}"
        _, tv, ach = random.choices(_SURNAMES_POOL, weights=_SURNAMES_WEIGHTS, k=1)[0]
        return Person(voornamen=firstname, achternaam=ach, tussenvoegsel=tv,
                      geboortedatum=dob.strftime("%Y-%m-%d"), geboorteplaats=birthplace,
                      geboorteland=birth_country, nationaliteit=nationality,
                      geslacht=gender, locale=locale)

    if locale == "moroccan":
        gender = random.choice(["M", "V"])
        pool = MOROCCAN_MALE_FIRSTNAMES if gender == "M" else MOROCCAN_FEMALE_FIRSTNAMES
        return Person(voornamen=random.choice(pool),
                      achternaam=random.choice(MOROCCAN_SURNAMES),
                      tussenvoegsel=None, geboortedatum=dob.strftime("%Y-%m-%d"),
                      geboorteplaats=birthplace, geboorteland=birth_country,
                      nationaliteit=nationality, geslacht=gender, locale=locale)

    if locale == "turkish":
        gender = random.choice(["M", "V"])
        pool = TURKISH_MALE_FIRSTNAMES if gender == "M" else TURKISH_FEMALE_FIRSTNAMES
        return Person(voornamen=random.choice(pool),
                      achternaam=random.choice(TURKISH_SURNAMES),
                      tussenvoegsel=None, geboortedatum=dob.strftime("%Y-%m-%d"),
                      geboorteplaats=birthplace, geboorteland=birth_country,
                      nationaliteit=nationality, geslacht=gender, locale=locale)

    if locale == "surinamese":
        gender = random.choice(["M", "V"])
        pool = SURINAMESE_MALE_FIRSTNAMES if gender == "M" else SURINAMESE_FEMALE_FIRSTNAMES
        return Person(voornamen=random.choice(pool),
                      achternaam=random.choice(SURINAMESE_SURNAMES),
                      tussenvoegsel=None, geboortedatum=dob.strftime("%Y-%m-%d"),
                      geboorteplaats=birthplace, geboorteland=birth_country,
                      nationaliteit=nationality, geslacht=gender, locale=locale)

    if locale == "indonesian":
        gender = random.choice(["M", "V"])
        pool = INDONESIAN_MALE_FIRSTNAMES if gender == "M" else INDONESIAN_FEMALE_FIRSTNAMES
        return Person(voornamen=random.choice(pool),
                      achternaam=random.choice(INDONESIAN_SURNAMES),
                      tussenvoegsel=None, geboortedatum=dob.strftime("%Y-%m-%d"),
                      geboorteplaats=birthplace, geboorteland=birth_country,
                      nationaliteit=nationality, geslacht=gender, locale=locale)

    # other
    gender = random.choice(["M", "V"])
    return Person(
        voornamen=(fake_de.first_name_male() if gender == "M" else fake_de.first_name_female()),
        achternaam=fake_de.last_name(), tussenvoegsel=None,
        geboortedatum=dob.strftime("%Y-%m-%d"), geboorteplaats=birthplace,
        geboorteland=birth_country, nationaliteit=nationality,
        geslacht=gender, locale=locale,
    )


# Plausible Dutch first-name spelling variants (data-entry / transliteration errors).
# Only covers names that realistically appear in Dutch administrative records with
# multiple accepted spellings, not invented combinations.
_FIRSTNAME_VARIANTS: dict[str, list[str]] = {
    "Erik":    ["Eric", "Eryk"],
    "Eric":    ["Erik"],
    "Sophie":  ["Sofie", "Sophy"],
    "Sofie":   ["Sophie"],
    "Pieter":  ["Peter", "Pietter"],
    "Peter":   ["Pieter"],
    "Martijn": ["Martin"],
    "Stefan":  ["Stephan", "Steffan"],
    "Stephan": ["Stefan", "Steven"],
    "Michel":  ["Michael", "Michiel"],
    "Michiel": ["Michel"],
    "Jan":     ["Jean", "Yan"],
    "Kees":    ["Cees"],
    "Cees":    ["Kees", "Ces"],
    "Luuk":    ["Luk", "Luke"],
    "Stijn":   ["Steijn", "Styn"],
    "Daan":    ["Dan"],
    "Luca":    ["Luka"],
    "Sanne":   ["Zanne", "Sanna"],
    "Anouk":   ["Annouk", "Anuk"],
    "Marieke": ["Marike", "Mariecke"],
    "Chantal": ["Chantel"],
    "Noor":    ["Nor"],
    "Emma":    ["Ema"],
    "Sara":    ["Sarah"],
    "Sarah":   ["Sara"],
    "Lisa":    ["Liza"],
    "Lotte":   ["Lot"],
    "Eva":     ["Ewa"],
}


def _dutch_phonetic_achternaam(name: str) -> str:
    """Apply one plausible Dutch phonetic substitution to a surname.

    Covers the most common spelling variation patterns found in Dutch
    administrative records: ij/y alternation, ie shortening, double-consonant
    dropping, and the archaic -gh suffix.
    """
    subs = [
        ("ij", "y"),  # Dijkstra → Dykstra, Rijk → Ryk
        ("ie", "y"),  # Vries → Vrys
        ("ei", "ai"), # Meijer → Maijer
        ("kk", "k"),  # Bakker → Baker
        ("ss", "s"),  # Visser → Viser
        ("nn", "n"),  # catches e.g. Janssen variants where two n's drop to one
        ("gh", "g"),  # Bergh → Berg (archaic Dutch)
    ]
    lower = name.lower()
    applicable = [(old, new) for old, new in subs if old in lower]
    if not applicable:
        # Fallback: drop one character from a doubled letter if present
        for i in range(len(name) - 1):
            if name[i].lower() == name[i + 1].lower() and name[i].isalpha():
                return name[:i] + name[i + 1:]
        return name
    old, new = random.choice(applicable)
    idx = lower.index(old)
    if name[idx].isupper() and new:
        new = new[0].upper() + new[1:]
    return name[:idx] + new + name[idx + len(old):]


def perturb_name(person: Person) -> dict:
    """Return a name-variant dict simulating data entry discrepancies."""
    voornamen     = person.voornamen
    achternaam    = person.achternaam
    tussenvoegsel = person.tussenvoegsel

    style = random.choice([
        "abbreviate_first", "drop_tussenvoegsel", "capitalize_tussenvoegsel",
        "swap_tussenvoegsel_abbrev", "phonetic_achternaam", "initial_only",
        "romanization_variant", "firstname_variant",
    ])

    if style == "abbreviate_first":
        parts = voornamen.split()
        voornamen = ".".join(p[0] for p in parts) + "."
    elif style == "drop_tussenvoegsel":
        tussenvoegsel = None
    elif style == "capitalize_tussenvoegsel" and tussenvoegsel:
        tussenvoegsel = tussenvoegsel.title()
    elif style == "swap_tussenvoegsel_abbrev" and tussenvoegsel:
        abbrevs = {"van den": "v/d", "van der": "v/d", "van de": "v/d", "van": "v."}
        tussenvoegsel = abbrevs.get(tussenvoegsel.lower(), tussenvoegsel)
    elif style == "phonetic_achternaam" and len(achternaam) > 3:
        achternaam = _dutch_phonetic_achternaam(achternaam)
    elif style == "initial_only":
        voornamen = voornamen[0] + "."
    elif style == "romanization_variant":
        variants = (MOROCCAN_SURNAME_VARIANTS.get(achternaam)
                    or TURKISH_SURNAME_VARIANTS.get(achternaam))
        if variants:
            achternaam = random.choice(variants)
    elif style == "firstname_variant":
        first_part = voornamen.split()[0]
        variants = _FIRSTNAME_VARIANTS.get(first_part)
        if variants:
            rest = voornamen.split()[1:]
            voornamen = " ".join([random.choice(variants)] + rest)

    return {"voornamen": voornamen, "achternaam": achternaam, "tussenvoegsel": tussenvoegsel}


def alias_variants(person: Person, n: int = 2) -> list[str]:
    """
    Generate n romanization/alias variants of a person's name.
    Used for SIS II and HKS alias_namen fields.
    """
    variants = []
    # Romanization variants
    sur_variants = (MOROCCAN_SURNAME_VARIANTS.get(person.achternaam)
                    or TURKISH_SURNAME_VARIANTS.get(person.achternaam)
                    or [])
    for sv in random.sample(sur_variants, min(len(sur_variants), n)):
        variants.append(f"{person.voornamen} {sv}")

    # Name-order transposition (family name first, common in non-Western registries)
    if len(variants) < n:
        variants.append(f"{person.achternaam} {person.voornamen}")

    # Abbreviated first name
    if len(variants) < n:
        abbrev = person.voornamen[0] + "."
        variants.append(f"{abbrev} {person.achternaam}")

    return variants[:n]


# ---------------------------------------------------------------------------
# Identifier generators
# ---------------------------------------------------------------------------


def bsn() -> str:
    """Generate a syntactically valid BSN (passes elfproef / 11-test)."""
    while True:
        digits = [random.randint(0, 9) for _ in range(8)]
        total = sum(d * w for d, w in zip(digits, [9, 8, 7, 6, 5, 4, 3, 2]))
        last = total % 11
        if last <= 9:
            candidate = "".join(str(d) for d in digits) + str(last)
            if candidate != "000000000":
                return candidate


def iban_nl() -> str:
    bank    = random.choice(NL_BANKS)
    account = str(random.randint(0, 9_999_999_999)).zfill(10)
    check   = str(random.randint(10, 99))
    return f"NL{check}{bank}{account}"


def postcode() -> str:
    number = random.randint(1000, 9999)
    forbidden = {"SS", "SD", "SA"}
    while True:
        letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
        if letters not in forbidden:
            return f"{number}{letters}"


def phone_nl() -> str:
    """Dutch mobile number in E.164: +316XXXXXXXX."""
    digits = "".join(str(random.randint(0, 9)) for _ in range(8))
    return f"+316{digits}"


def street_address() -> tuple[str, str, Optional[str]]:
    """Return (straatnaam, huisnummer, toevoeging).

    Draws from the Kaggle-sourced address pool when available, otherwise
    from the built-in STREET_NAMES list.
    """
    if _ADDRESSES:
        street, _pc, _city = random.choice(_ADDRESSES)
    else:
        street = random.choice(STREET_NAMES)
    number     = str(random.randint(1, 250))
    toevoeging = random.choice([None, None, None, "A", "B", "bis", "huis", "1", "2"])
    return street, number, toevoeging


def document_number(doc_type: str = "Paspoort") -> str:
    if doc_type == "Paspoort":
        # NL passport: letter + 8 digits
        return random.choice("ABCDEFGHJKLMNPRST") + "".join(
            str(random.randint(0, 9)) for _ in range(8)
        )
    if doc_type == "Identiteitskaart":
        return "".join(random.choices(string.ascii_uppercase, k=2)) + "".join(
            str(random.randint(0, 9)) for _ in range(7)
        )
    # Generic
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=9))


def license_plate() -> str:
    """Generate a synthetic Dutch license plate in one of several modern formats."""
    plate_chars = "ABCDEFGHJKLMNPRSTUVWXYZ"  # no I, O, Q
    fmt = random.choice(["99LLL9", "9LLL99", "LL999L"])
    if fmt == "99LLL9":
        return (f"{random.randint(10,99)}-"
                f"{''.join(random.choices(plate_chars, k=3))}-"
                f"{random.randint(1, 9)}")
    if fmt == "9LLL99":
        return (f"{random.randint(1,9)}-"
                f"{''.join(random.choices(plate_chars, k=3))}-"
                f"{random.randint(10, 99)}")
    # LL-999-L
    return (f"{''.join(random.choices(plate_chars, k=2))}-"
            f"{random.randint(100, 999)}-"
            f"{random.choice(plate_chars)}")


# Extended OCR confusion pairs for Dutch ANPR plates, derived from:
#   - arXiv:2203.14298 (Pareto analysis of ANPR character misclassifications)
#   - arXiv:2412.12572 (confusion matrix, deep-learning ANPR)
#   - Dutch plate font redesign context (2002, P/R gap)
# Each key maps to a list of substitutes; random.choice() selects one.
_OCR_CONFUSION: dict[str, list[str]] = {
    "0": ["O", "Q"],
    "O": ["0", "Q"],
    "Q": ["O", "0"],
    "1": ["I", "T"],
    "I": ["1", "T"],
    "T": ["1", "I"],
    "8": ["B", "E", "3", "6"],
    "B": ["8", "P", "R"],
    "E": ["8", "F"],
    "F": ["E"],
    "3": ["8", "9"],
    "6": ["8", "b"],
    "5": ["S", "3"],
    "S": ["5"],
    "2": ["Z"],
    "Z": ["2"],
    "P": ["R", "B"],
    "R": ["P"],
    "C": ["G"],
    "G": ["C"],
    "M": ["W"],
    "W": ["M", "V"],
    "V": ["W"],
    "D": ["O"],
    "A": ["4"],
    "4": ["A", "9"],
    "9": ["4", "q"],
    "K": ["X"],
    "X": ["K"],
}


def ocr_confuse_plate(plate: str) -> str:
    """Apply one random OCR character confusion to a normalized plate string."""
    normalized  = plate.replace("-", "")
    confusable  = [(i, c) for i, c in enumerate(normalized) if c in _OCR_CONFUSION]
    if not confusable:
        return plate  # nothing to confuse
    idx, char = random.choice(confusable)
    chars = list(normalized)
    chars[idx] = random.choice(_OCR_CONFUSION[char])
    confused = "".join(chars)
    # Re-insert hyphens at same positions as original
    result, ci = [], 0
    for c in plate:
        if c == "-":
            result.append("-")
        else:
            result.append(confused[ci])
            ci += 1
    return "".join(result)


def msisdn() -> str:
    """Dutch mobile MSISDN in E.164."""
    return phone_nl()


def imsi(carrier_mnc: str = "20416") -> str:
    """Generate a plausible IMSI: MCC(204) + MNC(2) + MSIN(10)."""
    msin = "".join(str(random.randint(0, 9)) for _ in range(10))
    return f"{carrier_mnc}{msin}"


def imei() -> str:
    """Generate a plausible 15-digit IMEI."""
    return "".join(str(random.randint(0, 9)) for _ in range(15))


def iccid() -> str:
    """Generate a plausible 19-digit ICCID (SIM card serial)."""
    # 89 (telecom) + 31 (NL country code) + carrier code (2) + account (14) + check (1)
    return "89" + "31" + "".join(str(random.randint(0, 9)) for _ in range(15))


def random_datetime(start_date: date, end_date: date) -> str:
    """Random ISO 8601 datetime between two dates."""
    delta = (end_date - start_date).days
    d = start_date + timedelta(days=random.randint(0, max(delta, 0)))
    h = random.randint(0, 23)
    m = random.randint(0, 59)
    s = random.randint(0, 59)
    return f"{d.strftime('%Y-%m-%d')}T{h:02d}:{m:02d}:{s:02d}"


def estimated_dob(year: int) -> str:
    """Return 1 January of given year, the standard fallback for unknown DOBs."""
    return f"{year}-01-01"
