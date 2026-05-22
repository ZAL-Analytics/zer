#!/usr/bin/env python3
"""
Raw dataset generator for zer.

Produces one dataset per provider at five scale tiers for external use.
No special modes (no graph-mode, stream-mode, or snapshot logic).

Output layout:
  data/raw/1k/{provider}/
  data/raw/5k/{provider}/
  data/raw/10k/{provider}/
  data/raw/25k/{provider}/
  data/raw/50k/{provider}/

Usage:
  python data_generator/generate_raw.py                        # all sizes, all providers
  python data_generator/generate_raw.py --sizes 1k 50k         # subset of sizes
  python data_generator/generate_raw.py --providers brp anpr   # subset of providers
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Size tier definitions
# ---------------------------------------------------------------------------

# Each entry: label → {provider: kwargs dict}
# For KvK the record count is expressed as --companies; persons are scaled at 12%.
# For ANPR the record count is --passages.
# All others use --records.

SIZES = {
    "1k":  1_000,
    "5k":  5_000,
    "10k": 10_000,
    "25k": 25_000,
    "50k": 50_000,
}

# Seeds are deterministic per size so outputs are reproducible.
SIZE_SEEDS = {
    "1k":  1,
    "5k":  5,
    "10k": 10,
    "25k": 25,
    "50k": 50,
}

PROVIDERS = ["kvk", "brp", "sis", "hks", "interpol", "anpr", "cdr", "sim", "fiu"]


def build_command(provider: str, n: int, output_dir: str, seed: int) -> list[str]:
    base = ["python", f"data_generator/generate_{provider}.py",
            "--output-dir", output_dir,
            "--seed", str(seed)]

    if provider == "kvk":
        persons = max(1, round(n * 0.12))
        return base + ["--companies", str(n), "--persons", str(persons),
                       "--multi-fraction", "0.15", "--duplicate-fraction", "0.10"]

    if provider == "anpr":
        return base + ["--passages", str(n)]

    if provider == "brp":
        return base + ["--records", str(n), "--duplicate-fraction", "0.10"]

    if provider == "sis":
        return base + ["--records", str(n),
                       "--alias-fraction", "0.60",
                       "--estimated-dob-frac", "0.30"]

    if provider == "hks":
        return base + ["--records", str(n),
                       "--alias-fraction", "0.70",
                       "--bsn-fraction", "0.60"]

    # brp, interpol, cdr, sim, fiu, plain --records
    return base + ["--records", str(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate raw provider datasets at multiple scale tiers"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=list(SIZES.keys()),
        default=list(SIZES.keys()),
        help="Scale tiers to generate (default: all)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=PROVIDERS,
        help="Providers to generate (default: all)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    errors: list[str] = []

    for size_label in args.sizes:
        n = SIZES[size_label]
        seed = SIZE_SEEDS[size_label]
        for provider in args.providers:
            output_dir = str(repo_root / "data" / "raw" / size_label / provider)
            cmd = build_command(provider, n, output_dir, seed)
            print(f"\n>>> {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(repo_root))
            if result.returncode != 0:
                msg = f"FAILED: {provider} @ {size_label}"
                print(msg, file=sys.stderr)
                errors.append(msg)

    print()
    if errors:
        print("The following generators failed:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All raw datasets generated successfully.")


if __name__ == "__main__":
    main()
