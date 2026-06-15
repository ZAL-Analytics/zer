#!/usr/bin/env bash
# generate_data.sh, generate datasets used by crate examples, tests, demos, and benchmarks.
#
# Usage:
#   ./scripts/generate_data.sh [OPTIONS]
#
# Options:
#   --examples             Generate shared example datasets (data/v1.1/examples/).
#   --tests                Generate crate-level test datasets (data/v1.1/tests/).
#   --demos                Generate demo datasets (persons + linkage + multi-source).
#   --benchmarks           Generate benchmark scenarios.
#   -h, --help             Show this help and exit.
#
# If no flags are given, all categories are generated.
#
# Examples:
#   # Generate everything (default)
#   ./scripts/generate_data.sh
#
#   # Only example and test datasets (needed before running crate examples/tests)
#   ./scripts/generate_data.sh --examples --tests
#
#   # Only demos datasets
#   ./scripts/generate_data.sh --demos
#
#   # Benchmarks only
#   ./scripts/generate_data.sh --benchmarks
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Argument parsing ──────────────────────────────────────────────────────────
DO_EXAMPLES=0
DO_TESTS=0
DO_DEMO=0
DO_BENCHMARK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --examples)   DO_EXAMPLES=1;  shift ;;
        --tests)      DO_TESTS=1;     shift ;;
        --demos)      DO_DEMO=1;      shift ;;
        --benchmarks) DO_BENCHMARK=1; shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^[^#]/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1 ;;
    esac
done

# If no flags given, run everything.
_any_flag=$(( DO_EXAMPLES + DO_TESTS + DO_DEMO + DO_BENCHMARK ))
if [[ "$_any_flag" -eq 0 ]]; then
    DO_EXAMPLES=1
    DO_TESTS=1
    DO_DEMO=1
    DO_BENCHMARK=1
fi

# ── Environment setup ─────────────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "WARNING: .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install faker kagglehub"
fi

python -c "import faker" 2>/dev/null || pip install faker
python -c "import kagglehub" 2>/dev/null || {
  echo "INFO: kagglehub not installed, installing for richer synthetic name/address data."
  pip install kagglehub
}

if python -c "import kagglehub" 2>/dev/null; then
  echo ">>> Ensuring Kaggle base datasets are cached in data/base/…"
  python -c "import sys; sys.path.insert(0, 'data_generator'); import _common"
fi

run() {
  echo ">>> $*"
  "$@"
}

# ── Dataset generators ────────────────────────────────────────────────────────

demos() {
    echo "--- Demo datasets ---"
    run python data_generator/generate_demo_persons.py --records 1000 --seed 42
    run python data_generator/generate_demo_linkage.py --persons 600  --seed 7
    run python data_generator/generate_demo_multi_source.py --brp 400 --kvk 300 --seed 11
}

benchmarks() {
    echo "--- Benchmarks ---"
    # BRP single-source scenarios
    run python data_generator/generate_bench.py --scenario brp/dedupe           --seed 41
    run python data_generator/generate_bench.py --scenario brp/link             --seed 42
    run python data_generator/generate_bench.py --scenario brp/link_and_dedupe  --seed 43

    # Cross-schema linking scenarios
    run python data_generator/generate_bench.py --scenario brp_kvk/link         --seed 44
    run python data_generator/generate_bench.py --scenario brp_sis/link         --seed 45
    run python data_generator/generate_bench.py --scenario brp_hks/link         --seed 46
    run python data_generator/generate_bench.py --scenario brp_kvk_hks/link_and_dedupe --seed 47

    # KvK single-source dedupe
    run python data_generator/generate_bench.py --scenario kvk/dedupe           --seed 48

    # Micro scenarios for CI smoke tests (~1 K records each)
    run python data_generator/generate_bench.py --scenario micro/brp/dedupe          --seed 8
    run python data_generator/generate_bench.py --scenario micro/brp/link            --seed 8
    run python data_generator/generate_bench.py --scenario micro/brp/link_and_dedupe --seed 8
    run python data_generator/generate_bench.py --scenario micro/brp_sis/link        --seed 8
}

examples_data() {
    echo "--- Example datasets (data/v1.1/examples/) ---"
    run python data_generator/generate_examples_tests.py --examples
}

tests_data() {
    echo "--- Test datasets (data/v1.1/tests/) ---"
    run python data_generator/generate_examples_tests.py --tests
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

[[ "$DO_EXAMPLES"  -eq 1 ]] && examples_data
[[ "$DO_TESTS"     -eq 1 ]] && tests_data
[[ "$DO_DEMO"      -eq 1 ]] && demos
[[ "$DO_BENCHMARK" -eq 1 ]] && benchmarks

echo ""
echo "Done."
