#!/usr/bin/env bash
# run_benchmark.sh, zer-bench driver
#
# Usage:
#   ./scripts/run_benchmark.sh [OPTIONS]
#
# Options:
#   --scenario SLUG|all      Benchmark scenario, e.g. brp/dedupe or brp_kvk/link.
#                            Use 'all' to run all full-size scenarios back-to-back.
#                            With --type=throughput, 'all' runs only the dedupe
#                            scenarios (brp/dedupe and kvk/dedupe).
#                            Use --list to see all available scenarios.
#   --list                   Print all available scenarios and exit.
#   --type TYPE              Benchmark type (default: accuracy).
#                            accuracy    measure precision/recall/F1 against ground truth
#                            throughput  measure raw processing speed (pairs/s, vectors/s)
#   --compare-libs LIB[,LIB...]  Comma-separated competitor libraries to benchmark
#                            alongside zer (e.g. --compare-libs=splink).  A comparison
#                            table is printed at the end.  Supported for both
#                            --type=accuracy and --type=throughput.
#   --target TARGET          zer compute target: cpu (default), cuda, vulkan, avx2.
#   --judge-target TARGET    Enable the MiniLM neural judge and set its provider:
#                            cpu (default), cuda, tensorrt.
#                            For both accuracy and throughput: runs zer without then
#                            with the judge and prints a side-by-side comparison.
#   --out DIR                Output directory (default: bench_results/data/{type}_{scenario}_{ts}).
#   --release                Build in release mode (default).
#   --debug                  Build in debug mode.
#   -h, --help               Show this help and exit.
#
# Examples:
#   # List all available scenarios
#   ./scripts/run_benchmark.sh --list
#
#   # Accuracy benchmark: zer only (default type)
#   ./scripts/run_benchmark.sh --scenario=brp/dedupe
#
#   # Accuracy benchmark: zer vs splink, then compare
#   ./scripts/run_benchmark.sh --scenario=brp/dedupe --compare-libs=splink
#
#   # Throughput: zer only on CUDA
#   ./scripts/run_benchmark.sh --scenario=brp/dedupe --type=throughput --target=cuda
#
#   # Throughput: zer vs splink (CPU)
#   ./scripts/run_benchmark.sh --scenario=brp/dedupe --type=throughput --compare-libs=splink
#
#   # Judge comparison (accuracy: no-judge then with-judge side-by-side)
#   ./scripts/run_benchmark.sh --scenario=brp/dedupe --judge-target=cpu
#
#   # Judge on TensorRT vs baseline
#   ./scripts/run_benchmark.sh --scenario=brp/dedupe --judge-target=tensorrt
#
#   # Run all accuracy scenarios back-to-back
#   ./scripts/run_benchmark.sh --scenario=all
#
#   # Run all accuracy scenarios against splink
#   ./scripts/run_benchmark.sh --scenario=all --compare-libs=splink
#
#   # Run all throughput scenarios (brp/dedupe and kvk/dedupe)
#   ./scripts/run_benchmark.sh --scenario=all --type=throughput
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=$(nproc)

# ── Defaults ──────────────────────────────────────────────────────────────────
SCENARIO=""
TYPE="accuracy"
LIBRARIES=""
TARGET="cpu"
JUDGE_TARGET=""
OUT=""
BUILD_MODE="--release"
LIST_SCENARIOS=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario=*)     SCENARIO="${1#*=}";      shift ;;
        --scenario)       SCENARIO="$2";           shift 2 ;;
        --type=*)         TYPE="${1#*=}";          shift ;;
        --type)           TYPE="$2";               shift 2 ;;
        --compare-libs=*) LIBRARIES="${1#*=}";     shift ;;
        --compare-libs)   LIBRARIES="$2";          shift 2 ;;
        --target=*)       TARGET="${1#*=}";        shift ;;
        --target)         TARGET="$2";             shift 2 ;;
        --judge-target=*) JUDGE_TARGET="${1#*=}";  shift ;;
        --judge-target)   JUDGE_TARGET="$2";       shift 2 ;;
        --out=*)          OUT="${1#*=}";           shift ;;
        --out)            OUT="$2";                shift 2 ;;
        --release)        BUILD_MODE="--release";  shift ;;
        --debug)          BUILD_MODE="";           shift ;;
        --list)           LIST_SCENARIOS=1;        shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^[^#]/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# ── Parse comma-separated library list ────────────────────────────────────────
LIB_LIST=()
if [[ -n "$LIBRARIES" ]]; then
    IFS=',' read -ra LIB_LIST <<< "$LIBRARIES"
fi

# ── Auto output folder ────────────────────────────────────────────────────────
if [[ -z "$OUT" ]]; then
    _ts=$(date +%s)
    _parts=("$TYPE")
    [[ -n "$SCENARIO" ]] && _parts+=("${SCENARIO//\//_}")
    _parts+=("$_ts")
    OUT="bench_results/data/$(IFS='_'; echo "${_parts[*]}")"
fi

# ── Cargo / binary setup ──────────────────────────────────────────────────────
FEATURES="$TARGET"
[[ -n "$JUDGE_TARGET" ]] && FEATURES="${FEATURES},judge_${JUDGE_TARGET}"
# Accuracy runs collect all scored pairs for PR-AUC; throughput runs skip this.
[[ "$TYPE" == "accuracy" ]] && FEATURES="${FEATURES},collect-pairs"

CARGO_ARGS=($BUILD_MODE -p zer-bench)
[[ "$FEATURES" != "cpu" ]] && CARGO_ARGS+=(--features "$FEATURES")

BENCH_BIN="cargo run ${CARGO_ARGS[*]} --"

# ── List mode ─────────────────────────────────────────────────────────────────
if [[ "$LIST_SCENARIOS" -eq 1 ]]; then
    $BENCH_BIN accuracy --list-scenarios
    exit 0
fi

# ── Core runners (all write to _OUTDIR set by the caller) ────────────────────

_zer_accuracy() {
    # $1: "judge" to include judge, anything else (or absent) for no judge
    local use_judge="${1:-}"
    local args=()
    [[ -n "$SCENARIO" ]] && args+=(--scenario "$SCENARIO")
    if [[ "$use_judge" == "judge" ]]; then
        args+=(--judge-target "${JUDGE_TARGET:-cpu}")
        echo ">>> zer accuracy  +judge (${JUDGE_TARGET:-cpu})"
    else
        echo ">>> zer accuracy"
    fi
    $BENCH_BIN accuracy "${args[@]}" --out "$_OUTDIR"
}

_lib_accuracy() {
    local lib="$1"
    local args=(--library "$lib")
    [[ -n "$SCENARIO" ]] && args+=(--scenario "$SCENARIO")
    echo ">>> $lib accuracy"
    $BENCH_BIN library "${args[@]}" --out "$_OUTDIR"
}

_zer_throughput() {
    local args=(--target "$TARGET")
    if [[ -n "$SCENARIO" ]]; then
        args+=(--scenario "$SCENARIO")
        # Resolve a single dataset file from the scenario path
        local ds="data/benchmarks/$SCENARIO/source.csv"
        [[ ! -f "$ds" ]] && ds="data/benchmarks/$SCENARIO/source_a.csv"
        [[ -f "$ds" ]] && args+=(--dataset "$ds")
    fi
    [[ -n "$JUDGE_TARGET" ]] && args+=(--judge-target "$JUDGE_TARGET")
    local _judge_info=""
    [[ -n "$JUDGE_TARGET" ]] && _judge_info=", judge: $JUDGE_TARGET"
    echo ">>> zer throughput  (target: $TARGET${_judge_info})"
    $BENCH_BIN throughput "${args[@]}" --out "$_OUTDIR"
}

_compare() {
    echo ">>> compare"
    $BENCH_BIN compare --results "$_OUTDIR"
}

_lib_throughput() {
    local lib="$1"
    local args=(--library "$lib" --mode throughput)
    [[ -n "$SCENARIO" ]] && args+=(--scenario "$SCENARIO")
    if [[ -n "$SCENARIO" ]]; then
        local ds="data/benchmarks/$SCENARIO/source.csv"
        [[ ! -f "$ds" ]] && ds="data/benchmarks/$SCENARIO/source_a.csv"
        [[ -f "$ds" ]] && args+=(--dataset "$ds")
    fi
    echo ">>> $lib throughput"
    $BENCH_BIN library "${args[@]}" --out "$_OUTDIR"
}

# ── Benchmark type runners ────────────────────────────────────────────────────

do_accuracy() {
    _OUTDIR="$1"
    mkdir -p "$_OUTDIR"

    if [[ -n "$JUDGE_TARGET" && ${#LIB_LIST[@]} -eq 0 ]]; then
        # Judge comparison: zer without judge then with judge
        echo "=== [1/2] zer (no judge) ==="
        _zer_accuracy
        echo "=== [2/2] zer (with judge: $JUDGE_TARGET) ==="
        _zer_accuracy judge
        _compare

    elif [[ ${#LIB_LIST[@]} -gt 0 ]]; then
        # zer + competitor libraries (optionally also with judge)
        local judge_slots=0
        [[ -n "$JUDGE_TARGET" ]] && judge_slots=1
        local total=$(( ${#LIB_LIST[@]} + 1 + judge_slots ))
        local i=1
        if [[ -n "$JUDGE_TARGET" ]]; then
            echo "=== [$i/$total] zer (no judge) ==="
            _zer_accuracy
            i=$((i + 1))
            echo "=== [$i/$total] zer (with judge: $JUDGE_TARGET) ==="
            _zer_accuracy judge
            i=$((i + 1))
        else
            echo "=== [$i/$total] zer ==="
            _zer_accuracy
            i=$((i + 1))
        fi
        for lib in "${LIB_LIST[@]}"; do
            echo "=== [$i/$total] $lib ==="
            _lib_accuracy "$lib"
            i=$((i + 1))
        done
        _compare

    else
        # zer only (no judge)
        _zer_accuracy
    fi
}

_zer_throughput_no_judge() {
    local args=(--target "$TARGET")
    if [[ -n "$SCENARIO" ]]; then
        args+=(--scenario "$SCENARIO")
        local ds="data/benchmarks/$SCENARIO/source.csv"
        [[ ! -f "$ds" ]] && ds="data/benchmarks/$SCENARIO/source_a.csv"
        [[ -f "$ds" ]] && args+=(--dataset "$ds")
    fi
    echo ">>> zer throughput  (target: $TARGET)"
    $BENCH_BIN throughput "${args[@]}" --out "$_OUTDIR"
}

do_throughput() {
    _OUTDIR="$1"
    mkdir -p "$_OUTDIR"

    if [[ -n "$JUDGE_TARGET" && ${#LIB_LIST[@]} -eq 0 ]]; then
        # Judge comparison: zer without judge then with judge
        echo "=== [1/2] zer (no judge) ==="
        _zer_throughput_no_judge
        echo "=== [2/2] zer (with judge: $JUDGE_TARGET) ==="
        _zer_throughput
        _compare

    elif [[ ${#LIB_LIST[@]} -gt 0 ]]; then
        local judge_slots=0
        [[ -n "$JUDGE_TARGET" ]] && judge_slots=1
        local total=$(( ${#LIB_LIST[@]} + 1 + judge_slots ))
        local i=1
        if [[ -n "$JUDGE_TARGET" ]]; then
            echo "=== [$i/$total] zer (no judge) ==="
            _zer_throughput_no_judge
            i=$((i + 1))
            echo "=== [$i/$total] zer (with judge: $JUDGE_TARGET) ==="
            _zer_throughput
            i=$((i + 1))
        else
            echo "=== [$i/$total] zer ==="
            _zer_throughput
            i=$((i + 1))
        fi
        for lib in "${LIB_LIST[@]}"; do
            echo "=== [$i/$total] $lib ==="
            _lib_throughput "$lib"
            i=$((i + 1))
        done
        _compare

    else
        _zer_throughput
    fi
}

# ── Full-size scenario list (used by --scenario=all) ─────────────────────────
ALL_FULL_SCENARIOS=(
    "brp/dedupe"
    "brp/link"
    "brp/link_and_dedupe"
    "brp_kvk/link"
    "brp_sis/link"
    "brp_hks/link"
    "brp_kvk_hks/link_and_dedupe"
    "kvk/dedupe"
)

# ── Dedupe-only scenario list (used by --scenario=all --type=throughput) ──────
ALL_THROUGHPUT_SCENARIOS=(
    "brp/dedupe"
    "kvk/dedupe"
)

# ── Throughput-only validation ────────────────────────────────────────────────
# Throughput benchmarks only measure the core dedupe pipeline loop.
# Link and link_and_dedupe scenarios are not supported.
if [[ "$TYPE" == "throughput" ]] && [[ -n "$SCENARIO" ]] && [[ "$SCENARIO" != "all" ]]; then
    _scenario_mode="${SCENARIO##*/}"
    if [[ "$_scenario_mode" != "dedupe" ]]; then
        echo "Error: --type=throughput only supports 'dedupe' scenarios." >&2
        echo "  Scenario '$SCENARIO' has mode '$_scenario_mode'." >&2
        echo "  Link and link_and_dedupe scenarios are not supported for throughput benchmarks." >&2
        exit 1
    fi
fi

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$TYPE" in
    accuracy)
        if [[ "$SCENARIO" == "all" ]]; then
            _base_out="$OUT"
            _total=${#ALL_FULL_SCENARIOS[@]}
            _i=1
            for _s in "${ALL_FULL_SCENARIOS[@]}"; do
                echo ""
                echo "════════════════════════════════════════════════════════════════"
                echo "  Scenario [$_i/$_total]: $_s"
                echo "════════════════════════════════════════════════════════════════"
                SCENARIO="$_s"
                _scenario_out="${_base_out}/${_s//\//_}"
                do_accuracy "$_scenario_out"
                _i=$((_i + 1))
            done
            echo ""
            echo "Done. All scenario results in: ${_base_out}/"
        else
            do_accuracy "$OUT"
            echo ""
            echo "Done. Results in: $OUT/"
        fi
        ;;
    throughput)
        if [[ "$SCENARIO" == "all" ]]; then
            _base_out="$OUT"
            _total=${#ALL_THROUGHPUT_SCENARIOS[@]}
            _i=1
            for _s in "${ALL_THROUGHPUT_SCENARIOS[@]}"; do
                echo ""
                echo "════════════════════════════════════════════════════════════════"
                echo "  Scenario [$_i/$_total]: $_s"
                echo "════════════════════════════════════════════════════════════════"
                SCENARIO="$_s"
                _scenario_out="${_base_out}/${_s//\//_}"
                do_throughput "$_scenario_out"
                _i=$((_i + 1))
            done
            echo ""
            echo "Done. All scenario results in: ${_base_out}/"
        else
            do_throughput "$OUT"
            echo ""
            echo "Done. Results in: $OUT/"
        fi
        ;;
    *)
        echo "Unknown --type: $TYPE  (valid: accuracy, throughput)" >&2
        exit 1
        ;;
esac
