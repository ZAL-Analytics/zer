#!/usr/bin/env bash
# Run a single demo binary from demos/ for one or more compute backends.
#
# Usage:
#   ./scripts/run_demo.sh --program=hello_backend
#   ./scripts/run_demo.sh --program=multi_source_linkage --target=cuda
#   ./scripts/run_demo.sh --program=multi_source_linkage --target=all
#   ./scripts/run_demo.sh --program=multi_source_linkage --target=avx2 --log-dir=logs/demos
#   ./scripts/run_demo.sh --program=person_deduplication --target=all --log-dir=/tmp/zer-logs
#   ./scripts/run_demo.sh --program=scoring_walkthrough --target=cpu --judge-target=cuda
#
# Flags:
#   --program=<name>          Name of the demo directory under demos/ (required)
#   --target=<t>              Backend to build for (default: cpu)
#                             Valid values: cpu  avx2  cuda  vulkan  all
#   --judge-target=<t>        ORT execution provider for zer-judge (default: absent → cpu)
#                             Valid values: cpu  cuda  tensorrt  rocm  directml  openvino
#   --log-dir=<path>          Write output to <path>/<program>_<target>.log
#                             Relative paths resolve from the invocation directory.
#                             Without this flag, output streams to stdout.
#   --timeout=<seconds>       Per-run timeout in seconds (default: 120; 0 = no timeout)
#   --list                    List available example programs and exit
#
# Backend notes:
#   cpu   , no extra feature flags (scalar fallback)
#   avx2  , --features avx2   (requires x86-64 with AVX2)
#   cuda  , --features cuda   (requires CUDA toolkit + nvcc)
#   vulkan, --features vulkan (requires Vulkan SDK + glslc)
#   all   , runs all four in sequence
#
# If a program's Cargo.toml does not declare a requested feature, that target
# is skipped with a notice rather than failing.
#
# Exit code: 0 if every run passes, 1 if any fail or are skipped.

set -euo pipefail
trap 'kill 0' INT TERM

INVOCATION_DIR="$(pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$REPO_ROOT/demos"
cd "$REPO_ROOT"

# ── Colours ───────────────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
    C_PASS="\033[0;32m"
    C_FAIL="\033[0;31m"
    C_SKIP="\033[0;33m"
    C_BOLD="\033[1m"
    C_DIM="\033[2m"
    C_RESET="\033[0m"
else
    C_PASS="" C_FAIL="" C_SKIP="" C_BOLD="" C_DIM="" C_RESET=""
fi

# ── Argument parsing ──────────────────────────────────────────────────────────

PROGRAM=""
RAW_TARGETS=()
JUDGE_TARGET=""
LOG_DIR=""
TIMEOUT=120
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --program)
            [[ -z "${2-}" ]] && { echo "error: --program requires a value" >&2; exit 1; }
            PROGRAM="$2"; shift 2 ;;
        --program=*)
            PROGRAM="${1#--program=}"; shift ;;
        --target)
            [[ -z "${2-}" ]] && { echo "error: --target requires a value" >&2; exit 1; }
            RAW_TARGETS+=("$2"); shift 2 ;;
        --target=*)
            RAW_TARGETS+=("${1#--target=}"); shift ;;
        --judge-target)
            [[ -z "${2-}" ]] && { echo "error: --judge-target requires a value" >&2; exit 1; }
            JUDGE_TARGET="$2"; shift 2 ;;
        --judge-target=*)
            JUDGE_TARGET="${1#--judge-target=}"; shift ;;
        --log-dir)
            [[ -z "${2-}" ]] && { echo "error: --log-dir requires a value" >&2; exit 1; }
            LOG_DIR="$2"; shift 2 ;;
        --log-dir=*)
            LOG_DIR="${1#--log-dir=}"; shift ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --timeout=*)
            TIMEOUT="${1#--timeout=}"; shift ;;
        --list|-l)
            LIST_ONLY=true; shift ;;
        --help|-h)
            sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

# ── List mode ─────────────────────────────────────────────────────────────────

if $LIST_ONLY; then
    printf "${C_BOLD}Available demos:${C_RESET}\n\n"
    for dir in "$EXAMPLES_DIR"/*/; do
        toml="$dir/Cargo.toml"
        # Skip library-only crates (no [[bin]], e.g. common).
        [[ -f "$toml" ]] && grep -q '^\[\[bin\]\]' "$toml" || continue
        name="$(basename "$dir")"
        features="$(grep -E '^(cpu|avx2|cuda|vulkan)\s*=' "$toml" 2>/dev/null \
                    | sed 's/\s*=.*//' | tr '\n' ' ' | sed 's/ $//' || true)"
        if [[ -n "$features" ]]; then
            printf "  ${C_BOLD}%-30s${C_RESET}${C_DIM}features: %s${C_RESET}\n" "$name" "$features"
        else
            printf "  ${C_BOLD}%s${C_RESET}\n" "$name"
        fi
    done
    printf "\n"
    exit 0
fi

# ── Validate --program ────────────────────────────────────────────────────────

if [[ -z "$PROGRAM" ]]; then
    echo "error: --program is required (use --list to see available programs)" >&2
    exit 1
fi

PROGRAM_DIR="$EXAMPLES_DIR/$PROGRAM"
PROGRAM_TOML="$PROGRAM_DIR/Cargo.toml"

if [[ ! -d "$PROGRAM_DIR" ]]; then
    echo "error: example '$PROGRAM' not found in $EXAMPLES_DIR" >&2
    exit 1
fi
if [[ ! -f "$PROGRAM_TOML" ]]; then
    echo "error: $PROGRAM_TOML not found" >&2
    exit 1
fi

# ── Resolve log directory ─────────────────────────────────────────────────────

if [[ -n "$LOG_DIR" ]]; then
    if [[ "$LOG_DIR" != /* ]]; then
        LOG_DIR="$INVOCATION_DIR/$LOG_DIR"
    fi
    mkdir -p "$LOG_DIR"
fi

# ── Target expansion ──────────────────────────────────────────────────────────

ALL_DEVICES=(cpu avx2 cuda vulkan)

expand_targets() {
    local seen=()
    for t in "${RAW_TARGETS[@]}"; do
        if [[ "$t" == "all" ]]; then
            for d in "${ALL_DEVICES[@]}"; do
                [[ " ${seen[*]-} " != *" $d "* ]] && seen+=("$d")
            done
        else
            case "$t" in
                cpu|avx2|cuda|vulkan) ;;
                *) echo "error: unknown target '$t' (valid: cpu avx2 cuda vulkan all)" >&2; exit 1 ;;
            esac
            [[ " ${seen[*]-} " != *" $t "* ]] && seen+=("$t")
        fi
    done
    printf '%s\n' "${seen[@]}"
}

[[ ${#RAW_TARGETS[@]} -eq 0 ]] && RAW_TARGETS=(cpu)
mapfile -t DEVICES < <(expand_targets)

features_for() {
    case "$1" in
        cpu)    echo "" ;;
        avx2)   echo "avx2" ;;
        cuda)   echo "cuda" ;;
        vulkan) echo "vulkan" ;;
    esac
}

judge_feature_for() {
    case "$1" in
        cpu)      echo "" ;;
        cuda)     echo "judge_cuda" ;;
        tensorrt) echo "judge_tensorrt" ;;
        rocm)     echo "judge_rocm" ;;
        directml) echo "judge_directml" ;;
        openvino) echo "judge_openvino" ;;
    esac
}

# Runs a command with a timeout, skipping the timeout wrapper when $1 is 0.
maybe_timeout() {
    local t="$1"; shift
    if [[ "$t" -eq 0 ]]; then
        "$@"
    else
        timeout "$t" "$@"
    fi
}

# Returns 0 if the package declares the given feature, 1 otherwise.
# cpu with no feature flag always passes (default build target).
package_has_feature() {
    local feature="$1"
    [[ -z "$feature" ]] && return 0
    grep -qE "^${feature}\s*=" "$PROGRAM_TOML" 2>/dev/null
}

# ── Run ───────────────────────────────────────────────────────────────────────

GLOBAL_FAIL=0

printf "${C_BOLD}program:${C_RESET}       %s\n" "$PROGRAM"
printf "${C_BOLD}targets:${C_RESET}       %s\n" "${DEVICES[*]}"
if [[ -n "$JUDGE_TARGET" ]]; then
    printf "${C_BOLD}judge-target:${C_RESET}  %s\n" "$JUDGE_TARGET"
fi
printf "\n"

for device in "${DEVICES[@]}"; do
    features="$(features_for "$device")"
    feat_display="${features:-none}"

    logfile=""
    if [[ -n "$LOG_DIR" ]]; then
        logfile="$LOG_DIR/${PROGRAM}_${device}.log"
    fi

    # Header
    printf "${C_BOLD}══ target: %s${C_RESET}${C_DIM}  (features: %s)${C_RESET}" \
           "$device" "$feat_display"
    if [[ -n "$logfile" ]]; then
        printf "${C_DIM}  → %s${C_RESET}" "$logfile"
        printf '══ target: %s  (features: %s)\n\n' "$device" "$feat_display" >"$logfile"
    fi
    printf "\n\n"

    # Skip if the package doesn't declare the required feature.
    if ! package_has_feature "$features"; then
        printf "${C_SKIP}skipped${C_RESET}${C_DIM}: '%s' does not declare feature '%s'${C_RESET}\n\n" \
               "$PROGRAM" "$features"
        [[ -n "$logfile" ]] && printf 'skipped: %s does not declare feature %s\n\n' \
               "$PROGRAM" "$features" >>"$logfile"
        GLOBAL_FAIL=1
        continue
    fi

    PACKAGE="${PROGRAM//_/-}"
    cmd=(cargo run -p "$PACKAGE")
    [[ -n "$features" ]] && cmd+=(--features "$features")
    judge_feature="$(judge_feature_for "${JUDGE_TARGET:-cpu}")"
    if [[ -n "$judge_feature" ]] && package_has_feature "$judge_feature"; then
        cmd+=(--features "$judge_feature")
    fi
    cmd+=(-- --target="$device")
    [[ -n "$JUDGE_TARGET" ]] && cmd+=(--judge-target="$JUDGE_TARGET")

    set +e
    if [[ -n "$logfile" ]]; then
        maybe_timeout "$TIMEOUT" "${cmd[@]}" >>"$logfile" 2>&1
    else
        maybe_timeout "$TIMEOUT" "${cmd[@]}"
    fi
    exit_code=$?
    set -e

    if [[ $exit_code -eq 124 ]]; then
        printf "${C_FAIL}timed out after %ds${C_RESET}\n" "$TIMEOUT"
        [[ -n "$logfile" ]] && printf 'timed out after %ds\n' "$TIMEOUT" >>"$logfile"
        GLOBAL_FAIL=1
    elif [[ $exit_code -ne 0 ]]; then
        printf "${C_FAIL}exit %d${C_RESET}\n" "$exit_code"
        [[ -n "$logfile" ]] && printf 'exit %d\n' "$exit_code" >>"$logfile"
        GLOBAL_FAIL=1
    else
        printf "${C_PASS}ok${C_RESET}\n"
        [[ -n "$logfile" ]] && printf 'ok\n' >>"$logfile"
    fi

    printf "\n"
    [[ -n "$logfile" ]] && printf '\n' >>"$logfile"
done

exit $GLOBAL_FAIL
