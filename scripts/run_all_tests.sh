#!/usr/bin/env bash
# Run all workspace tests for one or more compute backends.
#
# Usage:
#   ./scripts/run_all_tests.sh                        # default: run all tests on cpu, show output
#   ./scripts/run_all_tests.sh --target avx2          # run with AVX2 features
#   ./scripts/run_all_tests.sh --target cuda          # run with CUDA features
#   ./scripts/run_all_tests.sh --target vulkan        # run with Vulkan features
#   ./scripts/run_all_tests.sh --target all           # cpu + avx2 + cuda + vulkan
#   ./scripts/run_all_tests.sh --target cpu --target avx2   # explicit subset
#   ./scripts/run_all_tests.sh --judge-target cuda    # set ORT judge backend (exported as JUDGE_TARGET env var)
#   ./scripts/run_all_tests.sh --build-only           # check pass/fail without showing output
#   ./scripts/run_all_tests.sh --log-dir=../logs      # write output to per-target .log files
#   ./scripts/run_all_tests.sh --log-dir logs         # same, space form
#   ./scripts/run_all_tests.sh --list                 # list crates, don't run
#
# Valid targets: cpu  avx2  cuda  vulkan  all
# Valid judge-targets: cpu  cuda  tensorrt  rocm  directml  openvino  (no 'all', single value only)
#   cpu   , no feature flags (scalar CPU fallback)
#   avx2  , --features avx2
#   cuda  , --features cuda    (requires CUDA toolkit + nvcc at build time)
#   vulkan, --features vulkan  (requires Vulkan SDK + glslc at build time)
#   all   , runs all four in sequence
#
# With --log-dir, one <target>.log file is written per target.  The console
# still shows progress headers and pass/fail status; test output goes only
# to the log file.  Relative paths are resolved from the directory where the
# script is invoked, not from the repository root.
#
# Exit code: 0 if every run passes, 1 if any fail.

set -euo pipefail

# Kill all child processes when the user presses Ctrl+C or the script is
# terminated, so cargo/timeout subprocesses don't linger in the background.
trap 'kill 0' INT TERM

# Capture invocation dir before we cd to repo root, so relative --log-dir works.
INVOCATION_DIR="$(pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Colours ───────────────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
    C_PASS="\033[0;32m"
    C_FAIL="\033[0;31m"
    C_BOLD="\033[1m"
    C_DIM="\033[2m"
    C_RESET="\033[0m"
else
    C_PASS="" C_FAIL="" C_BOLD="" C_DIM="" C_RESET=""
fi

# ── Argument parsing ──────────────────────────────────────────────────────────

RAW_TARGETS=()
JUDGE_TARGET=""
LIST_ONLY=false
BUILD_ONLY=false
TIMEOUT=120
LOG_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --build-only)
            BUILD_ONLY=true; shift ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --log-dir)
            [[ -z "${2-}" ]] && { echo "error: --log-dir requires a value" >&2; exit 1; }
            LOG_DIR="$2"; shift 2 ;;
        --log-dir=*)
            LOG_DIR="${1#--log-dir=}"; shift ;;
        --list|-l)
            LIST_ONLY=true; shift ;;
        --help|-h)
            sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

# ── Validate --judge-target ───────────────────────────────────────────────────

if [[ -n "$JUDGE_TARGET" ]]; then
    case "$JUDGE_TARGET" in
        cpu|cuda|tensorrt|rocm|directml|openvino) ;;
        all) echo "error: --judge-target does not support 'all'; pick one: cpu cuda tensorrt rocm directml openvino" >&2; exit 1 ;;
        *)   echo "error: unknown --judge-target='$JUDGE_TARGET' (valid: cpu cuda tensorrt rocm directml openvino)" >&2; exit 1 ;;
    esac
fi

# ── Resolve and create log directory ─────────────────────────────────────────

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
                [[ " ${seen[*]} " != *" $d "* ]] && seen+=("$d")
            done
        else
            case "$t" in
                cpu|avx2|cuda|vulkan) ;;
                *) echo "error: unknown target '$t' (valid: cpu avx2 cuda vulkan all)" >&2; exit 1 ;;
            esac
            [[ " ${seen[*]} " != *" $t "* ]] && seen+=("$t")
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

# CUDA/Vulkan initialise hardware and compile shaders at test time, allow 300 s
# per crate; keep the default 120 s for cpu/avx2.  When TIMEOUT=0, no timeout.
timeout_for() {
    [[ "$TIMEOUT" -eq 0 ]] && { echo 0; return; }
    case "$1" in
        cuda|vulkan) echo 300 ;;
        *)           echo "${TIMEOUT}" ;;
    esac
}

# ── Discover crates ───────────────────────────────────────────────────────────

mapfile -t CRATES < <(
    find "$REPO_ROOT/crates" -maxdepth 2 -name "Cargo.toml" | sort |
    while read -r f; do
        grep -m1 '^name' "$f" | sed 's/.*"\(.*\)".*/\1/'
    done
)

# ── List mode ─────────────────────────────────────────────────────────────────

if $LIST_ONLY; then
    printf "${C_BOLD}Workspace crates:${C_RESET}\n\n"
    for crate in "${CRATES[@]}"; do
        printf "  ${C_BOLD}%s${C_RESET}\n" "$crate"
    done
    printf "\n${C_BOLD}Selected targets:${C_RESET}      %s\n" "${DEVICES[*]}"
    if [[ -n "$JUDGE_TARGET" ]]; then
        printf "${C_BOLD}Selected judge-target:${C_RESET}  %s\n" "$JUDGE_TARGET"
    fi
    exit 0
fi

# ── Build cargo command ───────────────────────────────────────────────────────

# Pre-compile heavy crates for the given feature set without running tests.
# Covers both zer-compute (shaders + GPU code) and zer (the facade that
# re-exports the feature), so neither cold-compiles inside the per-crate timeout.
warmup_compile() {
    local features="$1" logfile="$2"
    [[ -z "$features" ]] && return 0

    local out
    out="$(mktemp)"

    for crate in zer-compute zer; do
        local crate_toml="$REPO_ROOT/crates/$crate/Cargo.toml"
        grep -q "^${features}\s*=" "$crate_toml" 2>/dev/null || continue

        printf "${C_DIM}  (pre-compiling %s/%s…)${C_RESET}" "$crate" "$features"
        if cargo test -p "$crate" --features "$features" --no-run >>"$out" 2>&1; then
            printf "${C_DIM} done${C_RESET}\n"
        else
            printf "${C_DIM} done (warnings)${C_RESET}\n"
        fi
    done

    [[ -n "$logfile" && -s "$out" ]] && cat "$out" >>"$logfile"
    rm -f "$out"
}

# Returns the cargo test command for a given crate/features, with --features
# only if the crate actually declares that feature.  Judge features are added
# independently when the crate declares the corresponding judge_* feature.
make_cmd() {
    local crate="$1" features="$2"
    local cmd=(cargo test -p "$crate")

    local crate_toml="$REPO_ROOT/crates/$crate/Cargo.toml"
    if [[ -n "$features" ]] && grep -q "^${features}\s*=" "$crate_toml" 2>/dev/null; then
        cmd+=(--features "$features")
    fi

    local judge_feature
    judge_feature="$(judge_feature_for "${JUDGE_TARGET:-cpu}")"
    if [[ -n "$judge_feature" ]] && grep -q "^${judge_feature}\s*=" "$crate_toml" 2>/dev/null; then
        cmd+=(--features "$judge_feature")
    fi

    printf '%s\n' "${cmd[@]}"
}

# ── Normal mode (default): stream output ──────────────────────────────────────

run_normal() {
    local GLOBAL_FAIL=0

    [[ -n "$JUDGE_TARGET" ]] && export JUDGE_TARGET

    for device in "${DEVICES[@]}"; do
        local features
        features="$(features_for "$device")"
        local feat_display="${features:-none}"
        local judge_display="${JUDGE_TARGET:-cpu}"
        local logfile=""
        [[ -n "$LOG_DIR" ]] && logfile="$LOG_DIR/$device.log"

        printf "${C_BOLD}══ target: %s${C_RESET}${C_DIM}  (features: %s, judge: %s)${C_RESET}" \
               "$device" "$feat_display" "$judge_display"
        if [[ -n "$logfile" ]]; then
            printf "${C_DIM}  → %s${C_RESET}" "$logfile"
            printf '══ target: %s  (features: %s, judge: %s)\n\n' "$device" "$feat_display" "$judge_display" >"$logfile"
        fi
        printf "\n"
        warmup_compile "$features" "$logfile"
        printf "\n"

        local crate_timeout
        crate_timeout="$(timeout_for "$device")"

        for crate in "${CRATES[@]}"; do
            mapfile -t cmd < <(make_cmd "$crate" "$features")

            printf "${C_BOLD}── %s${C_RESET}\n" "$crate"
            [[ -n "$logfile" ]] && printf '── %s\n' "$crate" >>"$logfile"

            set +e
            if [[ -n "$logfile" ]]; then
                maybe_timeout "$crate_timeout" "${cmd[@]}" >>"$logfile" 2>&1
            else
                maybe_timeout "$crate_timeout" "${cmd[@]}"
            fi
            local exit_code=$?
            set -e

            if [[ $exit_code -eq 124 ]]; then
                printf "\n${C_FAIL}timed out after %ds${C_RESET}\n" "$crate_timeout"
                [[ -n "$logfile" ]] && printf '\ntimed out after %ds\n' "$crate_timeout" >>"$logfile"
                GLOBAL_FAIL=1
            elif [[ $exit_code -ne 0 ]]; then
                printf "\n${C_FAIL}exit %d${C_RESET}\n" "$exit_code"
                [[ -n "$logfile" ]] && printf '\nexit %d\n' "$exit_code" >>"$logfile"
                GLOBAL_FAIL=1
            fi

            printf "\n"
            [[ -n "$logfile" ]] && printf '\n' >>"$logfile"
        done
    done

    return $GLOBAL_FAIL
}

# ── Build-only mode: quiet pass/fail summary ──────────────────────────────────

run_build_only() {
    local GLOBAL_FAIL=0

    [[ -n "$JUDGE_TARGET" ]] && export JUDGE_TARGET

    for device in "${DEVICES[@]}"; do
        local features
        features="$(features_for "$device")"
        local feat_display="${features:-none}"
        local judge_display="${JUDGE_TARGET:-cpu}"
        local logfile=""
        [[ -n "$LOG_DIR" ]] && logfile="$LOG_DIR/$device.log"

        printf "${C_BOLD}┌─ target: %s${C_RESET}${C_DIM}  (features: %s, judge: %s)${C_RESET}" \
               "$device" "$feat_display" "$judge_display"
        if [[ -n "$logfile" ]]; then
            printf "${C_DIM}  → %s${C_RESET}" "$logfile"
            printf '┌─ target: %s  (features: %s, judge: %s)\n' "$device" "$feat_display" "$judge_display" >"$logfile"
        fi
        printf "\n"
        warmup_compile "$features" "$logfile"

        local pass=0 fail=0
        local crate_timeout
        crate_timeout="$(timeout_for "$device")"

        for crate in "${CRATES[@]}"; do
            mapfile -t cmd < <(make_cmd "$crate" "$features")

            printf "${C_DIM}│  %-46s${C_RESET}" "$crate"
            [[ -n "$logfile" ]] && printf '── %s\n' "$crate" >>"$logfile"

            local tmpout
            tmpout="$(mktemp)"
            local start_s=$SECONDS
            set +e
            maybe_timeout "$crate_timeout" "${cmd[@]}" >"$tmpout" 2>&1
            local exit_code=$?
            set -e
            local elapsed=$(( SECONDS - start_s ))

            if [[ -n "$logfile" ]]; then
                [[ -s "$tmpout" ]] && cat "$tmpout" >>"$logfile"
                printf '\n' >>"$logfile"
            fi

            if [[ $exit_code -eq 0 ]]; then
                (( pass++ )) || true
                printf "  ${C_PASS}✓${C_RESET}  ${C_DIM}%ds${C_RESET}\n" "$elapsed"
            elif [[ $exit_code -eq 124 ]]; then
                (( fail++ )) || true
                GLOBAL_FAIL=1
                printf "  ${C_FAIL}✗  timed out after %ds${C_RESET}\n" "$crate_timeout"
                [[ -n "$logfile" ]] && printf 'timed out after %ds\n\n' "$crate_timeout" >>"$logfile"
            else
                (( fail++ )) || true
                GLOBAL_FAIL=1
                printf "  ${C_FAIL}✗  exit %d${C_RESET}\n" "$exit_code"
                if [[ -z "$logfile" && -s "$tmpout" ]]; then
                    printf "${C_DIM}"
                    tail -5 "$tmpout" | sed 's/^/│     /'
                    printf "${C_RESET}"
                fi
            fi

            rm -f "$tmpout"
        done

        local total=$(( pass + fail ))
        printf "${C_DIM}│${C_RESET}\n"
        printf "${C_BOLD}└─ %s:${C_RESET}  ${C_PASS}%d passed${C_RESET}" "$device" "$pass"
        [[ $fail -gt 0 ]] && printf "  ${C_FAIL}%d failed${C_RESET}" "$fail"
        printf "  ${C_DIM}(%d total)${C_RESET}\n\n" "$total"
    done

    return $GLOBAL_FAIL
}

# ── Data presence check ───────────────────────────────────────────────────────

if ! $LIST_ONLY && [[ ! -f "$REPO_ROOT/data/tests/brp/brp_persons.csv" ]]; then
    printf "${C_FAIL}error: test datasets not found (data/tests/ is missing or empty).${C_RESET}\n" >&2
    printf "Run first:  ./scripts/generate_data.sh --tests\n" >&2
    exit 1
fi

# ── Dispatch ──────────────────────────────────────────────────────────────────

if $BUILD_ONLY; then
    run_build_only
else
    run_normal
fi
