#!/usr/bin/env bash
# Run all workspace examples for one or more compute backends.
#
# Usage:
#   ./scripts/run_all_examples.sh                        # default: run all examples on cpu, show output
#   ./scripts/run_all_examples.sh --target avx2          # run on AVX2 backend
#   ./scripts/run_all_examples.sh --target cuda          # run on CUDA backend
#   ./scripts/run_all_examples.sh --target vulkan        # run on Vulkan backend
#   ./scripts/run_all_examples.sh --target all           # cpu + avx2 + cuda + vulkan
#   ./scripts/run_all_examples.sh --target cpu --target avx2   # explicit subset
#   ./scripts/run_all_examples.sh --judge-target cuda    # set ORT judge backend for all examples
#   ./scripts/run_all_examples.sh --build-only           # check pass/fail without showing output
#   ./scripts/run_all_examples.sh --log-dir=../logs      # write output to per-target .log files
#   ./scripts/run_all_examples.sh --log-dir logs         # same, space form
#   ./scripts/run_all_examples.sh --list                 # list examples, don't run
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
# still shows progress headers and pass/fail status; program output goes only
# to the log file.  Relative paths are resolved from the directory where the
# script is invoked, not from the repository root.
#
# Exit code: 0 if every run passes, 1 if any fail.

set -euo pipefail

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
TIMEOUT=60
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
            [[ -z "${2-}" ]] && { echo "error: --timeout requires a value" >&2; exit 1; }
            TIMEOUT="$2"; shift 2 ;;
        --timeout=*)
            TIMEOUT="${1#--timeout=}"; shift ;;
        --log-dir)
            [[ -z "${2-}" ]] && { echo "error: --log-dir requires a value" >&2; exit 1; }
            LOG_DIR="$2"; shift 2 ;;
        --log-dir=*)
            LOG_DIR="${1#--log-dir=}"; shift ;;
        --list|-l)
            LIST_ONLY=true; shift ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
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
    # Relative paths are resolved from where the user invoked the script.
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

# ── Discover examples ─────────────────────────────────────────────────────────

mapfile -t EXAMPLES < <(
    find "$REPO_ROOT/crates" -name "*.rs" -path "*/examples/*" |
    while read -r f; do
        crate_dir="$(dirname "$(dirname "$f")")"
        crate="$(grep -m1 '^name' "$crate_dir/Cargo.toml" | sed 's/.*"\(.*\)".*/\1/')"
        example="$(basename "$f" .rs)"
        # _save must run before _load within the same crate so that load examples
        # find the file that save produces. Replace _save suffix with _0_save in
        # the sort key so it sorts before _load ('0' < 'l' in ASCII).
        sortkey="$(printf '%s' "$example" | sed 's/_save$/_0_save/')"
        printf '%s %s %s\n' "$crate" "$sortkey" "$example"
    done | sort -k1,1 -k2,2 | awk '{print $1, $3}'
)

# ── List mode ─────────────────────────────────────────────────────────────────

if $LIST_ONLY; then
    printf "${C_BOLD}Workspace examples:${C_RESET}\n\n"
    for entry in "${EXAMPLES[@]}"; do
        read -r crate example <<< "$entry"
        printf "  ${C_BOLD}%s${C_RESET}${C_DIM}::%s${C_RESET}\n" "$crate" "$example"
    done
    printf "\n${C_BOLD}Selected targets:${C_RESET}      %s\n" "${DEVICES[*]}"
    if [[ -n "$JUDGE_TARGET" ]]; then
        printf "${C_BOLD}Selected judge-target:${C_RESET}  %s\n" "$JUDGE_TARGET"
    fi
    exit 0
fi

# ── Build cargo command ───────────────────────────────────────────────────────

# Returns the cargo command for a given crate/example/features, with --features
# only if the crate actually declares that feature.  Judge features are added
# independently when the crate declares the corresponding judge_* feature.
make_cmd() {
    local crate="$1" example="$2" features="$3"
    local cmd=(cargo run -p "$crate" --example "$example")

    local crate_toml="$REPO_ROOT/crates/$crate/Cargo.toml"
    if [[ -n "$features" ]] && grep -q "^${features}\s*=" "$crate_toml" 2>/dev/null; then
        cmd+=(--features "$features")
    fi

    local judge_feature
    judge_feature="$(judge_feature_for "${JUDGE_TARGET:-cpu}")"
    if [[ -n "$judge_feature" ]] && grep -q "^${judge_feature}\s*=" "$crate_toml" 2>/dev/null; then
        cmd+=(--features "$judge_feature")
    fi

    if [[ -n "$JUDGE_TARGET" ]]; then
        cmd+=(-- --judge-target="$JUDGE_TARGET")
    fi

    printf '%s\n' "${cmd[@]}"
}

# ── Normal mode (default): stream output ──────────────────────────────────────

run_normal() {
    local GLOBAL_FAIL=0

    for device in "${DEVICES[@]}"; do
        local features
        features="$(features_for "$device")"
        local feat_display="${features:-none}"
        local logfile=""
        [[ -n "$LOG_DIR" ]] && logfile="$LOG_DIR/$device.log"

        # Console header
        local judge_display="${JUDGE_TARGET:-cpu}"
        printf "${C_BOLD}══ target: %s${C_RESET}${C_DIM}  (features: %s, judge: %s)${C_RESET}" \
               "$device" "$feat_display" "$judge_display"
        if [[ -n "$logfile" ]]; then
            printf "${C_DIM}  → %s${C_RESET}" "$logfile"
            printf '══ target: %s  (features: %s, judge: %s)\n\n' "$device" "$feat_display" "$judge_display" >"$logfile"
        fi
        printf "\n\n"

        for entry in "${EXAMPLES[@]}"; do
            read -r crate example <<< "$entry"

            mapfile -t cmd < <(make_cmd "$crate" "$example" "$features")

            printf "${C_BOLD}── %s::%s${C_RESET}\n" "$crate" "$example"
            [[ -n "$logfile" ]] && printf '── %s::%s\n' "$crate" "$example" >>"$logfile"

            set +e
            if [[ -n "$logfile" ]]; then
                maybe_timeout "$TIMEOUT" "${cmd[@]}" >>"$logfile" 2>&1
            else
                maybe_timeout "$TIMEOUT" "${cmd[@]}"
            fi
            local exit_code=$?
            set -e

            if [[ $exit_code -eq 124 ]]; then
                printf "\n${C_FAIL}timed out after %ds${C_RESET}\n" "$TIMEOUT"
                [[ -n "$logfile" ]] && printf '\ntimed out after %ds\n' "$TIMEOUT" >>"$logfile"
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

    for device in "${DEVICES[@]}"; do
        local features
        features="$(features_for "$device")"
        local feat_display="${features:-none}"
        local logfile=""
        [[ -n "$LOG_DIR" ]] && logfile="$LOG_DIR/$device.log"

        printf "${C_BOLD}┌─ target: %s${C_RESET}${C_DIM}  (features: %s)${C_RESET}" \
               "$device" "$feat_display"
        if [[ -n "$logfile" ]]; then
            printf "${C_DIM}  → %s${C_RESET}" "$logfile"
            printf '┌─ target: %s  (features: %s)\n' "$device" "$feat_display" >"$logfile"
        fi
        printf "\n"

        local pass=0 fail=0

        for entry in "${EXAMPLES[@]}"; do
            read -r crate example <<< "$entry"

            mapfile -t cmd < <(make_cmd "$crate" "$example" "$features")

            printf "${C_DIM}│  %-46s${C_RESET}" "$crate::$example"
            [[ -n "$logfile" ]] && printf '── %s::%s\n' "$crate" "$example" >>"$logfile"

            local tmpout
            tmpout="$(mktemp)"
            local start_s=$SECONDS
            set +e
            maybe_timeout "$TIMEOUT" "${cmd[@]}" >"$tmpout" 2>&1
            local exit_code=$?
            set -e
            local elapsed=$(( SECONDS - start_s ))

            # Always flush captured output to log if one is set.
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
                printf "  ${C_FAIL}✗  timed out after %ds${C_RESET}\n" "$TIMEOUT"
                [[ -n "$logfile" ]] && printf 'timed out after %ds\n\n' "$TIMEOUT" >>"$logfile"
            else
                (( fail++ )) || true
                GLOBAL_FAIL=1
                printf "  ${C_FAIL}✗  exit %d${C_RESET}\n" "$exit_code"
                # Without a log file, show the last few lines on the console.
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

if ! $LIST_ONLY && [[ ! -f "$REPO_ROOT/data/examples/brp_q1/brp_persons.csv" ]]; then
    printf "${C_FAIL}error: example datasets not found (data/examples/ is missing or empty).${C_RESET}\n" >&2
    printf "Run first:  ./scripts/generate_data.sh --examples\n" >&2
    exit 1
fi

# ── Dispatch ──────────────────────────────────────────────────────────────────

if $BUILD_ONLY; then
    run_build_only
else
    run_normal
fi
