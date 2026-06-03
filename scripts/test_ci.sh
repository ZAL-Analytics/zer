#!/usr/bin/env bash
# Mirror all GitHub CI checks locally.
#
# Usage:
#   ./scripts/test_ci.sh          # check mode: mirrors CI exactly, exits 1 on any failure
#   ./scripts/test_ci.sh --fix    # fix mode: auto-fix formatting and clippy, then verify all checks pass
#
# Prerequisites for audit/deny:
#   cargo install cargo-audit
#   cargo install cargo-deny
#
# Excluded crates (match CI):
#   zer-judge  — requires ORT binary download + ONNX model files not in the repo
#   zer-bench  — benchmarking CLI, not unit-testable
#   demos/*    — require runtime data files

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FIX=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)    FIX=true; shift ;;
        --help|-h)
            sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

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

# ── Shared exclude flags (must match ci.yml) ──────────────────────────────────

EXCLUDE=(
    --exclude zer-judge
    --exclude zer-bench
    --exclude blocking-explorer
    --exclude demo-common
    --exclude cross-source-linkage
    --exclude custom-components
    --exclude hello-backend
    --exclude multi-source-linkage
    --exclude person-deduplication
    --exclude scoring-walkthrough
)

# ── Step runner ───────────────────────────────────────────────────────────────

PASS=0
FAIL=0

step() {
    local name="$1"; shift
    printf "${C_BOLD}── %s${C_RESET}\n" "$name"
    if "$@"; then
        printf "${C_PASS}   ✓ passed${C_RESET}\n\n"
        (( PASS++ )) || true
    else
        printf "${C_FAIL}   ✗ failed${C_RESET}\n\n"
        (( FAIL++ )) || true
    fi
}

step_skip() {
    local name="$1" reason="$2"
    printf "${C_DIM}── %-20s (skipped: %s)${C_RESET}\n\n" "$name" "$reason"
}

# ── Checks ────────────────────────────────────────────────────────────────────

if $FIX; then
    printf "${C_BOLD}Applying fixes…${C_RESET}\n\n"

    step "fmt (fix)"    cargo fmt --all
    # --allow-dirty: fmt above may have left unstaged changes; clippy --fix needs this
    step "clippy (fix)" cargo clippy --workspace "${EXCLUDE[@]}" --features cpu --release --fix --allow-dirty --allow-staged

    printf "${C_BOLD}Verifying all checks pass after fixes…${C_RESET}\n\n"
fi

step "fmt"    cargo fmt --all --check
step "clippy" cargo clippy --workspace "${EXCLUDE[@]}" --features cpu --release -- -D warnings

# ── Test dataset setup ────────────────────────────────────────────────────────

_ensure_datasets() {
    local tests_data="$REPO_ROOT/data/tests"

    if [[ -d "$tests_data" ]] && [[ -n "$(ls -A "$tests_data" 2>/dev/null)" ]]; then
        echo "already present in data/tests/ — skipping"
        return 0
    fi

    if command -v huggingface-cli &>/dev/null; then
        echo "downloading from HuggingFace…"
        if bash "$REPO_ROOT/scripts/download_dataset.sh"; then
            return 0
        fi
        echo "download failed, falling back to local generation…"
    fi

    echo "generating locally via scripts/generate_data.sh --tests…"
    bash "$REPO_ROOT/scripts/generate_data.sh" --tests
}

step "datasets" _ensure_datasets
[[ $FAIL -eq 0 ]] || { printf "${C_FAIL}  Aborting: test datasets unavailable.${C_RESET}\n\n"; exit 1; }

step "test"   cargo test   --workspace "${EXCLUDE[@]}" --features cpu --release

if command -v cargo-audit &>/dev/null; then
    step "audit" cargo audit \
        --ignore RUSTSEC-2025-0141 \
        --ignore RUSTSEC-2024-0436
else
    step_skip "audit" "cargo-audit not installed — run: cargo install cargo-audit"
fi

if command -v cargo-deny &>/dev/null; then
    step "deny" cargo deny check
else
    step_skip "deny" "cargo-deny not installed — run: cargo install cargo-deny"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

printf "${C_BOLD}══ %d passed" "$PASS"
[[ $FAIL -gt 0 ]] && printf "  ${C_FAIL}%d failed${C_RESET}${C_BOLD}" "$FAIL"
printf "${C_RESET}\n"

[[ $FAIL -eq 0 ]]
