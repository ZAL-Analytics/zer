#!/usr/bin/env bash
# upload_library.sh, publish all zer crates to crates.io in dependency order.
#
# Prerequisites:
#   - ZER_CRATES_IO_TOKEN env var set (add to ~/.bashrc: export ZER_CRATES_IO_TOKEN=...)
#   - cargo installed and on PATH
#   - All changes committed (cargo publish refuses dirty working trees)
#
# Usage:
#   bash scripts/upload_library.sh             # full publish
#   DRY_RUN=1 bash scripts/upload_library.sh   # dry-run only (no upload)
#   bash scripts/upload_library.sh --wait 120  # override inter-crate wait (seconds)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"
UPLOAD_LOG="${UPLOAD_LOG:-/tmp/cargo-publish-uploaded.txt}"
WAIT_TIME="${WAIT_TIME:-600}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--wait) WAIT_TIME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Create the log file if it doesn't exist yet
touch "$UPLOAD_LOG"

# ── Colour output ──────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[zer]${NC} $*"; }
warn()  { echo -e "${YELLOW}[zer]${NC} $*"; }
error() { echo -e "${RED}[zer]${NC} $*" >&2; }

# ── Prerequisites ──────────────────────────────────────────────────────────────
if [[ -z "${ZER_CRATES_IO_TOKEN:-}" ]]; then
    error "ZER_CRATES_IO_TOKEN is not set."
    error "Add this to your ~/.bashrc:"
    error "  export ZER_CRATES_IO_TOKEN=<your crates.io API token>"
    error "Then run: source ~/.bashrc"
    exit 1
fi

if ! command -v cargo &>/dev/null; then
    error "cargo not found on PATH. Install Rust: https://rustup.rs"
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    warn "DRY_RUN=1, no crates will be uploaded."
fi

info "Resume log: $UPLOAD_LOG"

# ── Helpers ────────────────────────────────────────────────────────────────────

# verify <crate-path>,cargo package check (no registry resolution)
verify() {
    local crate_path="$1"
    local crate_name
    crate_name="$(basename "$crate_path")"
    cargo package \
        --no-verify \
        --manifest-path "$crate_path/Cargo.toml" \
        > /dev/null 2>&1 \
        && info "  $crate_name: package OK" \
        || { error "  $crate_name: package FAILED"; exit 1; }
}

# publish <relative-crate-path>
publish() {
    local rel_path="$1"
    local crate_path="$REPO_ROOT/$rel_path"
    local crate_name
    crate_name="$(basename "$crate_path")"

    echo ""

    # Skip if already recorded in the upload log
    if grep -qx "$crate_name" "$UPLOAD_LOG" 2>/dev/null; then
        info "  $crate_name: already published, skipping."
        return
    fi

    # Verify just before publishing so deps are already live on crates.io
    verify "$crate_path"
    info "Publishing $crate_name ..."

    local args=(
        --token "$ZER_CRATES_IO_TOKEN"
        --manifest-path "$crate_path/Cargo.toml"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        cargo publish --dry-run "${args[@]}"
        info "$crate_name: dry-run OK"
        return
    fi

    cargo publish "${args[@]}"
    echo "$crate_name" >> "$UPLOAD_LOG"
    info "$crate_name published successfully."

    # crates.io needs time to index the new version before dependent crates
    # can resolve it. 600 s is enough in practice; increase if you see
    # "no matching version" errors on the next crate. Also prevent rate limit errors 
    # by spacing out the uploads.
    warn "Waiting ${WAIT_TIME}s for crates.io to index $crate_name ..."
    sleep "$WAIT_TIME"
}

# ── Sanity check: ensure the working tree is clean ────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
    if ! git -C "$REPO_ROOT" diff --quiet HEAD; then
        error "Working tree has uncommitted changes. Commit or stash before publishing."
        exit 1
    fi
fi

# ── Pre-flight: verify only the leaf crates (no internal deps) ────────────────
# Dependent crates are verified in publish() once their deps are live on crates.io
info "Verifying leaf crates with 'cargo package' ..."
verify "$REPO_ROOT/crates/zer-core"
verify "$REPO_ROOT/crates/zer-prof"

# ── Publish in topological order (leaf crates first) ──────────────────────────
#
# Tier 0, no internal deps
publish "crates/zer-core"
publish "crates/zer-prof"

# Tier 1, depends on zer-core only
publish "crates/zer-compare"
publish "crates/zer-blocking"
publish "crates/zer-schema"
publish "crates/zer-cluster"
publish "crates/zer-judge"
publish "crates/zer-adapters"

# Tier 2, depends on tier-0 + tier-1
publish "crates/zer-compute"   # zer-core, zer-compare, zer-prof
publish "crates/zer-pipeline"  # zer-core, zer-blocking, zer-compare, zer-schema, zer-cluster

# Tier 3, facade, depends on all of the above
publish "crates/zer-lib"

echo ""
info "All zer crates published!"
info "View on crates.io: https://crates.io/search?q=zer-"
