#!/usr/bin/env bash
# upload_library.sh, publish all zer crates to crates.io in dependency order.
#
# Prerequisites:
#   - ZER_CRATES_IO_TOKEN env var set (add to ~/.bashrc: export ZER_CRATES_IO_TOKEN=...)
#   - cargo installed and on PATH
#   - All changes committed (cargo publish refuses dirty working trees)
#
# Usage:
#   bash scripts/upload_library.sh           # full publish
#   DRY_RUN=1 bash scripts/upload_library.sh # dry-run only (no upload)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"

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

# ── Helpers ────────────────────────────────────────────────────────────────────

# publish <relative-crate-path>
publish() {
    local rel_path="$1"
    local crate_path="$REPO_ROOT/$rel_path"
    local crate_name
    crate_name="$(basename "$crate_path")"

    echo ""
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
    info "$crate_name published successfully."

    # crates.io needs time to index the new version before dependent crates
    # can resolve it. 60 s is enough in practice; increase if you see
    # "no matching version" errors on the next crate.
    warn "Waiting 60 s for crates.io to index $crate_name ..."
    sleep 60
}

# ── Sanity check: ensure the working tree is clean ────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
    if ! git -C "$REPO_ROOT" diff --quiet HEAD; then
        error "Working tree has uncommitted changes. Commit or stash before publishing."
        exit 1
    fi
fi

# ── Verify cargo package for every crate first ────────────────────────────────
info "Verifying all crates with 'cargo package' ..."
for crate_path in \
    "$REPO_ROOT/crates/zer-core" \
    "$REPO_ROOT/crates/zer-prof" \
    "$REPO_ROOT/crates/zer-compare" \
    "$REPO_ROOT/crates/zer-blocking" \
    "$REPO_ROOT/crates/zer-schema" \
    "$REPO_ROOT/crates/zer-cluster" \
    "$REPO_ROOT/crates/zer-compute" \
    "$REPO_ROOT/crates/zer-pipeline" \
    "$REPO_ROOT/crates/zer-judge" \
    "$REPO_ROOT/crates/zer-adapters" \
    "$REPO_ROOT/crates/zer-lib"
do
    crate_name="$(basename "$crate_path")"
    cargo package \
        --no-verify \
        --manifest-path "$crate_path/Cargo.toml" \
        --token "$ZER_CRATES_IO_TOKEN" \
        > /dev/null 2>&1 \
        && info "  $crate_name: package OK" \
        || { error "  $crate_name: package FAILED"; exit 1; }
done

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
