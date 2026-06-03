#!/usr/bin/env bash
# Download test datasets from HuggingFace.
#
# Usage:
#   ./scripts/download_dataset.sh              # download tests/ subset (default)
#   ./scripts/download_dataset.sh --examples   # download tests/ + examples/ subsets
#   ./scripts/download_dataset.sh --all        # download everything
#
# Prerequisites:
#   pip install 'huggingface_hub'
#
# Optional: set HF_TOKEN env var to authenticate (required for private/gated repos).
#   export HF_TOKEN=hf_...

set -euo pipefail

REPO="arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"

ALL=false
EXAMPLES=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)      ALL=true;      shift ;;
        --examples) EXAMPLES=true; shift ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

if ! command -v hf &>/dev/null; then
    echo "Error: hf not found. Install with: pip install 'huggingface_hub'" >&2
    exit 1
fi

INCLUDE_ARGS=()
if ! $ALL; then
    INCLUDE_ARGS=(--include "tests/**")
    $EXAMPLES && INCLUDE_ARGS+=(--include "examples/**")
fi

TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    TOKEN_ARGS=(--token "$HF_TOKEN")
fi

echo "Downloading datasets from ${REPO} into data/ ..."
hf download "$REPO" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    "${INCLUDE_ARGS[@]}" \
    "${TOKEN_ARGS[@]}"

echo "Done."
