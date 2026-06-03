#!/usr/bin/env bash
# Download test datasets from HuggingFace.
#
# Usage:
#   ./scripts/download_dataset.sh          # download tests/ subset (default)
#   ./scripts/download_dataset.sh --all    # download everything
#
# Prerequisites:
#   pip install 'huggingface_hub[cli]'
#
# Optional: set HF_TOKEN env var to authenticate (required for private/gated repos).
#   export HF_TOKEN=hf_...

set -euo pipefail

REPO="arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"

ALL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)    ALL=true;  shift ;;
        -h|--help)
            sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

if ! command -v huggingface-cli &>/dev/null; then
    echo "Error: huggingface-cli not found. Install with: pip install 'huggingface_hub[cli]'" >&2
    exit 1
fi

INCLUDE_ARGS=()
if ! $ALL; then
    INCLUDE_ARGS=(--include "tests/**")
fi

TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    TOKEN_ARGS=(--token "$HF_TOKEN")
fi

echo "Downloading datasets from ${REPO} into data/ ..."
huggingface-cli download "$REPO" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    "${INCLUDE_ARGS[@]}" \
    "${TOKEN_ARGS[@]}"

echo "Done."
