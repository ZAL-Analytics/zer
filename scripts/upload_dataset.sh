#!/usr/bin/env bash
set -euo pipefail

REPO="arsalan-anwari/dutch-law-enforcement-entity-resolution-dataset"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../data" && pwd)"

# Resolve CLI: prefer `hf` alias, fall back to `huggingface-cli`
if command -v hf &>/dev/null; then
    HF_CLI="hf"
elif command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli"
else
    echo "Error: HuggingFace CLI not found. Install it with: pip install huggingface_hub[cli]"
    exit 1
fi

COMMIT_MSG="Update dataset files $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

echo "Uploading data/ to https://huggingface.co/datasets/${REPO} ..."
$HF_CLI upload "$REPO" "$DATA_DIR" . \
    --repo-type dataset \
    --commit-message "$COMMIT_MSG"

echo "Done."
