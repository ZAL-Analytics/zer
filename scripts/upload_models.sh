#!/usr/bin/env bash
set -euo pipefail

HF_REPO="arsalan-anwari/zjudge"
MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../models" && pwd)"

if command -v hf &>/dev/null; then
    HF_CLI="hf"
elif command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli"
else
    echo "Error: HuggingFace CLI not found. Install it with: pip install huggingface_hub[cli]"
    exit 1
fi

if [[ -z "$HF_REPO" ]]; then
    echo "Error: HF_REPO is not set." >&2
    exit 1
fi

COMMIT_MSG="Update model files $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

echo "Uploading models/ to https://huggingface.co/${HF_REPO} ..."
$HF_CLI upload "$HF_REPO" "$MODELS_DIR" . \
    --repo-type model \
    --commit-message "$COMMIT_MSG"

echo "Done."
