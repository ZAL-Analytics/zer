#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../models" && pwd)"

HF_REPO="arsalan-anwari/zjudge"

if ! command -v huggingface-cli &>/dev/null; then
    echo "Error: huggingface-cli not found. Install it with: pip install huggingface_hub" >&2
    exit 1
fi

if [[ -z "$HF_REPO" ]]; then
    echo "Error: HF_REPO is not set." >&2
    exit 1
fi

huggingface-cli download "$HF_REPO" --local-dir "$MODELS_DIR"
echo "Done."
