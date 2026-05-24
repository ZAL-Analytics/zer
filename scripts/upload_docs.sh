#!/usr/bin/env bash
# Sync docs/sphinx/out/ to the vimexx server.
# Usage: bash scripts/upload_docs.sh [--dry-run]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_OUT="$REPO_ROOT/docs/sphinx/out/"
REMOTE_HOST="vimexx"
REMOTE_PATH="/home/u214998p479997/domains/zal-analytics.ch/public_html/docs/zer/"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "==> Dry run, no files will be transferred"
fi

# Verify local output exists
if [[ ! -d "$LOCAL_OUT" ]]; then
    echo "ERROR: $LOCAL_OUT does not exist. Run scripts/generate_docs.sh first." >&2
    exit 1
fi

# Verify SSH connectivity before syncing
echo "==> Checking SSH connection to $REMOTE_HOST..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" true 2>/dev/null; then
    echo "ERROR: Cannot connect to $REMOTE_HOST. Check your SSH config / key." >&2
    exit 1
fi

echo "==> Syncing docs to $REMOTE_HOST:$REMOTE_PATH ..."
rsync \
    --archive \
    --compress \
    --delete \
    --exclude='res/' \
    --human-readable \
    --progress \
    ${DRY_RUN} \
    -e "ssh" \
    "$LOCAL_OUT" \
    "$REMOTE_HOST:$REMOTE_PATH"

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "Upload complete. Docs live at:"
    echo "  https://zal-analytics.ch/docs/zer/"
fi
