#!/usr/bin/env bash
# Sync docs/sphinx/out/ to the vimexx server.
# Usage: bash scripts/upload_docs.sh [--dry-run] [--sync-res]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="vimexx"
REMOTE_PATH="/home/u214998p479997/domains/zal-analytics.ch/public_html/docs/zer/"

DRY_RUN=""
EXCLUDE_RES="--exclude='res/'"
VERSION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN="--dry-run"; echo "==> Dry run, no files will be transferred"; shift ;;
        --sync-res) EXCLUDE_RES=""; echo "==> Including res/ in sync"; shift ;;
        --version)  VERSION="${2:?'--version requires an argument'}"; shift 2 ;;
        *)          shift ;;
    esac
done

if [[ -n "$VERSION" ]]; then
    LOCAL_OUT="$REPO_ROOT/docs/sphinx/out/$VERSION/"
    REMOTE_DEST="${REMOTE_PATH}${VERSION}/"
else
    LOCAL_OUT="$REPO_ROOT/docs/sphinx/out/"
    REMOTE_DEST="$REMOTE_PATH"
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

echo "==> Syncing docs to $REMOTE_HOST:$REMOTE_DEST ..."
rsync \
    --archive \
    --compress \
    --delete \
    ${EXCLUDE_RES} \
    --human-readable \
    --progress \
    ${DRY_RUN} \
    -e "ssh" \
    "$LOCAL_OUT" \
    "$REMOTE_HOST:$REMOTE_DEST"

if [[ -z "$DRY_RUN" ]]; then
    VERSIONS_JSON="$REPO_ROOT/docs/sphinx/out/versions.json"

    # In versioned mode versions.json is NOT inside the synced subdirectory,
    # so upload it separately.
    if [[ -n "$VERSION" && -f "$VERSIONS_JSON" ]]; then
        echo "==> Uploading versions.json..."
        scp "$VERSIONS_JSON" "$REMOTE_HOST:${REMOTE_PATH}versions.json"
    fi

    # Update the 'latest' symlink and root index.html from the local file.
    if [[ -f "$VERSIONS_JSON" ]]; then
        LATEST_VER=$(python3 << PYEOF
import json
try:
    with open('$VERSIONS_JSON') as f:
        vs = json.load(f)
    print(next((v['version'] for v in vs if v.get('latest')), ''))
except Exception:
    pass
PYEOF
)
        if [[ -n "$LATEST_VER" ]]; then
            echo "==> Updating 'latest' symlink -> $LATEST_VER ..."
            ssh "$REMOTE_HOST" "ln -snf '${LATEST_VER}' '${REMOTE_PATH}latest'"
            echo "==> Ensuring root index.html redirect exists..."
            ssh "$REMOTE_HOST" "test -f '${REMOTE_PATH}index.html' || echo '<meta http-equiv=\"refresh\" content=\"0;url=/docs/zer/latest/\">' > '${REMOTE_PATH}index.html'"
        fi
    fi

    echo ""
    echo "Upload complete. Docs live at:"
    if [[ -n "$VERSION" ]]; then
        echo "  https://zal-analytics.ch/docs/zer/$VERSION/"
    else
        echo "  https://zal-analytics.ch/docs/zer/"
    fi
fi
