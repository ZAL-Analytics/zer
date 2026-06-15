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

if [[ -n "$VERSION" && -z "$DRY_RUN" ]]; then
    NEW_URL="/docs/zer/$VERSION/"

    echo "==> Updating versions.json on server..."
    ssh "$REMOTE_HOST" python3 << PYEOF
import json, os

path = '$REMOTE_PATH/versions.json'
ver  = '$VERSION'
url  = '$NEW_URL'

if os.path.exists(path):
    with open(path) as f:
        versions = json.load(f)
else:
    versions = []

versions = [v for v in versions if v.get('version') != ver]
versions.append({'version': ver, 'url': url})

def ver_key(v):
    try:
        return tuple(int(x) for x in v['version'].split('.'))
    except ValueError:
        return (0,)

latest = max(versions, key=ver_key)
for v in versions:
    v.pop('latest', None)
latest['latest'] = True
versions.sort(key=ver_key, reverse=True)

with open(path, 'w') as f:
    json.dump(versions, f, indent=2)
    f.write('\n')
print('  updated:', path)
PYEOF

    echo "==> Updating 'latest' symlink -> $VERSION ..."
    ssh "$REMOTE_HOST" "ln -snf '${VERSION}' '${REMOTE_PATH}latest'"

    echo "==> Ensuring root index.html redirect exists..."
    ssh "$REMOTE_HOST" "test -f '${REMOTE_PATH}index.html' || echo '<meta http-equiv=\"refresh\" content=\"0;url=/docs/zer/latest/\">' > '${REMOTE_PATH}index.html'"
fi

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "Upload complete. Docs live at:"
    if [[ -n "$VERSION" ]]; then
        echo "  https://zal-analytics.ch/docs/zer/$VERSION/"
    else
        echo "  https://zal-analytics.ch/docs/zer/"
    fi
fi
