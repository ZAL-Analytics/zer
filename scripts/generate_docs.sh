#!/usr/bin/env bash
# Build Sphinx docs and rustdoc into docs/sphinx/out/.
# Usage: bash scripts/generate_docs.sh [--open] [--with-benchmarks-figures]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPHINX_SRC="$REPO_ROOT/docs/sphinx"
RUSTDOC_CSS="$SPHINX_SRC/_static/rustdoc-override.css"

# --------------------------------------------------------------------------- #
# Parse flags
# --------------------------------------------------------------------------- #

WITH_BENCH_FIGURES=0
OPEN_AFTER=0
VERSION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-benchmarks-figures) WITH_BENCH_FIGURES=1; shift ;;
    --open)                    OPEN_AFTER=1; shift ;;
    --version)                 VERSION="${2:?'--version requires an argument'}"; shift 2 ;;
    *)                         shift ;;
  esac
done

if [[ -n "$VERSION" ]]; then
  OUT_DIR="$REPO_ROOT/docs/sphinx/out/$VERSION"
else
  OUT_DIR="$REPO_ROOT/docs/sphinx/out"
fi
API_DIR="$OUT_DIR/api"

# --------------------------------------------------------------------------- #
# Benchmark figures helper
# --------------------------------------------------------------------------- #

copy_bench_figure() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: benchmark figure not found: $src" >&2
    exit 1
  fi
  cp "$src" "$dst"
}

# --------------------------------------------------------------------------- #
# 1. Sphinx build
# --------------------------------------------------------------------------- #

echo "==> Building Sphinx docs..."
python -m sphinx \
    -b html \
    -d "$OUT_DIR/.doctrees" \
    "$SPHINX_SRC" \
    "$OUT_DIR"
echo "    Sphinx output: $OUT_DIR"

# --------------------------------------------------------------------------- #
# 1b. Copy benchmark figures (optional)
# --------------------------------------------------------------------------- #

if [[ $WITH_BENCH_FIGURES -eq 1 ]]; then
  echo "==> Copying benchmark figures..."
  PLOTS="$REPO_ROOT/bench_results/plots"
  RES_DIR="$OUT_DIR/res"
  mkdir -p "$RES_DIR"

  copy_bench_figure \
    "$PLOTS/accuracy_all/accuracy_comparison/accuracy_comparison.svg" \
    "$RES_DIR/accuracy_comparison.svg"

  copy_bench_figure \
    "$PLOTS/accuracy_all/pr_curves/pr_curves.svg" \
    "$RES_DIR/pr_curves.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/throughput_comparison.svg" \
    "$RES_DIR/throughput_comparison_cuda.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/brp_dedupe/zer_cuda/stage_pie/stage_pie.svg" \
    "$RES_DIR/throughput_stage_pie_brp_zer_cuda.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/brp_dedupe/splink/stage_pie/stage_pie.svg" \
    "$RES_DIR/throughput_stage_pie_brp_splink.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/kvk_dedupe/zer_cuda/stage_pie/stage_pie.svg" \
    "$RES_DIR/throughput_stage_pie_kvk_zer_cuda.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/kvk_dedupe/splink/stage_pie/stage_pie.svg" \
    "$RES_DIR/throughput_stage_pie_kvk_splink.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/brp_dedupe/zer_cuda/memory_timeline/memory_timeline.svg" \
    "$RES_DIR/throughput_mem_timeline_brp_zer_cuda.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/brp_dedupe/splink/memory_timeline/memory_timeline.svg" \
    "$RES_DIR/throughput_mem_timeline_brp_splink.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/kvk_dedupe/zer_cuda/memory_timeline/memory_timeline.svg" \
    "$RES_DIR/throughput_mem_timeline_kvk_zer_cuda.svg"

  copy_bench_figure \
    "$PLOTS/throughput_cuda/kvk_dedupe/splink/memory_timeline/memory_timeline.svg" \
    "$RES_DIR/throughput_mem_timeline_kvk_splink.svg"

  echo "    Benchmark figures: $RES_DIR"
fi

# --------------------------------------------------------------------------- #
# 2. rustdoc build
# --------------------------------------------------------------------------- #

echo "==> Removing stale rustdoc output for old crate name 'zer'..."
rm -rf "$REPO_ROOT/target/doc/zer"

echo "==> Building rustdoc..."
(
  cd "$REPO_ROOT"
  RUSTDOCFLAGS="--extend-css $RUSTDOC_CSS --default-theme dark" \
    cargo doc \
      --workspace \
      --no-deps \
      --exclude zer-test-utils \
      --target-dir "$REPO_ROOT/target" \
      2>&1
)

# Copy rustdoc output into the Sphinx output tree
mkdir -p "$API_DIR"
rsync -a --delete \
    "$REPO_ROOT/target/doc/" \
    "$API_DIR/"
echo "    rustdoc output: $API_DIR"

# --------------------------------------------------------------------------- #
# 3. Update out/versions.json (versioned builds only)
# --------------------------------------------------------------------------- #

if [[ -n "$VERSION" ]]; then
  echo "==> Updating docs/sphinx/out/versions.json ..."
  VERSIONS_JSON="$REPO_ROOT/docs/sphinx/out/versions.json"
  python3 << PYEOF
import json, os

path = '$VERSIONS_JSON'
ver  = '$VERSION'
url  = '/docs/zer/$VERSION/'

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
fi

echo ""
echo "Done. Serve locally with:"
echo "  python -m http.server 8081 --directory \"$REPO_ROOT/docs/sphinx/out/\""
