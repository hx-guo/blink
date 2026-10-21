#!/usr/bin/env bash
# Trial run of the whole pipeline over a date range, WITHOUT publishing anything.
#
#   scripts/pipeline/realtime/trial.sh CONFIG FROM TO [WORKERS]
#
# Same stages as run.sh (search, exposure, vet, catalog, names, export), but over FROM..TO with WORKERS
# parallel processes on this node, names written to the test registry of CONFIG, and the export packed
# into RUN_DIR/trial-<stamp>.tar.gz to be looked at locally (web/public/data, then `pnpm dev`).
# Refuses to run if CONFIG sets DEPLOY: a trial must not reach the web server.
set -euo pipefail

CONFIG=${1:?usage: trial.sh CONFIG FROM TO [WORKERS]}
FROM=${2:?usage: trial.sh CONFIG FROM TO [WORKERS]}
TO=${3:?usage: trial.sh CONFIG FROM TO [WORKERS]}
WORKERS=${4:-8}
# shellcheck source=config.example.sh
source "$CONFIG"
if [[ -n "${DEPLOY:-}" ]]; then
  echo "trial.sh: DEPLOY is set in $CONFIG; a trial run must not publish. Use a config without DEPLOY." >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/catalogs/$SLUG" "$REGISTRY"
cd "$RUN_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$RUN_DIR/logs/trial-$STAMP.log"
exec > >(tee -a "$LOG") 2>&1
echo "== trial $STAMP  $SLUG  $FROM .. $TO  ($WORKERS workers)"

# the test registry: a git repository that lists the instrument as public
if ! git -C "$REGISTRY" rev-parse --git-dir > /dev/null 2>&1; then
  git -C "$REGISTRY" init -q
fi
NAME=$(echo "$SLUG" | sed 's|_|/|')    # slug back to the instrument name: Insight-HXMT_HE -> Insight-HXMT/HE
grep -qxF "$NAME" "$REGISTRY/public_instruments.txt" 2>/dev/null || echo "$NAME" >> "$REGISTRY/public_instruments.txt"

parallel() {
  local pids=()
  for ((i = 0; i < WORKERS; i++)); do
    "$BLINK" "$@" --workers "$WORKERS" --worker "$i" > "$RUN_DIR/logs/trial-$STAMP-$1-$i.log" 2>&1 &
    pids+=($!)
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  if [[ $failed -ne 0 ]]; then
    echo "trial.sh: $1 failed, see $RUN_DIR/logs/trial-$STAMP-$1-*.log" >&2
    exit 1
  fi
  tail -n 1 "$RUN_DIR"/logs/trial-"$STAMP"-"$1"-*.log
}

time parallel search "$FROM" "$TO" --instrument "$INSTRUMENT"
time parallel exposure "$FROM" "$TO" --instrument "$INSTRUMENT"
time parallel vet "$FROM" "$TO" --instrument "$INSTRUMENT"

CATALOG="$RUN_DIR/catalogs/$SLUG/trial-$STAMP"
"$BLINK" catalog --out-dir "$CATALOG" --instrument "$INSTRUMENT"

set +e
# shellcheck disable=SC2086
"$BLINK" name "$CATALOG" --registry "$REGISTRY" --version "$SLUG/trial-$STAMP" --apply-new-only ${NAME_EXTRA:-}
status=$?
set -e
[[ $status -eq 0 || $status -eq 3 ]] || exit "$status"
if [[ -n $(git -C "$REGISTRY" status --porcelain) ]]; then
  git -C "$REGISTRY" add -A
  git -C "$REGISTRY" commit -q -m "name: $SLUG/trial-$STAMP"
fi

EXPORT="$RUN_DIR/web/trial-$STAMP"
"$BLINK" export "$CATALOG" --registry "$REGISTRY" --data "$RUN_DIR/data" -o "$EXPORT" --public-only
tar -czf "$RUN_DIR/trial-$STAMP.tar.gz" -C "$EXPORT" . -C "$CATALOG" manifest.json catalog.csv rejected.csv
echo "== done: catalog $CATALOG"
echo "   website data packed in $RUN_DIR/trial-$STAMP.tar.gz ($(du -h "$RUN_DIR/trial-$STAMP.tar.gz" | cut -f1))"
echo "   (nothing was copied to the web server)"
