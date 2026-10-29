#!/usr/bin/env bash
# One run of the real-time pipeline (S1-S7), meant for cron, e.g. hourly:
#   15 * * * *  /path/to/blink/scripts/pipeline/realtime/run.sh /path/to/config.sh
#
# Searches the last DAYS_BACK days again (the search, exposure and vet skip days whose outputs are
# newer than their data), rebuilds the catalog, names new final events, and publishes the website
# data. Retractions, merges and reinstatements of names are never applied here: they are reported
# (NOTIFY) and wait for `blink name ... --apply` by a person.
set -euo pipefail

CONFIG=${1:?usage: run.sh CONFIG}
# shellcheck source=config.example.sh
source "$CONFIG"
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/catalogs/$SLUG"
cd "$RUN_DIR"

# one run at a time; a run that finds the lock busy just ends
exec 9> "$RUN_DIR/.run.lock"
flock -n 9 || exit 0

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$RUN_DIR/logs/$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

notify() {
  echo "$1"
  if [[ -n "$NOTIFY" ]]; then
    { echo "$1"; echo; echo "log: $LOG"; tail -n 60 "$LOG"; } | bash -c "$NOTIFY" || true
  fi
}
trap 'notify "TGF pipeline run $STAMP FAILED at line $LINENO"' ERR

FROM=$(date -u -d "$DAYS_BACK days ago" +%F)
TO=$(date -u +%F)
echo "== $STAMP  $SLUG  $FROM .. $TO"

# S1 search (writes signals, hours and exposure), S5 exposure for days searched before, S2 vet
"$BLINK" search "$FROM" "$TO" --instrument "$INSTRUMENT"
"$BLINK" exposure "$FROM" "$TO" --instrument "$INSTRUMENT"
"$BLINK" vet "$FROM" "$TO" --instrument "$INSTRUMENT"

# S3-S4 catalog of all days
CATALOG="$RUN_DIR/catalogs/$SLUG/$STAMP"
"$BLINK" catalog --out-dir "$CATALOG" --instrument "$INSTRUMENT"
ln -sfn "$CATALOG" "$RUN_DIR/catalogs/$SLUG/latest"
python3 "$HERE/track_latency.py" "$CATALOG" "$RUN_DIR/latency.csv" --now "$STAMP"

# S4.5 names: safe changes only
VERSION="$SLUG/$STAMP"
# status 3 = changes held for review; the ERR trap must not count it as a failure
trap - ERR
set +e
# shellcheck disable=SC2086
"$BLINK" name "$CATALOG" --registry "$REGISTRY" --version "$VERSION" --apply-new-only ${NAME_EXTRA:-}
NAME_STATUS=$?
set -e
trap 'notify "TGF pipeline run $STAMP FAILED at line $LINENO"' ERR
if [[ $NAME_STATUS -ne 0 && $NAME_STATUS -ne 3 ]]; then
  notify "TGF pipeline run $STAMP: blink name failed with status $NAME_STATUS"
  exit "$NAME_STATUS"
fi
if git -C "$REGISTRY" rev-parse --git-dir > /dev/null 2>&1 && [[ -n $(git -C "$REGISTRY" status --porcelain) ]]; then
  git -C "$REGISTRY" add -A
  git -C "$REGISTRY" commit -q -m "name: $VERSION"
  python3 "$HERE/../check_registry.py" "$REGISTRY" HEAD~1
fi
if [[ $NAME_STATUS -eq 3 ]]; then
  notify "TGF pipeline run $STAMP: name changes held for review. Check the REINSTATE/RETRACT/MERGE lines, then run:
  $BLINK name $CATALOG --registry $REGISTRY --version $VERSION --apply"
fi

# S6 export to a new directory, then switch the served symlink in one rename
EXPORT="$RUN_DIR/web/$STAMP"
"$BLINK" export "$CATALOG" --registry "$REGISTRY" --data "$RUN_DIR/data" -o "$EXPORT" --public-only
ln -sfn "$EXPORT" "$WEB_DATA.tmp"
mv -T "$WEB_DATA.tmp" "$WEB_DATA"

# S7 copy to the web server
if [[ -n "${DEPLOY:-}" ]]; then
  rsync -a --delete-after --delay-updates -e "${DEPLOY_SSH:-ssh -o BatchMode=yes}" "$EXPORT/" "$DEPLOY"
  echo "deployed to $DEPLOY"
fi

# keep the last KEEP_DAYS of catalogs; of the exports (thousands of files each) only the served one
find "$RUN_DIR/catalogs/$SLUG" -mindepth 1 -maxdepth 1 -type d -mtime "+$KEEP_DAYS" -exec rm -rf {} +
find "$RUN_DIR/web" -mindepth 1 -maxdepth 1 -type d ! -path "$EXPORT" -exec rm -rf {} +
find "$RUN_DIR/logs" -name '*.log' -mtime "+$KEEP_DAYS" -delete
echo "== done $STAMP"
