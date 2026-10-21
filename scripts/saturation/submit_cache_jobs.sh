#!/usr/bin/env bash
# Submit one hep_sub job per missing date in clean_partials/.
# Idempotent: skips dates whose partial already exists.
#
# Usage:  submit_cache_jobs.sh [START_DATE] [END_DATE] [--dry-run]
# Defaults: 20170615 .. today.

set -euo pipefail

START="${1:-20170615}"
END="${2:-$(date -u +%Y%m%d)}"
DRY_RUN=""
for arg in "$@"; do
    [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

WORKER="$(realpath "$(dirname "$0")/build_clean_one_day.sh")"
LOG_DIR="logs/cache"

HEPSUB="/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin/hep_sub"
HEPSUB_GROUP="gecam"
HEPSUB_MEM_MB="3000"
HEPSUB_WT="short"

mkdir -p "$LOG_DIR"

cur="$START"
n_submitted=0
n_skipped=0
while [ "$cur" -le "$END" ]; do
    year="${cur:0:4}"
    partial="clean_partials/year_${year}/${cur}.parquet"
    if [ -s "$partial" ]; then
        n_skipped=$((n_skipped + 1))
    else
        if [ -n "$DRY_RUN" ]; then
            echo "[DRY] $HEPSUB -g $HEPSUB_GROUP -m $HEPSUB_MEM_MB -wt $HEPSUB_WT \\"
            echo "        -o $LOG_DIR/${cur}.out -e $LOG_DIR/${cur}.err \\"
            echo "        -argu $cur -- $WORKER"
        else
            "$HEPSUB" -g "$HEPSUB_GROUP" -m "$HEPSUB_MEM_MB" -wt "$HEPSUB_WT" \
                -o "$LOG_DIR/${cur}.out" -e "$LOG_DIR/${cur}.err" \
                -argu "$cur" -- "$WORKER"
        fi
        n_submitted=$((n_submitted + 1))
    fi
    cur=$(python3 -c "
from datetime import date, timedelta
d = date(int('$cur'[:4]), int('$cur'[4:6]), int('$cur'[6:8])) + timedelta(days=1)
print(d.strftime('%Y%m%d'))")
done

echo "submitted=$n_submitted skipped=$n_skipped (range: $START..$END)"
