#!/usr/bin/env bash
# Submit one hep_sub job per missing date in the per_sec_parquet/ output dir.
#
# Usage:
#   ./submit_extract_jobs.sh [START_DATE] [END_DATE] [--dry-run]
#
# Defaults:
#   START_DATE = 20170615
#   END_DATE   = $(date -u +%Y%m%d)
#
# Idempotent: skips dates whose output already exists. Re-run to backfill.

set -euo pipefail

START="${1:-20170615}"
END="${2:-$(date -u +%Y%m%d)}"
DRY_RUN=""
for arg in "$@"; do
    [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

WORKER="$(realpath "$(dirname "$0")/extract_per_sec_one.sh")"
OUTPUT_DIR="per_sec_parquet"
LOG_DIR="logs/extract"

# IHEP HEP cluster configuration
HEPSUB="/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin/hep_sub"
HEPSUB_GROUP="gecam"
HEPSUB_MEM_MB="6000"     # observed peak ~4 GB; 6 GB has safety margin
HEPSUB_WT="short"        # 30min wall, high concurrency; ~2% of days exceed → hlogin fan-out

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cur="$START"
n_submitted=0
n_skipped=0
while [ "$cur" -le "$END" ]; do
    out="$OUTPUT_DIR/${cur}.parquet"
    if [ -s "$out" ]; then
        n_skipped=$((n_skipped + 1))
    else
        # NOTE: `-argu` uses argparse nargs='+' which is greedy, so we put the
        # jobscript path AFTER a `--` separator to stop argument capture.
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
    # Advance one day via portable Python date arithmetic.
    cur=$(python3 -c "
from datetime import date, timedelta
d = date(int('$cur'[:4]), int('$cur'[4:6]), int('$cur'[6:8])) + timedelta(days=1)
print(d.strftime('%Y%m%d'))")
done

echo "submitted=$n_submitted skipped=$n_skipped (range: $START..$END)"
