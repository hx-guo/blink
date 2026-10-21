#!/bin/bash
# Re-run all hypothesis plot scripts sequentially with full HV cache.
# Each script loads/processes ~26 GB DataFrame; cannot run in parallel.
#
# Logs to /tmp/rerun_<script>.log
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

SCRIPTS=(
    "bursts/plot_sci_pred_M7merged_perdet_with_260226A.py"        # V8 baseline
    "saturation/plot_sci_pred_M7merged_perdet_dt_k1.py"               # dt k=1
    "saturation/plot_sci_pred_M7merged_perdet_cpure1_gamma1.py"       # c_pure=γ=1
    "saturation/plot_sci_pred_M7merged_perdet_combined_dt_constraints.py"  # dt + c_pure=γ=1
    "saturation/plot_all_hypotheses.py"                                # 7 hypotheses inside
    "saturation/plot_V10_crossdet.py"                                  # V10 free
    "saturation/plot_V10_b0_cpure1_gamma1.py"                          # V10 + 真香
    "saturation/plot_V10_all_constraints.py"                           # V10 + dt + 真香
)

TOTAL=${#SCRIPTS[@]}
START=$(date +%s)
for i in "${!SCRIPTS[@]}"; do
    S="${SCRIPTS[$i]}"
    NOW=$(date '+%H:%M:%S')
    ELAPSED=$(( $(date +%s) - START ))
    echo "[$NOW] [$((i+1))/$TOTAL] running scripts/$S  (elapsed ${ELAPSED}s)" >&2
    LOG=/tmp/rerun_$(basename "$S" .py).log
    python3 -u "scripts/$S" > "$LOG" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
        echo "[$NOW] [$((i+1))/$TOTAL] DONE $S  (RC=0)" >&2
    else
        echo "[$NOW] [$((i+1))/$TOTAL] FAIL $S  (RC=$RC) — see $LOG" >&2
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - START ))
echo "[$(date '+%H:%M:%S')] ALL DONE in ${TOTAL_ELAPSED}s" >&2
