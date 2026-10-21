#!/bin/bash
# Submit 1000 hep_sub jobs to extract HV in parallel for dates 20180203 → 20260517.
# Each job handles ~3 days, writes its own CSV part.
#
# Run this ON the IHEP server (lxlogin):
#   cd /scratchfs/gecam/guohx/n_below_study
#   bash submit_hv_jobs.sh

set -e
HEPSUB=/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin/hep_sub
WORK=/scratchfs/gecam/guohx/n_below_study
cd "$WORK"
mkdir -p hv_parts hv_logs

# Generate chunks list (start_date end_date idx, one chunk per line)
python3 - <<'EOF' > chunks.list
from datetime import date, timedelta
s = date(2018, 2, 3)
e = date(2026, 5, 17)
total = (e - s).days + 1
N_JOBS = 1000
per_job = (total + N_JOBS - 1) // N_JOBS
cur = s
i = 0
while cur <= e:
    end = min(cur + timedelta(days=per_job - 1), e)
    print(f"{cur.strftime('%Y%m%d')} {end.strftime('%Y%m%d')} {i:04d}")
    cur = end + timedelta(days=1)
    i += 1
EOF

NJOBS=$(wc -l < chunks.list)
echo "Submitting $NJOBS jobs..."

# Submit
SUBMITTED=0
while read start end idx; do
    $HEPSUB -g gecam -wt short \
            -o "hv_logs/p_${idx}.out" -e "hv_logs/p_${idx}.err" \
            "$WORK/run_hv_chunk.sh" \
            -argu "$start" "$end" "$idx" > /dev/null
    SUBMITTED=$((SUBMITTED + 1))
    if (( SUBMITTED % 100 == 0 )); then
        echo "  submitted $SUBMITTED/$NJOBS"
    fi
done < chunks.list

echo "All $SUBMITTED jobs submitted."
echo "Monitor with: condor_q -submitter guohx"
