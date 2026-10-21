#!/bin/bash
# farm wrapper for the scouting scripts: run_sci.sh SCRIPT [ARGS...]
set -euo pipefail
BASE=/scratchfs2/gecam/guohx/p221009
mkdir -p $BASE/sci/out
cd $BASE/blink/scripts
source $BASE/venv/bin/activate
export MPLBACKEND=Agg MPLCONFIGDIR=$BASE/sci/.mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname) start $(date -Is) $*"
/usr/bin/time -f "%C: maxrss %MkB wall %es" python "$@"
echo "done $(date -Is)"
