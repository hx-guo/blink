#!/bin/bash
# hep_sub worker — invoked as: run_hv_chunk.sh START END IDX
# Writes hv_parts/hv_part_IDX.csv
set -e
START=$1
END=$2
IDX=$3
cd /scratchfs/gecam/guohx/n_below_study
mkdir -p hv_parts
python3 extract_hv_chunk.py "$START" "$END" "hv_parts/hv_part_${IDX}.csv"
