#!/usr/bin/env bash
# Per-day worker wrapper for hep_sub. Sets up environment + cwd, then calls the
# Python extractor for exactly one UTC date.
#
# Usage:  extract_per_sec_one.sh YYYYMMDD
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 YYYYMMDD" >&2
    exit 2
fi

export BLINK_1B_ROOT=/hxmtfs/data/Archive_tmp/1B
export BLINK_1K_ROOT=/hxmt/work/HXMT-DATA/1K
cd /scratchfs/gecam/guohx/blink
python3 scripts/saturation/extract_per_sec_day.py "$1" --output-dir per_sec_parquet
