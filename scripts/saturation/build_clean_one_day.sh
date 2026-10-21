#!/usr/bin/env bash
# Per-day worker wrapper for hep_sub.
# Usage:  build_clean_one_day.sh YYYYMMDD
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 YYYYMMDD" >&2
    exit 2
fi

cd /scratchfs/gecam/guohx/blink
python3 scripts/saturation/build_clean_one_day.py "$1"
