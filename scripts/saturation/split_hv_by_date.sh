#!/bin/bash
# Stream-split the big HV gz into per-date plain CSV files (one per date).
# awk streams the input, no memory spike.
# Output: n_below_study/hv_by_date/YYYYMMDD.csv  (~3300 files, ~5 MB each)
set -e
SRC=n_below_study/hv_table_full.csv.gz
DST=n_below_study/hv_by_date
mkdir -p "$DST"

# Skip header (NR>1), append each row to "$DST/$1.csv"
gunzip -c "$SRC" | awk -F',' '
NR == 1 { header = $0; next }
{
    f = "'"$DST"'/" $1 ".csv"
    if (!(f in seen)) {
        print header > f
        seen[f] = 1
    }
    print >> f
}
END { print "split into " length(seen) " per-date files" }
'

echo "Per-date HV files in $DST"
ls "$DST" | wc -l
du -sh "$DST"
