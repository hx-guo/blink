#!/usr/bin/env bash
# Resume the v5 full-data pipeline after returning home with the Graphite drive.
#
#   bash scripts/saturation/v5_finish.sh
#
# Idempotent: rsync --partial continues interrupted files; per-year workers skip
# years whose npz already exists. Safe to re-run if interrupted.
#
# Progress as of 2026-05-28 (drive disconnected mid-download):
#   - 2017, 2018: cache downloaded + npz computed (n_below_study/v5_npz/)
#   - 2019: ~97% downloaded, no npz yet
#   - 2020-2023, 2025, 2026: queued
#   - 2024: still building on hlogin (build_clean_cache.py) — picked up if ready
set -euo pipefail

CACHE=/Volumes/Graphite/work/blink/blink_clean_relaxed
NPZ=n_below_study/v5_npz
GRID=n_below_study/aacgm_grid_2020.npz
SRV='hlogin08:/scratchfs/gecam/guohx/blink/n_below_study'
SSH_CIPHER='ssh -c chacha20-poly1305@openssh.com'

if [ ! -d "$CACHE" ]; then
    echo "ERROR: $CACHE not mounted. Plug in the Graphite drive first."
    exit 1
fi
mkdir -p "$NPZ"

echo "=== [1/4] rsync resume (--partial continues; sample05 excluded) ==="
# Wildcard expands on the server, so 2024 is included automatically once its
# build finishes. --exclude drops the 5% sample file.
rsync -a --partial --progress -e "$SSH_CIPHER" \
    --exclude 'clean_relaxed_*_sample05.parquet' \
    "$SRV/clean_relaxed_20*.parquet" \
    "$CACHE/"

echo
echo "=== [2/4] per-year v5 workers (skip years already done) ==="
# shellcheck disable=SC1091
source .venv/bin/activate
for f in "$CACHE"/clean_relaxed_20*.parquet; do
    base=$(basename "$f" .parquet)
    case "$base" in *sample*) continue;; esac
    yr=${base#clean_relaxed_}
    out="$NPZ/v5_agg_${yr}.npz"
    if [ -s "$out" ]; then
        echo "  skip $yr (npz exists)"
        continue
    fi
    echo "  --- worker $yr ---"
    MPLCONFIGDIR=/tmp python3 -u scripts/saturation/v5_aggregator_yearly.py \
        --input "$f" --aacgm-grid "$GRID" --output "$out"
done

echo
echo "=== [3/4] merge all yearly npz ==="
python3 scripts/saturation/v5_merge_npz.py \
    --input-glob "$NPZ/v5_agg_*.npz" \
    --output "$NPZ/v5_agg_full.npz"

echo
echo "=== [4/4] plot final triptych ==="
python3 scripts/plot_v5_final_full.py \
    --input "$NPZ/v5_agg_full.npz" \
    --output plots/v5_final_full.png

echo
echo "DONE -> plots/v5_final_full.png"
