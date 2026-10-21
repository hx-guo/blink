#!/usr/bin/env bash
# Energy-resolved injection run for talk-20 Tier 1 (GRB 250919A).
# 233 fake 30 ms gaps per box, centres T0-4.8..+19.9 s at 0.1 s spacing,
# excluding any centre within 0.1 s of a real FIFO reset of any box
# (data/talk20_spec/inject_offsets_250919a.txt, derived from `blink sat detect`).
# Usage: HXMT_1B_DIR="$PWD/data/1B" ./scripts/talks/talk20_spec_run_inject.sh
set -euo pipefail
BIN=${BLINK_BIN:-./target/release/blink}
TRIG=432865758
OUT=data/talk20_spec/inject_250919a
AT=$(cat data/talk20_spec/inject_offsets_250919a.txt)
mkdir -p "$OUT"
for BOX in a b c; do
  "$BIN" sat inject "$TRIG" --before 5 --after 20 --target "$BOX" --at="$AT" --width 0.03 \
    --events-out "$OUT/events_$BOX.csv" --gapcov-out "$OUT/gapcov_$BOX.csv" \
    --gapbins-out "$OUT/gapbins_$BOX.csv" --truth-out "$OUT/truth_$BOX.csv" \
    > "$OUT/stdout_$BOX.csv" 2> "$OUT/stderr_$BOX.log"
done
