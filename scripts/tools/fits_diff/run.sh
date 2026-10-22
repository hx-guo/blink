#!/usr/bin/env bash
# Read the same FITS files with two builds of blink and compare what the readers return.
#
#   scripts/tools/fits_diff/run.sh OLD_REF NEW_REF LIST WORKDIR
#
# LIST has lines "kind<TAB>path" (generate.py writes one for synthetic files, pick_real.py for archive
# files). Each ref is checked out as a worktree under WORKDIR, instrumented with the test-only digest
# harness (instrument.py), and its digest test is run in release mode, crate by crate. Results go to
# WORKDIR/{old,new}.tsv, with wall time and peak memory per crate in WORKDIR/*.time. The exit status
# is that of compare.py: nonzero when a file reads differently or fails on one side only.
# Set BLINK_FITS_DIGEST_DUMP=dir to write the full text of every result (large for real files).
set -euo pipefail
OLD_REF=$1; NEW_REF=$2; LIST=$(realpath "$3"); WORK=$(realpath -m "$4")
KIT=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$KIT" rev-parse --show-toplevel)
mkdir -p "$WORK"
# wall time and peak memory need GNU time; without it the runs go untimed
timed() { local out=$1; shift; if [ -x /usr/bin/time ]; then /usr/bin/time -v -o "$out" "$@"; else "$@"; fi; }

for side in old new; do
  ref=$([ $side = old ] && echo "$OLD_REF" || echo "$NEW_REF")
  tree=$WORK/$side
  if [ ! -d "$tree" ]; then
    git -C "$REPO" worktree add --detach "$tree" "$ref"
    python3 "$KIT/instrument.py" "$tree"
  fi
  rm -f "$WORK/$side.tsv"
  for crate in blink_hxmt_he blink_fermi_gbm blink_gecam blink_svom_grm blink_grid; do
    (cd "$tree" && CARGO_TARGET_DIR=$WORK/target-$side cargo test --release --quiet -p $crate --lib --no-run)
    (cd "$tree" && BLINK_FITS_DIGEST=$LIST BLINK_FITS_DIGEST_OUT=$WORK/$side.tsv CARGO_TARGET_DIR=$WORK/target-$side \
      timed "$WORK/$side-$crate.time" cargo test --release --quiet -p $crate --lib fits_digest -- --exact fits_digest::fits_digest) \
      > "$WORK/$side-$crate.log" 2>&1 || echo "$side $crate: digest run failed, see $WORK/$side-$crate.log"
    if [ -f "$WORK/$side-$crate.time" ]; then
      printf '%s %-16s %s, peak %s kB\n' $side $crate \
        "$(grep -o 'Elapsed.*' "$WORK/$side-$crate.time")" "$(grep -o 'Maximum resident[^:]*: [0-9]*' "$WORK/$side-$crate.time" | grep -o '[0-9]*$')"
    fi
  done
  sort -o "$WORK/$side.tsv" "$WORK/$side.tsv"
done
python3 "$KIT/compare.py" "$WORK/old.tsv" "$WORK/new.tsv" "$LIST"
