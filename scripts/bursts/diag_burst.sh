#!/usr/bin/env bash
# Run end-to-end burst saturation diagnostic:
#   1. blink sat report  → data pack
#   2. plot_burst_report.py  → diagnostic PNG
#
# Pack is written to a temp directory and removed after plotting unless
# --keep-pack is given. Run from the repo root, or from any directory
# (the script cd's to repo root before running).
#
# Usage:
#   scripts/bursts/diag_burst.sh <TRIGGER> --before <s> --after <s> -o <PNG> [opts...]
#
# Required:
#   <TRIGGER>            MET number or UTC datetime (e.g. 2026-06-01T19:12:49.900)
#   --before <s>         Seconds before trigger
#   --after <s>          Seconds after trigger
#   -o, --output <PNG>   Output PNG path
#
# Plot pass-through (optional):
#   --bin <s>            Box-panel histogram bin width  (default 0.2)
#   --bin-zoom <s>       Zoom panel bin width            (default 0.002)
#   --zoom-box <X>       Per-det zoom box: A/B/C/auto/none (default auto)
#   --no-c25             Skip C25 engineering-model overlay
#   --c25-json <PATH>    C25 params JSON                 (default /tmp/per_det_25param.json)
#   --aacgm-grid <PATH>  AACGM mlat grid NPZ             (default n_below_study/aacgm_grid_2020.npz)
#
# Pack handling (optional):
#   --keep-pack [<DIR>]  Keep the pack; if DIR given, write there (else keep tmp dir)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Run end-to-end burst saturation diagnostic.

Usage:
  scripts/bursts/diag_burst.sh <TRIGGER> --before <s> --after <s> -o <PNG> [opts]

Required:
  <TRIGGER>            MET number or UTC datetime (e.g. 2026-06-01T19:12:49.900)
  --before <s>         Seconds before trigger
  --after <s>          Seconds after trigger
  -o, --output <PNG>   Output PNG path

Plot pass-through (optional):
  --bin <s>            Box-panel histogram bin width    (default 0.2)
  --bin-zoom <s>       Zoom panel bin width             (default 0.002)
  --zoom-box <X>       Per-det zoom box A/B/C/auto/none (default auto)
  --no-c25             Skip C25 engineering-model overlay
  --c25-json <PATH>    C25 params JSON  (default /tmp/per_det_25param.json)
  --aacgm-grid <PATH>  AACGM mlat grid  (default n_below_study/aacgm_grid_2020.npz)

Pack handling (optional):
  --keep-pack [<DIR>]  Keep the pack; if DIR given, write there
EOF
}

# ── arg parsing ────────────────────────────────────────────────────────────
if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then
    usage
    exit 0
fi

trigger="$1"; shift
before=""; after=""; output=""
bin=""; bin_zoom=""; zoom_box=""
c25_json=""; aacgm_grid=""; no_c25=""
keep_pack=""; pack_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --before)      before="$2"; shift 2 ;;
        --after)       after="$2"; shift 2 ;;
        -o|--output)   output="$2"; shift 2 ;;
        --bin)         bin="$2"; shift 2 ;;
        --bin-zoom)    bin_zoom="$2"; shift 2 ;;
        --zoom-box)    zoom_box="$2"; shift 2 ;;
        --no-c25)      no_c25=1; shift ;;
        --c25-json)    c25_json="$2"; shift 2 ;;
        --aacgm-grid)  aacgm_grid="$2"; shift 2 ;;
        --keep-pack)
            keep_pack=1
            # If next arg is a non-flag, treat as pack dir
            if [[ $# -gt 1 && "$2" != --* && "$2" != -* ]]; then
                pack_dir="$2"; shift 2
            else
                shift
            fi
            ;;
        *) echo "error: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

[[ -z "$before" || -z "$after" || -z "$output" ]] && {
    echo "error: --before, --after, -o are required" >&2
    echo "usage: $0 <TRIGGER> --before <s> --after <s> -o <PNG> [opts...]" >&2
    exit 2
}

# ── prerequisites ──────────────────────────────────────────────────────────
cli="./target/release/blink"
[[ -x "$cli" ]] || {
    echo "error: $cli not found; run 'cargo build -p blink --release' first" >&2
    exit 1
}
[[ -f scripts/bursts/plot_burst_report.py ]] || {
    echo "error: scripts/bursts/plot_burst_report.py not found" >&2
    exit 1
}

# ── pack dir setup ─────────────────────────────────────────────────────────
if [[ -z "$pack_dir" ]]; then
    pack_dir=$(mktemp -d -t burst_pack.XXXXXX)
fi
mkdir -p "$pack_dir"

cleanup() {
    if [[ -z "$keep_pack" ]]; then
        rm -rf "$pack_dir"
    else
        echo "Pack kept at $pack_dir" >&2
    fi
}
trap cleanup EXIT

# ── step 1: report ─────────────────────────────────────────────────────────
echo "[1/2] blink sat report → $pack_dir" >&2
"$cli" sat report "$trigger" \
    --before "$before" --after "$after" -o "$pack_dir"

# ── step 2: plot ───────────────────────────────────────────────────────────
plot_args=(--pack "$pack_dir" -o "$output")
[[ -n "$bin"        ]] && plot_args+=(--bin "$bin")
[[ -n "$bin_zoom"   ]] && plot_args+=(--bin-zoom "$bin_zoom")
[[ -n "$zoom_box"   ]] && plot_args+=(--zoom-box "$zoom_box")
[[ -n "$no_c25"     ]] && plot_args+=(--no-c25)
[[ -n "$c25_json"   ]] && plot_args+=(--c25-json "$c25_json")
[[ -n "$aacgm_grid" ]] && plot_args+=(--aacgm-grid "$aacgm_grid")

echo "[2/2] plot_burst_report → $output" >&2
python3 scripts/bursts/plot_burst_report.py "${plot_args[@]}"

echo "Done." >&2
