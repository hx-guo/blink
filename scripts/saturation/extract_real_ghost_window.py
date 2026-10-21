#!/usr/bin/env python3
"""Extract the 21-event window centered on a real CRC-collision ghost
(GRB 221009A, Box A, ds=1 SEC pair k=1527, stime=160124920, ghost@2845)
and save the elapsed_fwd values to a tiny CSV so the plotter can use
real data instead of synthetic."""

import sys
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from find_real_ghost_cascade import load_1b, scan_for_cascade
import numpy as np
import glob

WINDOW = 21          # total events to keep
GHOST_OFFSET = 10    # ghost will end up at index GHOST_OFFSET in the window


def main():
    fp = sorted(glob.glob("data/1B/2022/20221009/0642/HXMT_1B_*.fits"))[0]
    pkts = load_1b(fp)
    cands = scan_for_cascade(pkts, min_cascade=1)
    # Pick the dominant cascade pair
    target = max(cands, key=lambda r: r["cascade"])
    ef = target["ef"]; al = target["accept_l"]; ag = target["accept_g"]
    print(f"Chosen pair: stime={target['st_a']}  cascade={target['cascade']}",
          file=sys.stderr)
    ghost_idx_full = int(np.argmax(ef * (~al)))
    print(f"Ghost at index {ghost_idx_full} of {len(ef)}; "
          f"ef={ef[ghost_idx_full]}", file=sys.stderr)

    lo = ghost_idx_full - GHOST_OFFSET
    hi = lo + WINDOW
    win_ef = ef[lo:hi].astype(np.int64)
    win_lis = al[lo:hi]
    # Greedy in the window: need the running_max coming in from before lo
    running_max_before = int(ef[:lo].max()) if lo > 0 else -1
    accept_g_local = np.zeros(WINDOW, dtype=bool)
    rmax = running_max_before
    for i, v in enumerate(win_ef):
        if v > rmax:
            accept_g_local[i] = True
            rmax = int(v)
    print(f"Window ef[{lo}..{hi}]: ghost at local index {GHOST_OFFSET}",
          file=sys.stderr)
    print(f"  values: {win_ef.tolist()}", file=sys.stderr)
    print(f"  greedy accept: {accept_g_local.tolist()}", file=sys.stderr)
    print(f"  LIS accept:    {win_lis.tolist()}", file=sys.stderr)
    print(f"  running_max coming into window: {running_max_before}",
          file=sys.stderr)
    n_greedy_reject = int((~accept_g_local & win_lis).sum())
    print(f"  In window: greedy rejects {n_greedy_reject} events that LIS keeps",
          file=sys.stderr)

    import json
    out = {
        "ef": win_ef.tolist(),
        "accept_greedy": accept_g_local.tolist(),
        "accept_lis": win_lis.tolist(),
        "ghost_local_idx": GHOST_OFFSET,
        "ghost_full_idx": ghost_idx_full,
        "full_n": len(ef),
        "cascade_full": int(((~ag) & al).sum()),
        "running_max_before": running_max_before,
        "stime": int(target["st_a"]),
    }
    with open("plots/fig2_real_ghost_window.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: plots/fig2_real_ghost_window.json", file=sys.stderr)


if __name__ == "__main__":
    main()
