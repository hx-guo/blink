#!/usr/bin/env python3
"""GRB 221009A: per-1-s variance of the recovered (observed + gap-fill) count.

Uses the three-table recovery error model unchanged (analysis/recovery_cov.py,
Cov = S diag(C) S^T + k-term + r-term, include_u=False). The tables come from

    blink sat reconstruct 339945422 --before 50 --after 753 \
        --gapcov-out  data/p221009/errors/gapcov.csv \
        --gapbins-out data/p221009/errors/gapbins.csv > data/p221009/errors/events.csv

(events.csv is byte-identical to data/cache_221009a_reconstruct.csv).

The full window is too large for one 1-ms assembly, so each 1-s bin that
touches a gap is assembled on its own, padded so that every gap overlapping
the bin is contained whole. The central-bin variance is then exact: its rows
in S only reference cells of the same gaps (measured cells: same time;
empty cells: interpolation end points inside the same gap), and the k/r
terms are gap-local. Bins with no gap have Var = observed count (Poisson).

Output: data/p221009/errors/recvar_1s.csv
    t (bin left edge, s rel. T0=339945423), n_all, n_obs, n_fill,
    var_rec (full model), var_obs (=n_obs), n_gap_cross, n_gap_degen
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
import recovery_cov as rc  # noqa: E402

T0 = 339945423.0
OUT = Path("data/p221009/errors")
BEFORE, AFTER = 50, 700


def load_tables():
    """gapcov + gapbins with a box-unique gap key.

    gap_id restarts at 0 for every target box, but gapbins.csv carries no box
    column and rc.load_bins keys by gap_id alone, so with box=None the bins of
    A/B/C gaps sharing an id would overwrite each other. Both tables are
    written in the same loop order (box, then gap), and only cross-ref gaps
    have bins, so the k-th run of equal gap_id in gapbins belongs to the k-th
    cross-ref row of gapcov; this is asserted, then every gap is re-keyed as
    box_index * 1_000_000 + gap_id.
    """
    blocks = rc.load_blocks(str(OUT / "gapcov.csv"))
    rows = pd.read_csv(OUT / "gapbins.csv", keep_default_na=False, dtype=str)
    run = (rows["gap_id"] != rows["gap_id"].shift()).cumsum().to_numpy()
    cross = [b for b in blocks if b["type"] == "crossref"]
    n_runs = int(run.max())
    assert n_runs == len(cross), (n_runs, len(cross))
    boxes = sorted({b["target_box"] for b in blocks})
    key = {bx: i * 1_000_000 for i, bx in enumerate(boxes)}
    bins = {}
    run_first = np.concatenate(([0], np.where(np.diff(run) != 0)[0] + 1, [len(rows)]))
    for r in range(n_runs):
        blk = cross[r]
        seg = rows.iloc[run_first[r]:run_first[r + 1]]
        assert int(seg["gap_id"].iloc[0]) == blk["gap_id"]
        gid = key[blk["target_box"]] + blk["gap_id"]
        d = {}
        for _, row in seg.iterrows():
            d[int(row["bin_index"])] = {
                "t_lo": float(row["t_lo"]),
                "n_m": int(row["n_m"]) if row["n_m"] != "" else None,
                "kind": row["kind"],
                "left_bin": int(row["left_bin"]) if row["left_bin"] != "" else None,
                "right_bin": int(row["right_bin"]) if row["right_bin"] != "" else None,
                "tau": float(row["tau"]) if row["tau"] != "" else None,
            }
        bins[gid] = d
    for b in blocks:
        b["gap_id"] = key[b["target_box"]] + b["gap_id"]
    return blocks, bins


def main():
    t_start = time.time()
    ev = pd.read_csv(OUT / "events.csv", usecols=["box", "type", "met", "channel"],
                     dtype={"box": "category", "type": "category",
                            "met": "float64", "channel": "int32"})
    ev = ev.sort_values("met", kind="stable").reset_index(drop=True)
    met = ev["met"].to_numpy()
    box = ev["box"].astype(str).to_numpy()
    typ = ev["type"].astype(str).to_numpy()
    chan = ev["channel"].to_numpy()
    print(f"loaded {len(met)} rows in {time.time()-t_start:.0f}s", file=sys.stderr)

    blocks, bins = load_tables()
    g_lo = np.array([b["t_start"] for b in blocks])
    g_hi = np.array([b["t_stop"] for b in blocks])

    edges = T0 + np.arange(-BEFORE, AFTER + 1, 1.0)
    x = edges[:-1] - T0
    is_evt = typ == "EVT"
    n_all = np.histogram(met, bins=edges)[0].astype(float)
    n_obs = np.histogram(met[is_evt], bins=edges)[0].astype(float)
    n_fill = n_all - n_obs
    var_rec = n_obs.copy()
    n_gc = np.zeros(len(x), int)
    n_gd = np.zeros(len(x), int)

    for i in range(len(x)):
        a, b = edges[i], edges[i + 1]
        hit = (g_hi > a) & (g_lo < b)
        if not hit.any():
            continue
        lo = min(a, g_lo[hit].min()) - 1e-4
        hi = max(b, g_hi[hit].max()) + 1e-4
        sub_edges = np.array(sorted({lo, a, b, hi}))
        j0, j1 = np.searchsorted(met, [lo, hi])
        sub_ev = {"box": box[j0:j1], "type": typ[j0:j1],
                  "met": met[j0:j1], "channel": chan[j0:j1]}
        # gaps fully or partly inside [lo, hi]; only those overlapping the
        # central bin matter for its variance, the rest only touch pad bins.
        inwin = np.where((g_hi > lo) & (g_lo < hi))[0]
        sub_blocks = [blocks[k] for k in inwin]
        sub_bins = {blocks[k]["gap_id"]: bins[blocks[k]["gap_id"]]
                    for k in inwin if blocks[k]["gap_id"] in bins}
        N, cov = rc.cov_matrix(sub_ev, sub_blocks, sub_bins, sub_edges,
                               box=None, include_u=False)
        ci = int(np.searchsorted(sub_edges, a))
        if abs(N[ci] - n_all[i]) > 0.5:
            raise RuntimeError(f"bin {x[i]}: mean mismatch {N[ci]} vs {n_all[i]}")
        var_rec[i] = cov[ci, ci]
        n_gc[i] = sum(blocks[k]["type"] == "crossref" for k in np.where(hit)[0])
        n_gd[i] = sum(blocks[k]["type"] != "crossref" for k in np.where(hit)[0])
        print(f"  t={x[i]:+.0f}  gaps {n_gc[i]}c/{n_gd[i]}d  "
              f"var/n_obs={var_rec[i]/max(n_obs[i],1):.3f}  "
              f"({time.time()-t_start:.0f}s)", file=sys.stderr)

    pd.DataFrame({"t": x, "n_all": n_all, "n_obs": n_obs, "n_fill": n_fill,
                  "var_rec": var_rec, "var_obs": n_obs,
                  "n_gap_cross": n_gc, "n_gap_degen": n_gd}).to_csv(
        OUT / "recvar_1s.csv", index=False, float_format="%.6g")
    print(f"wrote {OUT/'recvar_1s.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
