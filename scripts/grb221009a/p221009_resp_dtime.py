"""Dead-time counter systematics at the deepest seconds (referee2 M4b;
GRB 221009A).

Per box, the six DeadTime_PHODet_i counters (1B HE_Eng) should see the same
box-level busy time (A1005 3.8 / 3.14.1), yet they differ by 2-7% within a
box.  Here we characterise that spread and its effect on the dead-time
correction at f ~ 0.11:
  1. D_i / <D>_box per channel in several phases (pre-burst, rise, P1, P2,
     flare, tail): is the pattern fixed (multiplicative) or rate dependent?
  2. per second, across the six channels: D_i = c0 + c1 * n_i, n_i the
     channel's observed 1K event rate -- is the excess tied to the channel's
     own rate (per-channel occupancy) or not?
  3. the corrected rate with per-channel live fractions, sum_i n_i/f_i, and
     with min/max channel D, relative to the box-mean form N/<f> used in v3.
Counter resolution / quantisation numbers follow the design document
(D: 8 MHz enable, reported in 16-us units; L_cyc: 1 MHz enable).

    python p221009_resp_dtime.py [OUTDIR]"""
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

BASE = "/scratchfs2/gecam/guohx/p221009"
os.chdir(f"{BASE}/blink")
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION  # noqa: E402

T0 = 339945423.0
TR = np.arange(-60, 701)
PHASES = {"pre -60..-10": (-60, -10), "rise 180-216": (180, 216), "P1 226-228": (226, 229),
          "P2 257-264": (257, 265), "flare 440-508": (440, 508), "tail 600-700": (600, 700)}


def eng(box, code):
    fn = sorted(Path(f"data/1B/2022/20221009/{code}").glob(f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
    d = fits.open(fn)["HE_Eng"].data
    off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    t = np.round(d["Time"].astype(float) + off + MET_CORRECTION - T0).astype(int)
    g = [BOX_OFFSET[box] + k for k in range(6)]
    D = np.vstack([d[f"DeadTime_PHODet_{i}"] for i in g]).astype(float)
    L = d["Length_Time_Cycle"].astype(float)
    out = {}
    for k, tt in enumerate(t):
        out[tt] = (D[:, k], L[k])
    return out, g


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    z = np.load(f"{BASE}/cache/he1k_hour.npz")
    t, det = z["t"], z["det"].astype(int)
    s = (t >= TR[0]) & (t < TR[-1] + 1)
    n = np.zeros((18, TR.size))
    np.add.at(n, (det[s], np.floor(t[s] - TR[0]).astype(int)), 1)
    lines = []
    rows = []
    for box, code in BOX_CODE.items():
        e, g = eng(box, code)
        Dm = np.full((6, TR.size), np.nan)
        Lm = np.full(TR.size, np.nan)
        for j, tt in enumerate(TR):
            if tt in e:
                Dm[:, j], Lm[j] = e[tt]
        nb = n[g]
        f_i = 1 - Dm / Lm
        f_m = 1 - Dm.mean(0) / Lm
        lines.append(f"box {box} (dets {g[0]}-{g[-1]})")
        for pn, (a, b) in PHASES.items():
            m = (TR >= a) & (TR < b) & np.isfinite(Lm)
            ratio = (Dm[:, m] / Dm[:, m].mean(0)).mean(1)
            share = (nb[:, m].sum(1) / nb[:, m].sum())
            fm = np.nanmean(f_m[m])
            lines.append(f"  {pn:14s} <f>={fm:.3f}  D_i/<D>: " + " ".join(f"{x:.3f}" for x in ratio)
                         + "   obs share x6: " + " ".join(f"{x * 6:.2f}" for x in share))
        # per-second regression across channels and per-channel correction
        for j, tt in enumerate(TR):
            if not np.isfinite(Lm[j]) or nb[:, j].sum() < 60:
                continue
            x = nb[:, j]
            y = Dm[:, j] / Lm[j]
            A = np.column_stack([np.ones(6), x / x.mean()])
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            r_box = x.sum() / f_m[j]
            r_ch = np.sum(x / f_i[:, j])
            fmin = 1 - Dm[:, j].max() / Lm[j]
            fmax = 1 - Dm[:, j].min() / Lm[j]
            rows.append((tt, box, f_m[j], r_ch / r_box, f_m[j] / fmin, f_m[j] / fmax, c[0], c[1],
                         np.corrcoef(x, y)[0, 1]))
    rows = np.array(rows, dtype=object)
    lines.append("PER-SECOND: R(per-channel f_i) / R(box-mean f), and R if all channels used min / max D")
    for pn, ab in list(PHASES.items()) + [("f<0.5 bins", None)]:
        a, b = ab if ab is not None else (None, None)
        if b is None:
            m = np.array([r[2] < 0.5 for r in rows])
        else:
            m = np.array([a <= r[0] < b for r in rows])
        if not m.any():
            continue
        sub = rows[m]
        q = np.array([r[3] for r in sub], float)
        lo = np.array([r[4] for r in sub], float)
        hi = np.array([r[5] for r in sub], float)
        c1 = np.array([r[7] for r in sub], float)
        cc = np.array([r[8] for r in sub], float)
        lines.append(f"  {pn:14s} n={m.sum():3d}: per-channel/box {np.median(q):.4f} [{q.min():.4f},{q.max():.4f}]; "
                     f"max-D/box {np.median(lo):.3f}; min-D/box {np.median(hi):.3f}; "
                     f"slope of D_i/L on own rate (per mean rate) {np.median(c1):+.4f}, corr {np.median(cc):+.2f}")
    for tt in (226, 227, 228, 262):
        sub = [r for r in rows if r[0] == tt]
        lines.append(f"  t={tt}: " + "; ".join(f"{r[1]} f={r[2]:.3f} ch/box {r[3]:.4f} maxD {r[4]:.3f} minD {r[5]:.3f}"
                                             for r in sub))
    # design-document resolution numbers at T0+227 s
    j = int(227 - TR[0])
    lines.append("RESOLUTION (A1005): D counts busy time with an 8 MHz enable (0.125 us), reported as bits 22..7 "
                 "(16-us LSB, truncated); L_cyc counts a 1 MHz enable over the same gated window (cntdis), same master clock.")
    with open(f"{out}/dtime_summary.txt", "w") as f:
        for l in lines:
            print(l, flush=True)
            print(l, file=f)
    np.save(f"{out}/dtime_rows.npy", rows, allow_pickle=True)


if __name__ == "__main__":
    main()
