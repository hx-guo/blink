#!/usr/bin/env python3
"""GRB 221009A: HXMT/GECAM ratios with the timing group's time-varying HE
background (p221009_timing_bkg.background_clean: 6065 + 1.084*Cnt_LargeEvt,
interpolated across the contaminated main pulse / flare spike) in place of the
constant B = 9332 evt/s. GECAM LG dead-time corrected reference and binning as
in p221009_scale_gecam_dt.py. Output: bkgdrift.txt (data/p221009/scale/)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from p221009_scale_common import OUT, SCALE, robust, saa_mask
from p221009_timing_bkg import background_clean


def main():
    d = pd.read_csv(OUT / "gecam_1s.csv")
    g = pd.read_csv(OUT / "gecam_lg_reference_1s.csv")
    x = d.t.to_numpy()
    tl, r, m, ok, coef, resid = background_clean()
    Bt = np.interp(x, tl, m)
    B0 = 9331.5
    gross_dt = d.he_net_dt.to_numpy() + B0          # r_all / f_live
    gross = d.he_net.to_numpy() + B0
    net_dt = {"B const 9332": gross_dt - B0, "B(t) timing": gross_dt - Bt}
    net = {"B const 9332": gross - B0, "B(t) timing": gross - Bt}
    bad = saa_mask(x)
    lines = [f"timing bkg coef {coef}, resid {resid:.0f}; mean B(t)-B0: "
             + ", ".join(f"{a}-{b}: {np.mean(Bt[(x>=a)&(x<b)]-B0):+.0f}"
                         for a, b in [(-50, -10), (170, 310), (300, 440), (440, 610), (600, 700)])]
    lines.append(f"{'window':<16}{'HE bkg':<15}{'/LG raw':>16}{'/LG DT-corr':>16}{'n':>4}")
    for name, lo, hi in [("main 170-310", 170, 310), ("valley 277-300", 277, 300),
                         ("flare 440-610", 440, 610), ("flare pk 500-520", 500, 520)]:
        for bn in net_dt:
            sel = (x >= lo) & (x < hi) & (g.lg_net > 50) & (net["B const 9332"] > 0) & ~bad
            a = robust(net_dt[bn][sel] / (SCALE * g.lg_net[sel]))
            b = robust(net_dt[bn][sel] / (SCALE * g.lg_net_dc[sel]))
            lines.append(f"{name:<16}{bn:<15}{a[0]:>9.3f}±{a[1]:.3f}{b[0]:>9.3f}±{b[1]:.3f}{a[2]:>4}")
    # scale-flux faint decades with B(t)
    lines.append("scale-flux DT-corr decade medians (x1e-3) on LG DT-corr: B const | B(t)")
    for lo, hi in zip([30, 100, 300, 1000, 3000, 10000], [100, 300, 1000, 3000, 10000, 30000]):
        row = f"{lo:>6}-{hi:<6}"
        for bn in net_dt:
            gg = g.lg_net_dc.to_numpy()
            w = (net[bn] > 5e3) & (gg > 30) & ~bad & (gg >= lo) & (gg < hi)
            row += f"{np.nanmedian(gg[w] / net_dt[bn][w]) * 1e3:>8.2f} (n={w.sum():3d})"
        lines.append(row)
    txt = "\n".join(lines) + "\n"
    (OUT / "bkgdrift.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
