#!/usr/bin/env python3
"""Sketch figure: GRB 221009A main pulse, dead-time-corrected HE light curve
with statistical and systematic error bands, GECAM-C x200 with errors, the
per-bin ratio, and the fractional error terms. Reads the CSVs written by
p221009_err_budget.py / p221009_err_ratio.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/p221009/errors")
lc = pd.read_csv(OUT / "lc_221009a_1s_errors.csv")
rb = pd.read_csv(OUT / "ratio_bins_main_clean.csv")
lo, hi = 180, 300
d = lc[(lc.t >= lo) & (lc.t < hi)]
x = d.t + 0.5
y = d.dt_corr
tot = np.sqrt(d.sig_stat ** 2 + d.sig_sys ** 2)

fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.2, 1.4], "hspace": 0.06})
a1.fill_between(x, y - tot, y + tot, step="mid", color="C0", alpha=0.25, lw=0,
                label="stat ⊕ sys (1σ)")
a1.fill_between(x, y - d.sig_stat, y + d.sig_stat, step="mid", color="C0", alpha=0.6, lw=0,
                label="stat (1σ)")
a1.step(x, y, where="mid", color="navy", lw=0.9, label="HE dead-time corrected")
g = d.gecam_net * 200
a1.errorbar(x, g, yerr=d.gecam_sig * 200, fmt="none", ecolor="C1", elinewidth=0.8)
a1.step(x, g, where="mid", color="C1", lw=0.8, label="GECAM-C GRD01 LG ×200")
a1.set_ylabel("Net rate (evt/s)")
a1.legend(fontsize=8, loc="upper right")
a1.set_title("GRB 221009A main pulse, 1 s bins — error budget sketch", fontsize=10)

a2.errorbar(rb.t + 0.5, rb.r, yerr=rb.s_stat, fmt="o", ms=2.5, color="k", elinewidth=0.8,
            label="stat")
a2.errorbar(rb.t + 0.5, rb.r, yerr=rb.s_all, fmt="none", ecolor="0.6", elinewidth=0.6,
            label="stat ⊕ sys")
a2.axhline(1, color="gray", lw=0.5, ls="--")
a2.set_ylim(0.4, 1.8)
a2.set_ylabel("HXMT/GECAM")
a2.legend(fontsize=7, loc="upper right", ncol=2)

ok = np.isfinite(y) & (y > 0)
for col, lab, c in (("sig_obs", "obs. Poisson", "C0"), ("sig_fill", "gap-fill", "C2"),
                    ("sig_align", "frame alignment (sys)", "C3"),
                    ("sig_degsys", "degenerate-gap ramp (sys)", "C4"),
                    ("sig_det", "detector averaging (sys)", "C5"),
                    ("sig_bsys", "background drift (sys)", "C7")):
    a3.step(x, np.where(ok, d[col] / y, np.nan), where="mid", lw=0.9, color=c, label=lab)
a3.set_yscale("log")
a3.set_ylim(1e-4, 1)
a3.set_ylabel("σ / R_src")
a3.set_xlabel("Time since T0 (s)")
a3.legend(fontsize=7, ncol=3, loc="upper right")
for ax in (a1, a2, a3):
    ax.axvspan(231, 248, color="0.85", lw=0)
    ax.axvspan(269, 276, color="0.85", lw=0)
fig.savefig(OUT / "fig_mainpulse_errors.png", dpi=150, bbox_inches="tight")
print("saved", OUT / "fig_mainpulse_errors.png")
