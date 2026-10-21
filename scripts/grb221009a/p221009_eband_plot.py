#!/usr/bin/env python3
"""GRB 221009A: energy-resolved HE figures (candidate + diagnostic).

  eband_candidate.png : (a) dead-time-corrected net CsI band rates (obs + fill),
                        (b) hardness ratios from observed events, seconds with
                            measurable pile-up greyed out,
                        (c) pile-up indicator (pulse-width widening at fixed channel).
  eband_hr_vs_pileup.png : HR1 vs the pile-up indicator, coloured by phase.

Inputs: data/p221009/eband/hr_1s_const.csv (p221009_eband_analysis.py)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
import pubstyle  # noqa: E402

OUT = Path("data/p221009/eband")
# Channel bands. Edges were set with the normal-gain CsI LUT (70/150/300/700 keV ->
# raw ch 30/53/99/224), but HE ran in low-gain (HV ~-815 V vs ~-970 V) mode during the
# burst, so deposited energies are ~3-4x higher; labels give channels + approx. MeV.
BANDS = [("c70_150", "ch 30–53 (~0.2–0.5 MeV)", "#0072B2"),
         ("c150_300", "ch 53–99 (~0.5–1 MeV)", "#E69F00"),
         ("c300_700", "ch 99–224 (~1–2.5 MeV)", "#009E73")]
PU_CUT = 0.01          # pile-up indicator excess above background
HE_OFF = [(230, 249), (268, 277)]
XL = (170, 620)


def main():
    pubstyle.apply()
    a = pd.read_csv(OUT / "hr_1s_const.csv")
    t = a.t.to_numpy() + 0.5
    off = a.he_off.astype(bool).to_numpy()
    bk = (a.t >= -55) & (a.t < -5)
    clean = (a.pu_excess < PU_CUT) & (a.src_frac > 0.5) & ~off

    fig, ax = plt.subplots(3, 1, figsize=(pubstyle.COL_W, 4.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3, 1.2, 0.8], hspace=0.08))
    for x in ax:
        for lo, hi in HE_OFF:
            x.axvspan(lo, hi, color="0.85", lw=0, zorder=0)
    for col, lab, c in BANDS:
        rec = a["r_" + col]
        B = rec[bk].mean()
        corr = np.where(a.flive > 0.05, rec / a.flive, np.nan) - B
        corr = np.where(off, np.nan, corr)
        ax[0].step(t, np.where(corr > 0, corr, np.nan), where="mid", color=c, lw=0.9, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_ylim(3e2, 1e7)
    ax[0].set_ylabel("net rate (cts/s)")
    ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.62), frameon=False, fontsize=6)
    ax[0].text(0.02, 0.9, "(a) CsI (pw>90) bands, dead-time corrected", transform=ax[0].transAxes, fontsize=7)

    for name, c, lab in (("HR1", "#0072B2", "HR1 = ch 53–99 / 30–53"),
                         ("HR2", "#009E73", "HR2 = ch 99–224 / 53–99")):
        y, e = a[name], a[name + "_err"]
        ok = clean & (e < 0.1)
        ax[1].errorbar(t[ok], y[ok], e[ok], fmt="o", ms=1.8, lw=0.5, color=c, label=lab)
        bad = ~clean & ~off & (a.src_frac > 0.5)
        ax[1].plot(t[bad], y[bad], "o", ms=2.2, mfc="none", mec="0.55", mew=0.5)
    ax[1].plot([], [], "o", ms=2.2, mfc="none", mec="0.55", label="pile-up affected")
    ax[1].set_ylim(0.3, 2.4)
    ax[1].set_ylabel("hardness ratio")
    ax[1].legend(loc="upper right", frameon=False, fontsize=6)
    ax[1].text(0.02, 0.9, "(b) observed events only", transform=ax[1].transAxes, fontsize=7)

    pu = np.where(off, np.nan, a.pu_frac)
    ax[2].step(t, pu, where="mid", color="#20347e", lw=0.9)
    f_bkg = a.pu_frac[bk].mean()
    ax[2].axhline(f_bkg + PU_CUT, color="#D62728", lw=0.7, ls="--")
    ax[2].axhline(f_bkg, color="0.5", lw=0.6, ls=":")
    ax[2].set_yscale("log")
    ax[2].set_ylim(2e-3, 0.6)
    ax[2].set_ylabel("wide-pulse frac.")
    ax[2].text(0.02, 0.82, "(c) pile-up indicator", transform=ax[2].transAxes, fontsize=7)
    ax[2].set_xlabel(r"$t-T_0$ (s)")
    ax[2].set_xlim(*XL)
    fig.savefig(OUT / "eband_candidate.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "eband_candidate.pdf", bbox_inches="tight")

    # ---- HR1 vs pile-up indicator ----
    fig, axs = plt.subplots(figsize=(pubstyle.COL_W, 2.6))
    phases = [("main rise 180–219", 180, 219, "#0072B2"), ("P1 219–230", 219, 230, "#D62728"),
              ("P2 249–268", 249, 268, "#E69F00"), ("post-P2 277–440", 277, 440, "#7d4fd0"),
              ("flare 440–600", 440, 600, "#009E73")]
    for lab, lo, hi, c in phases:
        s = (a.t >= lo) & (a.t < hi) & ~off & (a.src_frac > 0.5) & (a.HR1_err < 0.1)
        axs.errorbar(a.pu_excess[s].clip(lower=1e-4), a.HR1[s], a.HR1_err[s], fmt="o", ms=2.2,
                     lw=0.5, color=c, label=lab)
    axs.axvline(PU_CUT, color="#D62728", ls="--", lw=0.7)
    axs.set_xscale("log")
    axs.set_xlabel("pile-up indicator excess")
    axs.set_ylabel("HR1 (ch 53–99 / 30–53)")
    axs.legend(frameon=False, fontsize=6, loc="upper left")
    fig.savefig(OUT / "eband_hr_vs_pileup.png", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "eband_candidate.png", OUT / "eband_hr_vs_pileup.png")


if __name__ == "__main__":
    main()
