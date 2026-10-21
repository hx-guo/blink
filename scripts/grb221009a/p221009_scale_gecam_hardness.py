#!/usr/bin/env python3
"""GRB 221009A: GECAM-C GRD01 hardness evolution, GECAM dead time, and the
HXMT/GECAM ratio per phase (tests of the x200-scale spectral-drift story).

GRD01 BTIME channels (GRD01 EBOUNDS): HG ch0-5 = 0-14,14-30,30-61,61-129,
129-223,223-351 keV, ch6 HG overflow; LG ch7-8 empty, ch9-12 = 0.38-0.78,
0.78-1.50, 1.50-3.21, 3.21-6.28 MeV, ch13 LG overflow.
The daily event file has a GTI hole over T0-68..T0+330 s, so only BTIME
(50 ms, with per-row HG/LG EXPOSURE) covers the main pulse.

Outputs (data/p221009/scale/): gecam_1s.csv, phases.csv, gecam_hardness.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from p221009_scale_common import (OUT, PHASES, SCALE, gecam_btime_channels,
                                  hxmt_1s, rebin_sum, robust, saa_mask)

LIVE_BKG = 0.993  # LG/HG-quiet live fraction of the background orbit (BTIME)


def main():
    edges = np.arange(-50, 701, 1.0)
    x = edges[:-1]
    h = hxmt_1s(edges)
    tg, c, cb, e = gecam_btime_channels()
    C = rebin_sum(tg, c, edges)            # gross counts / s (20 rows)
    CB = rebin_sum(tg, cb, edges)          # background counts / s
    nrow = rebin_sum(tg, np.ones_like(tg), edges)
    E = rebin_sum(tg, e, edges)            # live seconds (HG, LG)
    T = nrow * 0.05
    live = E / T[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = C / T[:, None]
        brate = CB / T[:, None]
        # dead-time corrected net per channel: gross/live - bkg/live_bkg
        net_raw = rate - brate
        net_dc = np.empty_like(rate)
        net_dc[:, :7] = rate[:, :7] / live[:, [0]] - brate[:, :7] / LIVE_BKG
        net_dc[:, 7:] = rate[:, 7:] / live[:, [1]] - brate[:, 7:] / LIVE_BKG

    lg_raw = net_raw[:, 9:13].sum(1)
    lg_dc = net_dc[:, 9:13].sum(1)
    hg_dc = net_dc[:, 2:5].sum(1)          # 30-223 keV, avoids the soft ch0-1 residual

    # ---- 1-s table ----
    cols = dict(t=x, he_obs=h["r_obs"], he_net=h["net"], he_net_dt=h["net_dt"],
                he_flive=h["flive"], lg_raw=lg_raw, lg_dc=lg_dc,
                live_hg=live[:, 0], live_lg=live[:, 1], hg_dc_30_223=hg_dc)
    for k in range(14):
        cols[f"ch{k}_gross"] = rate[:, k]
        cols[f"ch{k}_bkg"] = brate[:, k]
    hdr = ",".join(cols)
    np.savetxt(OUT / "gecam_1s.csv", np.column_stack(list(cols.values())),
               delimiter=",", header=hdr, comments="", fmt="%.6g")

    # ---- phase table ----
    bad = saa_mask(x)
    rows = []
    print(f"{'phase':<20}{'HR10/9':>14}{'HR(11+12)/(9+10)':>20}{'HG30-223/LG':>16}"
          f"{'LGlive':>12}{'HGlive':>8}{'HE/G raw':>16}{'HE/G LGdc':>16}{'n':>4}")
    for name, (lo, hi) in PHASES.items():
        w = (x >= lo) & (x < hi)
        g = C[w].sum(0); b = CB[w].sum(0)
        lv = E[w].sum(0) / T[w].sum()
        tt = T[w].sum()
        # counts-level net, dead-time corrected with the phase-mean live fraction
        def ch(ks, gain):
            n = (g[ks].sum() / lv[gain] - b[ks].sum() / LIVE_BKG)
            v = (g[ks].sum() / lv[gain] ** 2 + b[ks].sum() / LIVE_BKG ** 2)
            return n, v
        n9, v9 = ch([9], 1); n10, v10 = ch([10], 1); n1112, v1112 = ch([11, 12], 1)
        nlg, vlg = ch([9, 10, 11, 12], 1); nhg, vhg = ch([2, 3, 4], 0)
        hr1 = n10 / n9; ehr1 = hr1 * np.sqrt(v10 / n10**2 + v9 / n9**2)
        hr2 = n1112 / (n9 + n10)
        ehr2 = hr2 * np.sqrt(v1112 / n1112**2 + (v9 + v10) / (n9 + n10)**2)
        hgl = nhg / nlg; ehgl = hgl * np.sqrt(vhg / nhg**2 + vlg / nlg**2)
        # per-bin HXMT/GECAM, same selection as make_f10 (GECAM net > 50, HE net > 0)
        sel = w & ~bad & (lg_raw > 50) & (h["net"] > 0)
        r_raw = robust(h["net_dt"][sel] / (SCALE * lg_raw[sel]))
        r_dc = robust(h["net_dt"][sel] / (SCALE * lg_dc[sel]))
        lmin = np.nanmin(live[w, 1])
        rows.append((name, lo, hi, hr1, ehr1, hr2, ehr2, hgl, ehgl, lv[1], lmin, lv[0],
                     r_raw[0], r_raw[1], r_dc[0], r_dc[1], r_raw[2], nlg / 1.0))
        print(f"{name:<20}{hr1:7.3f}±{ehr1:5.3f}{hr2:12.3f}±{ehr2:5.3f}"
              f"{hgl:9.2f}±{ehgl:5.2f}{lv[1]:7.3f}/{lmin:4.2f}{lv[0]:8.3f}"
              f"{r_raw[0]:9.3f}±{r_raw[1]:4.2f}{r_dc[0]:9.3f}±{r_dc[1]:4.2f}{r_raw[2]:4d}")
    with open(OUT / "phases.csv", "w") as f:
        f.write("phase,lo,hi,hr_10_9,e,hr_1112_910,e,hg_lg,e,lg_live_mean,lg_live_min,"
                "hg_live_mean,he_g_raw_med,sig,he_g_lgdc_med,sig,n,lg_net_dc_counts\n")
        for r in rows:
            f.write(",".join(str(v) if isinstance(v, str) else f"{v:.6g}" for v in r) + "\n")

    # whole-window ratios as quoted in the paper, with/without GECAM LG dead time
    for lo, hi in [(170, 310), (440, 610)]:
        sel = (x >= lo) & (x < hi) & (lg_raw > 50) & (h["net"] > 0) & ~bad
        print(f"window {lo}-{hi}: HE_dt/G_raw = {robust(h['net_dt'][sel]/(SCALE*lg_raw[sel]))}"
              f"  HE_dt/G_dc = {robust(h['net_dt'][sel]/(SCALE*lg_dc[sel]))}")

    # ---- figure: adaptive 1-s/5-s hardness ----
    e5 = np.arange(-50, 701, 5.0)
    x5 = e5[:-1]
    C5 = rebin_sum(tg, c, e5); B5 = rebin_sum(tg, cb, e5)
    E5 = rebin_sum(tg, e, e5); T5 = rebin_sum(tg, np.ones_like(tg), e5) * 0.05
    with np.errstate(divide="ignore", invalid="ignore"):
        lv5 = E5 / T5[:, None]
        n5 = np.empty_like(C5)
        n5[:, :7] = C5[:, :7] / lv5[:, [0]] - B5[:, :7] / LIVE_BKG
        n5[:, 7:] = C5[:, 7:] / lv5[:, [1]] - B5[:, 7:] / LIVE_BKG
        hr1_5 = n5[:, 10] / n5[:, 9]
        hr2_5 = n5[:, 11:13].sum(1) / n5[:, 9:11].sum(1)
        hgl_5 = n5[:, 2:5].sum(1) / n5[:, 9:13].sum(1)
        e_hr1 = hr1_5 * np.sqrt((C5[:, 10] + B5[:, 10]) / n5[:, 10]**2
                                + (C5[:, 9] + B5[:, 9]) / n5[:, 9]**2)
        e_hgl = hgl_5 * np.sqrt((C5[:, 2:5].sum(1) + B5[:, 2:5].sum(1)) / n5[:, 2:5].sum(1)**2
                                + (C5[:, 9:13].sum(1) + B5[:, 9:13].sum(1)) / n5[:, 9:13].sum(1)**2)
    good = (n5[:, 9:13].sum(1) > 300) & (x5 >= 175)
    hg_ok = good & (lv5[:, 0] > 0.8)

    fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True,
                           gridspec_kw={"hspace": 0.06})
    ax[0].step(x, np.where(h["net_dt"] > 0, h["net_dt"], np.nan), where="post",
               color="C0", lw=1, label="HE rec, dead-time corrected")
    ax[0].step(x, np.where(lg_raw > 0, SCALE * lg_raw, np.nan), where="post",
               color="C1", lw=1, label="GECAM LG x200 (as in paper)")
    ax[0].step(x, np.where(lg_dc > 0, SCALE * lg_dc, np.nan), where="post",
               color="C3", lw=0.8, ls="--", label="GECAM LG x200, LG dead time corrected")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e3, 1e7); ax[0].set_ylabel("net rate")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].axvspan(-50, 172, color="0.6", alpha=0.15)
    ax[0].text(166, 2e3, "GECAM-C\nSAA", ha="center", fontsize=7, color="0.3")
    ax[1].errorbar(x5[good] + 2.5, hr1_5[good], yerr=e_hr1[good], fmt="o", ms=3, color="C2")
    ax[1].set_ylabel("LG ch10/ch9\n(0.78-1.5)/(0.38-0.78) MeV")
    ax[1].set_ylim(0, 1.2)
    ax[2].errorbar(x5[good] + 2.5, hr2_5[good], fmt="s", ms=3, color="C4")
    ax[2].set_ylabel("LG (1.5-6.3)/(0.38-1.5) MeV"); ax[2].set_ylim(0, 0.6)
    ax[3].errorbar(x5[hg_ok] + 2.5, hgl_5[hg_ok], yerr=e_hgl[hg_ok], fmt="o", ms=3,
                   color="C5", label="HG live > 0.8")
    gg = good & ~hg_ok
    ax[3].plot(x5[gg] + 2.5, hgl_5[gg], "x", color="0.5", label="HG live < 0.8 (unreliable)")
    ax[3].set_yscale("log"); ax[3].set_ylabel("HG(30-223 keV)/LG\n(dead-time corr.)")
    ax[3].legend(fontsize=8)
    ax[4].plot(x, live[:, 1], color="C1", label="GECAM LG live")
    ax[4].plot(x, live[:, 0], color="C5", label="GECAM HG live")
    ax[4].plot(x, h["flive"], color="C0", label="HE mean live")
    ax[4].set_ylim(0, 1.05); ax[4].set_ylabel("live fraction"); ax[4].legend(fontsize=8)
    for a in ax:
        for lo, hi in PHASES.values():
            a.axvline(lo, color="0.85", lw=0.6)
        a.grid(alpha=0.2)
    ax[4].set_xlim(160, 700)
    ax[4].set_xlabel("t - T0 (s), T0 = 2022-10-09T13:17:00 UTC")
    fig.savefig(OUT / "gecam_hardness.png", dpi=150, bbox_inches="tight")
    print("saved", OUT / "gecam_hardness.png")


if __name__ == "__main__":
    main()
