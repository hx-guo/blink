#!/usr/bin/env python3
"""Direct comparison of Ge 2023 vs this-work recovery at Ge's native 1 ms binning.

Top panel: HXMT reconstructed light curve binned at 1 ms (with Poisson errors)
overlaid with Ge's complete digitized recovery (red + black markers, scaled to
our energy band by the empirical ×N factor derived from the gap-exterior bins).

Bottom panel: scatter (Ge_rate, HXMT_rate) for the 5 ms bins containing a Ge
marker — should fall on the y=x line if the two algorithms agree.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../bursts", "../lib",)]  # sibling and shared script modules
from engineering_prediction import load_engineering_prediction, T_REF
from plot_hxmt_vs_ibis import compute_hxmt_light_travel, HXMT_TRIGGER_UTC_STR

HXMT_CACHE = "data/cache_frb200428_reconstruct_3box.csv"
HXMT_ORBIT_FILE = "data/hxmt_aux/HXMT_20200428T14_Orbit_FFFFFF_V1_1K.FITS"
GE_RED = "data/ge2023/ge2023_fig1_red.csv"
GE_BLK = "data/ge2023/ge2023_fig1_black.csv"
OUT = "/Users/skyair/Developer/ihep/paper-hxmt-saturation/figures/f13_ge_vs_ours_1ms.pdf"
GE_TO_OURS_MS = 415.5


def main():
    # ---- HXMT trigger + light-travel ----
    burst_utc = Time(HXMT_TRIGGER_UTC_STR, scale="utc")
    burst_met = (burst_utc - Time("2012-01-01", scale="utc")).sec
    lt_hxmt = compute_hxmt_light_travel(burst_met, burst_utc, HXMT_ORBIT_FILE)
    print(f"HXMT->geocentric light-travel: {lt_hxmt*1000:+.2f} ms", file=sys.stderr)

    # ---- load HXMT cache ----
    import csv
    obs_t, fill_t = [], []
    with open(HXMT_CACHE) as f:
        r = csv.reader(f); next(r)
        for row in r:
            t_rel = float(row[2]) - burst_met + lt_hxmt
            (obs_t if row[1] == "EVT" else fill_t).append(t_rel)
    obs_t = np.array(obs_t); fill_t = np.array(fill_t)
    all_t = np.concatenate([obs_t, fill_t])
    print(f"HXMT: obs={len(obs_t):,}, fill={len(fill_t):,}", file=sys.stderr)

    # ---- load Ge ----
    ge_r = np.loadtxt(GE_RED, delimiter=",", skiprows=1)
    ge_b = np.loadtxt(GE_BLK, delimiter=",", skiprows=1)
    # column: T_ms_Ge, T_ms_ours, rate_1e4
    ge_red_t_s = ge_r[:, 1] / 1000.0
    ge_blk_t_s = ge_b[:, 1] / 1000.0
    ge_red_rate = ge_r[:, 2] * 1e4   # cnts/s, 15-250 keV
    ge_blk_rate = ge_b[:, 2] * 1e4

    # ---- bin HXMT at 1 ms ----
    BIN_MS = 1
    bin_w = BIN_MS / 1000.0
    edges = np.arange(-0.1, 0.7 + bin_w, bin_w)
    x = edges[:-1]
    r_hxmt_obs = np.histogram(obs_t, bins=edges)[0] / bin_w
    r_hxmt_all = np.histogram(all_t, bins=edges)[0] / bin_w

    # background from far pre-burst
    bkg_mask = (x < 0.1) | (x > 0.8)
    bkg_hxmt = r_hxmt_obs[bkg_mask].mean() if bkg_mask.any() else 0
    net_hxmt_obs = r_hxmt_obs - bkg_hxmt
    net_hxmt_all = r_hxmt_all - bkg_hxmt
    print(f"HXMT background at {BIN_MS} ms: {bkg_hxmt:.0f} cnts/s", file=sys.stderr)

    # ---- energy-band scale from gap-exterior Ge blacks ----
    # detect the FIFO gap from the fill events
    if len(fill_t) > 0:
        gap_lo = fill_t.min()
        gap_hi = fill_t.max()
    else:
        gap_lo, gap_hi = 0.377, 0.617
    print(f"FIFO gap: [{gap_lo*1000:.0f}, {gap_hi*1000:.0f}] ms", file=sys.stderr)

    outside_gap = (ge_blk_t_s < gap_lo) | (ge_blk_t_s > gap_hi)
    idx = np.searchsorted(edges, ge_blk_t_s) - 1
    valid = (idx >= 0) & (idx < len(x)) & outside_gap
    if valid.any():
        h_at_blk = r_hxmt_obs[idx[valid]]
        g_at_blk = ge_blk_rate[valid]
        usable = (g_at_blk > 5e3) & (h_at_blk > 5e3)
        if usable.sum() >= 3:
            eband = float(np.median(h_at_blk[usable] / g_at_blk[usable]))
            print(f"energy-band scale: ×{eband:.2f} from {usable.sum()} pts",
                  file=sys.stderr)
        else:
            eband = 1.0
    else:
        eband = 1.0

    # ---- scatter: at each Ge marker time, look up our HXMT rate ----
    def lookup_hxmt(t_s, arr=r_hxmt_all):
        # use nearest 1 ms bin
        i = np.searchsorted(edges, t_s) - 1
        if i < 0 or i >= len(arr): return np.nan
        return arr[i]

    ge_pts_t = np.concatenate([ge_red_t_s, ge_blk_t_s])
    ge_pts_rate = np.concatenate([ge_red_rate, ge_blk_rate]) * eband
    ge_pts_color = ["red"] * len(ge_red_t_s) + ["black"] * len(ge_blk_t_s)
    hxmt_at_ge = np.array([lookup_hxmt(t) for t in ge_pts_t])
    valid_pair = np.isfinite(hxmt_at_ge)

    # ---- plot ----
    fig, (ax_lc, ax_sc) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.2, 1]})

    # Top: 1 ms light curve overlay
    ax_lc.step(x * 1000, net_hxmt_all, where="post", color="C0", lw=1.0,
               alpha=0.85,
               label=f"HXMT/HE recovered (this work, 1 ms bins)")
    ax_lc.step(x * 1000, net_hxmt_obs, where="post", color="navy", lw=0.8,
               alpha=0.5, label="HXMT/HE observed (1 ms)")
    # Ge as markers
    ax_lc.plot((ge_red_t_s - 0.4155) * 1000, ge_red_rate * eband - bkg_hxmt,
               "o", color="crimson", markersize=4, markeredgewidth=0,
               alpha=0.85,
               label=f"Ge 2023 recovered (×{eband:.1f} band)")
    ax_lc.plot((ge_blk_t_s - 0.4155) * 1000, ge_blk_rate * eband - bkg_hxmt,
               "s", color="black", markersize=4, markerfacecolor="none",
               markeredgewidth=0.8, alpha=0.85,
               label=f"Ge 2023 normal (×{eband:.1f} band)")
    ax_lc.axvspan((gap_lo - 0.4155) * 1000, (gap_hi - 0.4155) * 1000,
                  color="orange", alpha=0.10, zorder=0,
                  label="FIFO saturation gap")
    ax_lc.set_xlim(-60, 130)
    ax_lc.set_ylim(-1e4, 2.5e5)
    ax_lc.set_xlabel(r"Time $-$ $T_0^{\rm Ge}$ (ms)")
    ax_lc.set_ylabel("Net rate (cnts/s)")
    ax_lc.legend(fontsize=8, loc="upper right")
    ax_lc.set_title(f"1 ms binning (= Ge's native resolution)")
    ax_lc.grid(alpha=0.2)

    # Right: scatter Ge vs Ours, only for bins INSIDE the gap (where the
    # recovery is the entire signal) — the agreement test
    ge_pts_t_arr = np.array(ge_pts_t)
    in_gap = (ge_pts_t_arr > gap_lo) & (ge_pts_t_arr < gap_hi)
    hxmt_net_at_ge = hxmt_at_ge - bkg_hxmt
    ge_net = ge_pts_rate - bkg_hxmt
    for col, mask, label in [
            ("crimson", in_gap, "Inside FIFO gap"),
            ("0.5", ~in_gap, "Outside gap")]:
        m = mask & valid_pair
        ax_sc.scatter(ge_net[m], hxmt_net_at_ge[m], c=col, s=18,
                      alpha=0.7, label=f"{label} (n={m.sum()})")
    lim = max(ax_sc.get_xlim()[1], ax_sc.get_ylim()[1], 1e5)
    ax_sc.plot([-1e4, lim], [-1e4, lim], "k--", lw=0.8, alpha=0.6, label="y=x")
    ax_sc.set_xlabel(f"Ge 2023 rate (×{eband:.1f} band, cnts/s)")
    ax_sc.set_ylabel("HXMT recovered rate at same time (cnts/s)")
    ax_sc.legend(fontsize=9)
    ax_sc.set_title("Point-by-point agreement")
    ax_sc.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"saved: {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
