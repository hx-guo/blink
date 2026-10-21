#!/usr/bin/env python3
"""Plot HXMT/HE reconstructed light curve vs Fermi/GBM + engineering-channel prediction.

Usage:
    python3 scripts/gbm/plot_hxmt_vs_gbm.py --bin 0.5
    python3 scripts/gbm/plot_hxmt_vs_gbm.py --bin 1.0 --before 20 --after 200

Adds a 4th trace (purple step) showing $\widehat{S}_{rec}^{eng}$ from the C25 model
applied to per-second engineering counters, summed over 18 detectors.
"""

import argparse, subprocess, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime, timezone

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from engineering_prediction import load_engineering_prediction, T_REF

# ── Config ──
HXMT_MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)
HXMT_TRIGGER_UTC = "2026-02-26T10:37:53"
HXMT_EPOCH = "2026-02-26T10"

GBM_DIR = "data/fermi_gbm/bn260226443"
GBM_TRIGGER_MET = 793795080.95811
GBM_DETS = ["n0", "n3", "b0", "b1"]

HXMT_TRIGGER_MET = (datetime.strptime(HXMT_TRIGGER_UTC, "%Y-%m-%dT%H:%M:%S")
                     .replace(tzinfo=timezone.utc) - HXMT_MET_EPOCH).total_seconds()
HXMT_TRIGGER_UTC_LABEL = "2026-02-26T10:37:50"
GBM_TO_HXMT_OFFSET = 5.958


def load_hxmt_reconstruct(before, after):
    cmd = ["./target/release/blink", "sat", "reconstruct", HXMT_TRIGGER_UTC,
           "--before", str(before), "--after", str(after)]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.stderr:
        for line in proc.stderr.strip().split("\n"):
            print(f"    {line}", file=sys.stderr)
    obs, fill = [], []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 3 or p[0] == "box":
            continue
        met = float(p[2])
        t = met - HXMT_TRIGGER_MET
        if p[1] == "EVT":
            obs.append(t)
        elif p[1] == "FILL_GAP":
            fill.append(t)
    return np.array(obs), np.array(fill)


def load_gbm_tte(det, before, after, emin=None, emax=None):
    path = os.path.join(GBM_DIR, f"glg_tte_{det}_bn260226443_v00.fit")
    if not os.path.exists(path):
        return np.array([])
    with fits.open(path, memmap=True) as f:
        d = f["EVENTS"].data
        times = d["TIME"]
        if emin is not None or emax is not None:
            eb = f["EBOUNDS"].data
            ecen = (eb["E_MIN"] + eb["E_MAX"]) / 2
            en = ecen[d["PHA"]]
            sel = np.ones(len(times), bool)
            if emin is not None: sel &= en >= emin
            if emax is not None: sel &= en <= emax
            times = times[sel]
    t = (times - GBM_TRIGGER_MET) + GBM_TO_HXMT_OFFSET
    mask = (t >= -before) & (t <= after)
    return t[mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=float, default=0.5)
    parser.add_argument("--before", type=float, default=10.0)
    parser.add_argument("--after", type=float, default=80.0)
    parser.add_argument("--det", type=str, nargs="+", default=["n0", "n3", "b0"])
    parser.add_argument("--emin", type=float, default=None, help="GBM energy min (keV)")
    parser.add_argument("--emax", type=float, default=None, help="GBM energy max (keV)")
    parser.add_argument("--xlim", type=float, nargs=2, default=None,
                        help="display x-range (data/statistics unaffected)")
    parser.add_argument("--scale-range", type=float, nargs=2, default=None,
                        help="window for the GBM scalar normalization "
                             "(default: the mid-window between the bkg segments)")
    parser.add_argument("--pub", action="store_true",
                        help="publication style: larger fonts, no title")
    parser.add_argument("--bkg", type=float, nargs=4, default=[-10, -2, 60, 80],
                        metavar=("T1", "T2", "T3", "T4"))
    parser.add_argument("-o", "--output", default="hxmt_vs_gbm.png")
    args = parser.parse_args()
    if args.pub:
        matplotlib.rcParams.update({
            "font.size": 12, "axes.labelsize": 13, "axes.linewidth": 0.9,
            "xtick.labelsize": 11, "ytick.labelsize": 11,
            "legend.fontsize": 10, "pdf.fonttype": 42,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
        })

    print("Loading HXMT/HE reconstruct...", file=sys.stderr)
    hxmt_obs, hxmt_fill = load_hxmt_reconstruct(args.before, args.after)
    hxmt_all = np.concatenate([hxmt_obs, hxmt_fill]) if len(hxmt_fill) > 0 else hxmt_obs
    print(f"  HXMT: {len(hxmt_obs):,} obs + {len(hxmt_fill):,} fill = {len(hxmt_all):,}", file=sys.stderr)

    print("Loading engineering-channel prediction...", file=sys.stderr)
    t_years_const = (np.datetime64("2026-02-26") - T_REF).astype("timedelta64[D]").astype(float) / 365.25
    eng_t, eng_rate = load_engineering_prediction(
        date_str="20260226", hour_str="100000",
        trigger_met=HXMT_TRIGGER_MET, before=args.before, after=args.after,
        t_years_const=t_years_const,
        orbit_path="data/hxmt_aux/HXMT_20260226T10_Orbit_FFFFFF_V1_1K.FITS")
    if eng_t is None:
        print("  ERROR: engineering data missing — skipping that trace", file=sys.stderr)
    else:
        print(f"  Engineering 1-Hz frames: {len(eng_t)}", file=sys.stderr)

    gbm_events = {}
    for det in args.det:
        evts = load_gbm_tte(det, args.before, args.after, args.emin, args.emax)
        gbm_events[det] = evts
    gbm_combined = np.concatenate([gbm_events[d] for d in args.det])
    print(f"  GBM combined ({'+'.join(args.det)}): {len(gbm_combined):,}", file=sys.stderr)

    bin_w = args.bin
    edges = np.arange(-args.before, args.after + bin_w, bin_w)
    x = edges[:-1]
    t1, t2, t3, t4 = args.bkg

    r_hxmt_obs = np.histogram(hxmt_obs, bins=edges)[0] / bin_w
    r_hxmt_all = np.histogram(hxmt_all, bins=edges)[0] / bin_w
    r_gbm = np.histogram(gbm_combined, bins=edges)[0] / bin_w

    bkg_mask = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    n_bkg = bkg_mask.sum()
    bkg_hxmt = np.mean(r_hxmt_all[bkg_mask]) if n_bkg else 0
    bkg_gbm = np.mean(r_gbm[bkg_mask]) if n_bkg else 0
    print(f"  Background: HXMT={bkg_hxmt:.0f}, GBM={bkg_gbm:.0f} (from {n_bkg} bins)", file=sys.stderr)

    net_hxmt_obs = r_hxmt_obs - bkg_hxmt
    net_hxmt_all = r_hxmt_all - bkg_hxmt
    net_gbm = r_gbm - bkg_gbm

    if args.scale_range is not None:
        s1, s2 = args.scale_range
        burst_mask = (x >= s1) & (x < s2)
    else:
        burst_mask = (x >= t2) & (x < t3)
    # Exclude filler-containing bins from the normalization window so the
    # external scale does not depend on the recovery under test.
    fill_bins = np.histogram(hxmt_fill, bins=edges)[0] > 0
    n_excl = int((burst_mask & fill_bins).sum())
    burst_mask = burst_mask & ~fill_bins
    print(f"  Scale window: {int(burst_mask.sum())} bins used, "
          f"{n_excl} filler-containing bins excluded", file=sys.stderr)
    sum_hxmt = np.sum(net_hxmt_all[burst_mask])
    sum_gbm = np.sum(net_gbm[burst_mask])
    scale = sum_hxmt / sum_gbm if sum_gbm > 0 else 1.0
    net_gbm_scaled = net_gbm * scale
    print(f"  GBM scale: {scale:.2f}", file=sys.stderr)

    # Engineering background subtraction (using 1-Hz bins)
    if eng_t is not None:
        eng_bkg_mask = ((eng_t >= t1) & (eng_t < t2)) | ((eng_t >= t3) & (eng_t < t4))
        bkg_eng = np.mean(eng_rate[eng_bkg_mask]) if eng_bkg_mask.any() else 0.0
        net_eng = eng_rate - bkg_eng
        print(f"  Engineering background: {bkg_eng:.0f} evt/s ({eng_bkg_mask.sum()} bins)", file=sys.stderr)

    # ── Plot ──
    fig, (ax_lc, ax_ratio) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    # Blue family for HXMT/HE event-level: derive dark + light variants from
    # matplotlib C0 so they share hue and saturation, differing only in
    # lightness (HLS L). Reads as "the same blue, two intensities" rather
    # than "two different blues". Frees C1/C2 for the two equal-weight
    # cross-check references (GBM, engineering), both plotted with identical
    # line width.
    import colorsys
    import matplotlib.colors as _mc
    _h, _l, _s = colorsys.rgb_to_hls(*_mc.to_rgb("C0"))
    # Keep NAVY/SKY close to C0's natural L=0.41 so all four blue elements
    # (two fills + two lines) read as the same family. Earlier L=0.18/0.72
    # spread too wide, making the lines look like different colors from the
    # C0-based fills.
    NAVY     = colorsys.hls_to_rgb(_h, 0.25, _s)   # C0 hue, L=0.25 → darker C0
    SKY_BLUE = colorsys.hls_to_rgb(_h, 0.58, _s)   # C0 hue, L=0.58 → lighter C0
    CROSS_LW = 1.2                                  # identical width for cross-checks
    # Fills both use C0 itself (the canonical hue + saturation, mid lightness)
    # so they read unambiguously as "blue" regardless of alpha; bottom fill is
    # denser (observed) and the recovery layer above is lighter (alpha-modulated).
    ax_lc.fill_between(x, 0, net_hxmt_obs, step="post", alpha=0.55,
                       color="C0", zorder=1)
    ax_lc.fill_between(x, net_hxmt_obs, net_hxmt_all, step="post", alpha=0.30,
                       color="C0", zorder=2)
    ax_lc.step(x, net_hxmt_obs, where="post", color=NAVY, lw=1.0,
               label="HXMT/HE observed", zorder=3)
    ax_lc.step(x, net_hxmt_all, where="post", color=SKY_BLUE, lw=1.0,
               label=f"HXMT/HE + reconstructed (+{len(hxmt_fill):,})", zorder=4)
    ax_lc.step(x, net_gbm_scaled, where="post", color="C1", lw=CROSS_LW,
               label=f"Fermi/GBM {'+'.join(args.det)} " + rf"($\times${scale:.2f})", zorder=5)
    if eng_t is not None:
        # 1-Hz step trace, left-edge aligned: d["Time"]=N represents the
        # engineering cycle [N, N+0.94] starting at GPS PPS tick N. Plot step
        # from N to N+1 so the visual centre (N+0.5) matches the data's mean
        # event time (~N+0.47).
        eng_edges = np.concatenate([eng_t, [eng_t[-1] + 1.0]])
        eng_step_x = np.repeat(eng_edges, 2)[1:-1]
        eng_step_y = np.repeat(net_eng, 2)
        ax_lc.plot(eng_step_x, eng_step_y, color="C2", lw=CROSS_LW,
                   label=r"engineering $\widehat{S}_{\rm rec}^{\rm eng}$ (1 Hz, summed over 18 det)",
                   zorder=6)

    ax_lc.set_ylabel("Net count rate (evt/s)")
    ax_lc.legend(loc="upper right", fontsize=9.5)
    ax_lc.axhline(0, color="gray", lw=0.5, ls="--")
    if not args.pub:
        ax_lc.set_title(f"GRB 260226A: HXMT/HE event-level + engineering vs Fermi/GBM  [{bin_w}s bins, geocentric]",
                        fontweight="bold")

    # Ratio panel: two parallel cross-check ratios, colour-matched to the
    # upper-panel lines (C1 orange = GBM, C2 green = engineering).
    with np.errstate(divide='ignore', invalid='ignore'):
        peak_gbm = np.nanmax(net_gbm_scaled)
        threshold_gbm = max(peak_gbm * 0.05, 100)
        ratio_gbm = np.where(net_gbm_scaled > threshold_gbm,
                              net_hxmt_all / net_gbm_scaled, np.nan)
    ax_ratio.step(x, ratio_gbm, where="post", color="C1", lw=CROSS_LW,
                  label="HXMT / GBM")

    if eng_t is not None:
        # Upsample 1-Hz engineering rate to the 0.5-s plot grid by holding
        # eng_rate[i] across the [eng_t[i], eng_t[i]+1) interval.
        eng_t_min = int(np.floor(eng_t[0]))
        eng_t_max = int(np.floor(eng_t[-1])) + 1
        idx = np.floor(x).astype(int) - eng_t_min
        valid = (idx >= 0) & (idx < len(net_eng))
        eng_up = np.where(valid, net_eng[np.clip(idx, 0, len(net_eng) - 1)], np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            peak_eng = np.nanmax(eng_up)
            threshold_eng = max(peak_eng * 0.05, 100)
            ratio_eng = np.where(eng_up > threshold_eng,
                                  net_hxmt_all / eng_up, np.nan)
        ax_ratio.step(x, ratio_eng, where="post", color="C2", lw=CROSS_LW,
                      label="HXMT / engineering")

    ax_ratio.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax_ratio.set_ylabel("HXMT / ref.")
    ax_ratio.set_ylim(0.3, 1.8)

    # Annotation block with both ratio statistics
    annot_lines = []
    rg = ratio_gbm[~np.isnan(ratio_gbm)]
    if len(rg) > 0:
        annot_lines.append(f"HXMT/GBM         = {np.mean(rg):.2f} ± {np.std(rg):.2f} ({len(rg)} bins)")
        # Diagnostic: does the ratio differ between reconstructed
        # (filler-containing) and clean bins?
        ok = ~np.isnan(ratio_gbm)
        for tag, m in [("filler", ok & fill_bins), ("clean ", ok & ~fill_bins)]:
            if m.sum():
                print(f"  ratio [{tag}] = {np.mean(ratio_gbm[m]):.3f} "
                      f"± {np.std(ratio_gbm[m]):.3f} ({int(m.sum())} bins)",
                      file=sys.stderr)
    if eng_t is not None:
        re = ratio_eng[~np.isnan(ratio_eng)]
        if len(re) > 0:
            annot_lines.append(f"HXMT/engineering = {np.mean(re):.2f} ± {np.std(re):.2f} ({len(re)} bins)")
    if annot_lines:
        ax_ratio.text(0.98, 0.92, "\n".join(annot_lines),
                      transform=ax_ratio.transAxes, ha="right", va="top",
                      fontsize=8.5, family="monospace",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))
    ax_ratio.legend(loc="lower right", fontsize=9, framealpha=0.85)
    if args.xlim:
        ax_lc.set_xlim(*args.xlim)
        ax_ratio.set_xlim(*args.xlim)
        vis = (x >= args.xlim[0]) & (x < args.xlim[1])
        ax_lc.set_ylim(-2500, np.nanmax(net_hxmt_all[vis]) * 1.10)
    ax_ratio.set_xlabel(f"Time since HXMT trigger (s)  [$T_0$ = {HXMT_TRIGGER_UTC_LABEL} UTC]")
    if not args.xlim:
        ax_ratio.set_xlim(-args.before, args.after)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
