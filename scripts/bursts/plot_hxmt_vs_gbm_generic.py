#!/usr/bin/env python3
"""Generic HXMT/HE (1B reconstruction) vs Fermi/GBM NaI light-curve comparison
for an arbitrary burst. NaI-selected, deposited-energy, optionally band-resolved.

Time alignment is done in absolute time:
  t_rel_HXMT = hxmt_met - MET_HXMT(hxmt_t0_utc)
  t_rel_GBM  = (gbm_met - gbm_trig_met) + (gbm_trig_utc - hxmt_t0_utc)   [seconds]
so both axes are seconds since the HXMT T0. (Inter-spacecraft light travel,
<~20 ms for two LEO satellites, is neglected here; it matters at 5 ms, not 0.5 s.)

Reads a pre-dumped HXMT reconstruct CSV (box,type,met,channel,pulse_width,...).

Example:
  .venv/bin/python scripts/bursts/plot_hxmt_vs_gbm_generic.py \
    --recon-csv data/recon_cache/211211A_recon.csv \
    --hxmt-t0 2021-12-11T13:09:59.5 \
    --gbm-dir data/fermi_gbm/bn211211549 --gbm-trig bn211211549 \
    --gbm-trig-met 660921004.651 --gbm-trig-utc 2021-12-11T13:09:59.651 \
    --dets auto --before 20 --after 80 --bin 0.5 --bands \
    --title "GRB 211211A" -o GRB211211A_vs_gbm.png
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from plot_hxmt_vs_ibis_bands import channel_to_kev_lut, fit_background

NAI_PW = (54, 70)
BANDS = [(20, 50), (50, 100), (100, 200)]


def met_of_utc(utc):
    return (Time(utc, scale="utc").unix_tai
            - Time("2012-01-01T00:00:00", scale="utc").unix_tai)


def load_hxmt(recon_csv, hxmt_t0, E_lut):
    trig = met_of_utc(hxmt_t0)
    ot, oe, ft, fe = [], [], [], []
    with open(recon_csv) as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 5 or p[0] == "box":
                continue
            typ, met, ch, pw = p[1], float(p[2]), int(p[3]), int(p[4])
            if not (NAI_PW[0] <= pw <= NAI_PW[1]):
                continue
            t = met - trig
            e = float(E_lut[min(max(ch, 0), 255)])
            (ot, oe) if typ == "EVT" else (ft, fe)
            if typ == "EVT":
                ot.append(t); oe.append(e)
            elif typ == "FILL_GAP":
                ft.append(t); fe.append(e)
    return (np.array(ot), np.array(oe), np.array(ft), np.array(fe))


def load_gbm(gbm_dir, trig, dets, gbm_trig_met, gbm_trig_utc, hxmt_t0, before, after, shift=0.0):
    delta = (Time(gbm_trig_utc, scale="utc") - Time(hxmt_t0, scale="utc")).sec + shift
    ts, es = [], []
    for det in dets:
        path = os.path.join(gbm_dir, f"glg_tte_{det}_{trig}_v00.fit")
        if not os.path.exists(path):
            continue
        with fits.open(path, memmap=True) as f:
            d = f["EVENTS"].data
            eb = f["EBOUNDS"].data
            ecen = (eb["E_MIN"] + eb["E_MAX"]) / 2.0
            t = (d["TIME"] - gbm_trig_met) + delta
            e = ecen[d["PHA"]]
        m = (t >= -before) & (t <= after)
        ts.append(t[m]); es.append(e[m])
    return np.concatenate(ts), np.concatenate(es)


def pick_bright_nai(gbm_dir, trig, gbm_trig_met, gbm_trig_utc, hxmt_t0, peak_win):
    """Return the 2 NaI dets with the most counts in peak_win (rel HXMT T0)."""
    delta = (Time(gbm_trig_utc, scale="utc") - Time(hxmt_t0, scale="utc")).sec
    counts = {}
    for det in ["n0","n1","n2","n3","n4","n5","n6","n7","n8","n9","na","nb"]:
        path = os.path.join(gbm_dir, f"glg_tte_{det}_{trig}_v00.fit")
        if not os.path.exists(path):
            continue
        with fits.open(path, memmap=True) as f:
            t = (f["EVENTS"].data["TIME"] - gbm_trig_met) + delta
        counts[det] = int(((t >= peak_win[0]) & (t < peak_win[1])).sum())
    best = sorted(counts, key=counts.get, reverse=True)[:2]
    print(f"  NaI counts in peak {peak_win}: " +
          ", ".join(f"{d}={counts[d]}" for d in sorted(counts, key=counts.get, reverse=True)),
          file=sys.stderr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-csv", required=True)
    ap.add_argument("--hxmt-t0", required=True)
    ap.add_argument("--gbm-dir", required=True)
    ap.add_argument("--gbm-trig", required=True, help="e.g. bn211211549")
    ap.add_argument("--gbm-trig-met", type=float, required=True)
    ap.add_argument("--gbm-trig-utc", required=True)
    ap.add_argument("--dets", default="auto", help="'auto' or comma list e.g. n0,n3")
    ap.add_argument("--before", type=float, default=20)
    ap.add_argument("--after", type=float, default=80)
    ap.add_argument("--bin", type=float, default=0.5)
    ap.add_argument("--bkg", type=float, nargs=4, required=True)
    ap.add_argument("--scale-range", type=float, nargs=2, required=True)
    ap.add_argument("--peak-win", type=float, nargs=2, default=None,
                    help="window for auto detector pick (default = scale-range)")
    ap.add_argument("--bkg-deg", type=int, default=1)
    ap.add_argument("--bands", action="store_true")
    ap.add_argument("--title", default="")
    ap.add_argument("--gbm-shift", type=float, default=0.0,
                    help="extra seconds added to GBM time (fine alignment)")
    ap.add_argument("--xlim", type=float, nargs=2, default=None,
                    help="display range (s); background/scale still use full before/after")
    ap.add_argument("-o", "--output", default="hxmt_vs_gbm.png")
    args = ap.parse_args()

    E_lut = channel_to_kev_lut()
    ot, oe, ft, fe = load_hxmt(args.recon_csv, args.hxmt_t0, E_lut)
    at, ae = np.concatenate([ot, ft]), np.concatenate([oe, fe])

    peak_win = tuple(args.peak_win) if args.peak_win else tuple(args.scale_range)
    if args.dets == "auto":
        dets = pick_bright_nai(args.gbm_dir, args.gbm_trig, args.gbm_trig_met,
                               args.gbm_trig_utc, args.hxmt_t0, peak_win)
    else:
        dets = args.dets.split(",")
    gt, ge = load_gbm(args.gbm_dir, args.gbm_trig, dets, args.gbm_trig_met,
                      args.gbm_trig_utc, args.hxmt_t0, args.before, args.after, args.gbm_shift)
    print(f"  HXMT NaI: {len(ot):,} obs + {len(ft):,} fill;  "
          f"GBM NaI {'+'.join(dets)}: {len(gt):,}", file=sys.stderr)

    bands = BANDS if args.bands else [(20, 200)]
    edges = np.arange(-args.before, args.after + args.bin, args.bin)
    x = edges[:-1] + args.bin / 2
    t1, t2, t3, t4 = args.bkg
    bkgm = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    s1, s2 = args.scale_range
    sm = (x >= s1) & (x < s2)

    fig, axes = plt.subplots(len(bands), 1, figsize=(11, 3.4*len(bands)+0.6),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (elo, ehi) in zip(axes, bands):
        def rate(t, e):
            m = (e >= elo) & (e < ehi)
            return np.histogram(t[m], bins=edges)[0] / args.bin
        n_obs = rate(ot, oe); n_obs = n_obs - fit_background(x, n_obs, bkgm, args.bkg_deg)
        n_all = rate(at, ae); n_all = n_all - fit_background(x, n_all, bkgm, args.bkg_deg)
        r_g = rate(gt, ge)
        gm = (ge >= elo) & (ge < ehi)
        gerr = np.sqrt(np.histogram(gt[gm], bins=edges)[0]) / args.bin
        n_g = r_g - fit_background(x, r_g, bkgm, args.bkg_deg)
        scale = n_all[sm].sum() / n_g[sm].sum() if n_g[sm].sum() > 0 else 1.0
        print(f"  band {elo}-{ehi}: scale x{scale:.3f}", file=sys.stderr)
        ax.fill_between(x, n_obs, n_all, step="mid", color="C1", alpha=0.30, zorder=2)
        ax.step(x, n_obs, where="mid", color="navy", lw=0.8, label="HXMT/HE NaI observed", zorder=3)
        ax.step(x, n_all, where="mid", color="C1", lw=0.9, label="HXMT/HE NaI obs+reconstructed", zorder=4)
        ax.fill_between(x, (n_g-gerr)*scale, (n_g+gerr)*scale, step="mid",
                        color="tab:green", alpha=0.22, lw=0, zorder=1)
        ax.step(x, n_g*scale, where="mid", color="tab:green", lw=1.0,
                label=f"Fermi/GBM NaI ({'+'.join(dets)}) x{scale:.2f}", zorder=5)
        ax.axhline(0, color="grey", lw=0.5); ax.margins(x=0)
        ax.set_ylabel("net rate (evt/s)")
        lbl = f"{elo}-{ehi} keV (deposited)" if args.bands else "20-200 keV NaI"
        ax.text(0.01, 0.90, lbl, transform=ax.transAxes, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        if args.xlim:
            ax.set_xlim(*args.xlim)
            vis = (x >= args.xlim[0]) & (x < args.xlim[1])
            ytop = max(n_all[vis].max(), (n_g*scale)[vis].max()) * 1.12
            ybot = min(0, n_obs[vis].min() * 1.1)
            ax.set_ylim(ybot, ytop)
    axes[-1].set_xlabel(f"time since HXMT T0 (s)   [T0 = {args.hxmt_t0} UTC]")
    axes[0].set_title(f"{args.title}  —  HXMT/HE (NaI, 1B recon) vs Fermi/GBM  [{args.bin*1e3:.0f} ms bins]")
    fig.tight_layout(); fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
