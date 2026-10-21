#!/usr/bin/env python3
"""HXMT/HE (1B reconstruction) vs SVOM/GRM light-curve comparison for a burst.

SVOM/GRM: 3 gamma-ray detectors (EVENTS01-03), PI channel -> deposited energy
via EBOUNDS, per-event DEAD_TIME, ANTI_COIN flag. Epoch MJDREF (TT).
Both instruments are LEO, so inter-spacecraft light travel (<~20 ms) is neglected.

Alignment is absolute: t_rel = met - MET(hxmt_t0), each in its own mission clock.

Example:
  .venv/bin/python scripts/bursts/plot_hxmt_vs_svom.py \
    --recon-csv data/recon_cache/250919A_recon.csv --hxmt-t0 2025-09-19T00:29:15 \
    --svom-evt data/svom_grm/svom_grm_evt_250919_00_v01.fits \
    --before 30 --after 130 --bin 0.5 --bkg -30 -24 112 130 \
    --scale-range 12 25 --xlim -12 28 --title "GRB 250919A" -o out.png
"""
import argparse, sys
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
HXMT_EPOCH = Time("2012-01-01T00:00:00", scale="utc")


def hxmt_met_of(utc):
    return Time(utc, scale="utc").unix_tai - HXMT_EPOCH.unix_tai


def load_hxmt(recon_csv, hxmt_t0, E_lut):
    trig = hxmt_met_of(hxmt_t0)
    ot, oe, ft, fe = [], [], [], []
    with open(recon_csv) as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 5 or p[0] == "box":
                continue
            typ, met, ch, pw = p[1], float(p[2]), int(p[3]), int(p[4])
            if not (NAI_PW[0] <= pw <= NAI_PW[1]):
                continue
            t, e = met - trig, float(E_lut[min(max(ch, 0), 255)])
            if typ == "EVT":
                ot.append(t); oe.append(e)
            elif typ == "FILL_GAP":
                ft.append(t); fe.append(e)
    return np.array(ot), np.array(oe), np.array(ft), np.array(fe)


def load_svom(evt_path, hxmt_t0, before, after):
    """Combined 3-GRD SVOM/GRM events; return (t_rel, energy_keV). Real photons
    only (ANTI_COIN==0); deposited energy from EBOUNDS[PI]."""
    f = fits.open(evt_path, memmap=True)
    hdr = f["EBOUNDS"].header
    epoch = Time(hdr["MJDREFI"] + hdr["MJDREFF"], format="mjd", scale="tt")
    svom_trig = (Time(hxmt_t0, scale="utc").tt - epoch).sec
    eb = f["EBOUNDS"].data
    ecen = (eb["E_MIN"] + eb["E_MAX"]) / 2.0
    ts, es = [], []
    for i in (1, 2, 3):
        d = f[f"EVENTS{i:02d}"].data
        good = (d["ANTI_COIN"] == 0)
        t = d["TIME"][good].astype(float) - svom_trig
        pi = np.clip(d["PI"][good].astype(int), 0, len(ecen) - 1)
        e = ecen[pi]
        m = (t >= -before) & (t <= after)
        ts.append(t[m]); es.append(e[m])
    return np.concatenate(ts), np.concatenate(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-csv", required=True)
    ap.add_argument("--hxmt-t0", required=True)
    ap.add_argument("--svom-evt", required=True)
    ap.add_argument("--before", type=float, default=30)
    ap.add_argument("--after", type=float, default=130)
    ap.add_argument("--bin", type=float, default=0.5)
    ap.add_argument("--bkg", type=float, nargs=4, required=True)
    ap.add_argument("--scale-range", type=float, nargs=2, required=True)
    ap.add_argument("--bkg-deg", type=int, default=1)
    ap.add_argument("--exclude-fill", action="store_true",
                    help="normalize on the bright burst but drop bins with any "
                         "reconstructed (filler) events, i.e. the saturated bins")
    ap.add_argument("--scale", type=float, default=None,
                    help="manual scale factor for the external curve (overrides "
                         "the auto normalization); auto value is still printed")
    ap.add_argument("--bands", action="store_true")
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("-o", "--output", default="hxmt_vs_svom.png")
    args = ap.parse_args()

    E_lut = channel_to_kev_lut()
    ot, oe, ft, fe = load_hxmt(args.recon_csv, args.hxmt_t0, E_lut)
    at, ae = np.concatenate([ot, ft]), np.concatenate([oe, fe])
    st, se = load_svom(args.svom_evt, args.hxmt_t0, args.before, args.after)
    print(f"  HXMT NaI: {len(ot):,} obs + {len(ft):,} fill;  SVOM/GRM: {len(st):,}",
          file=sys.stderr)

    bands = BANDS if args.bands else [(20, 200)]
    edges = np.arange(-args.before, args.after + args.bin, args.bin)
    x = edges[:-1] + args.bin / 2
    t1, t2, t3, t4 = args.bkg
    bkgm = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    s1, s2 = args.scale_range
    sm = (x >= s1) & (x < s2)
    if args.exclude_fill:
        fill_rate = np.histogram(ft, bins=edges)[0]  # NaI filler bins (any band)
        sm = sm & (fill_rate == 0)
        print(f"  scale window: bright burst [{s1},{s2}]s minus "
              f"{int(((x>=s1)&(x<s2)&(fill_rate>0)).sum())} saturated bins; "
              f"{int(sm.sum())} bins used", file=sys.stderr)

    fig, axes = plt.subplots(len(bands), 1, figsize=(11, 3.4*len(bands)+0.6),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (elo, ehi) in zip(axes, bands):
        def rate(t, e):
            m = (e >= elo) & (e < ehi)
            return np.histogram(t[m], bins=edges)[0] / args.bin
        n_obs = rate(ot, oe); n_obs = n_obs - fit_background(x, n_obs, bkgm, args.bkg_deg)
        n_all = rate(at, ae); n_all = n_all - fit_background(x, n_all, bkgm, args.bkg_deg)
        r_s = rate(st, se)
        sm_e = (se >= elo) & (se < ehi)
        serr = np.sqrt(np.histogram(st[sm_e], bins=edges)[0]) / args.bin
        n_s = r_s - fit_background(x, r_s, bkgm, args.bkg_deg)
        auto = n_all[sm].sum() / n_s[sm].sum() if n_s[sm].sum() > 0 else 1.0
        scale = args.scale if args.scale is not None else auto
        # ratios at the saturated (filler) bins, for annotation
        fb = np.histogram(ft, bins=edges)[0] > 0
        sat = fb & (x >= -args.before) & (x < args.after)
        ratios = [n_all[i] / (n_s[i]*scale) for i in np.where(sat)[0] if n_s[i] > 0]
        tag = f"scale x{scale:.3f}" + (" (manual)" if args.scale is not None else " (auto)")
        rtxt = "  ".join(f"{x[i]:+.1f}s:{n_all[i]/(n_s[i]*scale):.2f}"
                         for i in np.where(sat)[0] if n_s[i] > 0)
        print(f"  band {elo}-{ehi}: {tag};  sat-bin HXMT/ext = {rtxt}", file=sys.stderr)
        ax.fill_between(x, n_obs, n_all, step="mid", color="C1", alpha=0.30, zorder=2)
        ax.step(x, n_obs, where="mid", color="navy", lw=0.8, label="HXMT/HE NaI observed", zorder=3)
        ax.step(x, n_all, where="mid", color="C1", lw=0.9, label="HXMT/HE NaI obs+reconstructed", zorder=4)
        ax.fill_between(x, (n_s-serr)*scale, (n_s+serr)*scale, step="mid",
                        color="tab:purple", alpha=0.20, lw=0, zorder=1)
        ax.step(x, n_s*scale, where="mid", color="tab:purple", lw=1.0,
                label=f"SVOM/GRM (3 GRD) x{scale:.2f}", zorder=5)
        if args.exclude_fill:
            for xb in x[np.histogram(ft, bins=edges)[0] > 0]:
                ax.axvspan(xb-args.bin/2, xb+args.bin/2, color="red", alpha=0.10, zorder=0)
        ax.axhline(0, color="grey", lw=0.5); ax.margins(x=0)
        ax.set_ylabel("net rate (evt/s)")
        lbl = f"{elo}-{ehi} keV (deposited)" if args.bands else "20-200 keV"
        ax.text(0.01, 0.90, lbl, transform=ax.transAxes, fontweight="bold")
        ax.text(0.01, 0.78, tag + (f"\nsat-bin HXMT/ext: {rtxt}" if rtxt else ""),
                transform=ax.transAxes, fontsize=7.5, va="top", color="dimgray")
        ax.legend(fontsize=7, loc="upper right")
        if args.xlim:
            ax.set_xlim(*args.xlim)
            vis = (x >= args.xlim[0]) & (x < args.xlim[1])
            ax.set_ylim(min(0, n_obs[vis].min()*1.1),
                        max(n_all[vis].max(), (n_s*scale)[vis].max())*1.12)
    axes[-1].set_xlabel(f"time since HXMT T0 (s)   [T0 = {args.hxmt_t0} UTC]")
    axes[0].set_title(f"{args.title}  —  HXMT/HE (NaI, 1B recon) vs SVOM/GRM  [{args.bin*1e3:.0f} ms bins]")
    fig.tight_layout(); fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
