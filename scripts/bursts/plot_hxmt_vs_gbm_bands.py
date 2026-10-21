#!/usr/bin/env python3
"""Band-resolved HXMT/HE (NaI) vs Fermi/GBM (NaI n0+n3) light curves for
GRB 260226A, in three DEPOSITED-energy bands.

HXMT/HE: NaI events (pulse_width in [54,70]) from the live 1B reconstruction
(observed + fillers with jointly-recovered channel+pulse_width); deposited
energy from the CALDB 3-piece-quadratic gain averaged over the 18 NaI units
(same machinery as plot_hxmt_vs_ibis_bands.py).

Fermi/GBM: NaI TTE (n0+n3) with deposited energy from the per-detector EBOUNDS
channel centres. Same deposited-energy caveat as the IBIS comparison: NaI(Tl)
and NaI(Tl)@GBM redistribute incident spectra similarly but not identically;
this is the standard deposited-energy comparison, not an incident unfold.

GRB 260226A reaches HE through the CsI side (off-axis; NaI fraction in the
burst window 8.3% vs quiet 7.9%), so the NaI-selected signal is a small but
shape-faithful subset. Per-band single scalar normalization is fit on the
multi-peak phase T0+20..40 s; fillers are ~6% of the NaI counts there, so the
scale is anchored by observed events (per-band scales ~x0.06/x0.13/x0.15).

Run from blink/:
    .venv/bin/python scripts/bursts/plot_hxmt_vs_gbm_bands.py -o GRB260226A_hxmt_vs_gbm_bands.png
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../gbm",)]  # sibling and shared script modules
from plot_hxmt_vs_ibis_bands import (  # noqa: E402
    channel_to_kev_lut, fit_background, NAI_PW, BANDS,
)
from plot_hxmt_vs_gbm import (  # noqa: E402
    HXMT_TRIGGER_UTC, HXMT_TRIGGER_MET, HXMT_TRIGGER_UTC_LABEL,
    GBM_DIR, GBM_TRIGGER_MET, GBM_TO_HXMT_OFFSET,
)

GBM_NAI_DETS = ["n0", "n3"]


def load_hxmt_nai(before, after):
    """Run the live reconstruction; NaI-select; return (obs_t, obs_e, fill_t, fill_e)."""
    cmd = ["./target/release/blink", "sat", "reconstruct", HXMT_TRIGGER_UTC,
           "--before", str(before), "--after", str(after)]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    E_lut = channel_to_kev_lut()
    obs_t, obs_e, fill_t, fill_e = [], [], [], []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 5 or p[0] == "box":
            continue
        typ, met, ch, pw = p[1], float(p[2]), int(p[3]), int(p[4])
        if not (NAI_PW[0] <= pw <= NAI_PW[1]):
            continue
        t = met - HXMT_TRIGGER_MET
        e = float(E_lut[min(max(ch, 0), 255)])
        if typ == "EVT":
            obs_t.append(t); obs_e.append(e)
        elif typ == "FILL_GAP":
            fill_t.append(t); fill_e.append(e)
    return (np.asarray(obs_t), np.asarray(obs_e),
            np.asarray(fill_t), np.asarray(fill_e))


def load_gbm_nai(before, after):
    """GBM NaI TTE events with deposited energy from EBOUNDS channel centres."""
    ts, es = [], []
    for det in GBM_NAI_DETS:
        path = os.path.join(GBM_DIR, f"glg_tte_{det}_bn260226443_v00.fit")
        with fits.open(path, memmap=True) as f:
            d = f["EVENTS"].data
            eb = f["EBOUNDS"].data
            ecen = (eb["E_MIN"] + eb["E_MAX"]) / 2.0
            t = (d["TIME"] - GBM_TRIGGER_MET) + GBM_TO_HXMT_OFFSET
            e = ecen[d["PHA"]]
        m = (t >= -before) & (t <= after)
        ts.append(t[m]); es.append(e[m])
    return np.concatenate(ts), np.concatenate(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=float, default=0.5)
    ap.add_argument("--bkg-deg", type=int, default=1)
    ap.add_argument("--before", type=float, default=8.0)
    ap.add_argument("--after", type=float, default=80.0)
    ap.add_argument("--bkg", type=float, nargs=4, default=[-8, -3, 65, 80])
    ap.add_argument("--scale-range", type=float, nargs=2, default=[20.0, 40.0])
    ap.add_argument("--sat-phase", type=float, nargs=2, default=[20.0, 40.0],
                    help="FIFO-reset multi-peak phase to shade")
    ap.add_argument("-o", "--output", default="hxmt_vs_gbm_bands.png")
    args = ap.parse_args()

    obs_t, obs_e, fill_t, fill_e = load_hxmt_nai(args.before, args.after)
    all_t = np.concatenate([obs_t, fill_t])
    all_e = np.concatenate([obs_e, fill_e])
    gbm_t, gbm_e = load_gbm_nai(args.before, args.after)
    print(f"  HXMT NaI: {len(obs_t):,} obs + {len(fill_t):,} fill;  "
          f"GBM NaI ({'+'.join(GBM_NAI_DETS)}): {len(gbm_t):,} events", file=sys.stderr)

    edges = np.arange(-args.before, args.after + args.bin, args.bin)
    x = edges[:-1] + args.bin / 2
    t1, t2, t3, t4 = args.bkg
    bkgm = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    s1, s2 = args.scale_range
    sm = (x >= s1) & (x < s2)

    matplotlib.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.linewidth": 0.9,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 9.5, "pdf.fonttype": 42,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(10, 10), sharex=True)
    for ax, (elo, ehi) in zip(axes, BANDS):
        def rate(t, e):
            m = (e >= elo) & (e < ehi)
            return np.histogram(t[m], bins=edges)[0] / args.bin
        r_obs = rate(obs_t, obs_e)
        r_all = rate(all_t, all_e)
        r_gbm = rate(gbm_t, gbm_e)
        gbm_m = (gbm_e >= elo) & (gbm_e < ehi)
        n_gbm_raw = np.histogram(gbm_t[gbm_m], bins=edges)[0]
        gbm_err = np.sqrt(n_gbm_raw) / args.bin
        n_obs = r_obs - fit_background(x, r_obs, bkgm, args.bkg_deg)
        n_all = r_all - fit_background(x, r_all, bkgm, args.bkg_deg)
        n_gbm = r_gbm - fit_background(x, r_gbm, bkgm, args.bkg_deg)
        # Exclude filler-containing bins from the scale fit so the
        # normalization does not depend on the reconstruction under test.
        fill_bins = (r_all - r_obs) > 1e-9
        smc = sm & ~fill_bins
        scale = n_all[smc].sum() / n_gbm[smc].sum() if n_gbm[smc].sum() > 0 else 1.0
        print(f"  band {elo}-{ehi} keV: scale x{scale:.3f} "
              f"({int(smc.sum())} bins used, {int((sm & fill_bins).sum())} "
              f"filler bins excluded)", file=sys.stderr)
        # Diagnostic: per-band ratio in filler vs clean bins
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = n_gbm * scale
            ok = ref > 0.05 * np.nanmax(ref)
            ratio = np.where(ok, n_all / ref, np.nan)
        for tag, m in [("all   ", ok), ("filler", ok & fill_bins),
                       ("clean ", ok & ~fill_bins)]:
            if m.sum():
                r = ratio[m]
                q1, q2, q3 = np.nanpercentile(r, [25, 50, 75])
                print(f"    ratio [{tag}] = {np.nanmean(r):.3f} "
                      f"± {np.nanstd(r):.3f}; median {q2:.2f}, "
                      f"sigma_IQR {(q3 - q1) / 1.349:.2f} ({int(m.sum())} bins)",
                      file=sys.stderr)

        ax.axvspan(*args.sat_phase, color="tab:red", alpha=0.05, zorder=0)
        ax.fill_between(x, 0, n_obs, step="mid", alpha=0.5, color="C0", zorder=1)
        ax.fill_between(x, n_obs, n_all, step="mid", alpha=0.28, color="C0", zorder=2)
        ax.step(x, n_obs, where="mid", color="navy", lw=0.9,
                label="HXMT/HE NaI observed", zorder=3)
        ax.step(x, n_all, where="mid", color="C0", lw=0.9,
                label="HXMT/HE NaI obs+recon", zorder=4)
        ax.fill_between(x, (n_gbm - gbm_err) * scale, (n_gbm + gbm_err) * scale,
                        step="mid", color="tab:orange", alpha=0.25, lw=0, zorder=3)
        ax.step(x, n_gbm * scale, where="mid", color="tab:orange", lw=1.1,
                label=f"Fermi/GBM NaI " + rf"$\times${scale:.2f} " + "(±√N band)", zorder=5)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("net rate (counts/s)")
        ax.set_xlim(max(-args.before, -5.0), args.after)
        ax.text(0.015, 0.90, f"{elo}–{ehi} keV (deposited)",
                transform=ax.transAxes, fontweight="bold")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(
        f"time since trigger (s)   [T0 = {HXMT_TRIGGER_UTC_LABEL} UTC]")
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
