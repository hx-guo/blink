#!/usr/bin/env python3
"""Band-resolved HXMT/HE (NaI) light curves for SGR 1935+2154 / FRB 200428,
in three DEPOSITED-energy bands — HE ONLY, NO background subtraction.

Raw (un-subtracted) count rate is plotted, so the background sits as the
baseline; the mean background level per band is drawn as a dashed reference
line rather than being subtracted. No cross-instrument comparison.

Reuses the HE loading / calibration / light-travel from
plot_hxmt_vs_ibis_bands.py.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from plot_hxmt_vs_ibis_bands import (  # noqa: E402
    channel_to_kev_lut, load_hxmt_nai, fit_background, CACHE_PW, BANDS, NAI_PW,
)
from plot_hxmt_vs_ibis import (  # noqa: E402
    compute_hxmt_light_travel, HXMT_TRIGGER_UTC_STR, HXMT_ORBIT_FILE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=float, default=0.005)
    ap.add_argument("--before", type=float, default=0.3)
    ap.add_argument("--after", type=float, default=0.9)
    ap.add_argument("--bkg", type=float, nargs=4, default=[-0.3, -0.05, 0.7, 0.9])
    ap.add_argument("--bkg-deg", type=int, default=1,
                    help="background polynomial degree (1 = linear fit)")
    ap.add_argument("--hide-bkg", action="store_true",
                    help="don't draw the background reference line")
    ap.add_argument("-o", "--output", default="he_nai_bands.png")
    args = ap.parse_args()

    trig = Time(HXMT_TRIGGER_UTC_STR, scale="utc")
    trig_met = (trig - Time("2012-01-01T00:00:00", scale="utc")).sec
    hlt = compute_hxmt_light_travel(trig_met, trig, HXMT_ORBIT_FILE)

    E_lut = channel_to_kev_lut()
    obs_t, obs_e, fill_t, fill_e = load_hxmt_nai(CACHE_PW, trig_met, E_lut, hlt)
    all_t = np.concatenate([obs_t, fill_t])
    all_e = np.concatenate([obs_e, fill_e])
    print(f"  HXMT NaI: {len(obs_t):,} obs + {len(fill_t):,} fill "
          f"(light-travel {hlt*1e3:+.1f} ms)", file=sys.stderr)

    edges = np.arange(-args.before, args.after + args.bin, args.bin)
    x = edges[:-1] + args.bin / 2
    t1, t2, t3, t4 = args.bkg
    bkgm = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))

    fig, axes = plt.subplots(len(BANDS), 1, figsize=(10, 10), sharex=True)
    for ax, (elo, ehi) in zip(axes, BANDS):
        mo = (obs_e >= elo) & (obs_e < ehi)
        ma = (all_e >= elo) & (all_e < ehi)
        r_obs = np.histogram(obs_t[mo], bins=edges)[0] / args.bin
        r_all = np.histogram(all_t[ma], bins=edges)[0] / args.bin
        bg = fit_background(x, r_all, bkgm, args.bkg_deg)   # 本底拟合(未扣)

        ax.fill_between(x, 0, r_obs, step="mid", alpha=0.5, color="C0", zorder=1)
        ax.fill_between(x, r_obs, r_all, step="mid", alpha=0.28, color="C0", zorder=2)
        ax.step(x, r_obs, where="mid", color="navy", lw=0.9,
                label="HXMT/HE NaI observed", zorder=3)
        ax.step(x, r_all, where="mid", color="C0", lw=0.9,
                label="HXMT/HE NaI reconstructed", zorder=4)
        if not args.hide_bkg:
            deg_lbl = "linear" if args.bkg_deg == 1 else f"deg-{args.bkg_deg}"
            ax.plot(x, bg, color="tab:red", ls="--", lw=1.1,
                    label=f"background ({deg_lbl} fit)", zorder=5)
        ax.set_ylabel("rate (counts/s)")
        ax.set_ylim(bottom=0)
        ax.text(0.015, 0.90, f"{elo}–{ehi} keV (deposited)",
                transform=ax.transAxes, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel(f"time since trigger (s)   [T0 = {HXMT_TRIGGER_UTC_STR} UTC]")
    axes[0].set_title(
        "FRB/XRB 200428  —  HXMT/HE (NaI) "
        f"[{args.bin*1e3:.0f} ms bins]  (NaI pw∈{list(NAI_PW)})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
