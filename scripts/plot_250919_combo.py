#!/usr/bin/env python3
"""GRB 250919A vs SVOM/GRM + Fermi/GBM: combined 4-panel figure at 0.1 s.

Top panel: all HXMT/HE events (event-level, no energy selection) versus
the two low-deadtime externals and the independent engineering-counter
rate prediction. Lower three panels: HXMT/HE-CsI recovery in three
deposited-energy sub-bands (70-150, 150-300, 300-700 keV). All panels at
0.1 s binning so the FIFO-reset notches and their filling are resolved.

Replaces the separate f15 (0.5 s bands) + f16 (0.1 s single band) pair.

Data are loaded over the full recon window (-30..+140 s) so the
background fit uses the config windows; only the saturated phase is
displayed.

Standard command (paper Fig. 7 position):
  .venv/bin/python scripts/plot_250919_combo.py \
      -o figures/f15_xsat_250919_bands.pdf
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).parent))
from plot_hxmt_csi_multi import (  # noqa: E402
    CONFIGS, load_hxmt_csi, LOADERS, hxmt_met,
)
from plot_hxmt_vs_ibis_bands import fit_background  # noqa: E402
from engineering_prediction import load_engineering_prediction, T_REF  # noqa: E402
import pubstyle  # noqa: E402

BIN = 0.1
BEFORE, AFTER = 30.0, 140.0
XLIM = (-3.0, 18.0)
BANDS = [(70, 150), (150, 300), (300, 700)]
OBS_C = "#20347e"     # observed: dark blue
REC_C = "#5b9bd5"     # reconstructed: light blue
ENG_C = "#2e8b57"     # engineering channel: green

# 250919A engineering-counter inputs (T0 = 2025-09-19T00:29:15).
ENG_DATE, ENG_HOUR = "20250919", "000000"
ENG_ORBIT = "data/hxmt_aux/HXMT_20250919T00_Orbit_FFFFFF_V1_1K.FITS"


def load_hxmt_all(recon, t0):
    """All reconstructed HXMT events (no CsI selection); returns (obs_t, fill_t)."""
    trig = hxmt_met(t0)
    ot, ft = [], []
    with open(recon) as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 5 or p[0] == "box":
                continue
            typ, met = p[1], float(p[2])
            (ot if typ == "EVT" else ft).append(met - trig)
    return np.array(ot), np.array(ft)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", default="250919A", choices=list(CONFIGS))
    ap.add_argument("-o", "--output", default="250919_combo.pdf")
    ap.add_argument("--xlim", type=float, nargs=2, default=list(XLIM))
    args = ap.parse_args()

    pubstyle.apply()

    cfg = CONFIGS[args.burst]
    t0 = cfg["t0"]
    lut = np.load(cfg["csi_lut"])
    ot, oe, ft, fe = load_hxmt_csi(cfg["recon"], t0, lut)   # CsI-selected
    at, ae = np.concatenate([ot, ft]), np.concatenate([oe, fe])
    all_ot, all_ft = load_hxmt_all(cfg["recon"], t0)         # all events
    all_at = np.concatenate([all_ot, all_ft])
    exts = []
    for e in cfg["externals"]:
        t, en = LOADERS[e["type"]](t0=t0, **{k: v for k, v in e.items()
                                             if k not in ("type", "color", "label")})
        exts.append((e, t, en))
    print(f"  HXMT all: {len(all_ot):,} obs + {len(all_ft):,} fill;  "
          f"CsI: {len(ot):,} obs + {len(ft):,} fill;  "
          + ";  ".join(f"{e['label']}:{len(t):,}" for e, t, _ in exts),
          file=sys.stderr)

    trig_met = hxmt_met(t0)
    ty = (np.datetime64("2025-09-19") - T_REF).astype(
        "timedelta64[D]").astype(float) / 365.25
    eng_t, eng_rate = load_engineering_prediction(
        date_str=ENG_DATE, hour_str=ENG_HOUR, trigger_met=trig_met,
        before=BEFORE, after=AFTER, t_years_const=ty, orbit_path=ENG_ORBIT)

    edges = np.arange(-BEFORE, AFTER + BIN, BIN)
    x = edges[:-1] + BIN / 2
    b = cfg["bkg"]
    bkgm = ((x >= b[0]) & (x < b[1])) | ((x >= b[2]) & (x < b[3]))
    s = cfg["scale_range"]
    fillbin = np.histogram(ft, bins=edges)[0] > 0
    sm = (x >= s[0]) & (x < s[1]) & (~fillbin)
    xl = tuple(args.xlim)
    fill_view = np.where(fillbin & (x >= xl[0]) & (x < xl[1]))[0]
    vis = (x >= xl[0]) & (x < xl[1])

    def net(t, e, lo, hi):
        m = (e >= lo) & (e < hi)
        r = np.histogram(t[m], bins=edges)[0] / BIN
        return r - np.polyval(np.polyfit(x[bkgm], r[bkgm], 1), x)

    def net_all(t):
        r = np.histogram(t, bins=edges)[0] / BIN
        return r - np.polyval(np.polyfit(x[bkgm], r[bkgm], 1), x)

    fig, axes = plt.subplots(
        4, 1, figsize=(pubstyle.FULL_W, 6.0), sharex=True,
        gridspec_kw={"hspace": 0.0})

    # ---- top panel: all events + engineering ----
    ax = axes[0]
    nHo, nHa = net_all(all_ot), net_all(all_at)
    ax.fill_between(x, nHo, nHa, step="mid", color=REC_C, alpha=0.30, zorder=2)
    ax.step(x, nHo, where="mid", color=OBS_C, lw=1.0,
            label="HXMT/HE observed", zorder=4)
    ax.step(x, nHa, where="mid", color=REC_C, lw=1.2,
            label="HXMT/HE reconstructed", zorder=5)
    ymax = nHa[vis].max()
    for e, t, en in exts:
        nE = net_all(t)
        sc = nHa[sm].sum() / nE[sm].sum() if nE[sm].sum() > 0 else 1.0
        ax.step(x, nE * sc, where="mid", color=e["color"], lw=1.0,
                label=f"{e['label']} " + rf"$\times${sc:.2f}", zorder=3)
    if eng_t is not None:
        ebm = ((eng_t >= b[0]) & (eng_t < b[1])) | ((eng_t >= b[2]) & (eng_t < b[3]))
        net_eng = eng_rate - np.mean(eng_rate[ebm])
        ax.step(eng_t, net_eng, where="post", color=ENG_C, lw=1.1,
                label=r"engineering $\widehat{S}_{\rm rec}^{\rm eng}$"
                      " (1 Hz, 18-det sum)", zorder=3)
        evis = (eng_t >= xl[0]) & (eng_t < xl[1])
        ymax = max(ymax, net_eng[evis].max())
    ax.axhline(0, color="grey", lw=0.5)
    ax.margins(x=0)
    ax.set_ylabel("net rate (counts/s)")
    ax.text(0.02, 0.92, f"all events, {BIN * 1e3:.0f} ms bins",
            transform=ax.transAxes, fontweight="bold", va="top", fontsize=9)
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(min(0, nHo[vis].min() * 1.1), ymax * 1.65)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune="both"))

    # ---- lower three panels: CsI bands ----
    for ax, (lo, hi) in zip(axes[1:], BANDS):
        nHo, nHa = net(ot, oe, lo, hi), net(at, ae, lo, hi)
        ax.fill_between(x, nHo, nHa, step="mid", color=REC_C, alpha=0.30, zorder=2)
        ax.step(x, nHo, where="mid", color=OBS_C, lw=1.0,
                label="HXMT/HE-CsI observed", zorder=4)
        ax.step(x, nHa, where="mid", color=REC_C, lw=1.2,
                label="HXMT/HE-CsI reconstructed", zorder=5)
        for e, t, en in exts:
            nE = net(t, en, lo, hi)
            sc = nHa[sm].sum() / nE[sm].sum() if nE[sm].sum() > 0 else 1.0
            ax.step(x, nE * sc, where="mid", color=e["color"], lw=1.0,
                    label=f"{e['label']} " + rf"$\times${sc:.2f}", zorder=3)
            rr = np.array([nHa[i] / (nE[i] * sc) for i in fill_view if nE[i] > 0])
            summ = (f"median {np.median(rr):.2f} [{rr.min():.2f},{rr.max():.2f}]"
                    f" n={len(rr)}" if len(rr) else "no filler bins in view")
            print(f"  {lo}-{hi}keV {e['label']}: scale x{sc:.3f}  "
                  f"sat-bin HXMT/ext {summ}", file=sys.stderr)
        ax.axhline(0, color="grey", lw=0.5)
        ax.margins(x=0)
        ax.set_ylabel("net rate (counts/s)")
        ax.text(0.02, 0.92, f"{lo:.0f}–{hi:.0f} keV (deposited), "
                f"{BIN * 1e3:.0f} ms bins", transform=ax.transAxes,
                fontweight="bold", va="top", fontsize=9)
        ax.legend(loc="upper right")
        ax.set_ylim(min(0, nHo[vis].min() * 1.1), nHa[vis].max() * 1.14)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=5, prune="both"))

    axes[-1].set_xlim(*xl)
    axes[-1].set_xlabel(f"time since HXMT $T_0$ (s)   [$T_0$ = {t0} UTC]")
    fig.subplots_adjust(left=0.068, right=0.988, top=0.99, bottom=0.062)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
