#!/usr/bin/env python3
"""HXMT/HE-CsI (1B reconstruction) vs external gamma-ray instruments.

CsI is HE's main GRB detector (the official duty pipeline uses CsI, not NaI).
CsI energy comes from an ensemble Channel->energy lookup built from the official
pha2pi product (see docs/energy-recovery-methods.md and the CSI note); it is
applied to both observed and reconstructed (filler) events for continuity.

Externals share a deposited-energy framework and are normalized per band on the
bright non-saturated bins (the saturated/filler bins are the test, shaded red):
  - Fermi/GBM   : TTE, energy from EBOUNDS[PHA]
  - SVOM/GRM    : evt (3 GRDs), energy from EBOUNDS[PI], ANTI_COIN==0

Both HXMT and the externals are LEO, so inter-spacecraft light travel (<~20 ms)
is neglected; it matters below ~0.1 s binning, not at 0.5 s.

Usage:
  .venv/bin/python scripts/bursts/plot_hxmt_csi_multi.py --burst 250919A \
      --bin 0.5 --bands --xlim -12 28 -o GRB250919A_csi_vs_grm_gbm.png
  .venv/bin/python scripts/bursts/plot_hxmt_csi_multi.py --burst 250919A \
      --bin 0.1 --band 300 700 --xlim 2 13 -o GRB250919A_csi_100ms.png
"""
import argparse, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits

HEP = Time("2012-01-01T00:00:00", scale="utc")
CSI_PW, CSI_CH = (90, 257), (30, 255)
BANDS_DEFAULT = [(70, 150), (150, 300), (300, 700)]

# Per-burst configuration. Externals are drawn/normalized independently.
CONFIGS = {
    "250919A": {
        "t0": "2025-09-19T00:29:15",
        "recon": "data/recon_cache/250919A_recon.csv",
        "csi_lut": "data/hxmt_aux/csi_ch2e_250919.npy",
        "bkg": [-30, -24, 112, 130], "scale_range": [-2, 15],
        "externals": [
            {"type": "gbm", "color": "#ff7f0e", "label": "Fermi/GBM n7+n8",
             "dir": "data/fermi_gbm/bn250919020", "trig": "bn250919020",
             "dets": ["n7", "n8"], "tmet": 779934537.28,
             "tutc": "2025-09-19T00:28:52.28"},
            {"type": "svom", "color": "#7d4fd0", "label": "SVOM/GRM",
             "evt": "data/svom_grm/svom_grm_evt_250919_00_v01.fits"},
        ],
    },
    "211211A": {
        "t0": "2021-12-11T13:09:59.5",
        "recon": "data/recon_cache/211211A_recon.csv",
        "csi_lut": "data/hxmt_aux/csi_ch2e_211211.npy",
        "bkg": [-40, -5, 70, 95], "scale_range": [12, 45],
        "externals": [
            {"type": "gbm", "color": "#ff7f0e", "label": "Fermi/GBM n2+na",
             "dir": "data/fermi_gbm/bn211211549", "trig": "bn211211549",
             "dets": ["n2", "na"], "tmet": 660921004.651,
             "tutc": "2021-12-11T13:09:59.651"},
        ],
    },
    "260226A": {
        "t0": "2026-02-26T10:37:53",
        "recon": "data/recon_cache/260226A_recon.csv",
        "csi_lut": "data/hxmt_aux/csi_ch2e_260226.npy",
        "bkg": [-8, -3, 65, 80], "scale_range": [24, 44],
        "externals": [
            {"type": "gbm", "color": "#ff7f0e", "label": "Fermi/GBM n0+n3",
             "dir": "data/fermi_gbm/bn260226443", "trig": "bn260226443",
             "dets": ["n0", "n3"], "tmet": 793795080.95811,
             "tutc": "2026-02-26T10:37:55.958"},   # +2.958s vs T0 (cross-corr aligned)
            # SVOM/GRM was Earth-occulted for 260226A (flat at background) — not usable.
        ],
    },
}


def hxmt_met(utc):
    return Time(utc, scale="utc").unix_tai - HEP.unix_tai


def load_hxmt_csi(recon, t0, lut):
    trig = hxmt_met(t0)
    ot, oe, ft, fe = [], [], [], []
    with open(recon) as fh:
        for line in fh:
            p = line.split(",")
            if len(p) < 5 or p[0] == "box":
                continue
            typ, met, ch, pw = p[1], float(p[2]), int(p[3]), int(p[4])
            if not (CSI_PW[0] <= pw <= CSI_PW[1] and CSI_CH[0] <= ch <= CSI_CH[1]):
                continue
            e = lut[ch]
            if not np.isfinite(e):
                continue
            (ot if typ == "EVT" else ft).append(met - trig)
            (oe if typ == "EVT" else fe).append(float(e))
    return (np.array(ot), np.array(oe), np.array(ft), np.array(fe))


def load_svom(evt, t0, **_):
    f = fits.open(evt); h = f["EBOUNDS"].header
    ep = Time(h["MJDREFI"] + h["MJDREFF"], format="mjd", scale="tt")
    trig = (Time(t0, scale="utc").tt - ep).sec
    ec = (f["EBOUNDS"].data["E_MIN"] + f["EBOUNDS"].data["E_MAX"]) / 2
    t, e = [], []
    for i in (1, 2, 3):
        d = f["EVENTS%02d" % i].data; g = d["ANTI_COIN"] == 0
        t.append(d["TIME"][g].astype(float) - trig)
        e.append(ec[np.clip(d["PI"][g], 0, len(ec) - 1)])
    return np.concatenate(t), np.concatenate(e)


def load_gbm(dir, trig, dets, tmet, tutc, t0, **_):
    delta = (Time(tutc, scale="utc") - Time(t0, scale="utc")).sec
    t, e = [], []
    for det in dets:
        h = fits.open(f"{dir}/glg_tte_{det}_{trig}_v00.fit")
        d = h["EVENTS"].data; eb = h["EBOUNDS"].data
        ec = (eb["E_MIN"] + eb["E_MAX"]) / 2
        t.append((d["TIME"] - tmet) + delta); e.append(ec[d["PHA"]])
    return np.concatenate(t), np.concatenate(e)


LOADERS = {"svom": load_svom, "gbm": load_gbm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", required=True, choices=list(CONFIGS))
    ap.add_argument("--bin", type=float, default=0.5)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--bands", action="store_true", help="3 default bands")
    g.add_argument("--band", type=float, nargs=2, help="single band lo hi keV")
    ap.add_argument("--before", type=float, default=30)
    ap.add_argument("--after", type=float, default=130)
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--bkg-deg", type=int, default=1)
    ap.add_argument("--pub", action="store_true",
                    help="publication style: larger fonts, thicker lines, no title")
    ap.add_argument("-o", "--output", default="hxmt_csi_multi.png")
    args = ap.parse_args()
    if args.pub:
        matplotlib.rcParams.update({
            "font.size": 12, "axes.labelsize": 13, "axes.linewidth": 0.9,
            "xtick.labelsize": 11, "ytick.labelsize": 11,
            "legend.fontsize": 9.5, "pdf.fonttype": 42,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
        })
    cfg = CONFIGS[args.burst]
    t0 = cfg["t0"]

    lut = np.load(cfg["csi_lut"])
    ot, oe, ft, fe = load_hxmt_csi(cfg["recon"], t0, lut)
    at, ae = np.concatenate([ot, ft]), np.concatenate([oe, fe])
    exts = []
    for e in cfg["externals"]:
        t, en = LOADERS[e["type"]](t0=t0, **{k: v for k, v in e.items()
                                             if k not in ("type", "color", "label")})
        exts.append((e, t, en))
    print(f"  HXMT CsI: {len(ot):,} obs + {len(ft):,} fill;  " +
          ";  ".join(f"{e['label']}:{len(t):,}" for e, t, _ in exts), file=sys.stderr)

    bands = BANDS_DEFAULT if args.bands else [tuple(args.band) if args.band else (70, 700)]
    edges = np.arange(-args.before, args.after + args.bin, args.bin)
    x = edges[:-1] + args.bin / 2
    b = cfg["bkg"]; bkgm = ((x >= b[0]) & (x < b[1])) | ((x >= b[2]) & (x < b[3]))
    s = cfg["scale_range"]; fillbin = np.histogram(ft, bins=edges)[0] > 0
    sm = (x >= s[0]) & (x < s[1]) & (~fillbin)

    def net(t, e, lo, hi):
        m = (e >= lo) & (e < hi)
        r = np.histogram(t[m], bins=edges)[0] / args.bin
        c = np.polyfit(x[bkgm], r[bkgm], args.bkg_deg)
        return r - np.polyval(c, x)

    fig, axes = plt.subplots(len(bands), 1, figsize=(12, 3.4 * len(bands) + 0.8),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (lo, hi) in zip(axes, bands):
        nHo, nHa = net(ot, oe, lo, hi), net(at, ae, lo, hi)
        xl = args.xlim or (-args.before, args.after)
        for i in np.where(fillbin & (x >= xl[0]) & (x < xl[1]))[0]:
            ax.axvspan(x[i] - args.bin/2, x[i] + args.bin/2, color="red", alpha=0.12, zorder=0)
        LW = 1.5 if args.pub else 1.0
        ax.fill_between(x, nHo, nHa, step="mid", color="#5b9bd5", alpha=0.30, zorder=2)
        ax.step(x, nHo, where="mid", color="#20347e", lw=LW*0.9, label="HXMT/HE-CsI observed", zorder=4)
        ax.step(x, nHa, where="mid", color="#5b9bd5", lw=LW, label="HXMT/HE-CsI obs+recon", zorder=5)
        for e, t, en in exts:
            nE = net(t, en, lo, hi)
            sc = nHa[sm].sum() / nE[sm].sum() if nE[sm].sum() > 0 else 1.0
            ax.step(x, nE * sc, where="mid", color=e["color"], lw=LW*0.9,
                    label=f"{e['label']} $\\times${sc:.2f}", zorder=3)
            rr = np.array([nHa[i]/(nE[i]*sc) for i in np.where(fillbin & (x >= xl[0]) & (x < xl[1]))[0]
                           if nE[i] > 0])
            summ = (f"median {np.median(rr):.2f}  [{rr.min():.2f},{rr.max():.2f}]  n={len(rr)}"
                    if len(rr) else "no filler bins in view")
            print(f"  {lo}-{hi}keV {e['label']}: scale x{sc:.3f}  sat-bin HXMT/ext {summ}", file=sys.stderr)
        ax.axhline(0, color="grey", lw=0.5); ax.margins(x=0); ax.set_ylabel("net rate (evt/s)")
        ax.text(0.02, 0.92, f"{lo:.0f}\u2013{hi:.0f} keV (deposited), {args.bin*1e3:.0f} ms bins",
                transform=ax.transAxes, fontweight="bold", va="top",
                fontsize=12 if args.pub else 10)
        ax.legend(fontsize=7, loc="upper right")
        if args.xlim:
            ax.set_xlim(*args.xlim)
            vis = (x >= args.xlim[0]) & (x < args.xlim[1])
            ax.set_ylim(min(0, nHo[vis].min()*1.1), nHa[vis].max()*1.12)
    axes[-1].set_xlabel(f"time since HXMT T0 (s)   [T0 = {t0} UTC]")
    if not args.pub:
        extl = " & ".join(e["label"] for e in cfg["externals"])
        axes[0].set_title(f"GRB {args.burst} — HXMT/HE-CsI recovery vs {extl}, deposited-energy")
    fig.tight_layout(); fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
