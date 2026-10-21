#!/usr/bin/env python3
"""Write OGIP PHA + BAK spectra of a saturated segment from the 1B reconstruction,
on the official HE-CsI DRM PI grid (Nor mode, 30-1100 keV, 256 channels, the
EBOUNDS listed in data/hxmt_aux/csi_calib_<burst>.txt), ready for
talk20_spec_tier3_fold_fit.py once a DRM for the burst direction is available.

Two source spectra per segment:
  <burst>_<seg>_rec.pha : observed + recovered fillers, EXPOSURE = segment duration
  <burst>_<seg>_obs.pha : observed only, EXPOSURE = segment duration  (what a 1K user
                          would fit, unaware of the FIFO dead time)
plus one BAK (per-channel linear background fit on the off-burst windows, 1 s bins,
integrated over the segment; STAT_ERR from the fit covariance, as in the HE duty
pipeline's gen_hxmt_spec.py).

CsI selection identical to Tier 2: pulse_width in [90,257], channel >= 30, energy from
the per-burst CsI LUT; PI = EBOUNDS bin containing that deposited energy.

Run from blink/:
  .venv/bin/python scripts/talks/talk20_spec_tier3_write_pha.py --burst 260226A --seg 19.1 22.6 --name sat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = Path(__file__).resolve().parent
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../bursts",)]  # sibling and shared script modules
from plot_hxmt_csi_multi import hxmt_met  # noqa: E402
from talk20_spec_tier2_real import CFG  # noqa: E402

ROOT = HERE.parent.parent
OUT = Path("/Users/skyair/Developer/ihep/talk-20/figs/spectrum/pha")
CSI_PW, CSI_CHMIN = (90, 257), 30


def read_ebounds(burst):
    fn = ROOT / f"data/hxmt_aux/csi_calib_{burst[:6]}.txt"
    # The calib file holds an EBOUNDS block (channels 1..256, 30-1100 keV)
    # terminated by EBOUNDS_END, then a CHMAP block whose rows have the same
    # 3-field shape. Reading past EBOUNDS_END silently doubled the channel
    # count and broke the match to the official DRM's 256-channel grid.
    emin, emax = [], []
    with open(fn) as f:
        for line in f:
            if line.strip() == "EBOUNDS_END":
                break
            p = line.split()
            if len(p) == 3 and p[0].isdigit():
                emin.append(float(p[1])); emax.append(float(p[2]))
    return np.array(emin), np.array(emax)


def write_pha(path, counts, exposure, backfile, err=None, respfile="none"):
    cols = [fits.Column(name="CHANNEL", format="I", array=np.arange(1, len(counts) + 1))]
    if err is None:
        cols.append(fits.Column(name="COUNTS", format="J", array=np.rint(counts).astype(np.int32)))
    else:
        cols.append(fits.Column(name="COUNTS", format="E", array=counts.astype(np.float32)))
        cols.append(fits.Column(name="STAT_ERR", format="E", array=err.astype(np.float32)))
    hdu = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")
    h = hdu.header
    h["TELESCOP"] = "HXMT"; h["INSTRUME"] = "HE"; h["DETNAM"] = "CsI"; h["FILTER"] = "NONE"
    h["EXPOSURE"] = exposure; h["AREASCAL"] = 1.0; h["BACKSCAL"] = 1.0; h["CORRSCAL"] = 1.0
    h["BACKFILE"] = backfile; h["RESPFILE"] = respfile; h["ANCRFILE"] = "none"
    h["HDUCLASS"] = "OGIP"; h["HDUCLAS1"] = "SPECTRUM"; h["HDUVERS"] = "1.2.1"
    h["HDUCLAS2"] = "TOTAL" if err is None else "BKG"; h["HDUCLAS3"] = "COUNT"
    h["CHANTYPE"] = "PI"; h["DETCHANS"] = len(counts); h["POISSERR"] = err is None
    h["TLMIN1"] = 1; h["TLMAX1"] = len(counts)
    h["GROUPING"] = 0; h["QUALITY"] = 0
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", required=True, choices=list(CFG))
    ap.add_argument("--seg", type=float, nargs=2, required=True)
    ap.add_argument("--name", default="sat")
    args = ap.parse_args()
    cfg = CFG[args.burst]
    OUT.mkdir(parents=True, exist_ok=True)
    trig = hxmt_met(cfg["t0"])
    lut = np.load(cfg["csi_lut"])
    emin, emax = read_ebounds(args.burst)
    nch = len(emin)

    df = pd.read_csv(cfg["recon"] / "events.csv", usecols=["type", "met", "channel", "pulse_width"])
    df = df[np.isfinite(df["met"])]
    ch = np.clip(df["channel"].to_numpy(int), 0, 255); pw = df["pulse_width"].to_numpy(int)
    e = lut[ch]
    ok = (pw >= CSI_PW[0]) & (pw <= CSI_PW[1]) & (ch >= CSI_CHMIN) & np.isfinite(e)
    t = df["met"].to_numpy() - trig
    typ = df["type"].to_numpy()
    pi = np.searchsorted(emax, e, side="right")          # 0-based PI bin index
    ok &= (pi >= 0) & (pi < nch) & (e >= emin[0])

    a, b = args.seg; dur = b - a
    in_seg = ok & (t >= a) & (t < b)
    S_rec = np.bincount(pi[in_seg], minlength=nch).astype(float)
    S_obs = np.bincount(pi[in_seg & (typ == "EVT")], minlength=nch).astype(float)

    # background: per-channel linear fit of the observed 1 s rate on the off-burst windows
    tmin, tmax = np.floor(t.min()), np.ceil(t.max())
    e1 = np.arange(tmin, tmax + 0.5, 1.0); x1 = e1[:-1] + 0.5
    bm = np.zeros(len(x1), bool)
    for lo, hi in cfg["bkg"]:
        bm |= (x1 >= lo) & (x1 < hi)
    H, _, _ = np.histogram2d(t[ok & (typ == "EVT")], pi[ok & (typ == "EVT")],
                             bins=[e1, np.arange(-0.5, nch, 1.0)])
    B = np.zeros(nch); Berr = np.zeros(nch)
    xm = 0.5 * (a + b)
    for c in range(nch):
        y = H[bm, c]
        if y.sum() == 0:
            continue
        coef, cov = np.polyfit(x1[bm], y, 1, cov=True)
        B[c] = np.polyval(coef, xm) * dur
        X = np.array([[xm, 1.0]])
        Berr[c] = np.sqrt(max((X @ cov @ X.T).item(), 0.0)) * dur
    tag = f"{args.burst}_{args.name}"
    write_pha(OUT / f"{tag}.bak", B, dur, "none", err=Berr)
    write_pha(OUT / f"{tag}_rec.pha", S_rec, dur, f"{tag}.bak")
    write_pha(OUT / f"{tag}_obs.pha", S_obs, dur, f"{tag}.bak")
    print(f"{tag}: segment {a}..{b} s ({dur:.2f} s); CsI counts rec {S_rec.sum():.0f}, obs {S_obs.sum():.0f}, "
          f"bkg {B.sum():.0f}; net rec {S_rec.sum() - B.sum():.0f}, net obs {S_obs.sum() - B.sum():.0f}")
    print(f"wrote {OUT / (tag + '_rec.pha')}, {OUT / (tag + '_obs.pha')}, {OUT / (tag + '.bak')}")


if __name__ == "__main__":
    main()
