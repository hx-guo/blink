#!/usr/bin/env python3
"""GRB 221009A: phase of the HE 1-Hz engineering frame relative to the event
time axis, and its effect on the dead-time-corrected ratio.

make_f10.load_eng_dtau assigns frame k the time met_k = Time + (UTC_Last_Bdc -
sTime_Last_Bdc) + MET_CORRECTION and applies it to the 1-s event bin
[round(met_k) - T0, +1). Here, per box, we count observed (non-filled) events
in a window [met_k + phi, met_k + phi + L_k) (L_k = Length_Time_Cycle * 16 us)
and find the phi that best matches the six-detector sum of Cnt_PHODet, on
frames free of FIFO gap-fill, HE-off and counter ceiling.
Outputs: engphase.txt, engphase.png (data/p221009/scale/)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from p221009_scale_common import (OUT, SCALE, TRIGGER_MET_ASTROPY, hxmt_events,
                                  gecam_btime_channels, rebin_sum, robust, saa_mask)
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION, L_CYC_TO_SEC

PHO_CEILING = 16384


def eng_frames(box):
    code = BOX_CODE[box]
    fn = sorted(Path(f"data/1B/2022/20221009/{code}").glob(f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
    d = fits.open(fn)["HE_Eng"].data
    off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met = d["Time"].astype(float) + off + MET_CORRECTION
    L = d["Length_Time_Cycle"].astype(float) * L_CYC_TO_SEC
    pho = np.stack([d[f"Cnt_PHODet_{BOX_OFFSET[box] + i}"].astype(float) for i in range(6)], 1)
    dt = np.stack([d[f"DeadTime_PHODet_{BOX_OFFSET[box] + i}"].astype(float) for i in range(6)], 1)
    lf = 1 - dt / d["Length_Time_Cycle"].astype(float)[:, None]
    return met - TRIGGER_MET_ASTROPY, L, pho, lf.mean(1)


def main():
    t, fill, ch, pw, bx = hxmt_events()
    phis = np.arange(-2.5, 2.5001, 0.02)
    log = []
    fig, ax = plt.subplots(figsize=(7, 4))
    best = {}
    for bi, box in enumerate(("A", "B", "C")):
        te, L, pho, lf = eng_frames(box)
        sel_b = bx == bi
        tobs = np.sort(t[sel_b & ~fill])
        tfill = np.sort(t[sel_b & fill])
        P = pho.sum(1)
        good = ((te > -45) & (te < 695) & (pho.max(1) < PHO_CEILING - 1) & (P > 0))
        # drop frames touching HE-off or containing gap-fill within +-3 s
        for k in np.where(good)[0]:
            a, b = te[k] - 3, te[k] + 4
            if np.searchsorted(tfill, b) - np.searchsorted(tfill, a) > 0:
                good[k] = False
        good &= ~saa_mask(te - 3, 7.0)
        cc = []
        for phi in phis:
            lo = te[good] + phi
            n = np.searchsorted(tobs, lo + L[good]) - np.searchsorted(tobs, lo)
            # compare frame-to-frame changes (removes slow background trends)
            cc.append(np.corrcoef(np.diff(n), np.diff(P[good]))[0, 1])
        cc = np.array(cc)
        k = np.argmax(cc)
        best[box] = phis[k]
        lo = te[good] + phis[k]
        n = np.searchsorted(tobs, lo + L[good]) - np.searchsorted(tobs, lo)
        ratio = np.median(n / P[good])
        s = (f"box {box}: {good.sum()} clean frames, best phi = {phis[k]:+.2f} s "
             f"(cc {cc[k]:.4f}; cc at phi=0: {cc[np.argmin(np.abs(phis))]:.4f}); "
             f"median obs/Cnt_PHODet at best phi = {ratio:.3f}; mean L = {L.mean():.4f} s")
        print(s); log.append(s)
        ax.plot(phis, cc, label=f"box {box}")
    ax.set_xlabel("phi (s): event window [met_k + phi, met_k + phi + L_k)")
    ax.set_ylabel("corr(diff obs counts, diff Cnt_PHODet)")
    ax.axvline(0, color="0.6", lw=0.6); ax.legend()
    fig.savefig(OUT / "engphase.png", dpi=140, bbox_inches="tight")

    # ---- effect on the dead-time-corrected ratio: shift the f_live grid ----
    phi = np.median(list(best.values()))
    edges = np.arange(-50, 701, 1.0); x = edges[:-1]
    r_all = np.histogram(t, edges)[0].astype(float)
    B = r_all[x < -10].mean()
    frames = [eng_frames(b) for b in ("A", "B", "C")]
    te = frames[0][0]
    lf18 = np.mean([np.interp(te, f[0], f[3]) for f in frames], axis=0)
    tg, c, cb, e = gecam_btime_channels()
    T = rebin_sum(tg, np.ones_like(tg), edges) * 0.05
    lg = rebin_sum(tg, (c - cb)[:, 9:13].sum(1), edges) / T
    live = rebin_sum(tg, e[:, 1], edges) / T
    G = rebin_sum(tg, c[:, 9:13].sum(1), edges) / T
    Bg = rebin_sum(tg, cb[:, 9:13].sum(1), edges) / T
    lg_dc = G / live - Bg / 0.993
    bad = saa_mask(x)
    for label, shift in [("current mapping (round(met_k))", None),
                         (f"frame centre at met_k + phi + L/2, phi={phi:+.2f}", phi)]:
        if shift is None:
            m = {int(round(tt)): v for tt, v in zip(te, lf18)}
            fl = np.array([m.get(int(round(xx)), np.nan) for xx in x])
        else:
            # bin [x, x+1) takes the overlap-weighted live fraction of frames
            fl = np.interp(x + 0.5, te + shift + 0.47, lf18)
        net_dt = np.where(fl > 0.05, r_all / fl - B, np.nan)
        out = [label]
        for name, lo, hi in [("main 170-310", 170, 310), ("P1 226-229", 226, 229),
                             ("P2 255-267", 255, 267), ("flare 440-610", 440, 610)]:
            sel = (x >= lo) & (x < hi) & (lg > 50) & (r_all - B > 0) & ~bad
            a = robust(net_dt[sel] / (SCALE * lg[sel])); b = robust(net_dt[sel] / (SCALE * lg_dc[sel]))
            out.append(f"   {name:<14} HE_dt/LGraw {a[0]:.3f}±{a[1]:.3f}  HE_dt/LGdc {b[0]:.3f}±{b[1]:.3f} (n={a[2]})")
        pk = (x >= 224) & (x < 229)
        out.append(f"   bins 224-228 HE_dt/LGdc: {np.round(net_dt[pk] / (SCALE * lg_dc[pk]), 3)}")
        for s in out:
            print(s)
        log += out
    (OUT / "engphase.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
