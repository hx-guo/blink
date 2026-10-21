#!/usr/bin/env python3
"""GRB 221009A: is DeadTime_PHODet per detector or shared by the 6 detectors
of a box (one event-processing module + ADC per box)? Checks the within-box
spread of D(i,t), its correlation with the per-detector counts, the
box-level processing time per event, and the per-box vs 18-mean live-fraction
correction. Deterministic. Output: data/p221009/errors/pdau_check.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION  # noqa: E402

T0 = 339945423.0
OUT = Path("data/p221009/errors")
CEIL = 16384  # 14-bit Cnt_PHODet


def main():
    lines = []
    p = lambda s: (print(s), lines.append(s))
    for box, code in BOX_CODE.items():
        path = sorted(Path(f"data/1B/2022/20221009/{code}").glob(
            f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
        d = fits.open(path, memmap=True)["HE_Eng"].data
        off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        t = d["Time"].astype(float) + off + MET_CORRECTION - T0
        L = d["Length_Time_Cycle"].astype(float)
        g = [BOX_OFFSET[box] + j for j in range(6)]
        D = np.array([d[f"DeadTime_PHODet_{k}"].astype(float) for k in g])
        C = np.array([d[f"Cnt_PHODet_{k}"].astype(float) for k in g])
        on = (t > -50) & (t < 700) & (D.mean(0) > 0)
        rel = D[:, on].std(0) / D[:, on].mean(0)
        crel = C[:, on].std(0) / C[:, on].mean(0)
        p(f"Box {box}: within-box D spread (std/mean) median {np.median(rel):.4f}, "
          f"max {rel.max():.4f}; Cnt spread median {np.median(crel):.3f}")
        # detector offsets of D relative to the box mean (constant pattern?)
        ratio = D[:, on] / D[:, on].mean(0)
        p(f"   D_i/<D> per det: " + " ".join(f"{np.median(r):.4f}±{r.std():.4f}" for r in ratio))
        # correlation of D_i - <D> with C_i - <C> on the unsaturated tail
        tail = on & (t >= 300) & (t <= 700) & (C.max(0) < CEIL - 1)
        dD = (D[:, tail] - D[:, tail].mean(0)).ravel()
        dC = (C[:, tail] - C[:, tail].mean(0)).ravel()
        p(f"   corr(D_i-<D>, C_i-<C>) on tail = {np.corrcoef(dD, dC)[0,1]:+.3f}")
        # box-level processing time per event: <D> * 16us / sum_i C_i (tail, no wrap)
        tau_box = D[:, tail].mean(0) * 16e-6 / C[:, tail].sum(0)
        tau_det = D[:, tail] * 16e-6 / C[:, tail]
        p(f"   tau_box = <D>/sum_i Cnt = {np.median(tau_box)*1e6:.2f} us "
          f"(equivalent per-det D_i/Cnt_i = {np.median(tau_det)*1e6:.1f} us)")
        for k in (222, 227):
            i = np.where(np.round(t) == k)[0][0]
            p(f"   T0+{k}: D = {D[:, i].min():.0f}..{D[:, i].max():.0f} (L={L[i]:.0f}); "
              f"Cnt = {C[:, i].min():.0f}..{C[:, i].max():.0f} (14-bit, wraps)")

    lc = pd.read_csv(OUT / "lc_221009a_1s_errors.csv")
    m = np.isfinite(lc.dt_corr) & (lc.dt_corr > 0)
    rd = (lc.dt_corr_box - lc.dt_corr) / lc.dt_corr
    p("Per-box correction sum_b N_b/f_b - B vs N/f_mean - B (relative):")
    for t0 in (222, 226, 227, 228, 258, 262, 510):
        r = lc[lc.t == t0].iloc[0]
        p(f"   T0+{t0}: f_mean={r.f_live:.4f}  per-box-minus-mean = "
          f"{(r.dt_corr_box - r.dt_corr) / r.dt_corr:+.5f}")
    mm = m & (lc.t >= 180) & (lc.t < 300) & (lc.t != 229)
    p(f"   main 180-300 (excl. 229): max |diff| {np.abs(rd[mm]).max():.5f}, "
      f"median {np.median(rd[mm]):+.6f}")
    (OUT / "pdau_check.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
