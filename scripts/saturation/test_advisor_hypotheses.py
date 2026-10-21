#!/usr/bin/env python3
"""Test advisor's hypotheses:
   1. b ≈ <OOC_rate>          (the constant b in our model = 241Am calibration source counter)
   2. cross-det terms matter   (sum of other 5 dets in the box helps predict PHO_i)

Uses 260226A 1-hour FITS engineering (no per-event Sci needed for these tests).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
MET_CORRECTION = 4.0


def load_260226A_eng_full():
    """Load 260226A engineering with OOC included."""
    rows = []
    for box, code in [("A","0766"),("B","1009"),("C","1781")]:
        fe = fits.open(f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits",
                        memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met = d["Time"].astype(float) + offset + MET_CORRECTION
        L = d["Length_Time_Cycle"].astype(float)
        for det in range(6):
            det_g = BOX_OFFSET[box] + det
            pho = d[f"Cnt_PHODet_{det_g}"].astype(float)
            ooc = d[f"Cnt_OOCDet_{det_g}"].astype(float)
            wide = d[f"Cnt_CsI_PHODet_{det_g}"].astype(float)
            large_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)
            large = unwrap_large(pho, large_raw)
            dt   = d[f"DeadTime_PHODet_{det_g}"].astype(float)
            for i in range(len(met)):
                rows.append(dict(box=box, det=det, met_sec=int(met[i]),
                                  L_cyc=L[i], length_s=L[i]*16e-6,
                                  PHO=pho[i], OOC=ooc[i],
                                  Wide=wide[i], Large=large[i],
                                  Dt=dt[i]))
        fe.close()
    df = pd.DataFrame(rows)
    df = df[df["L_cyc"] > 50_000]
    df = df[df["PHO"] > 0]
    df["dt_frac"] = df["Dt"] / df["L_cyc"]
    for c, s in [("pho_rate","PHO"),("ooc_rate","OOC"),
                 ("wide_rate","Wide"),("large_rate","Large")]:
        df[c] = df[s] / df["length_s"]
    return df


def fit(sub, predictor_cols, target_col):
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in predictor_cols])
    coef, *_ = np.linalg.lstsq(X, sub[target_col].values, rcond=None)
    pred = X @ coef
    rms = float(np.sqrt(np.mean((sub[target_col].values - pred)**2)))
    return coef, rms


def main():
    print("Loading 260226A engineering FITS (3 boxes × 6 dets × ~3600 sec)...")
    df = load_260226A_eng_full()
    print(f"  rows: {len(df):,}")

    # ============= Sanity: OOC stability =============
    print(f"\n{'='*70}")
    print("OOC rate per (box, det):  mean, std, std/mean")
    print(f"{'='*70}")
    print(f"  {'box-det':>8s}  {'mean OOC/s':>11s}  {'std OOC/s':>10s}  {'std/mean':>9s}")
    for box in "ABC":
        for det in range(6):
            sub = df[(df["box"]==box) & (df["det"]==det)]
            m = float(sub["ooc_rate"].mean())
            s = float(sub["ooc_rate"].std())
            print(f"  {box}-{det:>1d}      {m:>11.2f}  {s:>10.2f}  {s/m if m>0 else 0:>9.3f}")

    # ============= Guess A: b ≈ <OOC_rate> =============
    # Use full GRB engineering: regress PHO ~ Wide + Large (no Sci, no OOC) and see if b ≈ OOC_rate
    # Then regress PHO ~ Wide + Large + OOC and see if b drops to ~0 and OOC coefficient ≈ 1
    print(f"\n{'='*70}")
    print("Guess A: b ≈ <OOC_rate>?")
    print(f"{'='*70}")
    print(f"  {'box-det':>8s}  {'b (no OOC)':>11s}  {'b (with OOC)':>13s}  "
          f"{'c_OOC':>8s}  {'<OOC>':>8s}  {'b vs <OOC>':>11s}")
    for box in "ABC":
        for det in range(6):
            sub = df[(df["box"]==box) & (df["det"]==det)]
            # Model A1: PHO ~ Wide + Large
            cA1, _ = fit(sub, ["wide_rate", "large_rate"], "pho_rate")
            # Model A2: PHO ~ Wide + Large + OOC
            cA2, _ = fit(sub, ["wide_rate", "large_rate", "ooc_rate"], "pho_rate")
            ooc_mean = float(sub["ooc_rate"].mean())
            match = "✓" if abs(cA1[0] - ooc_mean) / max(ooc_mean, 1) < 0.5 else " "
            print(f"  {box}-{det:>1d}      {cA1[0]:>+11.1f}  {cA2[0]:>+13.1f}  "
                  f"{cA2[3]:>8.4f}  {ooc_mean:>8.1f}  {match}  Δ={(cA1[0]-ooc_mean):+.0f}")

    # ============= Guess B: cross-det terms matter =============
    # Build cross-det sums per (box, met_sec), then refit
    print(f"\n{'='*70}")
    print("Guess B: cross-det sums help predict PHO_i?")
    print(f"{'='*70}")
    # Cross-det sums (sum of other 5 dets)
    for c in ["wide_rate", "large_rate", "ooc_rate"]:
        bsum = df.groupby(["box", "met_sec"])[c].transform("sum")
        df[c + "_js"] = bsum - df[c]
    print(f"  {'box-det':>8s}  {'RMS own only':>13s}  {'RMS +cross':>12s}  Δ%   "
          f"{'c_Wide_js':>10s}  {'c_Large_js':>11s}  {'c_OOC_js':>9s}")
    for box in "ABC":
        for det in range(6):
            sub = df[(df["box"]==box) & (df["det"]==det)]
            # Model B1: own det only (with OOC)
            cB1, rmsB1 = fit(sub, ["wide_rate","large_rate","ooc_rate"], "pho_rate")
            # Model B2: own + cross
            cB2, rmsB2 = fit(sub, ["wide_rate","large_rate","ooc_rate",
                                     "wide_rate_js","large_rate_js","ooc_rate_js"], "pho_rate")
            delta = 100.0 * (rmsB2 - rmsB1) / rmsB1
            print(f"  {box}-{det:>1d}      {rmsB1:>13.2f}  {rmsB2:>12.2f}  {delta:+5.1f}%   "
                  f"{cB2[4]:>+10.4f}  {cB2[5]:>+11.4f}  {cB2[6]:>+9.4f}")

    # ============= Combined: dt correction + cross-det + OOC, pooled =============
    print(f"\n{'='*70}")
    print("Pooled (all 3 boxes × 6 dets):  PHO·(1−k·dt/L) ~ Wide + Large + OOC + cross + b")
    print(f"  Grid scan over k ∈ [-1, 8]")
    print(f"{'='*70}")
    k_grid = np.linspace(-1, 8, 91)
    cols = ["wide_rate", "large_rate", "ooc_rate",
            "wide_rate_js", "large_rate_js", "ooc_rate_js"]
    Xmat = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    pho = df["pho_rate"].values
    dtf = df["dt_frac"].values
    best = (None, None, float("inf"))
    for k in k_grid:
        lf = 1.0 - k * dtf
        if np.any(lf <= 0): continue
        target = pho * lf
        coef, *_ = np.linalg.lstsq(Xmat, target, rcond=None)
        pred_rhs = Xmat @ coef
        pred_pho = pred_rhs / lf
        rms = float(np.sqrt(np.mean((pho - pred_pho)**2)))
        if rms < best[2]: best = (k, coef, rms)
    k_opt, coef_opt, rms_opt = best
    print(f"  k_opt = {k_opt:+.2f}")
    names = ["b", "Wide", "Large", "OOC", "Wide_js", "Large_js", "OOC_js"]
    for n, c in zip(names, coef_opt):
        print(f"    {n:>10s} = {c:>+10.4f}")
    print(f"  RMS_PHO (this model) = {rms_opt:.2f}")
    # Baseline for comparison: no OOC, no cross, no dt
    cbase, _ = fit(df, ["wide_rate","large_rate"], "pho_rate")
    pred = Xmat[:, :3] @ cbase
    rms_base = float(np.sqrt(np.mean((pho - pred)**2)))
    print(f"  RMS_PHO (baseline = PHO ~ Wide+Large only) = {rms_base:.2f}")
    print(f"  Improvement = {100*(rms_opt-rms_base)/rms_base:+.2f}%")


if __name__ == "__main__":
    main()
