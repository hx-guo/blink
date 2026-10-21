#!/usr/bin/env python3
"""Proof-of-concept V11 model: V10 + OOC (Cal) using 2017-10-01 Box A.
   Box A is the only local (FITS + CSV) pair available; one day = ~3600 sec.

   Compares:
     V9   = PHO·lf ~ Sci_pure + Sci_ACD + Wide + Large + b              (5 params + k)
     V10  = V9 + Sci_pure_js + Sci_ACD_js + Wide_js + Large_js          (9 params + k)
     V11  = V10 + OOC + OOC_js                                          (11 params + k)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

DATE = "20171001"
BOX = "A"
PDA_CODE = "0766"
FITS_PATH = f"data/1B/2017/{DATE}/{PDA_CODE}/HXMT_1B_{PDA_CODE}_{DATE}T000000_G002572_000_003.fits"
CSV_PATH  = f"n_below_study/per_sec_csvs/{DATE}_box{BOX}.csv"

L_THRESH = 50_000
SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP = 400.0, 1000.0, 6000.0
K_GRID = np.linspace(-1.0, 8.0, 181)


def load_csv_with_ooc():
    print(f"Loading CSV {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH,
        dtype={"date":"string","box":"category","met_sec":"int64","det":"int8",
               "L_cycles":"int32","PHO":"int32","Wide":"int32","Large":"int32",
               "Dt":"int32","Sci":"int32","Sci_ACD1":"int32","Sci_ACDN":"int32"})
    print(f"  CSV rows: {len(df):,}, MET range: {df['met_sec'].min()} – {df['met_sec'].max()}")

    print(f"Loading FITS {FITS_PATH}...")
    fe = fits.open(FITS_PATH, memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met = (d["Time"].astype(float) + offset).astype(np.int64)
    L_fits = d["Length_Time_Cycle"].astype(np.int64)
    # Extract per-det OOC, build long-form (det, met_sec) → OOC
    ooc_rows = []
    for det in range(6):
        ooc = d[f"Cnt_OOCDet_{det}"].astype(np.int32)
        for i in range(len(met)):
            ooc_rows.append((det, int(met[i]), int(ooc[i])))
    ooc_df = pd.DataFrame(ooc_rows, columns=["det", "met_sec", "OOC"])
    ooc_df["det"] = ooc_df["det"].astype("int8")
    print(f"  FITS rows: {len(ooc_df):,}, MET range: {ooc_df['met_sec'].min()} – {ooc_df['met_sec'].max()}")
    fe.close()

    n_before = len(df)
    df = df.merge(ooc_df, on=["det", "met_sec"], how="inner")
    print(f"  Merged: {len(df):,} (lost {n_before-len(df):,} rows without OOC match)")
    return df


def build_predictors(df):
    df = df[df["L_cycles"] > L_THRESH].copy()
    # Sec total Sci (for box-rate cap)
    g = df.groupby("met_sec", observed=True)["Sci"].sum().rename("sci_sec_total")
    df = df.merge(g, on="met_sec")
    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["Sci_ACD"] = df["Sci_ACD1"] + df["Sci_ACDN"]
    df["length"] = df["L_cycles"].astype("float64") * 16e-6
    df["dt_frac"] = df["Dt"].astype("float64") / df["L_cycles"]
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("acd_rate","Sci_ACD"),("wide_rate","Wide"),
                    ("large_rate","Large"),("pho_rate","PHO"),
                    ("ooc_rate","OOC")]:
        df[c] = df[src] / df["length"]
    df["group_rate"] = df["sci_sec_total"] / df["length"]
    # Cross-det sums (sum over the other 5 dets at the same met_sec)
    for c in ["scipure_rate","acd_rate","wide_rate","large_rate","ooc_rate"]:
        bsum = df.groupby("met_sec")[c].transform("sum")
        df[c + "_js"] = bsum - df[c]
    return df


def fit_with_kscan(sub, cols):
    """Fit PHO·(1-k·dt/L) ~ [1, *cols], scanning k. Returns (k_opt, rms_opt, coef, names)."""
    Xmat = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in cols]).astype(np.float64)
    pho = sub["pho_rate"].values.astype(np.float64)
    dtf = sub["dt_frac"].values.astype(np.float64)
    best = (None, float("inf"), None)
    for k in K_GRID:
        lf = 1.0 - k * dtf
        if np.any(lf <= 0): continue
        target = pho * lf
        coef, *_ = np.linalg.lstsq(Xmat, target, rcond=None)
        pred_rhs = Xmat @ coef
        pred_pho = pred_rhs / lf
        rms = float(np.sqrt(np.mean((pho - pred_pho)**2)))
        if rms < best[1]: best = (float(k), rms, coef)
    return best[0], best[1], best[2], ["b"] + cols


def main():
    df = load_csv_with_ooc()
    df = build_predictors(df)
    print(f"\nClean-band rows: ", end="")
    clean = ((df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN)
              & (df["group_rate"] < BOX_RATE_CAP))
    sub = df[clean].copy()
    print(f"{len(sub):,} (out of {len(df):,})")
    print(f"  dt_frac: mean={sub['dt_frac'].mean()*100:.2f}%, max={sub['dt_frac'].max()*100:.2f}%")
    print(f"  ooc_rate per det:")
    for det in range(6):
        s = sub[sub["det"]==det]
        print(f"    det {det}: mean OOC/s = {s['ooc_rate'].mean():.1f},  std/mean = {s['ooc_rate'].std()/s['ooc_rate'].mean()*100:.1f}%")

    # ============= Fit V9 vs V10 vs V11, pooled across 6 dets =============
    print(f"\n{'='*86}")
    print(f"Pooled fits on Box A 2017-10-01 ({len(sub):,} rows)")
    print(f"{'='*86}")
    versions = [
        ("V9  baseline   ", ["scipure_rate","acd_rate","wide_rate","large_rate"]),
        ("V10 +cross-det ", ["scipure_rate","acd_rate","wide_rate","large_rate",
                              "scipure_rate_js","acd_rate_js","wide_rate_js","large_rate_js"]),
        ("V11 +cross +OOC", ["scipure_rate","acd_rate","wide_rate","large_rate",
                              "scipure_rate_js","acd_rate_js","wide_rate_js","large_rate_js",
                              "ooc_rate","ooc_rate_js"]),
    ]
    base_rms = None
    for label, cols in versions:
        k_opt, rms_opt, coef, names = fit_with_kscan(sub, cols)
        delta = "" if base_rms is None else f"  Δ={100*(rms_opt-base_rms)/base_rms:+.2f}%"
        print(f"\n  {label}  k_opt={k_opt:+.2f}  RMS_PHO={rms_opt:.2f}{delta}")
        for n, c in zip(names, coef):
            print(f"      {n:>20s} = {c:>+10.4f}")
        if base_rms is None: base_rms = rms_opt

    # ============= Per-det V9 vs V11 (smaller N, may be noisier) =============
    print(f"\n{'='*86}")
    print(f"Per-det fits on Box A 2017-10-01")
    print(f"{'='*86}")
    print(f"  {'det':>3s}  {'N':>6s}  {'V9 RMS':>8s}  {'V11 RMS':>8s}  {'Δ%':>7s}  "
          f"{'V11 b':>8s}  {'V11 c_OOC':>10s}  {'<OOC>':>7s}")
    for det in range(6):
        sdet = sub[sub["det"] == det]
        if len(sdet) < 100:
            print(f"  {det:>3d}  {len(sdet):>6d}  (too few)")
            continue
        _, rms9, _, _ = fit_with_kscan(sdet, ["scipure_rate","acd_rate","wide_rate","large_rate"])
        k11, rms11, c11, n11 = fit_with_kscan(sdet,
            ["scipure_rate","acd_rate","wide_rate","large_rate",
             "scipure_rate_js","acd_rate_js","wide_rate_js","large_rate_js",
             "ooc_rate","ooc_rate_js"])
        ooc_mean = float(sdet["ooc_rate"].mean())
        # c_OOC is at index for ooc_rate (predictor index 9 in cols, so coef index 9)
        b_v11 = c11[0]
        cooc_v11 = c11[9]   # index of "ooc_rate" in cols list (0=b, 1-4=scipure...large, 5-8=_js, 9=ooc_rate)
        print(f"  {det:>3d}  {len(sdet):>6d}  {rms9:>8.2f}  {rms11:>8.2f}  "
              f"{100*(rms11-rms9)/rms9:>+7.2f}  {b_v11:>+8.2f}  {cooc_v11:>+10.4f}  {ooc_mean:>7.1f}")


if __name__ == "__main__":
    main()
