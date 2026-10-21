#!/usr/bin/env python3
"""Test: can we merge Sci_pure + Sci_ACD1 into one channel?

M7 (3 channel): c_pure·Sci_pure + c_ACD1·Sci_ACD1 + c_ACDN·Sci_ACDN + ...
M5 (2 channel): c_lowACD·(Sci_pure + Sci_ACD1) + c_highACD·Sci_ACDN + ...

If c_ACD1 ≈ c_pure physically, M5 should match M7 RMS.
"""
from pathlib import Path
import numpy as np
import pandas as pd

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
SCI_LO, SCI_HI, BOX_RATE = 400.0, 1000.0, 6000.0


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Sci": "int32", "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, usecols=list(dtype), dtype=dtype))
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH]
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN].copy()
    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["Sci_lowACD"] = df["Sci_pure"] + df["Sci_ACD1"]   # merged
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("lowacd_rate","Sci_lowACD"),
                    ("acd1_rate","Sci_ACD1"),("acdn_rate","Sci_ACDN"),
                    ("wide_rate","Wide"),("large_rate","Large"),
                    ("pho_rate","PHO")]:
        df[c] = df[src] / df["length"]
    df["group_rate"] = df["sci_sec_total"] / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    hv = pd.read_csv(HV_TABLE,
                     dtype={"date":"string","met_sec":"int64",
                            **{f"hv{i}":"float32" for i in range(18)}})
    hv = hv.set_index(["date","met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    return df


def fit_m7(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd1_rate"],
                          sub["acdn_rate"], sub["wide_rate"], sub["large_rate"]])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    pred = X @ coef
    rms = np.sqrt(np.mean((sub["pho_rate"].values - pred)**2))
    return coef, rms


def fit_m5(sub):
    """5-parameter: merged low-ACD + high-ACD + Wide + Large + b"""
    X = np.column_stack([np.ones(len(sub)), sub["lowacd_rate"],
                          sub["acdn_rate"], sub["wide_rate"], sub["large_rate"]])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    pred = X @ coef
    rms = np.sqrt(np.mean((sub["pho_rate"].values - pred)**2))
    return coef, rms


def main():
    df = load()
    print(f"  rows: {len(df):,}\n")

    print("=== M7 (3 channel: pure/ACD1/ACDN) vs M5 (2 channel: lowACD/ACDN) ===\n")
    for box in "ABC":
        mask = ((df["box"] == box)
                & (df["sci_rate"] >= SCI_LO) & (df["sci_rate"] < SCI_HI)
                & (df["group_rate"] < BOX_RATE))
        sub = df[mask]

        m7, rms7 = fit_m7(sub)
        m5, rms5 = fit_m5(sub)
        improvement = 100 * (rms5 - rms7) / rms7

        print(f"  Box {box}  N={len(sub):,}")
        print(f"    M7: b={m7[0]:+6.1f}, c_pure={m7[1]:.3f}, c_ACD1={m7[2]:.3f}, "
              f"c_ACDN={m7[3]:.3f}, β={m7[4]:.3f}, γ={m7[5]:.3f}")
        print(f"        ratios: c_ACD1/c_pure = {m7[2]/m7[1]:.3f}, "
              f"c_ACDN/c_pure = {m7[3]/m7[1]:.3f}")
        print(f"        RMS = {rms7:.3f}")
        print(f"    M5: b={m5[0]:+6.1f}, c_lowACD={m5[1]:.3f}, c_ACDN={m5[2]:.3f}, "
              f"β={m5[3]:.3f}, γ={m5[4]:.3f}")
        print(f"        c_ACDN/c_lowACD = {m5[2]/m5[1]:.3f}")
        print(f"        RMS = {rms5:.3f}")
        print(f"    M5 - M7 RMS: {rms5-rms7:+.4f}  ({improvement:+.2f}%)")
        print()


if __name__ == "__main__":
    main()
