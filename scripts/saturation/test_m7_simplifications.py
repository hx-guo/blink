#!/usr/bin/env python3
"""Test M7 simplifications:
  Full M7    : 6 params per box * 3 = 18 params
  Merged ACD : 5 params per box * 3 = 15 params (Sci_ACD1 + Sci_ACDN combined)
  Global     : 5 global params for all 3 boxes
"""
from pathlib import Path
import numpy as np
import pandas as pd

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
SCI_LO, SCI_HI = 400.0, 1000.0
BOX_RATE_CAP = 6000.0


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Sci": "int32", "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
    parts = []
    for f in sorted(CSV_DIR.glob("*.csv")):
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
    df["Sci_ACD"] = df["Sci_ACD1"] + df["Sci_ACDN"]
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("acd_rate","Sci_ACD"),
                    ("acd1_rate","Sci_ACD1"),("acdn_rate","Sci_ACDN"),
                    ("wide_rate","Wide"),("large_rate","Large"),
                    ("pho_rate","PHO")]:
        df[c] = df[src] / df["length"]
    df["group_rate"] = df["sci_sec_total"] / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    hv = pd.read_csv(HV_TABLE, dtype={"date":"string","met_sec":"int64",
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


def regress(sub, cols):
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in cols])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def rms(sub, pred):
    return float(np.sqrt(np.mean((sub["pho_rate"].values - pred)**2)))


def main():
    df = load()
    print(f"rows: {len(df):,}")

    clean = ((df["sci_rate"] >= SCI_LO) & (df["sci_rate"] < SCI_HI)
             & (df["group_rate"] < BOX_RATE_CAP))
    test_broad = (df["sci_rate"] > 300)

    print("\n" + "="*78)
    print("FULL M7 (per-box, 6 params each)")
    print("="*78)
    M7_full = {}
    rms_full = {}
    for box in "ABC":
        s = df[clean & (df["box"]==box)]
        cols = ["scipure_rate","acd1_rate","acdn_rate","wide_rate","large_rate"]
        c = regress(s, cols)
        M7_full[box] = c
        pred_train = (c[0] + c[1]*s["scipure_rate"] + c[2]*s["acd1_rate"]
                      + c[3]*s["acdn_rate"] + c[4]*s["wide_rate"] + c[5]*s["large_rate"])
        rms_train = rms(s, pred_train)
        s2 = df[test_broad & (df["box"]==box)]
        pred_test = (c[0] + c[1]*s2["scipure_rate"] + c[2]*s2["acd1_rate"]
                     + c[3]*s2["acdn_rate"] + c[4]*s2["wide_rate"] + c[5]*s2["large_rate"])
        rms_test = rms(s2, pred_test)
        rms_full[box] = (rms_train, rms_test)
        print(f"  Box {box}: b={c[0]:+.1f}, c_pure={c[1]:.3f}, c_ACD1={c[2]:.3f}, "
              f"c_ACDN={c[3]:.3f}, β={c[4]:.3f}, γ={c[5]:.3f}")
        print(f"           RMS_train={rms_train:.2f}  RMS_test(Sci>300)={rms_test:.2f}")

    print("\n" + "="*78)
    print("MERGED ACD M7 (per-box, 5 params each: c_pure, c_ACD, β, γ, b)")
    print("="*78)
    rms_merged = {}
    M7_merged = {}
    for box in "ABC":
        s = df[clean & (df["box"]==box)]
        cols = ["scipure_rate","acd_rate","wide_rate","large_rate"]
        c = regress(s, cols)
        M7_merged[box] = c
        pred_train = (c[0] + c[1]*s["scipure_rate"] + c[2]*s["acd_rate"]
                      + c[3]*s["wide_rate"] + c[4]*s["large_rate"])
        rms_train = rms(s, pred_train)
        s2 = df[test_broad & (df["box"]==box)]
        pred_test = (c[0] + c[1]*s2["scipure_rate"] + c[2]*s2["acd_rate"]
                     + c[3]*s2["wide_rate"] + c[4]*s2["large_rate"])
        rms_test = rms(s2, pred_test)
        rms_merged[box] = (rms_train, rms_test)
        print(f"  Box {box}: b={c[0]:+.1f}, c_pure={c[1]:.3f}, c_ACD={c[2]:.3f}, "
              f"β={c[3]:.3f}, γ={c[4]:.3f}")
        print(f"           RMS_train={rms_train:.2f}  RMS_test(Sci>300)={rms_test:.2f}")

    print("\n" + "="*78)
    print("GLOBAL M7 (single fit across all 3 boxes, 5 global params)")
    print("="*78)
    s = df[clean]
    cols = ["scipure_rate","acd_rate","wide_rate","large_rate"]
    cg = regress(s, cols)
    print(f"  Global: b={cg[0]:+.1f}, c_pure={cg[1]:.3f}, c_ACD={cg[2]:.3f}, "
          f"β={cg[3]:.3f}, γ={cg[4]:.3f}")
    print(f"  Apply Global to each box:")
    rms_global = {}
    for box in "ABC":
        s_train = df[clean & (df["box"]==box)]
        pt = (cg[0] + cg[1]*s_train["scipure_rate"] + cg[2]*s_train["acd_rate"]
              + cg[3]*s_train["wide_rate"] + cg[4]*s_train["large_rate"])
        rms_tr = rms(s_train, pt)
        s_test = df[test_broad & (df["box"]==box)]
        pe = (cg[0] + cg[1]*s_test["scipure_rate"] + cg[2]*s_test["acd_rate"]
              + cg[3]*s_test["wide_rate"] + cg[4]*s_test["large_rate"])
        rms_ts = rms(s_test, pe)
        rms_global[box] = (rms_tr, rms_ts)
        print(f"    Box {box}: RMS_train={rms_tr:.2f}  RMS_test={rms_ts:.2f}")

    # Summary
    print("\n" + "="*78)
    print("SUMMARY: RMS comparison (PHO cnt/s/det), %deviation vs FULL")
    print("="*78)
    for region, label in [("train", "CLEAN train"), ("test", "test Sci>300")]:
        print(f"\n  {label}:")
        print(f"    {'Box':>3s}  {'FULL (18p)':>11s}  {'MERGED (15p)':>16s}  {'GLOBAL (5p)':>16s}")
        for box in "ABC":
            i = 0 if region == "train" else 1
            f = rms_full[box][i]
            m = rms_merged[box][i]
            g = rms_global[box][i]
            print(f"    {box:>3s}  {f:>11.2f}  {m:>9.2f} ({100*(m-f)/f:+6.2f}%)  "
                  f"{g:>9.2f} ({100*(g-f)/f:+6.2f}%)")


if __name__ == "__main__":
    main()
