#!/usr/bin/env python3
"""Re-evaluate M7 (ACD 3-channel decomposition) vs M1 on CLEAN band.

Earlier M7 vs M1 comparison was done on OLD band [400, 1500] which had FIFO
contamination. Need to check: is M7's improvement also a contamination artifact
(like M_uni_M1's γ_1 was), or genuine physics?

M1:  PHO = (1+α)·Sci + β·Wide + γ·Large + b
M7:  PHO = c_pure·Sci_pure + c_ACD1·Sci_ACD1 + c_ACDN·Sci_ACDN
          + β·Wide + γ·Large + b

If M7 still significantly beats M1 on CLEAN band → ACD breakdown is real physics
If M7 ≈ M1 on CLEAN band → ACD breakdown was absorbing FIFO contamination
"""
from pathlib import Path
import numpy as np
import pandas as pd

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

SCI_LO = 400.0
SCI_HI = 1000.0
BOX_RATE_CAP = 6000.0


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32",
             "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
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
    df["sci_rate"]      = df["Sci"]      / df["length"]
    df["scipure_rate"]  = df["Sci_pure"] / df["length"]
    df["acd1_rate"]     = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]     = df["Sci_ACDN"] / df["length"]
    df["wide_rate"]     = df["Wide"]     / df["length"]
    df["large_rate"]    = df["Large"]    / df["length"]
    df["pho_rate"]      = df["PHO"]      / df["length"]
    df["group_rate"]    = df["sci_sec_total"] / df["length"]
    df["det_global"]    = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
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
    print(f"  normal-mode rows: {len(df):,}")
    return df


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def fit_m7(sub):
    X = np.column_stack([np.ones(len(sub)),
                         sub["scipure_rate"].values,
                         sub["acd1_rate"].values,
                         sub["acdn_rate"].values,
                         sub["wide_rate"].values,
                         sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_m1(sub, coef):
    b, c1plus, beta, gamma = coef
    return b + c1plus*sub["sci_rate"].values + beta*sub["wide_rate"].values + gamma*sub["large_rate"].values


def predict_m7(sub, coef):
    b, c0, c1, cN, beta, gamma = coef
    return (b + c0*sub["scipure_rate"].values + c1*sub["acd1_rate"].values
            + cN*sub["acdn_rate"].values + beta*sub["wide_rate"].values
            + gamma*sub["large_rate"].values)


def main():
    df = load()
    print(f"\n========================================")
    print(f"  M1 vs M7 in CLEAN band")
    print(f"  Test set: Sci ∈ [{SCI_LO}, {SCI_HI}]/det, group_rate < {BOX_RATE_CAP}")
    print(f"========================================")

    for box in "ABC":
        clean = ((df["box"] == box)
                 & (df["sci_rate"] >= SCI_LO) & (df["sci_rate"] < SCI_HI)
                 & (df["group_rate"] < BOX_RATE_CAP))
        test_set = df[clean]

        # M1 fit
        coef_m1 = fit_m1(test_set)
        pred_m1 = predict_m1(test_set, coef_m1)
        rms_m1 = np.sqrt(np.mean((test_set["pho_rate"].values - pred_m1)**2))

        # M7 fit
        coef_m7 = fit_m7(test_set)
        pred_m7 = predict_m7(test_set, coef_m7)
        rms_m7 = np.sqrt(np.mean((test_set["pho_rate"].values - pred_m7)**2))

        # Also test on broader range for comparison
        broader = ((df["box"] == box) & (df["sci_rate"] > 300))
        test_broad = df[broader]
        pred_m1_b = predict_m1(test_broad, coef_m1)
        pred_m7_b = predict_m7(test_broad, coef_m7)
        rms_m1_b = np.sqrt(np.mean((test_broad["pho_rate"].values - pred_m1_b)**2))
        rms_m7_b = np.sqrt(np.mean((test_broad["pho_rate"].values - pred_m7_b)**2))

        print(f"\n  Box {box}  (CLEAN N={len(test_set):,})")
        print(f"    M1 coefs:  b={coef_m1[0]:+.1f}, 1+α={coef_m1[1]:.3f}, "
              f"β={coef_m1[2]:.3f}, γ={coef_m1[3]:.3f}")
        print(f"    M7 coefs:  b={coef_m7[0]:+.1f}, c_pure={coef_m7[1]:.3f}, "
              f"c_ACD1={coef_m7[2]:.3f}, c_ACDN={coef_m7[3]:.3f}, "
              f"β={coef_m7[4]:.3f}, γ={coef_m7[5]:.3f}")
        print(f"")
        print(f"    RMS on CLEAN band (training set):")
        print(f"      M1: {rms_m1:.2f}    M7: {rms_m7:.2f}    "
              f"improvement: {100*(1-rms_m7/rms_m1):.1f}%")
        print(f"    RMS on broader range (Sci > 300, applied):")
        print(f"      M1: {rms_m1_b:.2f}    M7: {rms_m7_b:.2f}    "
              f"improvement: {100*(1-rms_m7_b/rms_m1_b):.1f}%")

        # Compare per-event yield: c_pure, c_ACD1, c_ACDN
        # vs M1's effective 1+α
        # If c_pure ≈ c_ACD1 ≈ c_ACDN, then ACD breakdown is redundant
        # If they differ, ACD has real predictive power
        c_pure, c_ACD1, c_ACDN = coef_m7[1], coef_m7[2], coef_m7[3]
        print(f"    ACD channel yields:  c_pure={c_pure:.2f}  "
              f"c_ACD1/c_pure={c_ACD1/c_pure:.2f}×  c_ACDN/c_pure={c_ACDN/c_pure:.2f}×")


if __name__ == "__main__":
    main()
