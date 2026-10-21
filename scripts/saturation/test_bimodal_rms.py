#!/usr/bin/env python3
"""Statistical test for bimodality: compare RMS of single-mode vs per-mode
fits on the SAME test set.

If bimodal exists:
  Per-mode fit applied per-bin should have SIGNIFICANTLY lower RMS than
  single-mode fit, on the same test data.
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
             "Sci": "int32"}
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
    df["sci_rate"]    = df["Sci"]      / df["length"]
    df["wide_rate"]   = df["Wide"]     / df["length"]
    df["large_rate"]  = df["Large"]    / df["length"]
    df["pho_rate"]    = df["PHO"]      / df["length"]
    df["group_rate"]  = df["sci_sec_total"] / df["length"]
    df["large_frac"]  = df["large_rate"] / df["sci_rate"].clip(lower=1)
    df["det_global"]  = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
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


def predict_m1(sub, coef):
    b, c1plus, beta, gamma = coef
    return b + c1plus*sub["sci_rate"].values + beta*sub["wide_rate"].values + gamma*sub["large_rate"].values


def main():
    df = load()

    # Classify modes
    main_classify = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)].copy()
    by_date = main_classify.groupby("date").agg(
        large_frac=("large_frac", "median"),
        N=("large_frac", "count"),
    )
    by_date = by_date[by_date["N"] > 200]
    HIGH_TH, LOW_TH = 0.55, 0.40
    high_dates = set(by_date[by_date["large_frac"] > HIGH_TH].index)
    low_dates  = set(by_date[by_date["large_frac"] < LOW_TH].index)
    df["mode"] = np.where(df["date"].isin(high_dates), "HIGH",
                          np.where(df["date"].isin(low_dates), "LOW", "MID"))

    # ===========================
    # TEST: CLEAN band, same test set for both fits
    # ===========================
    print(f"\n========================================")
    print(f"  CONTROLLED COMPARISON: same test set, same range")
    print(f"  Test set: Sci ∈ [{SCI_LO}, {SCI_HI}]/det, group_rate < {BOX_RATE_CAP}")
    print(f"  (Excludes MID and FIFO-contaminated bins)")
    print(f"========================================")

    for box in "ABC":
        # Test set: clean band, HIGH+LOW modes only
        clean = ((df["box"] == box)
                 & (df["sci_rate"] >= SCI_LO) & (df["sci_rate"] < SCI_HI)
                 & (df["group_rate"] < BOX_RATE_CAP)
                 & (df["mode"].isin(["HIGH","LOW"])))
        test_set = df[clean]
        N_total = len(test_set)
        N_high = (test_set["mode"]=="HIGH").sum()
        N_low = (test_set["mode"]=="LOW").sum()

        # === Fit A: SINGLE M1 on HIGH+LOW combined ===
        coef_single = fit_m1(test_set)
        pred_single = predict_m1(test_set, coef_single)
        rms_single = np.sqrt(np.mean((test_set["pho_rate"].values - pred_single)**2))

        # === Fit B: per-mode M1 ===
        high_sub = test_set[test_set["mode"]=="HIGH"]
        low_sub = test_set[test_set["mode"]=="LOW"]
        coef_high = fit_m1(high_sub)
        coef_low = fit_m1(low_sub)

        # Apply per-mode coefs to corresponding bins
        pred_permode = np.where(test_set["mode"].values=="HIGH",
                                 predict_m1(test_set, coef_high),
                                 predict_m1(test_set, coef_low))
        rms_permode = np.sqrt(np.mean((test_set["pho_rate"].values - pred_permode)**2))

        # === Bonus: per-mode RMS on respective subset ===
        pred_h_on_h = predict_m1(high_sub, coef_high)
        rms_h_on_h = np.sqrt(np.mean((high_sub["pho_rate"].values - pred_h_on_h)**2))
        pred_l_on_l = predict_m1(low_sub, coef_low)
        rms_l_on_l = np.sqrt(np.mean((low_sub["pho_rate"].values - pred_l_on_l)**2))

        # === What if we apply single-mode to subset? ===
        pred_single_on_h = predict_m1(high_sub, coef_single)
        rms_single_on_h = np.sqrt(np.mean((high_sub["pho_rate"].values - pred_single_on_h)**2))
        pred_single_on_l = predict_m1(low_sub, coef_single)
        rms_single_on_l = np.sqrt(np.mean((low_sub["pho_rate"].values - pred_single_on_l)**2))

        print(f"\n  Box {box}  (N={N_total:,}: HIGH={N_high:,}, LOW={N_low:,})")
        print(f"    Single M1 (4 params): RMS = {rms_single:.2f}")
        print(f"      coefs: b={coef_single[0]:+.1f}, 1+α={coef_single[1]:.3f}, "
              f"β={coef_single[2]:.3f}, γ={coef_single[3]:.3f}")
        print(f"    Per-mode M1 (8 params): RMS = {rms_permode:.2f}  → "
              f"REDUCTION: {100*(1 - rms_permode/rms_single):.1f}%")
        print(f"      HIGH coefs: b={coef_high[0]:+.1f}, 1+α={coef_high[1]:.3f}, "
              f"β={coef_high[2]:.3f}, γ={coef_high[3]:.3f}")
        print(f"      LOW  coefs: b={coef_low[0]:+.1f}, 1+α={coef_low[1]:.3f}, "
              f"β={coef_low[2]:.3f}, γ={coef_low[3]:.3f}")
        print(f"    Breakdown by mode subset:")
        print(f"      single fit applied to HIGH subset: RMS = {rms_single_on_h:.2f}")
        print(f"      HIGH-only fit applied to HIGH subset: RMS = {rms_h_on_h:.2f}  "
              f"(reduction: {100*(1-rms_h_on_h/rms_single_on_h):.1f}%)")
        print(f"      single fit applied to LOW subset: RMS = {rms_single_on_l:.2f}")
        print(f"      LOW-only fit applied to LOW subset: RMS = {rms_l_on_l:.2f}  "
              f"(reduction: {100*(1-rms_l_on_l/rms_single_on_l):.1f}%)")


if __name__ == "__main__":
    main()
