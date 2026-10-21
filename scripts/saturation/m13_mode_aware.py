#!/usr/bin/env python3
"""M13: mode-aware unified model.

Two complementary tests:

(1) Per-mode fit (HIGH vs LOW) of M1 and M7 — confirms coefficient differences.

(2) Unified model M_uni that includes a mode-indicator feature derived from
    per-second observables:

        PHO = (1+α)·Sci + β·Wide + γ_0·Large + γ_1·Large²/Sci + b

    γ_eff = γ_0 + γ_1·(Large/Sci) varies with source spectrum automatically.
    No need to know the mode a priori — the data itself provides the indicator.

    Also try M7 + interaction:
        PHO = c0·Sp + c1·S1 + cN·SN + β·Wide + γ_0·Large + γ_1·Large²/Sci + b

Apply each model to ALL data (HIGH + LOW + MID combined) and check RMS.
The unified model should match per-mode fits without needing classification.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
SCI_LO_CLEAN = 400.0
SCI_HI_CLEAN = 1500.0  # tighter than M12 — avoid 1500-2000 saturation transition


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32",
             "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs...")
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

    df["Sci_pure"]     = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["sci_rate"]     = df["Sci"]      / df["length"]
    df["scipure_rate"] = df["Sci_pure"] / df["length"]
    df["acd1_rate"]    = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]    = df["Sci_ACDN"] / df["length"]
    df["wide_rate"]    = df["Wide"]     / df["length"]
    df["large_rate"]   = df["Large"]    / df["length"]
    df["pho_rate"]     = df["PHO"]      / df["length"]
    df["large_frac"]   = df["large_rate"] / df["sci_rate"]
    df["large2_over_sci"] = df["large_rate"]**2 / df["sci_rate"]
    df["det_global"]   = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

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
    print(f"normal-mode rows: {len(df):,}")
    return df


def classify_modes(df):
    """Classify dates by Large/Sci median in Sci 1000-1500 main band."""
    main = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)].copy()
    by_date = main.groupby("date").agg(
        large_frac=("large_frac", "median"),
        N=("large_frac", "count"),
    )
    by_date = by_date[by_date["N"] > 200]
    HIGH_TH, LOW_TH = 0.55, 0.40
    high_dates = set(by_date[by_date["large_frac"] > HIGH_TH].index)
    low_dates  = set(by_date[by_date["large_frac"] < LOW_TH].index)
    mid_dates  = set(by_date[(by_date["large_frac"] >= LOW_TH)
                              & (by_date["large_frac"] <= HIGH_TH)].index)
    return high_dates, low_dates, mid_dates


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_m1(sub, coef):
    b, c1plus, beta, gamma = coef
    return b + c1plus*sub["sci_rate"].values + beta*sub["wide_rate"].values + gamma*sub["large_rate"].values


def fit_m7(sub):
    X = np.column_stack([
        np.ones(len(sub)),
        sub["scipure_rate"].values,
        sub["acd1_rate"].values,
        sub["acdn_rate"].values,
        sub["wide_rate"].values,
        sub["large_rate"].values,
    ])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_m7(sub, coef):
    b, c0, c1, cN, beta, gamma = coef
    return (b + c0*sub["scipure_rate"].values + c1*sub["acd1_rate"].values
            + cN*sub["acdn_rate"].values + beta*sub["wide_rate"].values
            + gamma*sub["large_rate"].values)


def fit_munified_m1(sub):
    """M1 + Large²/Sci interaction term."""
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values,
                         sub["large2_over_sci"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_munified_m1(sub, coef):
    b, c1plus, beta, gamma0, gamma1 = coef
    return (b + c1plus*sub["sci_rate"].values + beta*sub["wide_rate"].values
            + gamma0*sub["large_rate"].values + gamma1*sub["large2_over_sci"].values)


def fit_munified_m7(sub):
    """M7 + Large²/Sci interaction term."""
    X = np.column_stack([
        np.ones(len(sub)),
        sub["scipure_rate"].values,
        sub["acd1_rate"].values,
        sub["acdn_rate"].values,
        sub["wide_rate"].values,
        sub["large_rate"].values,
        sub["large2_over_sci"].values,
    ])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_munified_m7(sub, coef):
    b, c0, c1, cN, beta, gamma0, gamma1 = coef
    return (b + c0*sub["scipure_rate"].values + c1*sub["acd1_rate"].values
            + cN*sub["acdn_rate"].values + beta*sub["wide_rate"].values
            + gamma0*sub["large_rate"].values + gamma1*sub["large2_over_sci"].values)


def median_per_bin(x, y, bins, min_count=50):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()
    high_dates, low_dates, mid_dates = classify_modes(df)
    df["mode"] = "MID"
    df.loc[df["date"].isin(high_dates), "mode"] = "HIGH"
    df.loc[df["date"].isin(low_dates), "mode"] = "LOW"
    print(f"\nMode counts: HIGH={(df['mode']=='HIGH').sum():,}, "
          f"LOW={(df['mode']=='LOW').sum():,}, "
          f"MID={(df['mode']=='MID').sum():,}")

    # ============ Per-mode fits (M1 and M7) ============
    print(f"\n=== Per-mode fits, CLEAN band Sci ∈ [{SCI_LO_CLEAN}, {SCI_HI_CLEAN}] ===")
    print(f"\n  M1 fits (4 params):")
    print(f"  {'Mode':>6s}  {'Box':>3s}  {'N':>10s}  {'b':>9s}  {'1+α':>8s}  {'β (Wide)':>10s}  {'γ (Large)':>10s}")
    m1_fits = {}
    for mode in ["HIGH", "LOW", "ALL"]:
        for box in "ABC":
            if mode == "ALL":
                mask = ((df["box"] == box) & (df["sci_rate"] >= SCI_LO_CLEAN)
                        & (df["sci_rate"] < SCI_HI_CLEAN))
            else:
                mask = ((df["box"] == box) & (df["mode"] == mode)
                        & (df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN))
            sub = df[mask]
            if len(sub) < 1000:
                continue
            coef = fit_m1(sub)
            m1_fits[(mode, box)] = coef
            print(f"  {mode:>6s}  {box:>3s}  {len(sub):>10,d}  "
                  f"{coef[0]:>+9.1f}  {coef[1]:>+8.4f}  {coef[2]:>+10.4f}  {coef[3]:>+10.4f}")

    print(f"\n  M7 fits (6 params):")
    print(f"  {'Mode':>6s}  {'Box':>3s}  {'b':>8s}  {'c_pure':>8s}  {'c_ACD1':>8s}  "
          f"{'c_ACDN':>8s}  {'β':>8s}  {'γ':>8s}")
    m7_fits = {}
    for mode in ["HIGH", "LOW", "ALL"]:
        for box in "ABC":
            if mode == "ALL":
                mask = ((df["box"] == box) & (df["sci_rate"] >= SCI_LO_CLEAN)
                        & (df["sci_rate"] < SCI_HI_CLEAN))
            else:
                mask = ((df["box"] == box) & (df["mode"] == mode)
                        & (df["sci_rate"] >= SCI_LO_CLEAN) & (df["sci_rate"] < SCI_HI_CLEAN))
            sub = df[mask]
            if len(sub) < 1000:
                continue
            coef = fit_m7(sub)
            m7_fits[(mode, box)] = coef
            b, c0, c1, cN, beta, gamma = coef
            print(f"  {mode:>6s}  {box:>3s}  {b:>+8.1f}  {c0:>+8.3f}  {c1:>+8.3f}  "
                  f"{cN:>+8.3f}  {beta:>+8.3f}  {gamma:>+8.3f}")

    # ============ Unified model fits (single fit on ALL data) ============
    print(f"\n=== Unified models (single fit on ALL data, with Large²/Sci interaction) ===")
    print(f"\n  M_uni_M1 = M1 + γ₁·Large²/Sci  (5 params):")
    print(f"  {'Box':>3s}  {'N':>10s}  {'b':>9s}  {'1+α':>8s}  {'β':>9s}  {'γ_0':>9s}  {'γ_1':>9s}")
    muni_m1_fits = {}
    for box in "ABC":
        mask = ((df["box"] == box) & (df["sci_rate"] >= SCI_LO_CLEAN)
                & (df["sci_rate"] < SCI_HI_CLEAN))
        sub = df[mask]
        coef = fit_munified_m1(sub)
        muni_m1_fits[box] = coef
        print(f"  {box:>3s}  {len(sub):>10,d}  "
              f"{coef[0]:>+9.1f}  {coef[1]:>+8.4f}  {coef[2]:>+9.4f}  "
              f"{coef[3]:>+9.4f}  {coef[4]:>+9.4f}")

    print(f"\n  M_uni_M7 = M7 + γ₁·Large²/Sci  (7 params):")
    print(f"  {'Box':>3s}  {'b':>8s}  {'c_pure':>8s}  {'c_ACD1':>8s}  {'c_ACDN':>8s}  "
          f"{'β':>8s}  {'γ_0':>8s}  {'γ_1':>8s}")
    muni_m7_fits = {}
    for box in "ABC":
        mask = ((df["box"] == box) & (df["sci_rate"] >= SCI_LO_CLEAN)
                & (df["sci_rate"] < SCI_HI_CLEAN))
        sub = df[mask]
        coef = fit_munified_m7(sub)
        muni_m7_fits[box] = coef
        b, c0, c1, cN, beta, g0, g1 = coef
        print(f"  {box:>3s}  {b:>+8.1f}  {c0:>+8.3f}  {c1:>+8.3f}  {cN:>+8.3f}  "
              f"{beta:>+8.3f}  {g0:>+8.3f}  {g1:>+8.3f}")

    # ============ Compute residuals for various models ============
    print(f"\n=== Compute residuals ===")
    models = ["M1_ALL", "M7_ALL", "M_uni_M1", "M_uni_M7", "M1_HIGH", "M1_LOW", "M7_HIGH", "M7_LOW"]
    for n in models:
        df[f"resid_{n}"] = np.nan

    for box in "ABC":
        mask_apply = df["box"] == box
        sub = df[mask_apply]
        # All-data fits
        if ("ALL", box) in m1_fits:
            df.loc[mask_apply, "resid_M1_ALL"] = sub["pho_rate"].values - predict_m1(sub, m1_fits[("ALL", box)])
            df.loc[mask_apply, "resid_M7_ALL"] = sub["pho_rate"].values - predict_m7(sub, m7_fits[("ALL", box)])
        # Unified
        df.loc[mask_apply, "resid_M_uni_M1"] = sub["pho_rate"].values - predict_munified_m1(sub, muni_m1_fits[box])
        df.loc[mask_apply, "resid_M_uni_M7"] = sub["pho_rate"].values - predict_munified_m7(sub, muni_m7_fits[box])
        # Per-mode (only apply to corresponding mode rows)
        for mode in ["HIGH", "LOW"]:
            if (mode, box) in m1_fits:
                mask_mode = mask_apply & (df["mode"] == mode)
                df.loc[mask_mode, f"resid_M1_{mode}"] = (df.loc[mask_mode, "pho_rate"].values
                                                         - predict_m1(df[mask_mode], m1_fits[(mode, box)]))
                df.loc[mask_mode, f"resid_M7_{mode}"] = (df.loc[mask_mode, "pho_rate"].values
                                                         - predict_m7(df[mask_mode], m7_fits[(mode, box)]))

    # ============ RMS table ============
    print(f"\n=== RMS by Sci bin, full data and each mode ===")
    bin_edges = [400, 600, 1000, 1500, 2000]

    print(f"\n  ALL data (HIGH + LOW + MID mixed):")
    print(f"  {'Sci bin':>13s}  {'N':>10s}  {'M1':>8s}  {'M7':>8s}  {'M_uni_M1':>9s}  {'M_uni_M7':>9s}")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1_ALL", "M7_ALL", "M_uni_M1", "M_uni_M7"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>8.1f}  {rmss[1]:>8.1f}  {rmss[2]:>9.1f}  {rmss[3]:>9.1f}")

    print(f"\n  HIGH mode data (29 dates):")
    print(f"  {'Sci bin':>13s}  {'N':>10s}  {'M1_ALL':>8s}  {'M7_ALL':>8s}  "
          f"{'M1_HIGH':>9s}  {'M7_HIGH':>9s}  {'M_uni_M1':>9s}  {'M_uni_M7':>9s}")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["mode"] == "HIGH") & (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1_ALL", "M7_ALL", "M1_HIGH", "M7_HIGH", "M_uni_M1", "M_uni_M7"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>8.1f}  {rmss[1]:>8.1f}  {rmss[2]:>9.1f}  {rmss[3]:>9.1f}  "
              f"{rmss[4]:>9.1f}  {rmss[5]:>9.1f}")

    print(f"\n  LOW mode data (25 dates):")
    print(f"  {'Sci bin':>13s}  {'N':>10s}  {'M1_ALL':>8s}  {'M7_ALL':>8s}  "
          f"{'M1_LOW':>9s}  {'M7_LOW':>9s}  {'M_uni_M1':>9s}  {'M_uni_M7':>9s}")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["mode"] == "LOW") & (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1_ALL", "M7_ALL", "M1_LOW", "M7_LOW", "M_uni_M1", "M_uni_M7"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>8.1f}  {rmss[1]:>8.1f}  {rmss[2]:>9.1f}  {rmss[3]:>9.1f}  "
              f"{rmss[4]:>9.1f}  {rmss[5]:>9.1f}")

    # ============ Plot: residuals on HIGH/LOW separated ============
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    SCI_MIN, SCI_MAX = 300, 3000
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])

    for ax, mode_label, mode_col in zip(axes, ["HIGH mode", "LOW mode"], ["HIGH", "LOW"]):
        sub_data = df[df["mode"] == mode_col]
        for name, color, label in [
            ("M1_ALL", "black", "M1 fit on ALL data"),
            ("M7_ALL", "C2", "M7 fit on ALL data"),
            ("M_uni_M1", "red", "M_uni_M1 (M1 + γ₁·Large²/Sci)"),
            ("M_uni_M7", "blue", "M_uni_M7 (M7 + γ₁·Large²/Sci)"),
            (f"M1_{mode_col}", "gray", f"M1 fit on {mode_col} only (oracle)"),
        ]:
            med = median_per_bin(sub_data["sci_rate"].values,
                                 sub_data[f"resid_{name}"].values, bins)
            ls = ":" if "oracle" in label else "-"
            ax.plot(bc, med, ls=ls, color=color, lw=2, label=label, alpha=0.85)
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.axvspan(SCI_LO_CLEAN, SCI_HI_CLEAN, alpha=0.1, color="green")
        ax.set_xscale("log")
        ax.set_xlim(SCI_MIN, SCI_MAX)
        ax.set_xlabel("Sci [cnt/s/det]")
        ax.set_ylabel("median PHO residual [cnt/s/det]")
        ax.set_title(f"{mode_label} data")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("M13: mode-aware unified model vs per-mode oracle fits", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m13_mode_aware.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
