#!/usr/bin/env python3
"""M12: refit all models in clean main band Sci ∈ [400, 2000].

Three regions in the data:
  - Sci < 400:     anomalous (Wide > Sci, ratios don't match main physics).
                   60k rows, 1.5%. Different operating mode or filter leak.
  - Sci 400-2000:  clean main band, 85% of data, dominated by normal observation.
  - Sci > 2000:    sparse + saturation region (~12k rows). FIFO/spectral
                   redistribution dominates; linear models break down.

This script refits:
  - M1 (4 params: b, α, β, γ)
  - M7 (6 params: c0, c1, cN, β, γ, b)
  - Additive (1 param: k)
  - Multiplicative (1 param: k)

on Sci ∈ [400, 2000] ONLY. Compares to previous fits on Sci > 300 (broader).
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

# CLEAN BAND
SCI_LO_CLEAN = 400.0
SCI_HI_CLEAN = 2000.0


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
    df["dt_rate"]      = df["Dt"]       / df["length"]
    df["dt_frac"]      = df["Dt"]       / df["L_cycles"]
    df["live_frac"]    = 1.0 - df["dt_frac"]
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


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


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


def fit_additive(sub):
    """PHO = Sci + Wide + Large + k·Dt"""
    y = sub["pho_rate"].values - sub["sci_rate"].values - sub["wide_rate"].values - sub["large_rate"].values
    x = sub["dt_rate"].values
    k = np.sum(x * y) / np.sum(x * x)
    return k


def fit_multiplicative(sub):
    """PHO·(1-Dt/L) = k·Sci + Wide + Large"""
    y = sub["pho_rate"].values * sub["live_frac"].values - sub["wide_rate"].values - sub["large_rate"].values
    x = sub["sci_rate"].values
    k = np.sum(x * y) / np.sum(x * x)
    return k


def predict_m1(sub, coef):
    b, c1plus, beta, gamma = coef
    return b + c1plus*sub["sci_rate"].values + beta*sub["wide_rate"].values + gamma*sub["large_rate"].values


def predict_m7(sub, coef):
    b, c0, c1, cN, beta, gamma = coef
    return (b + c0*sub["scipure_rate"].values + c1*sub["acd1_rate"].values
            + cN*sub["acdn_rate"].values + beta*sub["wide_rate"].values + gamma*sub["large_rate"].values)


def predict_additive(sub, k):
    return (sub["sci_rate"].values + sub["wide_rate"].values
            + sub["large_rate"].values + k * sub["dt_rate"].values)


def predict_multiplicative(sub, k):
    return ((k * sub["sci_rate"].values + sub["wide_rate"].values
             + sub["large_rate"].values) / sub["live_frac"].values)


def median_per_bin(x, y, bins, min_count=50):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    # ============ Compare two fit ranges ============
    fit_ranges = {
        "wide (Sci > 300)":   (300.0, 1e6),
        "CLEAN (400 < Sci < 2000)": (SCI_LO_CLEAN, SCI_HI_CLEAN),
    }

    print(f"\n=== Coefficients comparison: fit on Sci>300 vs Sci∈[400,2000] ===")
    print(f"{'Range':>25s}  {'Box':>3s}  {'b':>10s}  {'1+α':>9s}  {'β (Wide)':>10s}  "
          f"{'γ (Large)':>10s}  {'k_add':>8s}  {'k_mul':>8s}")
    fits = {}
    for range_name, (lo, hi) in fit_ranges.items():
        for box in "ABC":
            mask = (df["box"] == box) & (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
            sub = df[mask]
            m1_coef = fit_m1(sub)
            m7_coef = fit_m7(sub)
            k_add = fit_additive(sub)
            k_mul = fit_multiplicative(sub)
            fits[(range_name, box)] = (m1_coef, m7_coef, k_add, k_mul)
            print(f"  {range_name:>25s}  {box:>3s}  "
                  f"{m1_coef[0]:>+10.1f}  {m1_coef[1]:>+9.4f}  "
                  f"{m1_coef[2]:>+10.4f}  {m1_coef[3]:>+10.4f}  "
                  f"{k_add:>+8.4f}  {k_mul:>+8.4f}")

    print(f"\n=== M7 coefficients comparison ===")
    print(f"{'Range':>25s}  {'Box':>3s}  {'b':>10s}  {'c_pure':>9s}  {'c_ACD1':>9s}  "
          f"{'c_ACDN':>9s}  {'β':>9s}  {'γ':>9s}")
    for range_name, (lo, hi) in fit_ranges.items():
        for box in "ABC":
            m1_coef, m7_coef, _, _ = fits[(range_name, box)]
            b, c0, c1, cN, beta, gamma = m7_coef
            print(f"  {range_name:>25s}  {box:>3s}  "
                  f"{b:>+10.1f}  {c0:>+9.4f}  {c1:>+9.4f}  {cN:>+9.4f}  "
                  f"{beta:>+9.4f}  {gamma:>+9.4f}")

    # ============ Apply CLEAN-band fit, compute residuals ============
    # Use CLEAN fit and evaluate on FULL data range
    print(f"\n=== Apply CLEAN-band fits to all data, compute residuals ===")
    models = ["M1", "M7", "add", "mul"]
    for n in models:
        df[f"resid_{n}"] = np.nan

    for box in "ABC":
        m1_coef, m7_coef, k_add, k_mul = fits[("CLEAN (400 < Sci < 2000)", box)]
        mask_apply = df["box"] == box
        sub = df[mask_apply]
        df.loc[mask_apply, "resid_M1"] = sub["pho_rate"].values - predict_m1(sub, m1_coef)
        df.loc[mask_apply, "resid_M7"] = sub["pho_rate"].values - predict_m7(sub, m7_coef)
        df.loc[mask_apply, "resid_add"] = sub["pho_rate"].values - predict_additive(sub, k_add)
        df.loc[mask_apply, "resid_mul"] = sub["pho_rate"].values - predict_multiplicative(sub, k_mul)

    print(f"\n=== RMS by Sci bin (in PHO cnt/s/det units, models fit on Sci∈[400,2000]) ===")
    bin_edges = [100, 300, 400, 600, 1000, 1500, 2000, 2500, 4500]
    header = f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>8s}" for n in models)
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in models]
        flag = "  ← FIT" if (lo >= SCI_LO_CLEAN and hi <= SCI_HI_CLEAN) else ""
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>8.1f}" for r in rmss) + flag
        print(row)

    print(f"\n=== Median residual by Sci bin ===")
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in models]
        flag = "  ← FIT" if (lo >= SCI_LO_CLEAN and hi <= SCI_HI_CLEAN) else ""
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+8.1f}" for m in meds) + flag
        print(row)

    # ============ Plot: residual vs Sci, all models ============
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    SCI_MIN, SCI_MAX = 100, 4500
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 50)
    bc = 0.5 * (bins[:-1] + bins[1:])

    colors = {"M1": "black", "M7": "C2", "add": "red", "mul": "blue"}
    for name in models:
        med = median_per_bin(df["sci_rate"].values, df[f"resid_{name}"].values, bins)
        if name == "M1":
            label = "M1 (4 params)"
        elif name == "M7":
            label = "M7 (6 params: ACD 3-channel decomp)"
        elif name == "add":
            label = "Additive: PHO = Sci+Wide+Large+k·Dt (1 param)"
        elif name == "mul":
            label = "Multiplicative: PHO·live = k·Sci+Wide+Large (1 param)"
        ax.plot(bc, med, "-", color=colors[name], lw=2, label=label, alpha=0.85)

    # Shade fit region
    ax.axvspan(SCI_LO_CLEAN, SCI_HI_CLEAN, alpha=0.1, color="green",
               label=f"Fit region Sci∈[{SCI_LO_CLEAN:.0f}, {SCI_HI_CLEAN:.0f}]")
    # Mark Sci=400 anomaly boundary
    ax.axvline(SCI_LO_CLEAN, color="purple", ls=":", lw=1.5, alpha=0.6,
               label="Sci=400 anomaly cutoff")
    ax.axvline(2500, color="red", ls=":", lw=1.5, alpha=0.6,
               label="Sci=2500 saturation onset")

    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(SCI_MIN, SCI_MAX)
    ax.set_xlabel("Sci [cnt/s/det]")
    ax.set_ylabel("median PHO residual [cnt/s/det]")
    ax.set_title("M12: refit on CLEAN band Sci∈[400, 2000]\n"
                 "(in fit region the residuals should be ~0)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = OUT_DIR / "m12_clean_band.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
