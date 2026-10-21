#!/usr/bin/env python3
"""Test the two ORIGINAL conservation models (predating M1) against current data.

Original models (user's earlier paper):
  Red  (multiplicative): Sci = (PHO × (L - Dt) / L - CsI - Large) / k
  Blue (additive):       Sci = PHO − CsI − Large − k × Dt

Rearrange to PHO prediction (for fair RMS comparison with M1/M7/M11d):
  Multiplicative: PHO_pred = (k·Sci + CsI + Large) / (1 - Dt/L)
  Additive:       PHO_pred = Sci + CsI + Large + k·Dt

Both have only ONE free parameter k (vs M1's 4 parameters b, α, β, γ).

Ambiguity: what is "CsI" in the original?
  In HXMT HE phoswich (NaI + CsI), CsI back-scintillator catches high-energy
  events. So CsI is likely the "Large" variable. But it could also map to "Wide".

  Test BOTH mappings:
    (a) CsI = Large, "Large" term in formula = Wide
    (b) CsI = Wide,  "Large" term in formula = Large
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
MAIN_BAND_LO = 300.0


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32"}
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

    df["sci_rate"]    = df["Sci"]      / df["length"]
    df["wide_rate"]   = df["Wide"]     / df["length"]
    df["large_rate"]  = df["Large"]    / df["length"]
    df["pho_rate"]    = df["PHO"]      / df["length"]
    df["dt_rate"]     = df["Dt"]       / df["length"]  # dead-time cycle count rate
    df["dt_frac"]     = df["Dt"]       / df["L_cycles"]
    df["live_frac"]   = 1.0 - df["dt_frac"]
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
    print(f"normal-mode rows: {len(df):,}")
    return df


def fit_k_additive(sub, csi_col, large_col):
    """PHO = Sci + CsI + Large + k·Dt → fit k by least squares."""
    y = sub["pho_rate"].values - sub["sci_rate"].values - sub[csi_col].values - sub[large_col].values
    x = sub["dt_rate"].values
    # k = sum(x·y) / sum(x²)
    k = np.sum(x * y) / np.sum(x * x)
    return k


def fit_k_multiplicative(sub, csi_col, large_col):
    """PHO·(1-Dt/L) = k·Sci + CsI + Large → fit k.

    Note: the model says (PHO·live - CsI - Large) = k·Sci, so:
      y' = PHO·live - CsI - Large
      x' = Sci
      k = sum(x'·y') / sum(x'²)
    """
    y = (sub["pho_rate"].values * sub["live_frac"].values
         - sub[csi_col].values - sub[large_col].values)
    x = sub["sci_rate"].values
    k = np.sum(x * y) / np.sum(x * x)
    return k


def predict_additive(sub, k, csi_col, large_col):
    return (sub["sci_rate"].values + sub[csi_col].values
            + sub[large_col].values + k * sub["dt_rate"].values)


def predict_multiplicative(sub, k, csi_col, large_col):
    return ((k * sub["sci_rate"].values + sub[csi_col].values
             + sub[large_col].values) / sub["live_frac"].values)


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_m1(sub, coef):
    b, c1plus, beta, gamma = coef
    return b + c1plus*sub["sci_rate"].values + beta*sub["wide_rate"].values + gamma*sub["large_rate"].values


def median_per_bin(x, y, bins, min_count=50):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    # ============ Fit on full-band data (Sci > MAIN_BAND_LO), per Box ============
    print(f"\n=== Original models: fit per Box, MAIN_BAND_LO={MAIN_BAND_LO} ===")
    mappings = {
        "(a) CsI=Large, Large_term=Wide": ("large_rate", "wide_rate"),
        "(b) CsI=Wide, Large_term=Large": ("wide_rate", "large_rate"),
    }

    results = {}
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        sub = df[mask_fit]
        print(f"\n  Box {box} (N={len(sub):,}):")

        # M1 baseline
        m1_coef = fit_m1(sub)
        print(f"    M1 fit: b={m1_coef[0]:.1f}, 1+α={m1_coef[1]:.3f}, "
              f"β={m1_coef[2]:.3f}, γ={m1_coef[3]:.3f}")

        for label, (csi_col, large_col) in mappings.items():
            k_add = fit_k_additive(sub, csi_col, large_col)
            k_mul = fit_k_multiplicative(sub, csi_col, large_col)
            print(f"    {label}:  k_add={k_add:.4f},  k_mul={k_mul:.4f}")
            results[(box, label, "add")] = k_add
            results[(box, label, "mul")] = k_mul
            results[(box, label, "m1")] = m1_coef

    # ============ Apply and compute residuals ============
    model_names = ["M1"]
    for label in mappings:
        model_names.append(f"add_{label[:3]}")
        model_names.append(f"mul_{label[:3]}")
    for n in model_names:
        df[f"resid_{n}"] = np.nan

    for box in "ABC":
        mask_apply = df["box"] == box
        sub = df[mask_apply]
        m1_coef = results[(box, "(a) CsI=Large, Large_term=Wide", "m1")]
        pho_m1 = predict_m1(sub, m1_coef)
        df.loc[mask_apply, "resid_M1"] = sub["pho_rate"].values - pho_m1

        for label, (csi_col, large_col) in mappings.items():
            tag = label[:3]
            k_add = results[(box, label, "add")]
            k_mul = results[(box, label, "mul")]
            pho_add = predict_additive(sub, k_add, csi_col, large_col)
            pho_mul = predict_multiplicative(sub, k_mul, csi_col, large_col)
            df.loc[mask_apply, f"resid_add_{tag}"] = sub["pho_rate"].values - pho_add
            df.loc[mask_apply, f"resid_mul_{tag}"] = sub["pho_rate"].values - pho_mul

    # ============ RMS by Sci bin (in cnt/s/det PHO units) ============
    print(f"\n=== RMS of (PHO_obs - PHO_pred) by Sci bin [PHO cnt/s/det units] ===")
    bin_edges = [100, 300, 600, 1000, 1500, 2000, 2500, 4500]
    header = f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>10s}" for n in model_names)
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in model_names]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>10.1f}" for r in rmss)
        print(row)

    print(f"\n=== Median residual (PHO_obs - PHO_pred) by Sci bin ===")
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in model_names]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+10.1f}" for m in meds)
        print(row)

    # ============ Plot: residual vs Sci, all models ============
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    SCI_MIN, SCI_MAX = 100, 4500
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])

    colors = {
        "M1": "black",
        "add_(a)": "red",
        "add_(b)": "darkred",
        "mul_(a)": "blue",
        "mul_(b)": "darkblue",
    }
    linestyles = {
        "M1": "-",
        "add_(a)": "-",
        "add_(b)": "--",
        "mul_(a)": "-",
        "mul_(b)": "--",
    }

    # Pool across boxes
    for name in model_names:
        med = median_per_bin(df["sci_rate"].values, df[f"resid_{name}"].values, bins)
        if name == "M1":
            label = "M1 (4 params: b, α, β, γ)"
        elif name.startswith("add"):
            map_label = "CsI=Large" if "(a)" in name else "CsI=Wide"
            label = f"Additive: PHO = Sci + CsI + Large + k·Dt ({map_label}, 1 param)"
        elif name.startswith("mul"):
            map_label = "CsI=Large" if "(a)" in name else "CsI=Wide"
            label = f"Multiplicative: PHO·live = k·Sci + CsI + Large ({map_label}, 1 param)"
        ax.plot(bc, med, ls=linestyles[name], color=colors[name], lw=2,
                label=label, alpha=0.85)

    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(SCI_MIN, SCI_MAX)
    ax.set_xlabel("Sci [cnt/s/det]")
    ax.set_ylabel("median PHO residual [cnt/s/det]")
    ax.set_title("Original models (additive, multiplicative) vs M1 baseline\n"
                 "(Sci=400 sees Wide-channel come into log y-axis range)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = OUT_DIR / "m0_original_models.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
