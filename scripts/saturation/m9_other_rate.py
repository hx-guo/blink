#!/usr/bin/env python3
"""M9: redo pile-up test with orthogonal basis to avoid M8's multicollinearity.

Key changes from M8:
  1. Use `other_rate = group_rate - sci_rate` (5 OTHER dets in PDAU group).
     This isolates "OTHER" pile-up contribution from this det's own Sci.
  2. Compute residual in PHO units directly (no division by sci_coef).
     For cross-model comparison, convert to "Sci-equivalent" using a FIXED
     reference 1+α = 2.24 (M1 average), so all residuals are on the same scale.

Models:
  M1:  PHO = (1+α)·Sci + β·Wide + γ·Large + b
  M9a: PHO = (1+α)·Sci + β·Wide + γ·Large + δ·other_rate + b
  M9b: PHO = (1+α)·Sci + β·Wide + γ·Large + δ·other_rate + ε·other_rate²/R0 + b
  M9c: PHO = (1+α)·Sci + β·Wide + γ·Large + δ·other_rate + ε·(sci_rate × other_rate)/R0 + b

  R0 = 10000 cnt/s normalization.

Physical interpretation:
  δ > 0 : pile-up gain — OTHER dets being busy adds PHO to THIS det (sub-thresh → PHO band)
  δ < 0 : dead-time loss — OTHER dets being busy steals ADC cycles
  ε on other_rate²: quadratic pile-up
  ε on sci×other: joint pile-up (this det fires AND other fires)
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
R0 = 10000.0
SCI_REF = 2.24  # reference 1+α from M1, used to convert PHO-resid to Sci-equiv


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
    df["group_rate"]  = df["sci_sec_total"] / df["length"]
    df["other_rate"]  = (df["group_rate"] - df["sci_rate"]) / R0  # normalized
    df["other2"]      = df["other_rate"] ** 2
    df["sci_x_other"] = (df["sci_rate"] / R0) * df["other_rate"]
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


def fit_lstsq(sub, terms):
    var = {
        "1":       np.ones(len(sub)),
        "Sci":     sub["sci_rate"].values,
        "Wide":    sub["wide_rate"].values,
        "Large":   sub["large_rate"].values,
        "Other":   sub["other_rate"].values,
        "Other2":  sub["other2"].values,
        "SciOth":  sub["sci_x_other"].values,
    }
    X = np.column_stack([var[t] for t in terms])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    cond = np.linalg.cond(X)
    return dict(zip(terms, coef)), cond


def predict_pho(sub, coefs):
    pho_pred = np.zeros(len(sub))
    for t, c in coefs.items():
        v = {
            "1":       1.0,
            "Sci":     sub["sci_rate"].values,
            "Wide":    sub["wide_rate"].values,
            "Large":   sub["large_rate"].values,
            "Other":   sub["other_rate"].values,
            "Other2":  sub["other2"].values,
            "SciOth":  sub["sci_x_other"].values,
        }[t]
        pho_pred += c * v
    return pho_pred


def median_per_bin(x, y, bins, min_count=200):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    models = {
        "M1":   ["1", "Sci", "Wide", "Large"],
        "M9a":  ["1", "Sci", "Wide", "Large", "Other"],
        "M9b":  ["1", "Sci", "Wide", "Large", "Other", "Other2"],
        "M9c":  ["1", "Sci", "Wide", "Large", "Other", "Other2", "SciOth"],
    }

    print(f"\n=== Coefficients (per Box) ===")
    print(f"{'Box':>4s} {'Model':>5s}  {'cond':>10s}  "
          f"{'b':>10s} {'1+α':>9s} {'β':>9s} {'γ':>9s} "
          f"{'δ(Other)':>10s} {'ε(Oth²)':>10s} {'ζ(SciOth)':>11s}")
    residuals_pho = {}
    for box in "ABC":
        for name, terms in models.items():
            mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
            coefs, cond = fit_lstsq(df[mask_fit], terms)
            b      = coefs.get("1", 0)
            c_sci  = coefs.get("Sci", 1.0)
            beta   = coefs.get("Wide", 0)
            gamma  = coefs.get("Large", 0)
            d_o    = coefs.get("Other", np.nan)
            e_o2   = coefs.get("Other2", np.nan)
            z_so   = coefs.get("SciOth", np.nan)
            print(f"  {box}    {name:>5s}  {cond:>10.1f}  "
                  f"{b:>+10.1f} {c_sci:>+9.4f} {beta:>+9.4f} {gamma:>+9.4f} "
                  f"{d_o:>+10.2f} {e_o2:>+10.2f} {z_so:>+11.4f}")
            mask_apply = df["box"] == box
            pho_pred = predict_pho(df[mask_apply], coefs)
            resid_pho = df.loc[mask_apply, "pho_rate"].values - pho_pred
            residuals_pho[(box, name)] = (mask_apply, resid_pho)

    # Convert PHO residual to Sci-equivalent using FIXED SCI_REF for all models
    for name in models:
        col = f"resid_{name}"
        df[col] = np.nan
        for box in "ABC":
            mask, res_pho = residuals_pho[(box, name)]
            df.loc[mask, col] = -res_pho / SCI_REF  # negate: if PHO under-predicted, sci_pred over-predicted

    # ============ RMS by Sci bin ============
    print(f"\n=== RMS by Sci bin (Sci-equivalent units, common scale) ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    header = f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>9s}" for n in models)
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>9.1f}" for r in rmss)
        print(row)

    print(f"\n=== Median residual by Sci bin ===")
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+9.1f}" for m in meds)
        print(row)

    # ============ RMS by GROUP rate bin ============
    print(f"\n=== RMS by group_rate bin ===")
    g_edges = [1800, 4000, 6000, 8000, 10000, 14000, 25000]
    header2 = f"{'group_rate':>15s}  {'N':>10s}  " + "  ".join(f"{n:>9s}" for n in models)
    print(header2)
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>9.1f}" for r in rmss)
        print(row)

    print(f"\n=== Median residual by group_rate bin ===")
    print(header2)
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+9.1f}" for m in meds)
        print(row)

    # ============ Plot: residual vs group_rate, one panel per model ============
    fig, axes = plt.subplots(1, len(models), figsize=(20, 5), sharey=True)
    G_MIN, G_MAX = 1800, 25000
    bins = np.logspace(np.log10(G_MIN), np.log10(G_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])
    for ax, name in zip(axes, models):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub = df[df["box"] == box]
            med = median_per_bin(sub["group_rate"].values,
                                 sub[f"resid_{name}"].values, bins)
            ax.plot(bc, med, "-", color=color, lw=2, label=f"Box {box}")
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(G_MIN, G_MAX)
        ax.set_ylim(-700, 250)
        terms_str = " + ".join(t for t in models[name] if t != "1")
        ax.set_title(f"{name}: {terms_str}", fontsize=9)
        ax.set_xlabel("group_rate [cnt/s/box]")
        if name == "M1":
            ax.set_ylabel("Sci-equiv residual [cnt/s/det]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("M9: orthogonal basis using other_rate (5 OTHER dets in PDAU)",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m9_other_rate.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
