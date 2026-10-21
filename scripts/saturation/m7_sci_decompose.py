#!/usr/bin/env python3
"""M7: Decompose Sci into 3 disjoint channels to avoid multicollinearity.

Sci = Sci_pure + Sci_ACD1 + Sci_ACDN  (disjoint subsets, verified in diag_acdn_subset)

Models:
  M1:  PHO = (1+α)·Sci + β·Wide + γ·Large + b           [baseline]
  M7:  PHO = c0·Sci_pure + c1·Sci_ACD1 + cN·Sci_ACDN + β·Wide + γ·Large + b

  Physical meaning:
    c0 = PHO yield per pure-NaI event (no shield coincidence)
    c1 = PHO yield per NaI+1-HVT coincident event
    cN = PHO yield per NaI+multi-HVT coincident event

If the 3 channels really have different PHO yields, M7 should:
  (a) flatten the +150 bump at Sci 1000-2000 (ACDN-driven, mid range)
  (b) flatten any high-Sci residual driven by ACD1 (high range)
  (c) give physically interpretable coefficients (all > 1, monotone)

This is mathematically equivalent to M6b but with orthogonal basis →
stable individual coefficients (no shared variance).
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

    # Disjoint decomposition
    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    # Sanity
    assert (df["Sci_pure"] >= 0).all(), "Sci_pure went negative"

    df["sci_rate"]      = df["Sci"]      / df["length"]
    df["scipure_rate"]  = df["Sci_pure"] / df["length"]
    df["acd1_rate"]     = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]     = df["Sci_ACDN"] / df["length"]
    df["wide_rate"]     = df["Wide"]     / df["length"]
    df["large_rate"]    = df["Large"]    / df["length"]
    df["pho_rate"]      = df["PHO"]      / df["length"]
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
    print(f"normal-mode rows: {len(df):,}")
    return df


def fit_lstsq(sub, terms):
    var = {
        "1":       np.ones(len(sub)),
        "Sci":     sub["sci_rate"].values,
        "SciPure": sub["scipure_rate"].values,
        "ACD1":    sub["acd1_rate"].values,
        "ACDN":    sub["acdn_rate"].values,
        "Wide":    sub["wide_rate"].values,
        "Large":   sub["large_rate"].values,
    }
    X = np.column_stack([var[t] for t in terms])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    cond = np.linalg.cond(X)
    return dict(zip(terms, coef)), cond


def apply_fit(sub, coefs):
    """Predict PHO and back out sci-equivalent residual on full Sci scale."""
    pho = sub["pho_rate"].values
    contrib = {
        "1":       np.ones(len(sub)),
        "Sci":     sub["sci_rate"].values,
        "SciPure": sub["scipure_rate"].values,
        "ACD1":    sub["acd1_rate"].values,
        "ACDN":    sub["acdn_rate"].values,
        "Wide":    sub["wide_rate"].values,
        "Large":   sub["large_rate"].values,
    }
    pho_pred = np.zeros(len(sub))
    for t, c in coefs.items():
        pho_pred += c * contrib[t]
    # Residual normalized by an effective "Sci yield" so we can compare M1 and M7 on the
    # same axis. Use M1-style sci_pred = (pho - other) / coef_Sci. For M7, we need a
    # combined Sci-equivalent. Use total Sci coefficient as weighted average.
    if "Sci" in coefs:
        sci_eq_coef = coefs["Sci"]
        sci_used = sub["sci_rate"].values
    elif "SciPure" in coefs:
        # Compute weighted average yield = (c0·Sci_pure + c1·ACD1 + cN·ACDN) / Sci
        # But this varies per row. Instead, define residual in PHO units / mean(coef on Sci).
        # Simpler: report residual in PHO units relative to the mean coefficient.
        # For honest comparison: convert to Sci-equivalent by dividing by the row-weighted yield.
        c0 = coefs["SciPure"]; c1 = coefs.get("ACD1", c0); cN = coefs.get("ACDN", c0)
        sci_yield = (c0 * sub["scipure_rate"].values
                     + c1 * sub["acd1_rate"].values
                     + cN * sub["acdn_rate"].values) / np.maximum(sub["sci_rate"].values, 1e-6)
        sci_eq_coef = sci_yield  # per row
        sci_used = sub["sci_rate"].values
    else:
        sci_eq_coef = 1.0
        sci_used = 0.0
    # residual in "Sci-equivalent" units
    other = pho_pred - (sci_eq_coef * sci_used if np.isscalar(sci_eq_coef)
                        else sci_eq_coef * sci_used)
    sci_pred = (pho - (pho_pred - (sci_eq_coef if np.isscalar(sci_eq_coef) else sci_eq_coef)
                       * sci_used)) / (sci_eq_coef if np.isscalar(sci_eq_coef) else sci_eq_coef)
    return sci_pred - sub["sci_rate"].values


def median_per_bin(sci, y, bins, min_count=200):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (sci >= bins[i]) & (sci < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    models = {
        "M1":  ["1", "Sci", "Wide", "Large"],
        "M7":  ["1", "SciPure", "ACD1", "ACDN", "Wide", "Large"],
    }

    print(f"\n=== Coefficients (per Box) ===")
    print(f"{'Box':>4s} {'Model':>5s}  {'cond(X)':>10s}  "
          f"{'b':>10s} {'c_Sci/cPure':>12s} {'cACD1':>9s} {'cACDN':>9s} "
          f"{'β':>9s} {'γ':>9s}")
    residuals = {}
    for box in "ABC":
        for name, terms in models.items():
            mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
            coefs, cond = fit_lstsq(df[mask_fit], terms)
            b = coefs.get("1", 0)
            c_sci = coefs.get("Sci", coefs.get("SciPure", 0))
            c_acd1 = coefs.get("ACD1", np.nan)
            c_acdn = coefs.get("ACDN", np.nan)
            beta = coefs.get("Wide", 0)
            gamma = coefs.get("Large", 0)
            print(f"  {box}    {name:>5s}  {cond:>10.1f}  "
                  f"{b:>+10.1f} {c_sci:>+12.4f} {c_acd1:>+9.4f} {c_acdn:>+9.4f} "
                  f"{beta:>+9.4f} {gamma:>+9.4f}")
            mask_apply = df["box"] == box
            res = apply_fit(df[mask_apply], coefs)
            residuals[(box, name)] = (mask_apply, res)

    # Attach residuals to df
    for name in models:
        col = f"resid_{name}"
        df[col] = np.nan
        for box in "ABC":
            mask, res = residuals[(box, name)]
            df.loc[mask, col] = res

    # ============ RMS by Sci bin ============
    print(f"\n=== RMS by Sci bin ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    header = f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>10s}" for n in models) + "   delta%"
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in models]
        delta_pct = 100 * (rmss[1] - rmss[0]) / rmss[0]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>10.1f}" for r in rmss)
        row += f"   {delta_pct:>+6.1f}%"
        print(row)

    # ============ Median residual by Sci bin ============
    print(f"\n=== Median residual by Sci bin ===")
    print(header.rsplit("   ", 1)[0])
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+10.1f}" for m in meds)
        print(row)

    # ============ Plot: residual vs Sci, one panel per model ============
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    SCI_MIN, SCI_MAX = MAIN_BAND_LO, 4500.0
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])
    for ax, name in zip(axes, models):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub = df[df["box"] == box]
            med = median_per_bin(sub["sci_rate"].values, sub[f"resid_{name}"].values, bins)
            ax.plot(bc, med, "-", color=color, lw=2, label=f"Box {box}")
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(SCI_MIN, SCI_MAX)
        ax.set_ylim(-700, 250)
        terms_str = " + ".join(t for t in models[name] if t != "1")
        ax.set_title(f"{name}: {terms_str}", fontsize=10)
        ax.set_xlabel("Sci [cnt/s/det]")
        if name == "M1":
            ax.set_ylabel("median residual [cnt/s/det]")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("M7: orthogonal Sci decomposition (Sci_pure + ACD1 + ACDN)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m7_sci_decompose.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
