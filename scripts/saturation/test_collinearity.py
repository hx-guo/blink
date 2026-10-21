#!/usr/bin/env python3
"""Test whether the γ bimodality is a real discrete state or an OLS
multicollinearity artifact.

For each per-(date,box) observation:
  - Compute Pearson(Sci, Large), Pearson(Sci, Wide)
  - Also Sci range / std
  - Compare these stats between low_γ and high_γ obs
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PER_OBS_FIT = Path("plots/per_obs_free_gamma_fits.csv")
CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE_PARTIAL = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots")
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


def main():
    fits = pd.read_csv(PER_OBS_FIT)
    fits["state"] = pd.cut(fits["gamma"], bins=[-1, 0.4, 0.8, 2],
                          labels=["low_γ", "mixed", "high_γ"])

    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
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
    g = df.groupby(["date", "box", "met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date", "box", "met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN].copy()
    df["sci_rate"] = df["Sci"] / df["length"]
    df["wide_rate"] = df["Wide"] / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

    # HV filter to normal mode
    hv = pd.read_csv(HV_TABLE_PARTIAL,
                     dtype={"date": "string", "met_sec": "int64",
                            **{f"hv{i}": "float32" for i in range(18)}})
    hv = hv.set_index(["date", "met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    print(f"normal-mode rows: {len(df):,}")

    # Per (date, box) compute correlations and ranges
    print("Per-obs correlations + range...")
    results = []
    for (date, box), sub in df.groupby(["date", "box"], observed=True):
        if len(sub) < 500:
            continue
        results.append({
            "date": date, "box": box,
            "n": len(sub),
            "corr_sci_large": sub[["sci_rate","large_rate"]].corr().iloc[0, 1],
            "corr_sci_wide":  sub[["sci_rate","wide_rate"]].corr().iloc[0, 1],
            "corr_wide_large": sub[["wide_rate","large_rate"]].corr().iloc[0, 1],
            "sci_min": sub["sci_rate"].min(),
            "sci_max": sub["sci_rate"].max(),
            "sci_std": sub["sci_rate"].std(),
            "sci_range_ratio": sub["sci_rate"].std() / max(sub["sci_rate"].mean(), 1),
            "wide_mean": sub["wide_rate"].mean(),
            "large_mean": sub["large_rate"].mean(),
        })
    res = pd.DataFrame(results)
    print(f"  obs: {len(res):,}")

    # Merge with state
    res = res.merge(fits[["date", "box", "state", "gamma"]],
                    on=["date", "box"], how="left")
    res = res[res["state"].isin(["low_γ", "high_γ"])].copy()
    print(f"  with state: {len(res):,}")

    print(f"\nStats per state:")
    for col in ["corr_sci_large", "corr_sci_wide", "corr_wide_large",
                "sci_range_ratio", "sci_max", "wide_mean", "large_mean"]:
        for state in ["low_γ", "high_γ"]:
            d = res[res["state"] == state][col]
            print(f"  {state:>7s} {col:>20s}: n={len(d):>3d}  "
                  f"median={d.median():>8.4f}  mean={d.mean():>8.4f}  std={d.std():>8.4f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cols = ["corr_sci_large", "corr_sci_wide", "corr_wide_large",
            "sci_range_ratio", "wide_mean", "large_mean"]
    for ax, c in zip(axes.flat, cols):
        for state, color in zip(["low_γ", "high_γ"], ["C3", "C0"]):
            d = res[res["state"] == state][c]
            ax.hist(d, bins=30, alpha=0.5, color=color,
                    label=f"{state} (N={len(d)})")
        ax.set_xlabel(c)
        ax.set_ylabel("# obs")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Per-obs counter-correlations: low_γ vs high_γ")
    fig.tight_layout()
    out = OUT_DIR / "collinearity_test.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Save the analysis
    res.to_csv("plots/per_obs_collinearity.csv", index=False)


if __name__ == "__main__":
    main()
