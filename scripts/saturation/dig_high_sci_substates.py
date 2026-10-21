#!/usr/bin/env python3
"""At high Sci (>1500), the M1 residual vs Sci_others plot shows TWO parallel
clouds. Find what discriminates them.

Color resid vs Sci_others scatter by candidate discriminator variables:
  - large_frac, wide_frac, dt_frac, acd1_frac, acdn_frac
  - year, det_global, HV
  - Sci/Sci_others ratio (own brightness vs neighbors)
  - corr(Sci, Large) per-obs
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
HIGH_SCI = 1500.0


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
    df["sci_rate"]   = df["Sci"]   / df["length"]
    df["wide_rate"]  = df["Wide"]  / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"]   = df["PHO"]   / df["length"]
    df["wide_frac"]  = df["Wide"]  / df["PHO"].clip(lower=1)
    df["large_frac"] = df["Large"] / df["PHO"].clip(lower=1)
    df["dt_frac"]    = df["Dt"]    / df["PHO"].clip(lower=1)
    df["acd1_frac"]  = df["Sci_ACD1"] / df["Sci"].clip(lower=1)
    df["acdn_frac"]  = df["Sci_ACDN"] / df["Sci"].clip(lower=1)
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    df["year"] = df["date"].str.slice(0, 4).astype("int16")

    print("Computing Sci_others...")
    tot = df.groupby(["date","box","met_sec"], observed=True)["sci_rate"].sum()
    tot.name = "sci_tot_rate"
    df = df.merge(tot, on=["date","box","met_sec"])
    df["sci_others_rate"] = df["sci_tot_rate"] - df["sci_rate"]
    df["sci_self_frac"] = df["sci_rate"] / df["sci_tot_rate"].clip(lower=1)
    df = df.drop(columns="sci_tot_rate")

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


def main():
    df = load()

    # M1 fit per Box on main band, compute residual
    print(f"\n=== M1 fits per Box on Sci > {MAIN_BAND_LO} ===")
    for box in "ABC":
        sub = df[(df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)]
        X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                             sub["wide_rate"].values, sub["large_rate"].values])
        coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
        b, one_plus_a, beta, gamma = coef
        df.loc[df["box"] == box, "_b"] = b
        df.loc[df["box"] == box, "_alpha"] = one_plus_a - 1
        df.loc[df["box"] == box, "_beta"] = beta
        df.loc[df["box"] == box, "_gamma"] = gamma
        print(f"  Box {box}: b={b:.1f}, α={one_plus_a-1:.3f}, β={beta:.3f}, γ={gamma:.4f}")
    pho_corr = df["pho_rate"] - df["_beta"]*df["wide_rate"] - df["_gamma"]*df["large_rate"]
    df["sci_pred_m1"] = (pho_corr - df["_b"]) / (1 + df["_alpha"])
    df["resid_m1"] = df["sci_pred_m1"] - df["sci_rate"]

    # Filter to high-Sci subset
    hi = df[df["sci_rate"] > HIGH_SCI].copy()
    print(f"\nHigh-Sci subset (Sci > {HIGH_SCI}): N = {len(hi):,}")

    # Identify upper vs lower band by residual threshold
    # Use a Sci_others-dependent threshold: roughly at the saddle between the two bands
    # Empirically the saddle is around -200 cnt/s/det at Sci_others ~10000
    hi["band"] = np.where(hi["resid_m1"] > -200, "upper", "lower")
    print(hi["band"].value_counts())

    # Compare candidate variables between upper and lower
    cands = ["wide_frac", "large_frac", "dt_frac", "acd1_frac", "acdn_frac",
             "year", "det_global", "hv", "sci_self_frac", "sci_rate",
             "sci_others_rate", "PHO", "Wide", "Large"]
    print(f"\n=== Per-band stats ===")
    print(f"{'var':>15s}  {'upper med':>10s}  {'lower med':>10s}  ratio")
    for c in cands:
        u = hi.loc[hi["band"] == "upper", c].median()
        l = hi.loc[hi["band"] == "lower", c].median()
        ratio = u / l if abs(l) > 1e-6 else np.inf
        print(f"  {c:>15s}  {u:>10.4f}  {l:>10.4f}  {ratio:>6.3f}")

    # Plot: scatter resid vs Sci_others, color by each candidate
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    cmap = "plasma"
    candidates_to_plot = ["large_frac", "wide_frac", "dt_frac", "acdn_frac",
                          "year", "det_global", "hv", "sci_self_frac"]
    # Subsample for speed
    samp = hi.sample(min(80_000, len(hi)), random_state=42)
    for ax, c in zip(axes.flat, candidates_to_plot):
        x = samp["sci_others_rate"].values
        y = samp["resid_m1"].values
        col = samp[c].astype(float).values
        # robust color range: 1-99 percentile
        vmin, vmax = np.percentile(col, [1, 99])
        sc = ax.scatter(x, y, c=col, s=2, cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True, linewidths=0)
        fig.colorbar(sc, ax=ax, label=c)
        ax.axhline(0, color="r", ls="--", lw=1)
        ax.set_xlabel("Sci_others [cnt/s/det × 5 dets sum]")
        ax.set_ylabel("M1 residual [cnt/s/det]")
        ax.set_title(f"colored by {c}")
        ax.set_ylim(-1500, 1000)
        ax.grid(alpha=0.3)
    fig.suptitle(f"High-Sci subset (Sci > {HIGH_SCI}): which variable splits the two parallel bands?",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "high_sci_substates.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
