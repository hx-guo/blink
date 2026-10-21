#!/usr/bin/env python3
"""Diagnose the sub-band splitting inside each HV mode.

Within HV=normal data we still see two parallel bands. This script:
  1. computes residual = (Sci_obs - Sci_pred) for each point
  2. scans candidate explanatory variables for bimodal split
  3. plots residual vs each candidate to spot the discriminator

Candidates examined:
  - year (epoch / PMT outgassing)
  - det (per-detector calibration spread)
  - Box (A/B/C — already split, but cross-check)
  - Wide/PHO, Large/PHO, Dt/PHO ratios (event-type mix)
  - Sci_ACD1, Sci_ACDN (anti-coincidence vetoed events)
  - CRC_box (data quality flag)
  - hv exact value (within "normal" mode, does it split?)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BETA, GAMMA = 2.0, 1.2
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32",
             "Sci_ACD1": "int32", "Sci_ACDN": "int32", "CRC_box": "int8"}
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs...")
    parts = []
    for i, f in enumerate(files):
        try:
            parts.append(pd.read_csv(f, usecols=list(dtype), dtype=dtype))
        except Exception:
            pass
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(files)}")
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH].copy()
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]
    df["sci_rate"] = df["Sci"] / df["length"]
    df["pho_corr_rate"] = (df["PHO"] - BETA*df["Wide"] - GAMMA*df["Large"]) / df["length"]
    df["nb_rate"] = df["pho_corr_rate"] - df["sci_rate"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    df["year"] = df["date"].str.slice(0, 4).astype("int16")
    df["wide_frac"] = df["Wide"] / df["PHO"].clip(lower=1)
    df["large_frac"] = df["Large"] / df["PHO"].clip(lower=1)
    df["dt_frac"] = df["Dt"] / df["PHO"].clip(lower=1)
    df["acd1_frac"] = df["Sci_ACD1"] / df["Sci"].clip(lower=1)
    df["acdn_frac"] = df["Sci_ACDN"] / df["Sci"].clip(lower=1)
    print(f"  filtered rows: {len(df):,}")

    # join HV (use partial recovered file)
    hv_path = Path("n_below_study/hv_table_partial.csv.gz")
    if not hv_path.exists():
        hv_path = HV_TABLE
    print(f"Loading HV table {hv_path}...")
    hv = pd.read_csv(hv_path,
                     dtype={"date": "string", "met_sec": "int64",
                            **{f"hv{i}": "float32" for i in range(18)}})
    print(f"  HV rows: {len(hv):,}, unique dates: {hv['date'].nunique()}")
    hv = hv.set_index(["date","met_sec"]).sort_index()

    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values],
        names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[df["hv"].notna() & (df["hv"] < -900) & (df["hv"] > -1100)].copy()
    print(f"  normal-mode rows with HV: {len(df):,}")
    return df


def main():
    df = load()

    # Global fit (b, α) on all normal-mode data (per Box)
    print(f"\nGlobal normal-mode fits per Box:")
    fits = {}
    for box in "ABC":
        sub = df[df["box"] == box]
        X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values])
        coef, *_ = np.linalg.lstsq(X, sub["nb_rate"].values, rcond=None)
        fits[box] = coef
        print(f"  Box {box}: b={coef[0]:.1f}, α={coef[1]:.4f}")

    # Per-row Sci_pred and residual
    df["b_fit"] = df["box"].map(lambda b: fits[b][0])
    df["alpha_fit"] = df["box"].map(lambda b: fits[b][1])
    df["sci_pred"] = (df["pho_corr_rate"] - df["b_fit"]) / (1 + df["alpha_fit"])
    df["residual"] = df["sci_pred"] - df["sci_rate"]
    df["resid_frac"] = df["residual"] / df["sci_rate"].clip(lower=1)

    print(f"\nResidual stats: median={df['residual'].median():.1f}, "
          f"std={df['residual'].std():.1f} cnt/s/det")
    # Bimodality check on residual itself
    h, edges = np.histogram(df["residual"].clip(-1500, 1500), bins=80)
    print(f"Residual histogram peaks:")
    for i in np.argsort(-h)[:5]:
        print(f"  [{edges[i]:>6.0f}, {edges[i+1]:>6.0f}]: {h[i]:,}")

    # Spearman correlation residual vs candidate variables
    print(f"\nSpearman correlation of |residual| with candidate vars:")
    cands = ["year", "det_global", "L_cycles", "PHO", "Wide", "Large",
             "Dt", "Sci_ACD1", "Sci_ACDN", "CRC_box", "hv",
             "wide_frac", "large_frac", "dt_frac", "acd1_frac", "acdn_frac",
             "sci_rate"]
    abs_resid = df["residual"].abs()
    # subsample for speed
    samp = df.sample(min(500_000, len(df)), random_state=42).index
    for c in cands:
        try:
            corr = df.loc[samp, c].astype(float).corr(
                abs_resid.loc[samp], method="spearman")
            print(f"  {c:>15s}:  ρ_abs={corr:+.4f}")
        except Exception as e:
            print(f"  {c:>15s}:  err {e}")

    print(f"\nSpearman correlation of SIGNED residual with candidates:")
    for c in cands:
        try:
            corr = df.loc[samp, c].astype(float).corr(
                df.loc[samp, "residual"], method="spearman")
            print(f"  {c:>15s}:  ρ={corr:+.4f}")
        except Exception:
            pass

    # Grid plot: residual vs each candidate
    candidates_to_plot = ["year", "det_global", "hv", "large_frac",
                          "wide_frac", "dt_frac", "acd1_frac", "acdn_frac"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
    for ax, c in zip(axes.flat, candidates_to_plot):
        x = df[c].astype(float).values
        y = df["residual"].values
        # hexbin if continuous; categorical if discrete
        try:
            uniq = np.unique(x)
        except Exception:
            uniq = np.array([])
        if len(uniq) < 30:
            # boxplot per category
            data = [y[x == u] for u in uniq]
            ax.boxplot(data, positions=uniq, widths=(uniq.max()-uniq.min())*0.05+0.3,
                       showfliers=False)
            ax.set_xlabel(c)
        else:
            ax.hexbin(x, y, gridsize=80, cmap="viridis",
                      norm=LogNorm(vmin=1), mincnt=1)
            ax.set_xlabel(c)
        ax.axhline(0, color="r", ls="--", lw=1)
        ax.set_ylim(-1500, 1500)
        ax.grid(alpha=0.3)
    axes[0,0].set_ylabel("residual = Sci_pred - Sci_obs [cnt/s/det]")
    axes[1,0].set_ylabel("residual = Sci_pred - Sci_obs [cnt/s/det]")
    fig.suptitle("Residual diagnostic: what splits the sub-bands in normal mode?",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "subband_diagnostic.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
