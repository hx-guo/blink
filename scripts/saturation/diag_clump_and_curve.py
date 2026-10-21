#!/usr/bin/env python3
"""Two-pronged diagnostic:

A) Trace the LEFT CLUMP (Sci < 300 cnt/s/det in normal HV mode):
   - Compare HV, L_cycles, year, sci_sec_total, acdn_frac, wide_frac between
     clump bins and main-band bins.
   - Is it short L_cycles? HV transient? specific years?

B) Trace the UPPER-RIGHT NON-LINEARITY (Sci > 1500):
   - Fit linear M1 only on Sci<1500 (clean linear region)
   - Residual at high Sci shows deviation from linearity
   - Test second-order terms: PHO², Wide², Large², PHO·Wide, PHO·Large, Wide·Large
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
CLUMP_HI = 300.0           # Sci threshold to define "left clump"
LINEAR_HI = 1500.0         # Sci ceiling for linear fit region


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
    df["sci_rate"] = df["Sci"] / df["length"]
    df["wide_rate"] = df["Wide"] / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"] = df["PHO"] / df["length"]
    df["wide_frac"] = df["Wide"] / df["PHO"].clip(lower=1)
    df["large_frac"] = df["Large"] / df["PHO"].clip(lower=1)
    df["dt_frac"] = df["Dt"] / df["PHO"].clip(lower=1)
    df["acd1_frac"] = df["Sci_ACD1"] / df["Sci"].clip(lower=1)
    df["acdn_frac"] = df["Sci_ACDN"] / df["Sci"].clip(lower=1)
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    df["year"] = df["date"].str.slice(0, 4).astype("int16")

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

    # ==== A. CLUMP CHARACTERIZATION ====
    print(f"\n=== A. Left clump (Sci ≤ {CLUMP_HI}) vs main band (Sci > {CLUMP_HI}) ===")
    print(f"{'group':>15s} {'col':>15s} {'n':>10s} {'median':>10s} {'mean':>10s} {'std':>10s}")
    for col in ["hv", "L_cycles", "sci_sec_total", "year", "acdn_frac",
                "acd1_frac", "wide_frac", "large_frac", "dt_frac"]:
        for group, mask in [("clump", df["sci_rate"] <= CLUMP_HI),
                            ("main",  df["sci_rate"] > CLUMP_HI)]:
            d = df.loc[mask, col]
            print(f"  {group:>10s}  {col:>15s} {len(d):>10,d} "
                  f"{d.median():>10.4f} {d.mean():>10.4f} {d.std():>10.4f}")

    # ==== B. NON-LINEARITY AT HIGH SCI ====
    # Fit M1 on Sci ∈ [300, 1500] only (clean linear region), then look at residual at Sci > 1500
    print(f"\n=== B. Linear fit on {CLUMP_HI} < Sci < {LINEAR_HI}, residual at Sci > {LINEAR_HI} ===")
    fits_lin = {}
    for box in "ABC":
        sub = df[(df["box"] == box) &
                 (df["sci_rate"] > CLUMP_HI) &
                 (df["sci_rate"] < LINEAR_HI)]
        X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                             sub["wide_rate"].values, sub["large_rate"].values])
        coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
        b, one_plus_a, beta, gamma = coef
        fits_lin[box] = (b, one_plus_a - 1, beta, gamma)
        print(f"  Box {box}:  b={b:.1f}, α={one_plus_a-1:.3f}, β={beta:.3f}, γ={gamma:.4f}  "
              f"(N={len(sub):,})")

    # Now look at residual at high Sci using the LINEAR fit (extrapolated)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=False)
    for row_i, box in enumerate("ABC"):
        b, alpha, beta, gamma = fits_lin[box]
        sub = df[df["box"] == box]
        pho_corr = sub["pho_rate"].values - beta*sub["wide_rate"].values \
                    - gamma*sub["large_rate"].values
        sci_pred = (pho_corr - b) / (1 + alpha)
        sci = sub["sci_rate"].values
        resid = sci_pred - sci

        # Left panel: sci_pred vs sci, density scatter
        ax = axes[row_i, 0]
        SCI_MIN, SCI_MAX = 40.0, 5000.0
        Y_MIN, Y_MAX = 1.0, 5000.0
        sp_pos = np.maximum(sci_pred, Y_MIN * 0.5)
        keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
        ax.hexbin(sci[keep], sp_pos[keep], gridsize=80, xscale="log", yscale="log",
                  extent=(np.log10(SCI_MIN), np.log10(SCI_MAX),
                          np.log10(Y_MIN), np.log10(Y_MAX)),
                  cmap="viridis", norm=LogNorm(vmin=1), mincnt=1, rasterized=True)
        ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "r--", lw=1.5)
        ax.axvline(CLUMP_HI, color="gray", ls=":", lw=1)
        ax.axvline(LINEAR_HI, color="gray", ls=":", lw=1)
        ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlabel("Sci obs [cnt/s/det]")
        ax.set_ylabel(f"Box {box}\nSci_pred [cnt/s/det]")
        ax.set_title(f"Linear fit on [{CLUMP_HI},{LINEAR_HI}], applied to all Sci")
        ax.grid(alpha=0.3, which="both")

        # Right panel: residual (sci_pred - sci) vs sci
        ax = axes[row_i, 1]
        bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
        bc = 0.5*(bins[:-1] + bins[1:])
        med = []
        for i in range(len(bins)-1):
            m = (sci >= bins[i]) & (sci < bins[i+1])
            med.append(np.median(resid[m]) if m.sum() > 500 else np.nan)
        ax.plot(bc, np.array(med), "o-", color="orange", lw=2, ms=3,
                label="binned median residual")
        ax.axhline(0, color="r", ls="--", lw=1)
        ax.axvline(CLUMP_HI, color="gray", ls=":", lw=1, label=f"Sci={CLUMP_HI:.0f}")
        ax.axvline(LINEAR_HI, color="gray", ls=":", lw=1, label=f"Sci={LINEAR_HI:.0f}")
        ax.set_xscale("log")
        ax.set_xlim(SCI_MIN, SCI_MAX)
        ax.set_xlabel("Sci obs [cnt/s/det]")
        ax.set_ylabel("residual = Sci_pred - Sci [cnt/s/det]")
        ax.set_title(f"Box {box} residual structure")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "clump_and_curvature_diag.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
