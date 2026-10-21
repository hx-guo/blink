#!/usr/bin/env python3
"""Plot M1 residual directly vs group_rate (box-total Sci rate), to test
the hypothesis that the S-curve in residual-vs-Sci is actually a
group_rate effect, not a per-det Sci effect.

Also try residual / sci_rate (normalized fractional) to see if it
collapses by group_rate alone.

If hypothesis correct:
  - Residual should be a single S-curve when plotted against group_rate
  - The curve should be similar across Boxes
  - Adding group_rate as a 4th term in M1 → M8 should flatten the residual

For comparison: plot residual vs sci_rate ALSO, so we can compare the
two visualizations side by side.
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
    df["group_rate"]  = df["sci_sec_total"] / df["length"]
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


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_m1_resid(sub, coef):
    b, c1, beta, gamma = coef
    sci_pred = (sub["pho_rate"].values - b
                - beta*sub["wide_rate"].values
                - gamma*sub["large_rate"].values) / c1
    return sci_pred - sub["sci_rate"].values


def median_per_bin(x, y, bins, min_count=200):
    med = np.full(len(bins) - 1, np.nan)
    n   = np.zeros(len(bins) - 1, dtype=int)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        n[i] = m.sum()
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med, n


def main():
    df = load()

    # Fit M1 per box
    print("\n=== M1 coefs ===")
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        coef = fit_m1(df[mask_fit])
        print(f"  Box {box}: b={coef[0]:.1f}, 1+α={coef[1]:.4f}, β={coef[2]:.4f}, γ={coef[3]:.4f}")
        mask_apply = df["box"] == box
        df.loc[mask_apply, "resid_M1"] = predict_m1_resid(df[mask_apply], coef)

    # ============ Plot: residual vs sci_rate AND group_rate, per Box ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    SCI_MIN, SCI_MAX = 300, 4500
    GRP_MIN, GRP_MAX = 1800, 27000  # ~6× per-det range
    bins_sci = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bins_grp = np.logspace(np.log10(GRP_MIN), np.log10(GRP_MAX), 40)
    bc_sci = 0.5 * (bins_sci[:-1] + bins_sci[1:])
    bc_grp = 0.5 * (bins_grp[:-1] + bins_grp[1:])

    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = df[df["box"] == box]
        med_sci, _ = median_per_bin(sub["sci_rate"].values, sub["resid_M1"].values, bins_sci)
        med_grp, _ = median_per_bin(sub["group_rate"].values, sub["resid_M1"].values, bins_grp)
        axes[0].plot(bc_sci, med_sci, "-", color=color, lw=2, label=f"Box {box}")
        axes[1].plot(bc_grp, med_grp, "-", color=color, lw=2, label=f"Box {box}")

    for ax, title, xlab, xlim in zip(
        axes,
        ["resid_M1 vs per-det Sci", "resid_M1 vs GROUP Sci (6-det sum)"],
        ["Sci [cnt/s/det]", "group_rate [cnt/s/box]"],
        [(SCI_MIN, SCI_MAX), (GRP_MIN, GRP_MAX)]
    ):
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(-700, 200)
        ax.set_xlabel(xlab)
        ax.set_ylabel("median M1 residual [cnt/s/det]")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Test: is residual a per-det Sci effect, or a GROUP rate effect?",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "diag_resid_vs_group_rate.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # ============ Numerical: residual at fixed Sci, vs group_rate ============
    print(f"\n=== resid_M1 vs (Sci × group_rate) cross-tab ===")
    sci_edges = [300, 600, 1000, 1500, 2000, 3000]
    grp_edges = [1800, 4000, 6000, 8000, 10000, 14000, 25000]
    header_label = "group_rate / Sci"
    print(f"  {header_label:>18s}  " + "  ".join(f"{s:>8d}" for s in sci_edges[:-1]))
    for j in range(len(grp_edges) - 1):
        glo, ghi = grp_edges[j], grp_edges[j+1]
        row_vals = []
        for i in range(len(sci_edges) - 1):
            slo, shi = sci_edges[i], sci_edges[i+1]
            m = ((df["sci_rate"] >= slo) & (df["sci_rate"] < shi)
                 & (df["group_rate"] >= glo) & (df["group_rate"] < ghi))
            if m.sum() < 100:
                row_vals.append("---")
            else:
                row_vals.append(f"{df.loc[m, 'resid_M1'].median():+8.0f}")
        print(f"  {glo:>6d}-{ghi:>6d}      " + "  ".join(f"{v:>8s}" for v in row_vals))

    # ============ Same table, but with COUNT instead of median ============
    print(f"\n=== Counts in same bins (to gauge statistical weight) ===")
    print(f"  {header_label:>18s}  " + "  ".join(f"{s:>8d}" for s in sci_edges[:-1]))
    for j in range(len(grp_edges) - 1):
        glo, ghi = grp_edges[j], grp_edges[j+1]
        row_vals = []
        for i in range(len(sci_edges) - 1):
            slo, shi = sci_edges[i], sci_edges[i+1]
            m = ((df["sci_rate"] >= slo) & (df["sci_rate"] < shi)
                 & (df["group_rate"] >= glo) & (df["group_rate"] < ghi))
            n = m.sum()
            row_vals.append(f"{n/1000:>7.0f}k" if n > 1000 else (f"{n:>8d}" if n > 0 else "---"))
        print(f"  {glo:>6d}-{ghi:>6d}      " + "  ".join(f"{v:>8s}" for v in row_vals))


if __name__ == "__main__":
    main()
