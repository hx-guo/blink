#!/usr/bin/env python3
"""Hypothesis: 18 detectors have different collimators (some narrow-FOV,
some wide-FOV, some blocked). At high Sci, on-source detectors see source
directly (lower band, big negative residual); blocked/wide detectors see
mostly background (upper band, small residual).

Diagnostic:
  - Plot residual_M1 distribution per det_global (0..17), at high Sci
  - Compute median residual per det at high Sci
  - If certain dets cluster at residual ≈ 0 and others at residual ≈ -600,
    collimator effect is confirmed
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
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN].copy()
    df["sci_rate"]   = df["Sci"]   / df["length"]
    df["wide_rate"]  = df["Wide"]  / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"]   = df["PHO"]   / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

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
    pho_corr = df["pho_rate"] - df["_beta"]*df["wide_rate"] - df["_gamma"]*df["large_rate"]
    df["sci_pred_m1"] = (pho_corr - df["_b"]) / (1 + df["_alpha"])
    df["resid_m1"] = df["sci_pred_m1"] - df["sci_rate"]

    # Statistics per det_global at high Sci
    print(f"\n=== Per-detector residual stats at Sci > {HIGH_SCI} ===")
    hi = df[df["sci_rate"] > HIGH_SCI].copy()
    print(f"{'det':>4s} {'N':>10s} {'med_resid':>10s} {'med_self_frac':>14s} "
          f"{'med_sci':>10s}")
    stats = []
    for d in range(18):
        sub = hi[hi["det_global"] == d]
        if len(sub) == 0:
            continue
        s = {
            "det": d,
            "n": len(sub),
            "med_resid": sub["resid_m1"].median(),
            "med_self_frac": sub["sci_self_frac"].median(),
            "med_sci": sub["sci_rate"].median(),
        }
        stats.append(s)
        print(f"  {d:>2d}  {s['n']:>10,d}  {s['med_resid']:>10.1f}  "
              f"{s['med_self_frac']:>14.4f}  {s['med_sci']:>10.0f}")

    stats_df = pd.DataFrame(stats)

    # Plot: 18-panel grid showing residual histogram per det at high Sci
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=True, sharey=True)
    boxes = ["A"]*6 + ["B"]*6 + ["C"]*6
    for d in range(18):
        ax = axes[d // 6, d % 6]
        sub = hi[hi["det_global"] == d]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes, ha="center")
            continue
        ax.hist(sub["resid_m1"], bins=80, alpha=0.7, color="steelblue",
                range=(-1500, 500))
        ax.axvline(0, color="r", ls="--", lw=1)
        ax.axvline(sub["resid_m1"].median(), color="k", ls=":", lw=1.5,
                   label=f"med={sub['resid_m1'].median():.0f}")
        ax.set_title(f"Box {boxes[d]} det {d%6}  (global {d})\n"
                     f"N={len(sub):,}, self_frac={sub['sci_self_frac'].median():.3f}",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(-1500, 500)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Per-detector M1 residual distribution at Sci > {HIGH_SCI}",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "per_det_residual_hi.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Quick visualization: scatter med_resid vs med_self_frac
    fig, ax = plt.subplots(figsize=(8, 6))
    for box, color in zip("ABC", ["C0","C1","C2"]):
        d_range = {"A": range(0,6), "B": range(6,12), "C": range(12,18)}[box]
        m = stats_df[stats_df["det"].isin(d_range)]
        ax.scatter(m["med_self_frac"], m["med_resid"], s=80, c=color,
                   edgecolors="black", label=f"Box {box}")
        for _, row in m.iterrows():
            ax.annotate(f"d{int(row['det'])%6}", (row["med_self_frac"], row["med_resid"]),
                        fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("Median Sci_self / Sci_total (per det, high Sci)")
    ax.set_ylabel("Median M1 residual [cnt/s/det]")
    ax.set_title(f"Per-detector behavior at Sci > {HIGH_SCI}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = OUT_DIR / "per_det_residual_summary.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
