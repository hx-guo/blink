#!/usr/bin/env python3
"""Pooled bend analysis on FULL per-second dataset (272+ dates × 3 boxes).

Reproduces plots/pooled_bend.png with much more data — to characterize
non-linearity at high count rate.

Aggregates per-detector rows to per (date, box, sec) by summing
PHO/Wide/Large/Sci across the 6 detectors; L_cycles is box-level (same
for all dets in a box-second so just take det=0).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

L_THRESH = 50_000       # >0.8 sec live time
SCI_MIN = 100           # cnt/s/box minimum
PHO_MIN = 100

def load_aggregate():
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs ...")
    dtype = {
        "date": "string", "box": "category", "met_sec": "int64",
        "det": "int8", "L_cycles": "int32",
        "PHO": "int32", "Wide": "int32", "Large": "int32",
        "Sci": "int32",
    }
    use = list(dtype)
    parts = []
    for i, f in enumerate(files):
        try:
            parts.append(pd.read_csv(f, usecols=use, dtype=dtype))
        except Exception as e:
            print(f"  ERR {f.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")
    df = pd.concat(parts, ignore_index=True)
    print(f"  per-det rows = {len(df):,}")
    g = df.groupby(["date", "box", "met_sec"], observed=True).agg(
        L_cycles=("L_cycles", "max"),
        PHO=("PHO", "sum"),
        Wide=("Wide", "sum"),
        Large=("Large", "sum"),
        Sci=("Sci", "sum"),
    ).reset_index()
    print(f"  per-box-sec rows = {len(g):,}")
    return g


def main():
    g = load_aggregate()
    g["length"] = g["L_cycles"] * 16e-6
    g["sci_rate"] = g["Sci"] / g["length"]
    g["PHO_rate"] = g["PHO"] / g["length"]
    g["Wide_rate"] = g["Wide"] / g["length"]
    g["Large_rate"] = g["Large"] / g["length"]

    mask = (
        (g["L_cycles"] > L_THRESH)
        & (g["sci_rate"] > SCI_MIN)
        & (g["PHO_rate"] > PHO_MIN)
        & np.isfinite(g["sci_rate"])
    )
    big = g[mask].copy()
    print(f"\nAfter filter: {len(big):,} per-box-sec bins")
    print(f"  Sci rate: {big['sci_rate'].min():.0f}–{big['sci_rate'].max():.0f}  median {big['sci_rate'].median():.0f}")
    print(f"  PHO rate: {big['PHO_rate'].min():.0f}–{big['PHO_rate'].max():.0f}")
    print(f"  dates: {big['date'].nunique()}  boxes: {big['box'].nunique()}")

    sci = big["sci_rate"].values
    pho = big["PHO_rate"].values
    wide = big["Wide_rate"].values
    large = big["Large_rate"].values

    A1 = np.column_stack([np.ones(len(sci)), pho, wide, large])
    c1, *_ = np.linalg.lstsq(A1, sci, rcond=None)
    pred1 = A1 @ c1; resid1 = sci - pred1; rms1 = float(np.sqrt(np.mean(resid1**2)))

    A2 = np.column_stack([np.ones(len(sci)), pho, wide, large, pho**2, pho*large])
    c2, *_ = np.linalg.lstsq(A2, sci, rcond=None)
    pred2 = A2 @ c2; resid2 = sci - pred2; rms2 = float(np.sqrt(np.mean(resid2**2)))

    A3 = np.column_stack([np.ones(len(sci)), pho, wide, large,
                          pho**2, wide**2, large**2,
                          pho*wide, pho*large, wide*large])
    c3, *_ = np.linalg.lstsq(A3, sci, rcond=None)
    pred3 = A3 @ c3; resid3 = sci - pred3; rms3 = float(np.sqrt(np.mean(resid3**2)))

    print(f"\n{'Model':>22s}  RMS [cnt/s/box]")
    print(f"  {'Linear (4 coefs)':>22s}  {rms1:.0f}")
    print(f"  {'+ PHO² + PHO·L (6)':>22s}  {rms2:.0f}")
    print(f"  {'All 2nd-order (10)':>22s}  {rms3:.0f}")

    # === Plot ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        (pred1, resid1, rms1, "Linear (4 coefs)", "C0"),
        (pred2, resid2, rms2, "+ PHO² + PHO·L (6)", "C1"),
        (pred3, resid3, rms3, "All 2nd-order (10)", "C2"),
    ]
    # color by year for visual identity
    years = big["date"].str.slice(0, 4).astype(int).values
    year_uniq = sorted(np.unique(years))
    cmap = plt.cm.viridis
    yr2col = {y: cmap((i + 0.5) / len(year_uniq)) for i, y in enumerate(year_uniq)}

    for col, (pred, resid, rms, label, color) in enumerate(panels):
        ax = axes[0, col]
        for y in year_uniq:
            m = years == y
            ax.scatter(sci[m], pred[m], s=1.2, alpha=0.15, color=yr2col[y],
                       label=str(y) if col == 0 else None, rasterized=True)
        lo, hi = sci.min() * 0.95, sci.max() * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Sci observed [cnt/s/box]")
        ax.set_ylabel("Sci predicted")
        ax.set_title(f"{label}\nPooled RMS = {rms:.0f}")
        if col == 0:
            ax.legend(fontsize=7, markerscale=5, loc="upper left", ncol=2)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.scatter(sci, resid, s=0.8, alpha=0.03, color=color, rasterized=True)
        ax.axhline(0, color="r", ls="--", lw=1)
        bins = np.linspace(sci.min(), np.percentile(sci, 99.5), 40)
        bc = 0.5 * (bins[:-1] + bins[1:])
        med, qlo, qhi = [], [], []
        for i in range(len(bins) - 1):
            m = (sci >= bins[i]) & (sci < bins[i + 1])
            if m.sum() > 100:
                med.append(np.median(resid[m]))
                qlo.append(np.percentile(resid[m], 16))
                qhi.append(np.percentile(resid[m], 84))
            else:
                med.append(np.nan); qlo.append(np.nan); qhi.append(np.nan)
        med = np.array(med); qlo = np.array(qlo); qhi = np.array(qhi)
        ax.fill_between(bc, qlo, qhi, alpha=0.25, color="k")
        ax.plot(bc, med, "k-", lw=2)
        ax.set_xlabel("Sci observed [cnt/s/box]")
        ax.set_ylabel("Residual [cnt/s/box]")
        ax.set_title("Residual median ± 1σ band")
        ax.set_ylim(-2000, 2000)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Pooled bend analysis: {big['date'].nunique()} dates × 3 boxes ({len(big):,} bins)\n"
                 f"All 2nd-order RMS = {rms3:.0f}, vs linear {rms1:.0f}  (reduction: {(1 - rms3 / rms1) * 100:.0f}%)",
                 fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "pooled_bend_full.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
