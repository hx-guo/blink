#!/usr/bin/env python3
"""Pooled bend analysis on existing 4 dates × 3 boxes data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_sec_full.csv")
print(f"Loaded {len(df)} rows, dates: {sorted(df['date'].unique())}")

df["length"] = df["L_cycles"] * 16e-6
df["PHO_rate"] = df["PHO"] / df["length"]
df["Wide_rate"] = df["Wide"] / df["length"]
df["Large_rate"] = df["Large"] / df["length"]
df["sci_rate"] = df["evt"]

# Filter: drop SAA spikes & saturation, keep all rates otherwise
mask = ((df["sci_rate"] > 100) & (df["PHO_rate"] > 100)
        & (df["L_cycles"] > 50000) & np.isfinite(df["sci_rate"]))
big = df[mask].copy()
print(f"After filter: {len(big)} rows")
print(f"  Sci rate range: {big['sci_rate'].min():.0f}–{big['sci_rate'].max():.0f}")
print(f"  PHO rate range: {big['PHO_rate'].min():.0f}–{big['PHO_rate'].max():.0f}")
print(f"  median Sci: {big['sci_rate'].median():.0f}")

# Group statistics by date and box
print(f"\nPer (date, box) stats:")
for (date, box), g in big.groupby(["date", "box"]):
    print(f"  {date} {box}: N={len(g)}, Sci median={g['sci_rate'].median():.0f}, max={g['sci_rate'].max():.0f}")

# Fit 3 models on POOLED data
sci = big["sci_rate"].values
pho = big["PHO_rate"].values
wide = big["Wide_rate"].values
large = big["Large_rate"].values

A1 = np.column_stack([np.ones(len(sci)), pho, wide, large])
c1, *_ = np.linalg.lstsq(A1, sci, rcond=None)
pred1 = A1 @ c1
resid1 = sci - pred1
rms1 = np.sqrt(np.mean(resid1**2))

A2 = np.column_stack([np.ones(len(sci)), pho, wide, large,
                      pho**2, pho*large])
c2, *_ = np.linalg.lstsq(A2, sci, rcond=None)
pred2 = A2 @ c2
resid2 = sci - pred2
rms2 = np.sqrt(np.mean(resid2**2))

A3 = np.column_stack([np.ones(len(sci)), pho, wide, large,
                      pho**2, wide**2, large**2,
                      pho*wide, pho*large, wide*large])
c3, *_ = np.linalg.lstsq(A3, sci, rcond=None)
pred3 = A3 @ c3
resid3 = sci - pred3
rms3 = np.sqrt(np.mean(resid3**2))

print(f"\n{'Model':>20s}  RMS [cnt/s/box]")
print(f"  {'Linear (4 coefs)':>20s}  {rms1:.0f}")
print(f"  {'+ PHO² + PHO·L':>20s}  {rms2:.0f}")
print(f"  {'All 2nd-order (10)':>20s}  {rms3:.0f}")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
panels = [
    (pred1, resid1, rms1, "Linear (4 coefs)", "C0"),
    (pred2, resid2, rms2, "+ PHO² + PHO·L (6)", "C1"),
    (pred3, resid3, rms3, "All 2nd-order (10)", "C2"),
]
date_colors = {20171001: "C3", 20180315: "C0", 20200415: "C2", 20221009: "C1"}
for col, (pred, resid, rms, label, color) in enumerate(panels):
    # Top: pred vs obs colored by date
    ax = axes[0, col]
    for date in big["date"].unique():
        m = big["date"] == date
        ax.scatter(sci[m], pred[m], s=2, alpha=0.2,
                   color=date_colors[date], label=f"{date}", rasterized=True)
    lo, hi = sci.min() * 0.95, sci.max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci observed [cnt/s/box]")
    ax.set_ylabel("Sci predicted")
    ax.set_title(f"{label}\nPooled RMS = {rms:.0f}")
    if col == 0:
        ax.legend(fontsize=8, markerscale=4)
    ax.grid(alpha=0.3)

    # Bottom: residual vs Sci with binned median
    ax = axes[1, col]
    ax.scatter(sci, resid, s=1.5, alpha=0.05, color=color, rasterized=True)
    ax.axhline(0, color="r", ls="--", lw=1)
    bins = np.linspace(sci.min(), sci.quantile(0.99) if hasattr(sci, "quantile") else np.percentile(sci, 99), 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (sci >= bins[i]) & (sci < bins[i+1])
        med.append(np.median(resid[m]) if m.sum() > 30 else np.nan)
    ax.plot(bc, med, "k-", lw=2)
    ax.set_xlabel("Sci observed [cnt/s/box]")
    ax.set_ylabel("Residual [cnt/s/box]")
    ax.set_title("Residual median curve")
    ax.set_ylim(-1500, 1500)
    ax.grid(alpha=0.3)

fig.suptitle(f"Pooled bend analysis: 4 dates × 3 boxes ({len(big)} bins)\n"
             f"All 2nd-order RMS = {rms3:.0f}, vs linear {rms1:.0f}",
             fontsize=11)
fig.tight_layout()
out = "plots/pooled_bend.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")
