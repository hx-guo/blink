#!/usr/bin/env python3
"""274-date parameter time evolution analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter

df = pd.read_csv("coef_table_big.csv")
print(f"Rows: {len(df)}, dates: {df['date'].nunique()}, range: {df['date'].min()} to {df['date'].max()}")

def date_to_year(s):
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.year + (dt - datetime(dt.year, 1, 1)).total_seconds() / (365.25 * 86400)

df["year"] = df["date"].apply(date_to_year)
df = df.sort_values("year").reset_index(drop=True)

# Per-detector time evolution
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}

for ax, param, title in zip(axes.flat,
                             ["a0", "a1", "a2", "a3"],
                             ["a₀ (intercept) [cnt/s]", "a₁ (PHO coef)",
                              "a₂ (Wide coef)", "a₃ (Large coef)"]):
    # Per-box mean across 6 detectors per date
    for box in "ABC":
        sub = df[df["box"] == box].groupby("year")[param].agg(["mean", "std", "count"]).reset_index()
        ax.fill_between(sub["year"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                         color=box_colors[box], alpha=0.15)
        ax.plot(sub["year"], sub["mean"], "o", color=box_colors[box],
                 markersize=2.5, alpha=0.5)

    # Smoothed trend per box (rolling median over time)
    for box in "ABC":
        sub = df[df["box"] == box].groupby("year")[param].mean().reset_index().sort_values("year")
        if len(sub) >= 31:
            window = min(31, len(sub) // 4 * 2 + 1)
            smooth = savgol_filter(sub[param].values, window, 3)
            ax.plot(sub["year"], smooth, "-", color=box_colors[box], lw=2.5,
                    label=f"Box {box} (smoothed)")

    ax.set_xlabel("Year")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

fig.suptitle(f"Parameter time evolution: {df['date'].nunique()} dates × 18 detectors = {len(df)} measurements",
             fontsize=12)
fig.tight_layout()
out = "plots/274dates_evolution.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")

# === Statistical summary ===
print(f"\n=== Linear time trends per box × parameter ===")
print(f"{'Box':>3s} {'Param':>5s}  {'slope/yr':>10s}  {'R²':>5s}  {'mean':>8s}  {'std':>7s}  {'8yr drift':>10s}")
for box in "ABC":
    for param in ["a0", "a1", "a2", "a3"]:
        sub = df[df["box"] == box]
        agg = sub.groupby("year")[param].mean().reset_index()
        x = agg["year"].values; y = agg[param].values
        A = np.column_stack([np.ones_like(x), x])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ c
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        drift_8yr = c[1] * 8
        print(f"  {box} {param:>5s}  {c[1]:>+10.5f}  {r2:>5.3f}  {y.mean():>8.3f}  {y.std():>7.3f}  {drift_8yr:>+10.3f}")
