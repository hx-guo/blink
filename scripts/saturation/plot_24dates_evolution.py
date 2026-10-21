#!/usr/bin/env python3
"""24-date parameter evolution analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("coef_table_24dates.csv")
print(f"Loaded {len(df)} rows: {df['date'].nunique()} dates × {df.groupby(['box','det']).ngroups} det slots")

def date_to_year(s):
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.year + (dt - datetime(dt.year, 1, 1)).total_seconds() / (365.25 * 86400)

df["year"] = df["date"].apply(date_to_year)

# Per-detector parameter drift
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}
box_markers = {"A": "o", "B": "s", "C": "^"}

for ax, param, title in zip(axes.flat,
                             ["a0", "a1", "a2", "a3"],
                             ["a₀ (intercept)", "a₁ (PHO coef)",
                              "a₂ (Wide coef)", "a₃ (Large coef)"]):
    # Per-detector lines (faint)
    for box in "ABC":
        for det in range(6):
            sub = df[(df["box"] == box) & (df["det"] == det)].sort_values("year")
            if len(sub) < 2: continue
            ax.plot(sub["year"], sub[param], "-",
                    color=box_colors[box], alpha=0.2, lw=0.5)
            ax.scatter(sub["year"], sub[param], marker=box_markers[box],
                        color=box_colors[box], s=10, alpha=0.4)

    # Per-box mean ± std
    for box in "ABC":
        sub = df[df["box"] == box].groupby("year")[param].agg(["mean", "std"]).reset_index()
        ax.errorbar(sub["year"], sub["mean"], yerr=sub["std"], fmt=box_markers[box]+"-",
                     color=box_colors[box], lw=2, markersize=6, capsize=2,
                     label=f"Box {box} (mean ± std across 6 det)")
    ax.set_xlabel("Year")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle(f"Parameter evolution over 8 years (24 dates × 18 detectors = 432 measurements)",
             fontsize=12)
fig.tight_layout()
out = "plots/24dates_param_evolution.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")

# === Compute trend statistics ===
print(f"\n=== Linear time trend (year coefficient) per (box, parameter) ===")
print(f"{'Box':>3s} {'Param':>5s} {'slope/yr':>10s}  {'R²':>5s}  {'mean':>7s}  {'std':>6s}")
for box in "ABC":
    for param in ["a0", "a1", "a2", "a3"]:
        sub = df[df["box"] == box]
        # Aggregate per-date (mean across 6 detectors)
        agg = sub.groupby("year")[param].mean().reset_index()
        x = agg["year"].values; y = agg[param].values
        A = np.column_stack([np.ones_like(x), x])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope_yr = c[1]
        pred = A @ c
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  {box} {param:>5s} {slope_yr:>+10.4f}  {r2:>5.3f}  "
              f"{y.mean():>7.3f}  {y.std():>6.3f}")

# === Stability check: how much does each (box, det) vary across 24 dates? ===
print(f"\n=== Per-detector parameter spread across 24 dates ===")
print(f"{'Det':>3s} {'a0_std':>7s} {'a1_std':>7s} {'a2_std':>7s} {'a3_std':>7s}  {'rms_med':>8s}")
for box in "ABC":
    for det in range(6):
        sub = df[(df["box"] == box) & (df["det"] == det)]
        if len(sub) < 5: continue
        print(f"  {box}{det} {sub['a0'].std():>7.1f} {sub['a1'].std():>7.4f} "
              f"{sub['a2'].std():>7.3f} {sub['a3'].std():>7.4f}  {sub['rms'].median():>8.1f}")

# === Are some detectors much more stable than others? ===
# Sort by total variation
print(f"\n=== Detectors with most/least time-stable a₁ (PHO coef) ===")
stab = df.groupby(["box", "det"])["a1"].std().sort_values()
print("Most stable:")
print(stab.head(5))
print("Least stable:")
print(stab.tail(5))
