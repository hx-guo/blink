#!/usr/bin/env python3
"""Universal formula demo on a quiet day (2018-03-15).
Uses per-second engineering + Sci data (summed over 6 detectors per box).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Per-detector universal coefficients (from 274-day fit median)
COEFS = {
    "A": {"a0": +107.0, "a1": 0.676, "a2": -1.318, "a3": -0.633},
    "B": {"a0": +116.0, "a1": 0.646, "a2": -1.247, "a3": -0.579},
    "C": {"a0": +128.0, "a1": 0.599, "a2": -1.236, "a3": -0.509},
}

# Load
df = pd.read_csv("per_sec_full.csv")
print(f"Available dates in per_sec_full: {sorted(df['date'].unique())}")
day = 20180315
sub = df[df["date"] == day].copy()
print(f"\n{day}: {len(sub)} per-(box, second) rows")

# Convert to rates
sub["length"] = sub["L_cycles"] * 16e-6
for col in ["PHO", "Wide", "Large", "Dt"]:
    sub[f"{col}_rate"] = sub[col] / sub["length"]
# Sci is already per second
sub["sci_rate"] = sub["evt"]  # evt was counted per integer-second bin

# Apply universal formula at box level (6× detector-level constants)
def predict_box(row):
    c = COEFS[row["box"]]
    # Per-box: Sci_total = 6·a0 + a1·PHO_sum + a2·W_sum + a3·L_sum
    return 6 * c["a0"] + c["a1"] * row["PHO_rate"] + c["a2"] * row["Wide_rate"] + c["a3"] * row["Large_rate"]

sub["sci_pred"] = sub.apply(predict_box, axis=1)
sub["resid"] = sub["sci_rate"] - sub["sci_pred"]

# Sort by box and time
sub = sub.sort_values(["box", "met_sec"])

# === Plot ===
fig, axes = plt.subplots(3, 3, figsize=(16, 11))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}

for col, box in enumerate("ABC"):
    bd = sub[sub["box"] == box].copy()
    if len(bd) == 0: continue
    t0 = bd["met_sec"].min()
    bd["t_min"] = (bd["met_sec"] - t0) / 60.0  # minutes

    # Top: light curves
    ax = axes[0, col]
    ax.plot(bd["t_min"], bd["sci_rate"], "C0-", lw=0.7, alpha=0.9, label="Sci observed")
    ax.plot(bd["t_min"], bd["sci_pred"], "C3-", lw=0.7, alpha=0.7, label="Sci predicted")
    ax.set_xlabel("Minutes since file start")
    ax.set_ylabel(f"Box {box}: Sci [cnt/s/box]")
    ax.set_title(f"Box {box}: light curve")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # Middle: scatter
    ax = axes[1, col]
    ax.scatter(bd["sci_rate"], bd["sci_pred"], s=3, alpha=0.4,
                color=box_colors[box], rasterized=True)
    lo, hi = max(bd["sci_rate"].min() * 0.9, 0), bd["sci_rate"].quantile(0.99) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci observed [cnt/s/box]")
    ax.set_ylabel("Sci predicted [cnt/s/box]")
    rms_box = np.sqrt(np.mean(bd["resid"] ** 2))
    ax.set_title(f"Box {box}: predicted vs observed (RMS={rms_box:.0f})")
    ax.grid(alpha=0.3)

    # Bottom: residual histogram
    ax = axes[2, col]
    ax.hist(bd["resid"], bins=50, color=box_colors[box], alpha=0.7, edgecolor="k")
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.axvline(bd["resid"].mean(), color="k", ls="-", lw=1.5,
                label=f"mean = {bd['resid'].mean():+.0f}")
    ax.set_xlabel("Sci_obs − Sci_pred [cnt/s/box]")
    ax.set_ylabel("count")
    ax.set_title(f"Box {box}: residual distribution\n"
                  f"σ = {bd['resid'].std():.0f} cnt/s/box ({bd['resid'].std()/6:.0f} per-det)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# Filter for non-saturated for overall RMS
all_resid = sub["resid"].values
filt = (sub["sci_rate"] > sub["sci_rate"].quantile(0.05)) & \
       (sub["sci_rate"] < sub["sci_rate"].quantile(0.95))
rms_filt = np.sqrt(np.mean(sub["resid"][filt] ** 2))
rms_per_det = rms_filt / 6  # box has 6 dets, summing decreases per-det rate

fig.suptitle(f"Universal formula on QUIET day 2018-03-15 (3 boxes, 1 hour)\n"
             f"Sci_pred = 6·a₀ + a₁·PHO + a₂·Wide + a₃·Large  (per-box constants)\n"
             f"5-95% bin RMS = {rms_filt:.0f} cnt/s/box (~{rms_per_det:.0f} cnt/s/det)",
             fontsize=11)
fig.tight_layout()
out = "plots/universal_quiet_180315.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")

print(f"\n=== Summary 2018-03-15 ===")
print(f"  Per-box RMS: {rms_filt:.1f} cnt/s/box")
print(f"  Per-detector equivalent: ~{rms_per_det:.1f} cnt/s/det")
for box in "ABC":
    bd = sub[sub["box"] == box]
    if len(bd) > 0:
        print(f"  Box {box}: <Sci>={bd['sci_rate'].mean():.0f}, <pred>={bd['sci_pred'].mean():.0f}, "
              f"σ_resid={bd['resid'].std():.0f} cnt/s/box, "
              f"bias={bd['resid'].mean():+.0f}")
