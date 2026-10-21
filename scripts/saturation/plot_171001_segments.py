#!/usr/bin/env python3
"""Split 2017-10-01 hour into time segments to find clean steady region."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_sec_full.csv")
sub = df[(df["date"] == 20171001) & (df["box"] == "A")].copy().sort_values("met_sec")
sub["length"] = sub["L_cycles"] * 16e-6
sub["PHO_rate"] = sub["PHO"] / sub["length"]
sub["Wide_rate"] = sub["Wide"] / sub["length"]
sub["Large_rate"] = sub["Large"] / sub["length"]
sub["sci_rate"] = sub["evt"]
sub["t_min"] = (sub["met_sec"] - sub["met_sec"].min()) / 60

# Light curve
fig, axes = plt.subplots(3, 2, figsize=(14, 11))

ax = axes[0, 0]
ax.plot(sub["t_min"], sub["sci_rate"], lw=0.5, color="C0", label="Sci")
ax.plot(sub["t_min"], sub["PHO_rate"]/3, lw=0.5, alpha=0.7, color="C3", label="PHO/3")
ax.set_xlabel("min"); ax.set_ylabel("rate [cnt/s/box]")
ax.set_title("2017-10-01 Box A — light curve (find segments)")
ax.legend(); ax.grid(alpha=0.3)

# Find natural segmentation by Sci rate ranges
# Box A: Sci 4000-7600 -- the "two clusters" probably correspond to two regimes
# Plot histogram of Sci rate
ax = axes[0, 1]
ax.hist(sub["sci_rate"], bins=60, color="C0", edgecolor="k")
ax.set_xlabel("Sci rate [cnt/s/box]")
ax.set_ylabel("count")
ax.set_title("Sci rate distribution (look for bimodality)")
ax.grid(alpha=0.3)

# Apply linear fit per segment
def fit_and_resid(seg, label, ax_pred, ax_resid, color):
    if len(seg) < 50: return
    sci = seg["sci_rate"].values
    pho = seg["PHO_rate"].values
    wide = seg["Wide_rate"].values
    large = seg["Large_rate"].values
    A = np.column_stack([np.ones(len(sci)), pho, wide, large])
    c, *_ = np.linalg.lstsq(A, sci, rcond=None)
    pred = A @ c
    resid = sci - pred
    rms = np.sqrt(np.mean(resid**2))

    ax_pred.scatter(sci, pred, s=3, alpha=0.4, color=color, label=f"{label} N={len(sci)}, RMS={rms:.0f}")
    ax_resid.scatter(sci, resid, s=3, alpha=0.4, color=color, label=label)

    # binned median
    bins = np.linspace(sci.min(), sci.max(), 25)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (sci >= bins[i]) & (sci < bins[i + 1])
        med.append(np.median(resid[m]) if m.sum() > 5 else np.nan)
    ax_resid.plot(bc, med, "-", color=color, lw=2)

# Compare 4 strategies
ax_pred = axes[1, 0]
ax_resid = axes[1, 1]
n = len(sub)
fit_and_resid(sub.iloc[:n//3], "First 1/3", ax_pred, ax_resid, "C0")
fit_and_resid(sub.iloc[n//3:2*n//3], "Middle 1/3", ax_pred, ax_resid, "C1")
fit_and_resid(sub.iloc[2*n//3:], "Last 1/3", ax_pred, ax_resid, "C2")
for ax in [ax_pred, ax_resid]:
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax_pred.set_xlabel("Sci obs"); ax_pred.set_ylabel("Sci pred")
ax_pred.set_title("Linear fit: time segments (1/3, 1/3, 1/3)")
ax_resid.set_xlabel("Sci obs"); ax_resid.set_ylabel("residual")
ax_resid.set_title("Residual vs Sci")
ax_resid.axhline(0, color="r", ls="--")

# Bottom: Sci-rate threshold split (low vs high)
ax_pred = axes[2, 0]
ax_resid = axes[2, 1]
median_sci = sub["sci_rate"].median()
low = sub[sub["sci_rate"] < median_sci]
high = sub[sub["sci_rate"] >= median_sci]
fit_and_resid(low, f"Low half (Sci<{median_sci:.0f})", ax_pred, ax_resid, "C0")
fit_and_resid(high, f"High half (Sci≥{median_sci:.0f})", ax_pred, ax_resid, "C3")
for ax in [ax_pred, ax_resid]:
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax_pred.set_xlabel("Sci obs"); ax_pred.set_ylabel("Sci pred")
ax_pred.set_title("Linear fit: low-rate vs high-rate halves")
ax_resid.set_xlabel("Sci obs"); ax_resid.set_ylabel("residual")
ax_resid.set_title("Residual vs Sci")
ax_resid.axhline(0, color="r", ls="--")

fig.tight_layout()
out = "plots/171001_segments.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")
