#!/usr/bin/env python3
"""Why does 2020-04-15 have two branches? Investigate."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_sec_full.csv")
sub = df[df["date"] == 20200415].copy()
sub["length"] = sub["L_cycles"] * 16e-6
sub["PHO_rate"] = sub["PHO"] / sub["length"]
sub["Wide_rate"] = sub["Wide"] / sub["length"]
sub["Large_rate"] = sub["Large"] / sub["length"]
sub["sci_rate"] = sub["evt"]
sub["t_min"] = (sub["met_sec"] - sub["met_sec"].min()) / 60.0
# Hardness ratio
sub["W_PHO"] = sub["Wide_rate"] / sub["PHO_rate"]
sub["L_PHO"] = sub["Large_rate"] / sub["PHO_rate"]
sub["S_PHO"] = sub["sci_rate"] / sub["PHO_rate"]

# Use Box A only for clarity
ba = sub[sub["box"] == "A"].sort_values("t_min")
print(f"2020-04-15 Box A: {len(ba)} bins, t range 0–{ba['t_min'].max():.1f} min")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# (1) Light curve PHO + Sci
ax = axes[0, 0]
ax.plot(ba["t_min"], ba["PHO_rate"], lw=0.5, color="C3", label="PHO", alpha=0.8)
ax.plot(ba["t_min"], ba["sci_rate"], lw=0.5, color="C0", label="Sci")
ax.plot(ba["t_min"], ba["Wide_rate"]*5, lw=0.5, color="C1", label="Wide×5")
ax.plot(ba["t_min"], ba["Large_rate"], lw=0.5, color="C2", label="Large")
ax.set_xlabel("min"); ax.set_ylabel("rate [cnt/s/box]")
ax.set_yscale("log")
ax.set_title("Light curves (Box A)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (2) Sci vs PHO scatter, colored by time
ax = axes[0, 1]
sc = ax.scatter(ba["PHO_rate"], ba["sci_rate"], c=ba["t_min"], cmap="viridis",
                 s=3, alpha=0.5)
plt.colorbar(sc, ax=ax, label="time [min]")
ax.set_xlabel("PHO rate"); ax.set_ylabel("Sci rate")
ax.set_title("Sci vs PHO colored by time — TWO BRANCHES?")
ax.grid(alpha=0.3)

# (3) Sci/PHO ratio over time
ax = axes[1, 0]
ax.plot(ba["t_min"], ba["S_PHO"], lw=0.7, color="C0")
ax.set_xlabel("min"); ax.set_ylabel("Sci / PHO")
ax.set_title("Sci/PHO ratio over time")
ax.grid(alpha=0.3)

# (4) Large/PHO ratio (hardness) over time
ax = axes[1, 1]
ax.plot(ba["t_min"], ba["L_PHO"], lw=0.7, color="C2", label="Large/PHO")
ax.plot(ba["t_min"], ba["W_PHO"], lw=0.7, color="C1", label="Wide/PHO")
ax.set_xlabel("min"); ax.set_ylabel("ratio")
ax.set_title("Spectral ratios over time (hardness indicator)")
ax.legend(); ax.grid(alpha=0.3)

# (5) PHO histogram - bimodal?
ax = axes[2, 0]
ax.hist(ba["PHO_rate"], bins=80, color="C3", edgecolor="k")
ax.set_xlabel("PHO rate")
ax.set_ylabel("count")
ax.set_title("PHO rate distribution")
ax.grid(alpha=0.3)

# (6) split by clear feature: e.g. t_min < 22 (flare) vs t_min > 22
# Looking at flare day 200415, flare onset is at ~22 min from earlier light curve
# Look at the (Sci_obs, Sci_pred) for two halves with universal coefs
COEFS_A = {"a0": +107.0, "a1": 0.676, "a2": -1.318, "a3": -0.633}
ba["sci_pred"] = (6 * COEFS_A["a0"] +
                   COEFS_A["a1"] * ba["PHO_rate"] +
                   COEFS_A["a2"] * ba["Wide_rate"] +
                   COEFS_A["a3"] * ba["Large_rate"])
ba["resid"] = ba["sci_rate"] - ba["sci_pred"]

ax = axes[2, 1]
sc = ax.scatter(ba["sci_rate"], ba["sci_pred"], c=ba["t_min"], cmap="viridis",
                 s=3, alpha=0.5)
plt.colorbar(sc, ax=ax, label="time [min]")
lo, hi = ba["sci_rate"].min(), ba["sci_rate"].max()
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlabel("Sci obs"); ax.set_ylabel("Sci pred")
ax.set_title("pred vs obs colored by time")
ax.grid(alpha=0.3)

fig.suptitle("2020-04-15 Box A: investigating two branches", fontsize=11)
fig.tight_layout()
out = "plots/200415_branches.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")

# Identify branches: split by hardness L/PHO
# Use median as boundary
median_L_PHO = ba["L_PHO"].median()
hard = ba[ba["L_PHO"] > median_L_PHO]
soft = ba[ba["L_PHO"] <= median_L_PHO]
print(f"\nHard branch (L/PHO > {median_L_PHO:.3f}): N={len(hard)}, median t={hard['t_min'].median():.1f}, "
      f"Sci median={hard['sci_rate'].median():.0f}")
print(f"Soft branch (L/PHO <= {median_L_PHO:.3f}): N={len(soft)}, median t={soft['t_min'].median():.1f}, "
      f"Sci median={soft['sci_rate'].median():.0f}")
