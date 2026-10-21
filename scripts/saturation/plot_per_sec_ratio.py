#!/usr/bin/env python3
"""Test FIFO/MCU saturation hypothesis: does Sci/PHO (or Sci/N_n_predicted)
   drop at high PHO rate per second?

   Sci ≡ EVT (per second, summed over 6 detectors per box)
   N_n_predicted = PHO - 2·Wide - 1.2·Large (from our universal model)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_sec_full.csv")

# All values are sums over 6 detectors. Convert to rate (cnt/s)
# L_cycles is in 16us units, length_s = L_cycles * 16e-6
df["length"] = df["L_cycles"] * 16e-6
# PHO/Wide/Large are integer counts in length seconds; rate = count / length
for col in ["PHO", "Wide", "Large", "Dt"]:
    df[f"{col}_rate"] = df[col] / df["length"]
# evt is per integer second (from solve), per-second rate ≈ evt
df["sci_rate"] = df["evt"]  # already per second

# Predicted N_normal (would-be Sci if no FIFO saturation, no CRC, etc)
df["N_n_pred"] = df["PHO_rate"] - 2 * df["Wide_rate"] - 1.2 * df["Large_rate"]
# Below = N_n_pred - sci_rate; if positive, Sci is "missing" events
df["Sci_loss"] = df["N_n_pred"] - df["sci_rate"]
df["loss_frac"] = df["Sci_loss"] / df["N_n_pred"].clip(1)

# Filter: reasonable bins (no SAA edge, no zero stuff)
df = df[(df["L_cycles"] > 50000) & (df["sci_rate"] > 100) & (df["PHO_rate"] > 100)]
print(f"After filter: {len(df)} per-second rows")

# === Plot: Sci/N_n_pred vs PHO_rate per box ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
date_styles = {
    "20171001": ("C0", "2017-10-01"),
    "20180315": ("C1", "2018-03-15"),
    "20200415": ("C2", "2020-04-15"),
    "20221009": ("C3", "2022-10-09"),
}

# Top row: per-date Sci/N_n_pred vs PHO_rate (one panel per box)
for col, box in enumerate("ABC"):
    ax = axes[0, col]
    for date, (color, label) in date_styles.items():
        sub = df[(df["date"] == int(date)) & (df["box"] == box)].copy()
        if len(sub) < 100: continue
        # Bin by PHO_rate, compute median Sci/N_n_pred
        bins = np.logspace(np.log10(max(sub["PHO_rate"].min(), 500)),
                           np.log10(sub["PHO_rate"].quantile(0.99) + 1), 30)
        sub["bin"] = pd.cut(sub["PHO_rate"], bins)
        ratio = sub["sci_rate"] / sub["N_n_pred"].clip(1)
        sub["ratio"] = ratio
        # only keep where ratio is sensible
        sub = sub[(sub["ratio"] > 0) & (sub["ratio"] < 2)]
        agg = sub.groupby("bin", observed=True).agg(pho=("PHO_rate", "mean"),
                                                      ratio=("ratio", "median"),
                                                      n=("ratio", "size")).reset_index()
        agg = agg[agg["n"] >= 5]
        ax.plot(agg["pho"], agg["ratio"], "-", color=color, marker="o", markersize=3,
                lw=1.2, label=label)
    ax.axhline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("PHO rate per box [cnt/s] (sum 6 dets)")
    ax.set_ylabel("Sci / N_n_pred (median)")
    ax.set_title(f"Box {box}")
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.3)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

# Bottom row: scatter cloud, all dates pooled, per box
for col, box in enumerate("ABC"):
    ax = axes[1, col]
    for date, (color, label) in date_styles.items():
        sub = df[(df["date"] == int(date)) & (df["box"] == box)].copy()
        if len(sub) < 100: continue
        ratio = sub["sci_rate"] / sub["N_n_pred"].clip(1)
        ax.scatter(sub["PHO_rate"], ratio, s=2, alpha=0.15, color=color, rasterized=True)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("PHO rate [cnt/s]")
    ax.set_ylabel("Sci / N_n_pred")
    ax.set_title(f"Box {box} scatter")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.5)
    ax.grid(alpha=0.3, which="both")

fig.suptitle("FIFO/MCU saturation test: does Sci/(PHO-2W-1.2L) drop at high PHO?\n"
             "Hypothesis: ratio < 1 at high rate = pre-saturation FIFO buffering loss",
             fontsize=11)
fig.tight_layout()
out = "plots/per_sec_ratio.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# === Summary: ratio at low vs high rate ===
print(f"\n=== Sci / N_n_pred summary per box ===")
for date, (color, label) in date_styles.items():
    for box in "ABC":
        sub = df[(df["date"] == int(date)) & (df["box"] == box)].copy()
        if len(sub) < 100: continue
        ratio = sub["sci_rate"] / sub["N_n_pred"].clip(1)
        sub["ratio"] = ratio
        sub = sub[(sub["ratio"] > 0) & (sub["ratio"] < 2)]
        # split by PHO_rate quartile
        q25 = sub["PHO_rate"].quantile(0.25)
        q75 = sub["PHO_rate"].quantile(0.75)
        ratio_low = sub[sub["PHO_rate"] < q25]["ratio"].median()
        ratio_high = sub[sub["PHO_rate"] > q75]["ratio"].median()
        print(f"  {label} box{box}: low-PHO ({sub['PHO_rate'].quantile(0.1):.0f}-{q25:.0f}) ratio={ratio_low:.4f}  "
              f"high-PHO ({q75:.0f}-{sub['PHO_rate'].quantile(0.9):.0f}) ratio={ratio_high:.4f}  Δ={ratio_high-ratio_low:+.4f}")
