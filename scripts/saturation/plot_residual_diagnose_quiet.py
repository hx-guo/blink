#!/usr/bin/env python3
"""Find what drives the residual non-linearity on the quiet day."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COEFS = {
    "A": {"a0": +107.0, "a1": 0.676, "a2": -1.318, "a3": -0.633},
    "B": {"a0": +116.0, "a1": 0.646, "a2": -1.247, "a3": -0.579},
    "C": {"a0": +128.0, "a1": 0.599, "a2": -1.236, "a3": -0.509},
}

df = pd.read_csv("per_sec_full.csv")
sub = df[df["date"] == 20180315].copy()
sub["length"] = sub["L_cycles"] * 16e-6
for col in ["PHO", "Wide", "Large", "Dt"]:
    sub[f"{col}_rate"] = sub[col] / sub["length"]
sub["sci_rate"] = sub["evt"]

def predict_box(row):
    c = COEFS[row["box"]]
    return 6 * c["a0"] + c["a1"] * row["PHO_rate"] + c["a2"] * row["Wide_rate"] + c["a3"] * row["Large_rate"]

sub["sci_pred"] = sub.apply(predict_box, axis=1)
sub["resid"] = sub["sci_rate"] - sub["sci_pred"]

# Hardness, Wide-fraction, Dead-fraction
sub["hardness"] = sub["Large_rate"] / sub["sci_rate"].clip(1)
sub["wide_frac"] = sub["Wide_rate"] / sub["PHO_rate"].clip(1)
sub["large_frac"] = sub["Large_rate"] / sub["PHO_rate"].clip(1)
sub["dead_frac"] = sub["Dt"] / (sub["L_cycles"] * 16e-6) / 16e-6  # proper fraction
# Actually Dt is already in seconds (from script that multiplied by 16e-6); fraction = Dt/length
sub["dead_frac"] = sub["Dt"] / sub["length"]

# === Plot residual vs candidate factors ===
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}

for ax, factor, label in zip(
    axes.flat,
    ["sci_rate", "PHO_rate", "Wide_rate", "Large_rate",
     "wide_frac", "large_frac", "hardness", "dead_frac"],
    ["Sci rate", "PHO rate", "Wide rate", "Large rate",
     "Wide / PHO", "Large / PHO", "Large / Sci", "Dt / Length"],
):
    for box in "ABC":
        bd = sub[sub["box"] == box]
        ax.scatter(bd[factor], bd["resid"], s=3, alpha=0.3,
                    color=box_colors[box], label=f"Box {box}", rasterized=True)
    # Pearson on log if rate, linear if fraction
    all_x = sub[factor].values; all_y = sub["resid"].values
    finite = np.isfinite(all_x) & np.isfinite(all_y)
    rho = np.corrcoef(all_x[finite], all_y[finite])[0, 1] if finite.sum() > 10 else np.nan
    ax.axhline(0, color="r", ls="--", lw=1)
    # binned median
    bins = np.linspace(np.percentile(all_x, 1), np.percentile(all_x, 99), 25)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (all_x >= bins[i]) & (all_x < bins[i + 1])
        med.append(np.median(all_y[m]) if m.sum() > 10 else np.nan)
    ax.plot(bc, med, "k-", lw=2, alpha=0.85)
    ax.set_xlabel(label)
    ax.set_ylabel("Sci_obs − Sci_pred [cnt/s/box]")
    ax.set_title(f"{label}   ρ = {rho:+.3f}")
    ax.grid(alpha=0.3)
    if ax == axes[0, 0]:
        ax.legend(fontsize=8, loc="lower right")

fig.suptitle(f"What drives residual on quiet day 2018-03-15? (Sci_obs − Sci_pred per second per box)",
             fontsize=11)
fig.tight_layout()
out = "plots/residual_diagnose_180315.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")

# Top correlations
print(f"\n=== Pearson ρ(residual, factor) ===")
for f in ["sci_rate", "PHO_rate", "Wide_rate", "Large_rate",
          "wide_frac", "large_frac", "hardness", "dead_frac"]:
    rho = sub[[f, "resid"]].corr().iloc[0, 1]
    print(f"  {f:>15s}: ρ = {rho:+.3f}")
