#!/usr/bin/env python3
"""Bend analysis on 2017-10-01 — sustained high count rate, no major burst."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_sec_full.csv")
sub = df[df["date"] == 20171001].copy()
sub["length"] = sub["L_cycles"] * 16e-6
for col in ["PHO", "Wide", "Large", "Dt"]:
    sub[f"{col}_rate"] = sub[col] / sub["length"]
sub["sci_rate"] = sub["evt"]

print(f"2017-10-01 data: {len(sub)} per-(box, sec) rows")
for box in "ABC":
    bd = sub[sub["box"] == box]
    print(f"  Box {box}: Sci range {bd['sci_rate'].min():.0f}–{bd['sci_rate'].max():.0f}, "
          f"PHO range {bd['PHO_rate'].min():.0f}–{bd['PHO_rate'].max():.0f}, "
          f"median Sci={bd['sci_rate'].median():.0f}")

# Light curve for Box A
fig0, ax = plt.subplots(figsize=(13, 4))
ba = sub[sub["box"] == "A"].sort_values("met_sec")
t = (ba["met_sec"] - ba["met_sec"].min()) / 60
ax.plot(t, ba["sci_rate"], lw=0.5, label="Sci")
ax.plot(t, ba["PHO_rate"], lw=0.5, alpha=0.7, label="PHO")
ax.set_xlabel("min")
ax.set_yscale("log")
ax.set_ylabel("rate [cnt/s/box]")
ax.set_title("2017-10-01 Box A (sustained high rate)")
ax.legend(); ax.grid(alpha=0.3)
fig0.tight_layout()
fig0.savefig("plots/171001_lightcurve.png", dpi=120)

# === Bend analysis (Box A, full range) ===
ba = sub[sub["box"] == "A"]
sci = ba["sci_rate"].values
pho = ba["PHO_rate"].values
wide = ba["Wide_rate"].values
large = ba["Large_rate"].values

mask = (sci > 100) & (pho > 100) & np.isfinite(sci) & np.isfinite(pho)
print(f"\nBox A: {mask.sum()} bins after filter")
print(f"  Sci range: {sci[mask].min():.0f}–{sci[mask].max():.0f}, p99={np.percentile(sci[mask],99):.0f}")
print(f"  PHO range: {pho[mask].min():.0f}–{pho[mask].max():.0f}, p99={np.percentile(pho[mask],99):.0f}")

sci_m = sci[mask]; pho_m = pho[mask]; wide_m = wide[mask]; large_m = large[mask]

# Fit 3 models
A1 = np.column_stack([np.ones(len(sci_m)), pho_m, wide_m, large_m])
c1, *_ = np.linalg.lstsq(A1, sci_m, rcond=None)
pred1 = A1 @ c1
resid1 = sci_m - pred1
rms1 = np.sqrt(np.mean(resid1**2))

A2 = np.column_stack([np.ones(len(sci_m)), pho_m, wide_m, large_m,
                      pho_m**2, pho_m*large_m])
c2, *_ = np.linalg.lstsq(A2, sci_m, rcond=None)
pred2 = A2 @ c2
resid2 = sci_m - pred2
rms2 = np.sqrt(np.mean(resid2**2))

A3 = np.column_stack([np.ones(len(sci_m)), pho_m, wide_m, large_m,
                      pho_m**2, wide_m**2, large_m**2,
                      pho_m*wide_m, pho_m*large_m, wide_m*large_m])
c3, *_ = np.linalg.lstsq(A3, sci_m, rcond=None)
pred3 = A3 @ c3
resid3 = sci_m - pred3
rms3 = np.sqrt(np.mean(resid3**2))

print(f"\nLinear (4 coefs):     RMS = {rms1:.1f}")
print(f"+ PHO² + PHO·L:       RMS = {rms2:.1f}")
print(f"All 2nd-order (10):    RMS = {rms3:.1f}")
print(f"\nLinear coefs: a0={c1[0]:.0f}, a1={c1[1]:.4f}, a2={c1[2]:.4f}, a3={c1[3]:.4f}")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
panels = [
    (pred1, resid1, rms1, "Linear (4 coefs)", "C0"),
    (pred2, resid2, rms2, "+ PHO² + PHO·L", "C1"),
    (pred3, resid3, rms3, "All 2nd-order (10)", "C2"),
]
for col, (pred, resid, rms, label, color) in enumerate(panels):
    ax = axes[0, col]
    ax.scatter(sci_m, pred, s=2, alpha=0.2, color=color, rasterized=True)
    lo, hi = sci_m.min() * 0.95, sci_m.max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci observed [cnt/s/box]")
    ax.set_ylabel("Sci predicted")
    ax.set_title(f"{label}\nRMS = {rms:.0f}")
    ax.grid(alpha=0.3)

    ax = axes[1, col]
    ax.scatter(sci_m, resid, s=2, alpha=0.15, color=color, rasterized=True)
    ax.axhline(0, color="r", ls="--", lw=1)
    bins = np.linspace(sci_m.min(), sci_m.max(), 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (sci_m >= bins[i]) & (sci_m < bins[i+1])
        med.append(np.median(resid[m]) if m.sum() > 5 else np.nan)
    ax.plot(bc, med, "k-", lw=2)
    ax.set_xlabel("Sci observed [cnt/s/box]")
    ax.set_ylabel("Residual [cnt/s/box]")
    ax.set_title("Residual median curve")
    ax.grid(alpha=0.3)
    ax.set_ylim(-1500, 1500)

fig.suptitle(f"2017-10-01 Box A: sustained high rate (Sci 3000-7000)\n"
             f"Wider rate range than 260226 — bend should be more visible",
             fontsize=11)
fig.tight_layout()
out = "plots/bend_171001.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")
print(f"\nQuadratic coefs: a0={c2[0]:.0f}, PHO²={c2[4]:.4e}, PHO·L={c2[5]:.4e}")
