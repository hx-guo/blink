#!/usr/bin/env python3
"""Where does the bend come from? Test on 260226 box A with successive models."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0

# Load Box A only
fe = fits.open("data/1B/2026/20260226/0766/HXMT_1B_0766_20260226T100000_G076262_000_004.fits", memmap=True)
d = fe["HE_Eng"].data
offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
L_cycles = d["Length_Time_Cycle"].astype(float)
length_s = L_cycles * 16e-6
PHO = np.column_stack([d[f"Cnt_PHODet_{i}"].astype(float) for i in range(6)])
Wide = np.column_stack([d[f"Cnt_CsI_PHODet_{i}"].astype(float) for i in range(6)])
Large_raw = np.column_stack([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in range(6)])
Large = np.column_stack([unwrap_large(PHO[:, i], Large_raw[:, i]) for i in range(6)])
fe.close()

df = pd.read_csv("/tmp/260226_boxA_acd.csv", usecols=["type", "met", "det_id"],
                 dtype={"type": "category", "met": "float64", "det_id": "int8"})
df = df[df["type"] == "EVT"]

Sci = np.zeros((len(met_eng), 6))
for det in range(6):
    evts = np.sort(df.loc[df["det_id"] == det, "met"].values)
    for i in range(len(met_eng)):
        t0 = met_eng[i]; t1 = t0 + length_s[i]
        Sci[i, det] = np.searchsorted(evts, t1) - np.searchsorted(evts, t0)

# Flatten to per-(second, det) rates
sci = (Sci / length_s[:, None]).flatten()
pho = (PHO / length_s[:, None]).flatten()
wide = (Wide / length_s[:, None]).flatten()
large = (Large / length_s[:, None]).flatten()

# Filter
mask = (sci > 100) & (pho > 100) & (sci < np.percentile(sci, 95))
sci_f = sci[mask]; pho_f = pho[mask]; wide_f = wide[mask]; large_f = large[mask]
print(f"After filter: {mask.sum()} bins")

# === Three models ===
# Model 1: Linear (4 coefs)
A1 = np.column_stack([np.ones(len(sci_f)), pho_f, wide_f, large_f])
c1, *_ = np.linalg.lstsq(A1, sci_f, rcond=None)
pred1 = A1 @ c1
resid1 = sci_f - pred1
rms1 = np.sqrt(np.mean(resid1 ** 2))
print(f"Linear (4 coefs):     RMS = {rms1:.1f}")

# Model 2: + quadratic PHO² and PHO·L
A2 = np.column_stack([np.ones(len(sci_f)), pho_f, wide_f, large_f,
                      pho_f ** 2, pho_f * large_f])
c2, *_ = np.linalg.lstsq(A2, sci_f, rcond=None)
pred2 = A2 @ c2
resid2 = sci_f - pred2
rms2 = np.sqrt(np.mean(resid2 ** 2))
print(f"+ PHO² + PHO·L:       RMS = {rms2:.1f}")

# Model 3: All 2nd-order interactions (10 coefs)
A3 = np.column_stack([np.ones(len(sci_f)), pho_f, wide_f, large_f,
                      pho_f ** 2, wide_f ** 2, large_f ** 2,
                      pho_f * wide_f, pho_f * large_f, wide_f * large_f])
c3, *_ = np.linalg.lstsq(A3, sci_f, rcond=None)
pred3 = A3 @ c3
resid3 = sci_f - pred3
rms3 = np.sqrt(np.mean(resid3 ** 2))
print(f"All 2nd-order (10):    RMS = {rms3:.1f}")

# === Plot ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

models = [
    (pred1, resid1, rms1, "Linear (4 coefs)", "C0"),
    (pred2, resid2, rms2, "+ PHO² + PHO·L (6 coefs)", "C1"),
    (pred3, resid3, rms3, "All 2nd-order (10 coefs)", "C2"),
]

for col, (pred, resid, rms, label, color) in enumerate(models):
    # Top: pred vs obs
    ax = axes[0, col]
    ax.scatter(sci_f, pred, s=2, alpha=0.2, color=color, rasterized=True)
    lo, hi = sci_f.min(), sci_f.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("Sci predicted")
    ax.set_title(f"{label}\nRMS = {rms:.1f} cnt/s")
    ax.grid(alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # Bottom: residual vs Sci_obs
    ax = axes[1, col]
    ax.scatter(sci_f, resid, s=2, alpha=0.15, color=color, rasterized=True)
    ax.axhline(0, color="r", ls="--", lw=1)
    # Binned median
    bins = np.linspace(sci_f.min(), sci_f.max(), 25)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (sci_f >= bins[i]) & (sci_f < bins[i + 1])
        med.append(np.median(resid[m]) if m.sum() > 5 else np.nan)
    ax.plot(bc, med, "k-", lw=2)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("Residual [cnt/s/det]")
    ax.set_title(f"Residual median curve")
    ax.set_ylim(-200, 200)
    ax.grid(alpha=0.3)

fig.suptitle("Where does the bend come from? 260226 Box A, per-date refit, no ACD filter\n"
             "Watch the median residual curve in bottom row — bend disappears with quadratic terms",
             fontsize=11)
fig.tight_layout()
out = "plots/bend_analysis.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")

# Print quadratic coefficients to interpret physics
print(f"\n=== Quadratic model coefficients (for physical interpretation) ===")
print(f"  Sci = {c2[0]:+.0f} + {c2[1]:+.4f}·PHO + {c2[2]:+.4f}·W + {c2[3]:+.4f}·L")
print(f"        + {c2[4]:+.3e}·PHO² + {c2[5]:+.3e}·PHO·L")
# What's the magnitude at typical (PHO=1500, L=500)?
typ_pho2 = c2[4] * 1500 ** 2
typ_phoL = c2[5] * 1500 * 500
print(f"  At PHO=1500, L=500: PHO² term = {typ_pho2:+.0f} cnt/s,  PHO·L term = {typ_phoL:+.0f} cnt/s")
