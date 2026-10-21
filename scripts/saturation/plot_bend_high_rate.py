#!/usr/bin/env python3
"""Analyze bend with full rate range (no 5-95% filter)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0

# Load Box A only — this is the burst-saturated date 260226
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

# Per-(second, det) rates
sci = (Sci / length_s[:, None]).flatten()
pho = (PHO / length_s[:, None]).flatten()
wide = (Wide / length_s[:, None]).flatten()
large = (Large / length_s[:, None]).flatten()

print(f"Total bins: {len(sci)}")
print(f"Sci range: {sci.min():.0f} – {sci.max():.0f}, median = {np.median(sci):.0f}, p99 = {np.percentile(sci, 99):.0f}")

# Two filters: relaxed (most data) and "non-saturated" (PHO check)
# The "saturation" we want to avoid is FIFO-saturation, not high rate per se
# Use a filter that excludes only obvious FIFO saturation events
mask_full = (sci > 50) & (pho > 100) & (sci < 5000) & np.isfinite(sci) & np.isfinite(pho)
mask_quiet = (sci > 50) & (pho > 100) & (sci < 1500) & np.isfinite(sci) & np.isfinite(pho)
print(f"  Full range bins (Sci<5000): {mask_full.sum()}")
print(f"  Quiet bins (Sci<1500): {mask_quiet.sum()}")

# === Fit on quiet, predict on full range ===
A_q = np.column_stack([np.ones(mask_quiet.sum()), pho[mask_quiet], wide[mask_quiet], large[mask_quiet]])
c_q, *_ = np.linalg.lstsq(A_q, sci[mask_quiet], rcond=None)
print(f"\nFit on quiet (Sci<1500): a0={c_q[0]:.0f}, a1={c_q[1]:.4f}, a2={c_q[2]:.4f}, a3={c_q[3]:.4f}")

# Apply on full range
pred_lin = c_q[0] + c_q[1] * pho + c_q[2] * wide + c_q[3] * large

# === Fit on full range ===
A_f = np.column_stack([np.ones(mask_full.sum()), pho[mask_full], wide[mask_full], large[mask_full]])
c_f, *_ = np.linalg.lstsq(A_f, sci[mask_full], rcond=None)
A2_f = np.column_stack([np.ones(mask_full.sum()), pho[mask_full], wide[mask_full], large[mask_full],
                        pho[mask_full]**2, pho[mask_full]*large[mask_full]])
c2_f, *_ = np.linalg.lstsq(A2_f, sci[mask_full], rcond=None)
pred_f_lin = c_f[0] + c_f[1] * pho + c_f[2] * wide + c_f[3] * large
pred_f_quad = c2_f[0] + c2_f[1] * pho + c2_f[2] * wide + c2_f[3] * large + c2_f[4] * pho**2 + c2_f[5] * pho * large

# === Plot ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

panels = [
    (mask_full, pred_lin, "Linear refit on QUIET only,\napplied to full range",
     "C0", "If bend exists, this should make it most visible"),
    (mask_full, pred_f_lin, "Linear refit on FULL range",
     "C1", "Still has bend if non-linearity is real"),
    (mask_full, pred_f_quad, "Quadratic refit on FULL range",
     "C2", "Bend should disappear with PHO² + PHO·L"),
]

for col, (mask, pred, label, color, note) in enumerate(panels):
    # Top: pred vs obs
    ax = axes[0, col]
    sci_m = sci[mask]; pred_m = pred[mask]
    ax.scatter(sci_m, pred_m, s=1, alpha=0.1, color=color, rasterized=True)
    lo, hi = max(sci_m.min(), 100), sci_m.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("Sci predicted")
    rms = np.sqrt(np.mean((sci_m - pred_m) ** 2))
    ax.set_title(f"{label}\nRMS = {rms:.0f}")
    ax.grid(alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # Bottom: residual vs Sci
    ax = axes[1, col]
    resid = sci_m - pred_m
    ax.scatter(sci_m, resid, s=1, alpha=0.1, color=color, rasterized=True)
    ax.axhline(0, color="r", ls="--", lw=1)
    bins = np.linspace(sci_m.min(), sci_m.max(), 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (sci_m >= bins[i]) & (sci_m < bins[i + 1])
        med.append(np.median(resid[m]) if m.sum() > 5 else np.nan)
    ax.plot(bc, med, "k-", lw=2)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("Residual [cnt/s/det]")
    ax.set_title(f"{note}")
    ax.set_ylim(-1500, 500)
    ax.grid(alpha=0.3)

fig.suptitle("Bend visibility: 260226 Box A, full rate range up to 5000 cnt/s/det\n"
             "Top: pred vs obs. Bottom: residual vs obs (the median curve shows the bend).",
             fontsize=11)
fig.tight_layout()
out = "plots/bend_high_rate.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")
print(f"\nQuadratic coefs (full range):")
print(f"  Sci = {c2_f[0]:+.0f} + {c2_f[1]:+.4f}·PHO + {c2_f[2]:+.4f}·W + {c2_f[3]:+.4f}·L")
print(f"        + {c2_f[4]:+.3e}·PHO² + {c2_f[5]:+.3e}·PHO·L")
