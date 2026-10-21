#!/usr/bin/env python3
"""Bend analysis on 2020-04-15 (magnetar giant flare day, sustained high rate).
   Use only quiet portion BEFORE flare to avoid FIFO saturation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0

# Load Box A
fe = fits.open("data/1B/2020/20200415/0766/HXMT_1B_0766_20200415T080000_G024828_000_004.fits", memmap=True)
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

df = pd.read_csv("/tmp/200415_boxA_acd.csv", usecols=["type", "met", "det_id"],
                 dtype={"type": "category", "met": "float64", "det_id": "int8"})
df = df[df["type"] == "EVT"]

Sci = np.zeros((len(met_eng), 6))
for det in range(6):
    evts = np.sort(df.loc[df["det_id"] == det, "met"].values)
    for i in range(len(met_eng)):
        t0 = met_eng[i]; t1 = t0 + length_s[i]
        Sci[i, det] = np.searchsorted(evts, t1) - np.searchsorted(evts, t0)

# Light curve to spot the giant flare
sci_total_per_sec = Sci.sum(axis=1)
fig0, ax = plt.subplots(figsize=(13, 4))
t_min = (met_eng - met_eng[0]) / 60
ax.plot(t_min, sci_total_per_sec, lw=0.5)
ax.set_yscale("log")
ax.set_xlabel("Minutes since start")
ax.set_ylabel("Sci total rate per box [cnt/s]")
ax.set_title("2020-04-15 Box A light curve — find quiet vs flare")
ax.grid(alpha=0.3)
fig0.tight_layout()
fig0.savefig("plots/200415_lightcurve.png", dpi=120)
print(f"Box A flare peak: {sci_total_per_sec.max():.0f} cnt/s/box")
print(f"Box A quiescent (median): {np.median(sci_total_per_sec):.0f}")

# Per-(sec, det) rates
sci = (Sci / length_s[:, None]).flatten()
pho = (PHO / length_s[:, None]).flatten()
wide = (Wide / length_s[:, None]).flatten()
large = (Large / length_s[:, None]).flatten()

print(f"\nSci per-det range: {sci[sci>0].min():.0f} – {sci.max():.0f}, p99 = {np.percentile(sci, 99):.0f}")
print(f"PHO per-det range: {pho[pho>0].min():.0f} – {pho.max():.0f}")

# Choose mask: exclude only obvious FIFO saturation (Sci suddenly drops while PHO is high)
# A simple cut: PHO < 25000 cnt/s/det (well below MCU limit which is ~2500/det)
# Actually MCU readout limit is ~15000 evt/s/box = 2500/det. So PHO ~25000 is when buffering kicks in.
mask = (sci > 50) & (pho > 100) & np.isfinite(sci) & np.isfinite(pho)
print(f"After basic filter: {mask.sum()} bins (full range)")

sci_m = sci[mask]; pho_m = pho[mask]; wide_m = wide[mask]; large_m = large[mask]

# === Three models ===
A1 = np.column_stack([np.ones(len(sci_m)), pho_m, wide_m, large_m])
c1, *_ = np.linalg.lstsq(A1, sci_m, rcond=None)
pred1 = A1 @ c1
resid1 = sci_m - pred1
rms1 = np.sqrt(np.mean(resid1 ** 2))

A2 = np.column_stack([np.ones(len(sci_m)), pho_m, wide_m, large_m,
                      pho_m**2, pho_m*large_m])
c2, *_ = np.linalg.lstsq(A2, sci_m, rcond=None)
pred2 = A2 @ c2
resid2 = sci_m - pred2
rms2 = np.sqrt(np.mean(resid2 ** 2))

A3 = np.column_stack([np.ones(len(sci_m)), pho_m, wide_m, large_m,
                      pho_m**2, wide_m**2, large_m**2,
                      pho_m*wide_m, pho_m*large_m, wide_m*large_m])
c3, *_ = np.linalg.lstsq(A3, sci_m, rcond=None)
pred3 = A3 @ c3
resid3 = sci_m - pred3
rms3 = np.sqrt(np.mean(resid3 ** 2))

print(f"\nLinear (4 coefs):     RMS = {rms1:.1f}")
print(f"+ PHO² + PHO·L:       RMS = {rms2:.1f}")
print(f"All 2nd-order (10):    RMS = {rms3:.1f}")

# === Plot ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

panels = [
    (pred1, resid1, rms1, "Linear (4 coefs)", "C0"),
    (pred2, resid2, rms2, "+ PHO² + PHO·L", "C1"),
    (pred3, resid3, rms3, "All 2nd-order (10)", "C2"),
]

for col, (pred, resid, rms, label, color) in enumerate(panels):
    # Top: pred vs obs
    ax = axes[0, col]
    ax.scatter(sci_m, pred, s=1, alpha=0.1, color=color, rasterized=True)
    lo, hi = max(sci_m.min(), 100), sci_m.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("Sci predicted")
    ax.set_title(f"{label}\nRMS = {rms:.0f} cnt/s")
    ax.grid(alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xscale("log"); ax.set_yscale("log")

    # Bottom: residual median curve
    ax = axes[1, col]
    ax.scatter(sci_m, resid, s=1, alpha=0.05, color=color, rasterized=True)
    ax.axhline(0, color="r", ls="--", lw=1)
    bins = np.logspace(np.log10(max(sci_m.min(), 100)), np.log10(sci_m.max()), 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (sci_m >= bins[i]) & (sci_m < bins[i + 1])
        med.append(np.median(resid[m]) if m.sum() > 5 else np.nan)
    ax.plot(bc, med, "k-", lw=2)
    ax.set_xlabel("Sci observed [cnt/s/det]")
    ax.set_ylabel("Residual [cnt/s/det]")
    ax.set_xscale("log")
    ax.set_title(f"Residual median curve — bend?")
    ax.grid(alpha=0.3, which="both")

fig.suptitle("Bend on 2020-04-15 Box A (magnetar giant flare, full range)\n"
             f"Sci range: {sci_m.min():.0f} – {sci_m.max():.0f} cnt/s/det",
             fontsize=11)
fig.tight_layout()
out = "plots/bend_200415.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")
print(f"\nQuadratic coefs: {c2}")
