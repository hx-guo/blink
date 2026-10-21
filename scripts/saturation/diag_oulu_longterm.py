#!/usr/bin/env python3
"""OULU neutron monitor 1964-2026 (62 yr) — find the functional form of solar
modulation. Test whether a fixed-period sinusoid even works over 5-6 cycles, and
measure cycle asymmetry (fast fall / slow rise of cosmic rays)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# parse OULU daily
dates, cr = [], []
for line in open("/tmp/oulu_long.txt"):
    s = line.strip()
    if not s or s.startswith("#") or ";" not in s:
        continue
    d, v = s.split(";")
    try:
        cr.append(float(v)); dates.append(np.datetime64(d.strip().replace(" ", "T")))
    except ValueError:
        continue
dates = np.array(dates); cr = np.array(cr)
t = (dates - dates[0]).astype("timedelta64[D]").astype(float) / 365.25
ycal = 1964.25 + t

# monthly mean
months = dates.astype("datetime64[M]")
um = np.unique(months)
mt, mv = [], []
for m in um:
    sel = months == m
    if sel.sum() > 10:
        mt.append((m - dates[0].astype("datetime64[M]")).astype(int) / 12.0)
        mv.append(np.nanmean(cr[sel]))
mt = np.array(mt); mv = np.array(mv); myr = 1964.25 + mt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))

# Panel 1: full 62-yr series + single-sinusoid fit attempt
ax1.plot(myr, mv, "-", color="steelblue", lw=1.0, label="OULU monthly")
# try fixed-period sinusoid
def sinP(tt, k0, a, b, P):
    return k0 + a*np.cos(2*np.pi*tt/P) + b*np.sin(2*np.pi*tt/P)
popt, _ = curve_fit(sinP, mt, mv, p0=[100, 5, 5, 10.8], maxfev=20000)
ax1.plot(myr, sinP(mt, *popt), "r--", lw=1.5, label=f"single sinusoid P={popt[3]:.1f}yr (RMS={np.std(mv-sinP(mt,*popt)):.2f})")
# solar minima (CR maxima)
for ymin in [1965, 1976, 1987, 1997, 2009, 2020]:
    ax1.axvline(ymin, color="green", ls=":", lw=0.8, alpha=0.6)
ax1.axvspan(2017.5, 2026.4, color="orange", alpha=0.15, label="HXMT era")
ax1.set_xlabel("year"); ax1.set_ylabel("OULU count rate")
ax1.set_title("OULU 1964-2026: single fixed-period sinusoid CANNOT fit 62 yr (period drifts)", fontsize=12)
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# Panel 2: cycle asymmetry — CR peaks (solar minima) and troughs
# detect peaks/troughs by smoothing
from scipy.ndimage import uniform_filter1d
sm = uniform_filter1d(mv, 13)
peaks, troughs = [], []
for i in range(13, len(sm)-13):
    if sm[i] == max(sm[i-13:i+13]): peaks.append(i)
    if sm[i] == min(sm[i-13:i+13]): troughs.append(i)
peaks = sorted(set(peaks)); troughs = sorted(set(troughs))
ax2.plot(myr, mv, "-", color="steelblue", lw=1.0)
ax2.plot(myr, sm, "-", color="navy", lw=2, label="13-mo smooth")
ax2.plot(myr[peaks], mv[peaks], "g^", ms=10, label="CR max (solar min)")
ax2.plot(myr[troughs], mv[troughs], "rv", ms=10, label="CR min (solar max)")
ax2.axvspan(2017.5, 2026.4, color="orange", alpha=0.15)
ax2.set_xlabel("year"); ax2.set_ylabel("OULU count rate")
ax2.set_title("cycle peaks/troughs — measure rise vs fall asymmetry", fontsize=12)
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

fig.suptitle("OULU 62-yr: functional form of solar modulation", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("plots/diag_oulu_longterm.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("Saved plots/diag_oulu_longterm.png")

# print cycle timing
print(f"\nsingle-sinusoid P fit = {popt[3]:.2f} yr, RMS = {np.std(mv-sinP(mt,*popt)):.2f}")
print(f"CR maxima (solar minima) years: {[f'{myr[p]:.1f}' for p in peaks]}")
print(f"CR minima (solar maxima) years: {[f'{myr[p]:.1f}' for p in troughs]}")
pk = [myr[p] for p in peaks]
if len(pk) > 1:
    print(f"peak-to-peak intervals (yr): {[f'{pk[i+1]-pk[i]:.1f}' for i in range(len(pk)-1)]}")
