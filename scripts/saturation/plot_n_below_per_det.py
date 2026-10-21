#!/usr/bin/env python3
"""Per-detector engineering vs sci comparison: N_below = (PHO - Wide - Large) - Sci

Each of 18 detectors gets its own panel (Box A: 0-5, B: 6-11, C: 12-17).
Linear fit:  N_below = b + alpha * Sci
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import csv
import os
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
BOXES = [
    ("A", "0766", "/tmp/260226_boxA_full.csv", 0),
    ("B", "1009", "/tmp/260226_boxB_full.csv", 6),
    ("C", "1781", "/tmp/260226_boxC_full.csv", 12),
]

sat_intervals = {"A": [], "B": [], "C": []}
with open("/tmp/detect_260226a.csv") as f:
    for r in csv.DictReader(f):
        sat_intervals[r["box"]].append((float(r["start_met"]), float(r["stop_met"])))
for k in sat_intervals:
    sat_intervals[k].sort()


def overlaps_saturation(t0, t1, intervals):
    for s, e in intervals:
        if s < t1 and e > t0:
            return True
    return False


# Load all 18 detectors
det_data = []  # list of dicts
for box_name, eng_code, sci_csv, det_off in BOXES:
    print(f"Loading Box {box_name}...")
    eng_file = f"data/1B/2026/20260226/{eng_code}/HXMT_1B_{eng_code}_20260226T100000_G076262_000_004.fits"
    fe = fits.open(eng_file, memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
    L_cycles = d["Length_Time_Cycle"].astype(float)
    length_s = L_cycles * 16e-6

    det_ids = [det_off + i for i in range(6)]
    PHO = np.column_stack([d[f"Cnt_PHODet_{i}"].astype(float) for i in det_ids])
    Wide = np.column_stack([d[f"Cnt_CsI_PHODet_{i}"].astype(float) for i in det_ids])
    Large_raw = np.column_stack([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in det_ids])
    Large = np.column_stack([unwrap_large(PHO[:, i], Large_raw[:, i]) for i in range(6)])

    det_evts = {i: [] for i in range(6)}
    with open(sci_csv) as f:
        for r in csv.DictReader(f):
            if r["type"] == "EVT":
                det_evts[int(r["det_id"])].append(float(r["met"]))
    for k in det_evts:
        det_evts[k] = np.sort(np.array(det_evts[k]))

    Sci = np.zeros((len(met_eng), 6))
    for i in range(len(met_eng)):
        t0 = met_eng[i]
        t1 = t0 + length_s[i]
        for det in range(6):
            Sci[i, det] = np.searchsorted(det_evts[det], t1) - np.searchsorted(det_evts[det], t0)

    valid_box = np.ones(len(met_eng), dtype=bool)
    for i in range(len(met_eng)):
        t0 = met_eng[i]
        t1 = t0 + length_s[i]
        if overlaps_saturation(t0, t1, sat_intervals[box_name]):
            valid_box[i] = False
    valid_box &= (L_cycles > 50000) & (Sci.sum(axis=1) > 100)

    N_below = (PHO - Wide - Large) - Sci

    for det_local in range(6):
        det_global = det_ids[det_local]
        det_data.append({
            "box": box_name,
            "det_local": det_local,
            "det_global": det_global,
            "sci_rate": Sci[valid_box, det_local] / length_s[valid_box],
            "nb_rate": N_below[valid_box, det_local] / length_s[valid_box],
            "pho_rate": PHO[valid_box, det_local] / length_s[valid_box],
            "met": met_eng[valid_box],
        })

    fe.close()


# === Plot 1: 6 cols × 3 rows = 18 panels, one per detector ===
fig, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True)
fits_table = []
for idx, dd in enumerate(det_data):
    box_idx = "ABC".index(dd["box"])
    ax = axes[box_idx, dd["det_local"]]

    sci = dd["sci_rate"]
    nb = dd["nb_rate"]

    ax.scatter(sci, nb, s=2, alpha=0.25, color="C0", rasterized=True)

    # Linear fit
    X = np.column_stack([np.ones_like(sci), sci])
    coef, *_ = np.linalg.lstsq(X, nb, rcond=None)
    b, alpha = coef
    pred = X @ coef
    resid = nb - pred
    rms_abs = np.sqrt(np.mean(resid ** 2))
    rms_rel = rms_abs / np.mean(nb)
    fits_table.append((dd["box"], dd["det_local"], dd["det_global"], b, alpha, rms_abs, rms_rel))

    xs = np.linspace(sci.min() * 0.9, sci.max() * 1.05, 100)
    ax.plot(xs, b + alpha * xs, "r-", lw=1.2, alpha=0.85)

    # Binned median
    nb_bins = 12
    bins = np.linspace(sci.min(), np.percentile(sci, 99), nb_bins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(nb_bins):
        m = (sci >= bins[i]) & (sci < bins[i + 1])
        if m.sum() > 5:
            med.append(np.median(nb[m]))
        else:
            med.append(np.nan)
    ax.plot(bc, med, "k-", lw=1.4, alpha=0.85, zorder=5)

    ax.set_title(f"{dd['box']}{dd['det_local']}  b={b:.0f}, α={alpha:.3f}, RMS={rms_rel*100:.1f}%",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, np.percentile(sci, 99.5) * 1.05)

for r in range(3):
    axes[r, 0].set_ylabel(f"Box {'ABC'[r]}\n$N_\\mathrm{{below}}$ rate [cnt/s]")
for c in range(6):
    axes[2, c].set_xlabel("Sci rate [cnt/s]")

fig.suptitle("Per-detector: $N_\\mathrm{below} = (\\mathrm{PHO} - \\mathrm{Wide} - \\mathrm{Large}) - \\mathrm{Sci}$  vs Sci rate (260226A non-saturated)",
             fontsize=12)
fig.tight_layout()
out1 = "plots/n_below_per_det_260226.png"
os.makedirs("plots", exist_ok=True)
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"Saved: {out1}")

# === Plot 2: parameter scatter — (b, α) across 18 detectors ===
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}
for box, det_local, det_global, b, alpha, rms_abs, rms_rel in fits_table:
    axL.scatter(b, alpha, color=box_colors[box], s=80, edgecolor="k", linewidth=0.6,
                label=f"Box {box}" if det_local == 0 else None)
    axL.annotate(f"{box}{det_local}", (b, alpha), fontsize=7,
                 xytext=(3, 3), textcoords="offset points")
    axR.scatter(b, rms_rel * 100, color=box_colors[box], s=80, edgecolor="k", linewidth=0.6)
    axR.annotate(f"{box}{det_local}", (b, rms_rel * 100), fontsize=7,
                 xytext=(3, 3), textcoords="offset points")

axL.set_xlabel("b [cnt/s/det]  (constant background)")
axL.set_ylabel("α  (Sci-correlated fraction)")
axL.set_title("Per-detector fit parameters")
axL.legend()
axL.grid(alpha=0.3)

axR.set_xlabel("b [cnt/s/det]")
axR.set_ylabel("RMS residual / <N_below>  [%]")
axR.set_title("Fit quality")
axR.grid(alpha=0.3)

fig2.tight_layout()
out2 = "plots/n_below_params_per_det_260226.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")

# === Print fit table ===
print("\n=== Per-detector linear fit  N_below = b + alpha * Sci ===")
print(f"{'Box':>3s} {'Det':>3s} {'global':>6s} {'b[cnt/s]':>9s} {'alpha':>7s} {'RMS[cnt/s]':>11s} {'RMS%':>6s}")
for box, det_local, det_global, b, alpha, rms_abs, rms_rel in fits_table:
    print(f"{box:>3s} {det_local:>3d} {det_global:>6d} {b:>9.1f} {alpha:>7.3f} {rms_abs:>11.1f} {rms_rel * 100:>5.1f}%")

# Box-level stats
print("\n=== Box-level summary ===")
for box in "ABC":
    rows = [r for r in fits_table if r[0] == box]
    bs = np.array([r[3] for r in rows])
    alphas = np.array([r[4] for r in rows])
    rms_pct = np.array([r[6] for r in rows]) * 100
    print(f"Box {box}: b = {bs.mean():.1f} ± {bs.std():.1f}, α = {alphas.mean():.3f} ± {alphas.std():.3f}, RMS = {rms_pct.mean():.1f}%")
