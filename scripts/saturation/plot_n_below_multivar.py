#!/usr/bin/env python3
"""Multivariate regression for N_below per detector.

Test which factors actually explain the residual structure beyond linear-in-Sci:
  N_below_rate = b + α·Sci + β·Wide + γ·Large + δ·DeadFrac + ε·t

Then make residual diagnostic plots vs each candidate.
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


det_data = []
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
    Dt = np.column_stack([d[f"DeadTime_PHODet_{i}"].astype(float) for i in det_ids]) * 16e-6  # seconds

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
        v = valid_box
        det_global = det_ids[det_local]
        det_data.append({
            "box": box_name,
            "det_local": det_local,
            "det_global": det_global,
            "sci": Sci[v, det_local] / length_s[v],
            "nb": N_below[v, det_local] / length_s[v],
            "wide": Wide[v, det_local] / length_s[v],
            "large": Large[v, det_local] / length_s[v],
            "deadfrac": Dt[v, det_local] / length_s[v],
            "t_rel": met_eng[v] - met_eng[v].min(),
            "pho": PHO[v, det_local] / length_s[v],
        })
    fe.close()


# === Multivariate fit per detector ===
# Model: nb = b + a*Sci + c*Wide + d*Large + e*DeadFrac + f*t (linear)
print("\n=== Multivariate fit  N_below = b + a·Sci + c·Wide + d·Large + e·DeadFrac + f·t ===")
print(f"{'Box':>3s} {'D':>2s}  {'b':>6s} {'a(Sci)':>8s} {'c(W)':>7s} {'d(L)':>7s} {'e(Df)':>8s} {'f(t)':>9s}  {'RMS%':>5s}  {'RMS_lin%':>9s}")

results = []
for dd in det_data:
    sci = dd["sci"]; nb = dd["nb"]; wide = dd["wide"]
    large = dd["large"]; df = dd["deadfrac"]; t = dd["t_rel"]
    # Standardize t to avoid scale issues, but keep b interpretable as cnt/s
    X = np.column_stack([np.ones_like(sci), sci, wide, large, df, t / 1000.0])
    coef, *_ = np.linalg.lstsq(X, nb, rcond=None)
    pred = X @ coef
    resid = nb - pred
    rms_full = np.sqrt(np.mean(resid ** 2)) / np.mean(nb) * 100
    # Linear-only baseline
    Xlin = np.column_stack([np.ones_like(sci), sci])
    coeflin, *_ = np.linalg.lstsq(Xlin, nb, rcond=None)
    rms_lin = np.sqrt(np.mean((nb - Xlin @ coeflin) ** 2)) / np.mean(nb) * 100
    results.append({"dd": dd, "coef": coef, "rms_full": rms_full, "rms_lin": rms_lin, "resid_lin": nb - Xlin @ coeflin})
    print(f"{dd['box']:>3s} {dd['det_local']:>2d}  {coef[0]:>6.1f} {coef[1]:>8.3f} {coef[2]:>7.3f} "
          f"{coef[3]:>7.3f} {coef[4]:>8.1f} {coef[5]:>9.2f}  {rms_full:>4.1f}%  {rms_lin:>7.1f}%")

# === Plot 1: residual of linear fit vs each candidate factor (per box average, plus all 18 lines faint) ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
factors = ["wide", "large", "deadfrac", "t_rel", "pho"]
factor_labels = {
    "wide": "Wide rate [cnt/s]",
    "large": "Large rate [cnt/s]",
    "deadfrac": "Dead fraction = Dt/Length",
    "t_rel": "MET - start [s]",
    "pho": "PHO rate [cnt/s]",
}
plot_factors = ["wide", "large", "deadfrac", "t_rel", "pho"]
det_colors = plt.cm.tab10(np.arange(6))
box_idx = {"A": 0, "B": 1, "C": 2}

# Top row: residual vs Wide rate, Large rate, DeadFrac
# Bottom row: residual vs t, PHO rate, hardness ratio
for k, factor in enumerate(plot_factors[:5]):
    if k < 3:
        ax = axes[0, k]
    elif k == 3:
        ax = axes[1, 0]
    else:
        ax = axes[1, k - 3]

    for r in results:
        x = r["dd"][factor]
        y = r["resid_lin"]
        c = det_colors[r["dd"]["det_local"]]
        marker = ["o", "s", "^"][box_idx[r["dd"]["box"]]]
        ax.scatter(x, y, s=2, alpha=0.15, color=c, rasterized=True)

    # binned median across all 18 detectors
    all_x = np.concatenate([r["dd"][factor] for r in results])
    all_y = np.concatenate([r["resid_lin"] for r in results])
    bins = np.linspace(np.percentile(all_x, 1), np.percentile(all_x, 99), 25)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (all_x >= bins[i]) & (all_x < bins[i + 1])
        med.append(np.median(all_y[m]) if m.sum() > 5 else np.nan)
    ax.plot(bc, med, "k-", lw=2, label="binned median (all dets)")

    # Pearson correlation across all detectors
    valid_corr = np.isfinite(all_x) & np.isfinite(all_y)
    if valid_corr.sum() > 100:
        rho = np.corrcoef(all_x[valid_corr], all_y[valid_corr])[0, 1]
    else:
        rho = np.nan
    ax.set_title(f"{factor_labels[factor]}   ρ={rho:.3f}")
    ax.axhline(0, color="r", ls="--", lw=0.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel(factor_labels[factor])
    ax.set_ylabel("Resid (linear) [cnt/s]")
    ax.legend(loc="upper right", fontsize=8)

# Last subplot: hardness ratio = Large/(Sci+1)
ax = axes[1, 2]
for r in results:
    x = r["dd"]["large"] / np.maximum(r["dd"]["sci"], 1)
    y = r["resid_lin"]
    c = det_colors[r["dd"]["det_local"]]
    ax.scatter(x, y, s=2, alpha=0.15, color=c, rasterized=True)
all_x = np.concatenate([r["dd"]["large"] / np.maximum(r["dd"]["sci"], 1) for r in results])
all_y = np.concatenate([r["resid_lin"] for r in results])
bins = np.linspace(np.percentile(all_x, 1), np.percentile(all_x, 99), 25)
bc = 0.5 * (bins[:-1] + bins[1:])
med = []
for i in range(len(bins) - 1):
    m = (all_x >= bins[i]) & (all_x < bins[i + 1])
    med.append(np.median(all_y[m]) if m.sum() > 5 else np.nan)
ax.plot(bc, med, "k-", lw=2, label="binned median")
rho = np.corrcoef(all_x[np.isfinite(all_x) & np.isfinite(all_y)],
                   all_y[np.isfinite(all_x) & np.isfinite(all_y)])[0, 1]
ax.set_title(f"Hardness = Large/Sci   ρ={rho:.3f}")
ax.axhline(0, color="r", ls="--", lw=0.8)
ax.grid(alpha=0.3)
ax.set_xlabel("Large / Sci")
ax.set_ylabel("Resid (linear) [cnt/s]")

fig.suptitle("Linear-fit residuals  (N_below_obs − [b + α·Sci])  vs candidate factors\n"
             "Color = detector id within box (0..5); all 3 boxes overlaid", fontsize=11)
fig.tight_layout()
out = "plots/n_below_residual_diagnostics_260226.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# === Summary: how much improvement does each extra term give? ===
# Try adding ONE term at a time to baseline (b + a*Sci)
print("\n=== Single-term improvement over baseline (b + a·Sci) ===")
extra_terms = [("Wide", "wide"), ("Large", "large"), ("DeadFrac", "deadfrac"),
               ("Hardness=L/Sci", None), ("Time", "t_rel")]
for name, key in extra_terms:
    rms_pcts = []
    for dd in det_data:
        sci = dd["sci"]; nb = dd["nb"]
        if name == "Hardness=L/Sci":
            extra = dd["large"] / np.maximum(dd["sci"], 1)
        else:
            extra = dd[key]
        X = np.column_stack([np.ones_like(sci), sci, extra])
        c, *_ = np.linalg.lstsq(X, nb, rcond=None)
        rms = np.sqrt(np.mean((nb - X @ c) ** 2)) / np.mean(nb) * 100
        rms_pcts.append(rms)
    print(f"  + {name:>16s}:  mean RMS = {np.mean(rms_pcts):.1f}%  (baseline ~26%)")

print("Done.")
