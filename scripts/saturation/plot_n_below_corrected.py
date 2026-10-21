#!/usr/bin/env python3
"""Test corrected conservation: PHO = N_n + β·Wide + Large + N_below
=> N_below = PHO - β·Wide - Large - Sci

Per detector, find optimal β by minimizing RMS of linear fit
(N_below_β = b + α·Sci) — i.e. β that makes residual independent of Wide.

Compare β=1 (original) vs β=2 (re-trigger / pile-up hypothesis) vs fitted β.
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

    for det_local in range(6):
        v = valid_box
        det_data.append({
            "box": box_name,
            "det_local": det_local,
            "det_global": det_ids[det_local],
            "sci": Sci[v, det_local] / length_s[v],
            "wide": Wide[v, det_local] / length_s[v],
            "large": Large[v, det_local] / length_s[v],
            "pho": PHO[v, det_local] / length_s[v],
            "t_rel": met_eng[v] - met_eng[v].min(),
        })
    fe.close()


def fit_linear_in_sci(nb_rate, sci_rate):
    """Fit nb = b + α·Sci, return (b, α, RMS_pct)."""
    X = np.column_stack([np.ones_like(sci_rate), sci_rate])
    coef, *_ = np.linalg.lstsq(X, nb_rate, rcond=None)
    pred = X @ coef
    rms = np.sqrt(np.mean((nb_rate - pred) ** 2))
    return coef[0], coef[1], rms / np.mean(nb_rate) * 100


def find_optimal_beta(pho, wide, large, sci):
    """For N_below_β = PHO - β·Wide - Large - Sci, find β that minimizes RMS
    of linear-in-Sci fit. This is when residual is uncorrelated with Wide."""
    # Closed form: solve d/dβ RMS² = 0
    # Equivalent: regress N_below_old (with β=1) against (Sci, Wide) and read β-1 = c(W)
    nb1 = pho - 1 * wide - large - sci
    X = np.column_stack([np.ones_like(sci), sci, wide])
    coef, *_ = np.linalg.lstsq(X, nb1, rcond=None)
    return 1.0 + coef[2], coef[0], coef[1]  # β, b, α


# Fit per detector
print("\n=== Per-detector: find β that absorbs Wide-correlated residual ===")
print(f"{'Box':>3s} {'D':>2s}  {'β_fit':>6s}  {'b':>6s}  {'α':>6s}  "
      f"{'RMS(β=1)':>9s} {'RMS(β=2)':>9s} {'RMS(β=fit)':>11s}")

results = []
for dd in det_data:
    pho = dd["pho"]; wide = dd["wide"]; large = dd["large"]; sci = dd["sci"]

    # β = 1 (original)
    nb1 = pho - 1 * wide - large - sci
    b1, a1, rms1 = fit_linear_in_sci(nb1, sci)
    # β = 2 (hypothesis)
    nb2 = pho - 2 * wide - large - sci
    b2, a2, rms2 = fit_linear_in_sci(nb2, sci)
    # β = fitted
    beta_fit, b_fit, a_fit = find_optimal_beta(pho, wide, large, sci)
    nbf = pho - beta_fit * wide - large - sci
    bf, af, rmsf = fit_linear_in_sci(nbf, sci)

    results.append({
        "dd": dd, "beta": beta_fit,
        "b1": b1, "a1": a1, "rms1": rms1,
        "b2": b2, "a2": a2, "rms2": rms2,
        "bf": bf, "af": af, "rmsf": rmsf,
        "nb1": nb1, "nb2": nb2, "nbf": nbf,
    })
    print(f"{dd['box']:>3s} {dd['det_local']:>2d}  {beta_fit:>6.2f}  {bf:>6.1f}  {af:>6.3f}  "
          f"{rms1:>8.1f}% {rms2:>8.1f}% {rmsf:>10.1f}%")

print(f"\nBox-level β medians:")
for box in "ABC":
    bs = [r["beta"] for r in results if r["dd"]["box"] == box]
    print(f"  Box {box}: β = {np.median(bs):.2f} ± {np.std(bs):.2f}")
all_betas = [r["beta"] for r in results]
print(f"  All:   β = {np.median(all_betas):.2f} ± {np.std(all_betas):.2f}")

# === Plot 1: 18 panels with β=2 vs β=fit ===
fig, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True)
for r in results:
    dd = r["dd"]
    box_idx = "ABC".index(dd["box"])
    ax = axes[box_idx, dd["det_local"]]

    sci = dd["sci"]
    nb_orig = r["nb1"]   # β=1
    nb_corr = r["nbf"]   # β=fit

    ax.scatter(sci, nb_orig, s=2, alpha=0.18, color="C3", rasterized=True, label=f"β=1 (orig)")
    ax.scatter(sci, nb_corr, s=2, alpha=0.18, color="C0", rasterized=True, label=f"β={r['beta']:.2f}")

    # Fit lines
    xs = np.linspace(sci.min(), sci.max(), 100)
    ax.plot(xs, r["b1"] + r["a1"] * xs, "r--", lw=1.0, alpha=0.85)
    ax.plot(xs, r["bf"] + r["af"] * xs, "C0-", lw=1.4, alpha=0.85)

    ax.set_title(f"{dd['box']}{dd['det_local']}  β={r['beta']:.2f}, "
                 f"RMS {r['rms1']:.0f}%→{r['rmsf']:.0f}%", fontsize=8)
    ax.grid(alpha=0.3)
    if dd["det_local"] == 0:
        ax.set_ylabel(f"Box {dd['box']}\nN_below [cnt/s]")
    if box_idx == 2:
        ax.set_xlabel("Sci rate [cnt/s]")
    if dd["det_local"] == 0 and box_idx == 0:
        ax.legend(fontsize=7, loc="upper left", markerscale=4)

fig.suptitle("Corrected N_below = PHO − β·Wide − Large − Sci  (red: β=1 original; blue: β fitted per det)",
             fontsize=11)
fig.tight_layout()
out1 = "plots/n_below_beta_per_det_260226.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out1}")

# === Plot 2: β across detectors, RMS comparison ===
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
labels = [f"{r['dd']['box']}{r['dd']['det_local']}" for r in results]
xs = np.arange(len(results))
betas = [r["beta"] for r in results]
box_colors = {"A": "C0", "B": "C1", "C": "C2"}
colors = [box_colors[r["dd"]["box"]] for r in results]
axL.bar(xs, betas, color=colors)
axL.axhline(1.0, color="r", ls="--", label="β=1 (original)")
axL.axhline(2.0, color="g", ls="--", label="β=2 (re-trigger)")
axL.set_xticks(xs)
axL.set_xticklabels(labels, rotation=90, fontsize=8)
axL.set_ylabel("Fitted β")
axL.set_title("Per-detector β (Wide multiplier in PHO conservation)")
axL.legend()
axL.grid(alpha=0.3, axis="y")

rms1_arr = np.array([r["rms1"] for r in results])
rmsf_arr = np.array([r["rmsf"] for r in results])
rms2_arr = np.array([r["rms2"] for r in results])
axR.plot(xs, rms1_arr, "ro-", label=f"β=1 (orig, mean {rms1_arr.mean():.1f}%)")
axR.plot(xs, rms2_arr, "g^-", label=f"β=2 (mean {rms2_arr.mean():.1f}%)")
axR.plot(xs, rmsf_arr, "C0s-", label=f"β fitted (mean {rmsf_arr.mean():.1f}%)")
axR.set_xticks(xs)
axR.set_xticklabels(labels, rotation=90, fontsize=8)
axR.set_ylabel("RMS residual / <N_below> [%]")
axR.set_title("Linear-in-Sci fit quality after subtracting β·Wide")
axR.legend()
axR.grid(alpha=0.3)

fig2.tight_layout()
out2 = "plots/n_below_beta_summary_260226.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")
