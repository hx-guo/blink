#!/usr/bin/env python3
"""Test (β, γ) stability across dates spanning 6 years (Box A only).

For each date, fit per-detector (β, γ) on quiet bins (Sci between 5th and 95th
percentile to avoid SAA / burst / saturation), and compare γ stability.

If γ ≈ 1.21 across all dates, it's a true hardware constant (not requiring
hourly recalibration).
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import csv
import glob
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0

# (date_label, FITS path glob, Sci CSV, det_offset for Box A=0)
DATES = [
    ("2020-04-15", "data/1B/2020/20200415/0766/*.fits", "/tmp/200415_boxA.csv"),
    ("2020-04-28", "data/1B/2020/20200428/0766/*.fits", "/tmp/200428_boxA.csv"),
    ("2022-10-09", "data/1B/2022/20221009/0766/*.fits", "/tmp/221009_boxA.csv"),
    ("2026-02-26", "data/1B/2026/20260226/0766/*.fits", "/tmp/260226_boxA_full.csv"),
    ("2026-04-10", "data/1B/2026/20260410/0766/*.fits", "/tmp/260410_boxA.csv"),
]


def fit_linear(y, x):
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[0], c[1], y - X @ c


def analyze_date(label, fits_glob, sci_csv):
    """Return list of dicts (per detector) with β, γ, RMS for this date."""
    fits_files = sorted(glob.glob(fits_glob))
    if not fits_files:
        print(f"  {label}: no FITS file found ({fits_glob})")
        return None
    fe = fits.open(fits_files[0], memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
    L_cycles = d["Length_Time_Cycle"].astype(float)
    length_s = L_cycles * 16e-6

    PHO = np.column_stack([d[f"Cnt_PHODet_{i}"].astype(float) for i in range(6)])
    Wide = np.column_stack([d[f"Cnt_CsI_PHODet_{i}"].astype(float) for i in range(6)])
    Large_raw = np.column_stack([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in range(6)])
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
        t0 = met_eng[i]; t1 = t0 + length_s[i]
        for det in range(6):
            Sci[i, det] = np.searchsorted(det_evts[det], t1) - np.searchsorted(det_evts[det], t0)

    # Filter: reasonable Length, total Sci between 5th and 95th percentile
    # (excludes SAA crossings AND bursts which may be saturated)
    valid = (L_cycles > 50000) & (Sci.sum(axis=1) > 100)
    sci_total = Sci.sum(axis=1)
    p5, p95 = np.percentile(sci_total[valid], [5, 95])
    valid &= (sci_total >= p5) & (sci_total <= p95)
    fe.close()

    n_total = len(met_eng); n_valid = valid.sum()
    print(f"  {label}: {n_total} bins, {n_valid} kept (5–95% Sci range)")

    out = []
    for det in range(6):
        v = valid
        sci = Sci[v, det] / length_s[v]
        wide = Wide[v, det] / length_s[v]
        large = Large[v, det] / length_s[v]
        pho = PHO[v, det] / length_s[v]

        # Fit β, γ via 4-term regression
        nb_base = pho - wide - large - sci
        X = np.column_stack([np.ones_like(sci), sci, wide, large])
        c, *_ = np.linalg.lstsq(X, nb_base, rcond=None)
        beta = 1.0 + c[2]; gamma = 1.0 + c[3]

        # RMS at fitted (β, γ)
        nb_best = pho - beta * wide - gamma * large - sci
        _, _, r = fit_linear(nb_best, sci)
        rms = np.sqrt(np.mean(r ** 2))

        # Means for context
        out.append({
            "date": label, "det": det,
            "beta": beta, "gamma": gamma, "rms": rms,
            "sci_med": np.median(sci), "wide_med": np.median(wide),
            "large_med": np.median(large), "pho_med": np.median(pho),
            "n_valid": n_valid,
        })
    return out


print("Analyzing all dates (Box A, 6 detectors each)...\n")
all_results = []
for label, fits_glob, sci_csv in DATES:
    res = analyze_date(label, fits_glob, sci_csv)
    if res:
        all_results.extend(res)

print()
print("=== Per-detector × per-date (β, γ, RMS) ===")
print(f"{'Date':>11s}  {'D':>2s}  {'β':>5s} {'γ':>5s}  RMS  <Sci> <Wide> <Large>")
for r in all_results:
    print(f"  {r['date']}  {r['det']:>2d}  {r['beta']:>5.2f} {r['gamma']:>5.2f}  "
          f"{r['rms']:>4.0f} {r['sci_med']:>5.0f} {r['wide_med']:>5.0f} {r['large_med']:>5.0f}")

# === Plot 1: γ across dates per detector ===
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
det_colors = plt.cm.tab10(np.arange(6))
date_labels = [d[0] for d in DATES]
date_idx = {d: i for i, d in enumerate(date_labels)}

# Left: γ vs date, one line per detector
for det in range(6):
    rows = [r for r in all_results if r["det"] == det]
    rows.sort(key=lambda x: date_idx[x["date"]])
    xs = [date_idx[r["date"]] for r in rows]
    ys = [r["gamma"] for r in rows]
    axes[0].plot(xs, ys, "o-", color=det_colors[det], label=f"A{det}",
                 markersize=8, linewidth=1.5)
axes[0].set_xticks(range(len(date_labels)))
axes[0].set_xticklabels(date_labels, rotation=20)
axes[0].set_ylabel("γ (Large multiplier)")
axes[0].set_title("γ stability across 6 years (Box A, 6 detectors)")
axes[0].axhline(1.21, color="k", ls="--", alpha=0.5, label="γ=1.21 (260226)")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="best", fontsize=9, ncol=2)

# Right: β vs date
for det in range(6):
    rows = [r for r in all_results if r["det"] == det]
    rows.sort(key=lambda x: date_idx[x["date"]])
    xs = [date_idx[r["date"]] for r in rows]
    ys = [r["beta"] for r in rows]
    axes[1].plot(xs, ys, "o-", color=det_colors[det], label=f"A{det}",
                 markersize=8, linewidth=1.5)
axes[1].set_xticks(range(len(date_labels)))
axes[1].set_xticklabels(date_labels, rotation=20)
axes[1].set_ylabel("β (Wide multiplier)")
axes[1].set_title("β stability across 6 years (per-detector specific?)")
axes[1].axhline(2.0, color="k", ls="--", alpha=0.5, label="β=2.0")
axes[1].grid(alpha=0.3)
axes[1].legend(loc="best", fontsize=9, ncol=2)

fig.tight_layout()
out1 = "plots/beta_gamma_stability.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out1}")

# === Stats ===
print("\n=== γ stats per date (across 6 detectors) ===")
for label in date_labels:
    gs = [r["gamma"] for r in all_results if r["date"] == label]
    print(f"  {label}:  γ = {np.mean(gs):.3f} ± {np.std(gs):.3f}  (range {min(gs):.3f}–{max(gs):.3f})")

print("\n=== γ stats per detector (across 5 dates) ===")
for det in range(6):
    gs = [r["gamma"] for r in all_results if r["det"] == det]
    print(f"  A{det}:  γ = {np.mean(gs):.3f} ± {np.std(gs):.3f}  (range {min(gs):.3f}–{max(gs):.3f})")

print("\n=== β stats per detector (across 5 dates) ===")
for det in range(6):
    bs = [r["beta"] for r in all_results if r["det"] == det]
    print(f"  A{det}:  β = {np.mean(bs):.2f} ± {np.std(bs):.2f}  (range {min(bs):.2f}–{max(bs):.2f})")

all_g = [r["gamma"] for r in all_results]
all_b = [r["beta"] for r in all_results]
print(f"\nGRAND TOTAL:  γ = {np.mean(all_g):.3f} ± {np.std(all_g):.3f}, "
      f"β = {np.mean(all_b):.2f} ± {np.std(all_b):.2f}  ({len(all_g)} measurements)")
