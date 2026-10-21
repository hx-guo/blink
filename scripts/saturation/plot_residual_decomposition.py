#!/usr/bin/env python3
"""Decompose residual structure: is the 'S-bend' physics or pooling artifact?

(1) Plot residual colored by date — if dates separate clearly, pooling artifact
(2) Try physics-motivated minimal form:  Sci = κ·(PHO - 2·W - L) + C
    Just 2 parameters with clear physical meaning.
(3) Try ratio formulation:  Sci/PHO vs (Wide/PHO, Large/PHO)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
DATES = [
    ("2020-04-15", "data/1B/2020/20200415/{box}/*.fits", "/tmp/200415_box{B}.csv"),
    ("2020-04-28", "data/1B/2020/20200428/{box}/*.fits", "/tmp/200428_box{B}.csv"),
    ("2022-10-09", "data/1B/2022/20221009/{box}/*.fits", "/tmp/221009_box{B}.csv"),
    ("2026-02-26", "data/1B/2026/20260226/{box}/*.fits", "/tmp/260226_box{B}_full.csv"),
    ("2026-04-10", "data/1B/2026/20260410/{box}/*.fits", "/tmp/260410_box{B}.csv"),
]
BOX_FITS_CODES = {"A": "0766", "B": "1009", "C": "1781"}
BOX_OFFSETS = {"A": 0, "B": 6, "C": 12}


def load(date_label, fits_glob_tpl, sci_csv_tpl, box):
    fits_glob = fits_glob_tpl.format(box=BOX_FITS_CODES[box])
    sci_csv = sci_csv_tpl.format(B=box)
    fits_files = sorted(glob.glob(fits_glob))
    if not fits_files: return None
    fe = fits.open(fits_files[0], memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    L_cycles = d["Length_Time_Cycle"].astype(float)
    length_s = L_cycles * 16e-6
    det_off = BOX_OFFSETS[box]
    det_ids = [det_off + i for i in range(6)]
    PHO = np.column_stack([d[f"Cnt_PHODet_{i}"].astype(float) for i in det_ids])
    Wide = np.column_stack([d[f"Cnt_CsI_PHODet_{i}"].astype(float) for i in det_ids])
    Large_raw = np.column_stack([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in det_ids])
    Large = np.column_stack([unwrap_large(PHO[:, i], Large_raw[:, i]) for i in range(6)])
    met_eng = d["Time"].astype(float) + offset + MET_CORRECTION

    df = pd.read_csv(sci_csv, usecols=["type", "met", "det_id"],
                     dtype={"type": "category", "met": "float64", "det_id": "int8"})
    df = df[df["type"] == "EVT"]
    Sci = np.zeros((len(met_eng), 6))
    for det in range(6):
        evts = np.sort(df["met"].values[df["det_id"].values == det])
        for i in range(len(met_eng)):
            t0 = met_eng[i]; t1 = t0 + length_s[i]
            Sci[i, det] = np.searchsorted(evts, t1) - np.searchsorted(evts, t0)
    del df
    valid = (L_cycles > 50000) & (Sci.sum(axis=1) > 100)
    sci_total = Sci.sum(axis=1)
    if valid.sum() < 50:
        fe.close(); return None
    p5, p95 = np.percentile(sci_total[valid], [5, 95])
    valid &= (sci_total >= p5) & (sci_total <= p95)
    fe.close()
    return {"PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci,
            "length": length_s, "valid": valid}


print("Loading...")
all_rows = []
for date_label, fits_glob_tpl, sci_csv_tpl in DATES:
    for box in "ABC":
        D = load(date_label, fits_glob_tpl, sci_csv_tpl, box)
        if D is None: continue
        v = D["valid"]
        for det in range(6):
            sci = D["Sci"][v, det] / D["length"][v]
            wide = D["Wide"][v, det] / D["length"][v]
            large = D["Large"][v, det] / D["length"][v]
            pho = D["PHO"][v, det] / D["length"][v]
            for i in range(len(pho)):
                all_rows.append({"date": date_label, "box": box, "det": det,
                                 "PHO": pho[i], "Wide": wide[i], "Large": large[i],
                                 "Sci": sci[i]})
        print(f"  {date_label} Box {box} done")
big = pd.DataFrame(all_rows)
print(f"Total bins: {len(big)}")

# Linear model
A_lin = np.column_stack([np.ones(len(big)), big["PHO"], big["Wide"], big["Large"]])
c_lin, *_ = np.linalg.lstsq(A_lin, big["Sci"], rcond=None)
big["pred_lin"] = A_lin @ c_lin
big["resid_lin"] = big["Sci"] - big["pred_lin"]

# Physics-motivated minimal form: Sci = κ·(PHO − 2·W − L) + C
big["E"] = big["PHO"] - 2 * big["Wide"] - big["Large"]  # "engineering pseudo-Sci"
A_min = np.column_stack([np.ones(len(big)), big["E"]])
c_min, *_ = np.linalg.lstsq(A_min, big["Sci"], rcond=None)
big["pred_min"] = A_min @ c_min
big["resid_min"] = big["Sci"] - big["pred_min"]
rms_min = np.sqrt(np.mean(big["resid_min"] ** 2))
print(f"\nMinimal physics form: Sci = {c_min[0]:.1f} + {c_min[1]:.4f}·(PHO − 2·Wide − Large)")
print(f"  RMS = {rms_min:.1f} cnt/s   (vs linear 4-coef: 51.3)")

# === Plot 1: residual vs Sci, COLORED BY DATE ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

date_colors = {d[0]: c for d, c in zip(DATES,
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])}

# (1) Linear residual colored by date
ax = axes[0, 0]
for date in [d[0] for d in DATES]:
    sub = big[big["date"] == date]
    if len(sub) < 100: continue
    sample = sub.sample(min(8000, len(sub)), random_state=0)
    ax.scatter(sample["Sci"], sample["resid_lin"], s=1.5, alpha=0.2,
               color=date_colors[date], label=date, rasterized=True)
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Sci_obs [cnt/s]")
ax.set_ylabel("Residual (linear) [cnt/s]")
ax.set_title("Linear residual COLORED BY DATE\n(if structure is per-date, dates separate)")
ax.set_ylim(-200, 200)
ax.legend(fontsize=8, markerscale=4)
ax.grid(alpha=0.3)

# (2) Per-date median residual curves overlaid
ax = axes[0, 1]
sci_grid = np.linspace(400, 1700, 30)
for date in [d[0] for d in DATES]:
    sub = big[big["date"] == date]
    if len(sub) < 100: continue
    med = []
    for i in range(len(sci_grid) - 1):
        m = (sub["Sci"] >= sci_grid[i]) & (sub["Sci"] < sci_grid[i + 1])
        med.append(np.median(sub["resid_lin"][m]) if m.sum() > 30 else np.nan)
    ax.plot(0.5 * (sci_grid[:-1] + sci_grid[1:]), med, "o-",
            color=date_colors[date], lw=2, markersize=5, label=date)
# Pool median
med_all = []
for i in range(len(sci_grid) - 1):
    m = (big["Sci"] >= sci_grid[i]) & (big["Sci"] < sci_grid[i + 1])
    med_all.append(np.median(big["resid_lin"][m]) if m.sum() > 30 else np.nan)
ax.plot(0.5 * (sci_grid[:-1] + sci_grid[1:]), med_all, "k-", lw=3,
        label="POOLED median", alpha=0.6)
ax.axhline(0, color="r", ls="--", lw=0.8)
ax.set_xlabel("Sci_obs [cnt/s]")
ax.set_ylabel("Median residual per date [cnt/s]")
ax.set_title("Per-date median residual\n(check if dates have different shape)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(-100, 150)

# (3) Per-detector slot (90 slots) median residual, all overlaid
ax = axes[0, 2]
for (date, box, det), grp in big.groupby(["date", "box", "det"]):
    med = []
    for i in range(len(sci_grid) - 1):
        m = (grp["Sci"] >= sci_grid[i]) & (grp["Sci"] < sci_grid[i + 1])
        med.append(np.median(grp["resid_lin"][m]) if m.sum() > 5 else np.nan)
    ax.plot(0.5 * (sci_grid[:-1] + sci_grid[1:]), med, "-",
            color=date_colors[date], alpha=0.3, lw=0.5)
ax.plot(0.5 * (sci_grid[:-1] + sci_grid[1:]), med_all, "k-", lw=3,
        label="POOLED median", alpha=0.8)
ax.axhline(0, color="r", ls="--", lw=0.8)
ax.set_xlabel("Sci_obs [cnt/s]")
ax.set_ylabel("Median residual per detector slot")
ax.set_title("Per-(date, box, det) median residual (90 thin lines)\n+ pooled median (thick black)")
ax.grid(alpha=0.3)
ax.set_ylim(-200, 200)

# (4) Sci vs E = PHO - 2W - L scatter, ALL data
ax = axes[1, 0]
sample = big.sample(min(60000, len(big)), random_state=0)
ax.scatter(sample["E"], sample["Sci"], s=1, alpha=0.06, color="C0", rasterized=True)
xs = np.linspace(big["E"].min(), big["E"].max(), 100)
ax.plot(xs, c_min[0] + c_min[1] * xs, "r-", lw=2,
        label=f"y = {c_min[0]:.1f} + {c_min[1]:.4f}·E")
ax.set_xlabel("E = PHO − 2·Wide − Large [cnt/s]")
ax.set_ylabel("Sci [cnt/s]")
ax.set_title(f"Direct scatter Sci vs E\nRMS = {rms_min:.1f} (2 coeffs)")
ax.legend()
ax.grid(alpha=0.3)

# (5) Sci/PHO vs Large/PHO  (ratio space)
ax = axes[1, 1]
big["s_p"] = big["Sci"] / big["PHO"]
big["l_p"] = big["Large"] / big["PHO"]
big["w_p"] = big["Wide"] / big["PHO"]
sample2 = big.sample(min(40000, len(big)), random_state=0)
sc = ax.scatter(sample2["l_p"], sample2["s_p"], c=sample2["w_p"],
                s=1.5, alpha=0.3, cmap="viridis", vmin=0, vmax=0.05, rasterized=True)
plt.colorbar(sc, ax=ax, label="Wide/PHO")
ax.set_xlabel("Large / PHO")
ax.set_ylabel("Sci / PHO")
ax.set_title("Ratio space: Sci/PHO vs Large/PHO\n(detector-invariant if hypothesis holds)")
ax.grid(alpha=0.3)

# (6) Compare residual curves for several model complexities
ax = axes[1, 2]
# Compute pooled medians of various models
def compute_model(features):
    A = np.column_stack([np.ones(len(big))] + features)
    c, *_ = np.linalg.lstsq(A, big["Sci"], rcond=None)
    pred = A @ c
    return big["Sci"] - pred, np.sqrt(np.mean((big["Sci"] - pred) ** 2))

# Per-detector best-possible — we'll fit this as our floor
big["resid_perdet"] = 0.0
rms_perdets = []
for (date, box, det), grp in big.groupby(["date", "box", "det"]):
    A_sub = np.column_stack([np.ones(len(grp)), grp["PHO"], grp["Wide"], grp["Large"]])
    c_sub, *_ = np.linalg.lstsq(A_sub, grp["Sci"], rcond=None)
    big.loc[grp.index, "resid_perdet"] = grp["Sci"] - A_sub @ c_sub
    rms_perdets.append(np.sqrt(np.mean((grp["Sci"] - A_sub @ c_sub) ** 2)))

# Overlaid medians: linear, minimal physics, per-detector
for resid_col, label, color in [
    ("resid_lin", f"Linear 4-coef (RMS=51.3)", "C3"),
    ("resid_min", f"Minimal: κ·(PHO−2W−L)+C  (RMS={rms_min:.0f})", "C1"),
    ("resid_perdet", f"Per-detector (RMS={np.mean(rms_perdets):.0f})", "C2"),
]:
    med = []
    for i in range(len(sci_grid) - 1):
        m = (big["Sci"] >= sci_grid[i]) & (big["Sci"] < sci_grid[i + 1])
        med.append(np.median(big[resid_col][m]) if m.sum() > 30 else np.nan)
    ax.plot(0.5 * (sci_grid[:-1] + sci_grid[1:]), med, "-", color=color,
            lw=2, label=label)
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Sci_obs [cnt/s]")
ax.set_ylabel("Median residual [cnt/s]")
ax.set_title("Median residual: different model levels")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
ax.set_ylim(-50, 50)

fig.tight_layout()
out = "plots/residual_decomposition.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

print(f"\n=== Summary ===")
print(f"  Linear (4 coef):                  RMS = 51.3")
print(f"  Minimal physics: κ·(P-2W-L)+C:    RMS = {rms_min:.1f}  (only 2 coefs!)")
print(f"  All 2nd-order (10 coef):          RMS = 31.5")
print(f"  Per-detector best (4 each, 360):  RMS = {np.mean(rms_perdets):.1f}")
