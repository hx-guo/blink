#!/usr/bin/env python3
"""Compare 3 calibration levels:
  (1) Single global fit                     — pool all 90 slots
  (2) Per-detector (18 fits, pool 5 dates)  — calibrate ONCE per det, never re-fit
  (3) Per-(detector, date)                  — recalibrate each day per det

Question: is option (2) close enough to (3) to be worth it without recalibration?
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
big["det_key"] = big["box"] + big["det"].astype(str)  # A0, A1, ..., C5
print(f"Total bins: {len(big)}")

# === (1) Single global fit ===
A = np.column_stack([np.ones(len(big)), big["PHO"], big["Wide"], big["Large"]])
c1, *_ = np.linalg.lstsq(A, big["Sci"], rcond=None)
big["pred1"] = A @ c1
big["resid1"] = big["Sci"] - big["pred1"]

# === (2) Per-detector, pooled across 5 dates ===
big["pred2"] = 0.0
det_coefs = {}
for det_key, grp in big.groupby("det_key"):
    A_d = np.column_stack([np.ones(len(grp)), grp["PHO"], grp["Wide"], grp["Large"]])
    c_d, *_ = np.linalg.lstsq(A_d, grp["Sci"], rcond=None)
    det_coefs[det_key] = c_d
    big.loc[grp.index, "pred2"] = A_d @ c_d
big["resid2"] = big["Sci"] - big["pred2"]

# === (3) Per-(detector, date) — full recalibration ===
big["pred3"] = 0.0
for (det_key, date), grp in big.groupby(["det_key", "date"]):
    A_dd = np.column_stack([np.ones(len(grp)), grp["PHO"], grp["Wide"], grp["Large"]])
    c_dd, *_ = np.linalg.lstsq(A_dd, grp["Sci"], rcond=None)
    big.loc[grp.index, "pred3"] = A_dd @ c_dd
big["resid3"] = big["Sci"] - big["pred3"]

rms1 = np.sqrt(np.mean(big["resid1"] ** 2))
rms2 = np.sqrt(np.mean(big["resid2"] ** 2))
rms3 = np.sqrt(np.mean(big["resid3"] ** 2))

print(f"\n(1) Single global fit:                RMS = {rms1:.1f}")
print(f"(2) Per-detector (18 fits, fixed):    RMS = {rms2:.1f}")
print(f"(3) Per-(detector, date):             RMS = {rms3:.1f}")
print(f"\nImprovement (1) → (2): {rms1 - rms2:.1f} cnt/s ({(1-rms2/rms1)*100:.0f}%)")
print(f"Improvement (2) → (3): {rms2 - rms3:.1f} cnt/s ({(1-rms3/rms2)*100:.0f}%)")

# Per-date breakdown for option (2) — does it work uniformly across dates?
print(f"\n=== RMS per date for option (2) — does fixed calib work for new dates? ===")
print(f"{'Date':>11s}  {'RMS_global':>10s}  {'RMS_perdet':>10s}  {'RMS_perdet+date':>15s}")
for date, grp in big.groupby("date"):
    print(f"  {date}  {np.sqrt(np.mean(grp['resid1']**2)):>10.1f}  "
          f"{np.sqrt(np.mean(grp['resid2']**2)):>10.1f}  "
          f"{np.sqrt(np.mean(grp['resid3']**2)):>15.1f}")

# === Plot ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Predicted vs observed for each level
def hexpanel(ax, x, y, title):
    sample = np.random.choice(len(x), min(60000, len(x)), replace=False)
    xs = x.values[sample]; ys = y.values[sample]
    hb = ax.hexbin(xs, ys, gridsize=70, cmap="viridis", mincnt=1, bins="log")
    lo, hi = max(xs.min(), 300), min(xs.max(), 1900)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    bins = np.linspace(lo, hi, 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (xs >= bins[i]) & (xs < bins[i + 1])
        med.append(np.median(ys[m]) if m.sum() > 30 else np.nan)
    ax.plot(bc, med, "w-", lw=2.5)
    ax.plot(bc, med, "k-", lw=1.2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci_obs"); ax.set_ylabel("Sci_pred")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)

hexpanel(axes[0, 0], big["Sci"], big["pred1"], f"(1) Global  4 coef  RMS={rms1:.1f}")
hexpanel(axes[0, 1], big["Sci"], big["pred2"], f"(2) Per-det fixed  4×18 coef  RMS={rms2:.1f}")
hexpanel(axes[0, 2], big["Sci"], big["pred3"], f"(3) Per-(det, date)  4×90 coef  RMS={rms3:.1f}")

# Median residual curves
ax = axes[1, 0]
sci_grid = np.linspace(400, 1700, 30)
for resid_col, label, color in [
    ("resid1", f"Global (RMS={rms1:.1f})", "C3"),
    ("resid2", f"Per-det fixed (RMS={rms2:.1f})", "C0"),
    ("resid3", f"Per-(det, date) (RMS={rms3:.1f})", "C2"),
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
ax.set_title("Median residual vs Sci")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-50, 80)

# Residual histograms (show distribution narrowing)
ax = axes[1, 1]
bins_h = np.linspace(-200, 200, 80)
ax.hist(big["resid1"], bins=bins_h, alpha=0.5, color="C3", label=f"Global  σ={rms1:.0f}")
ax.hist(big["resid2"], bins=bins_h, alpha=0.5, color="C0", label=f"Per-det fixed  σ={rms2:.0f}")
ax.hist(big["resid3"], bins=bins_h, alpha=0.5, color="C2", label=f"Per-(det, date)  σ={rms3:.0f}")
ax.set_xlabel("Residual [cnt/s]")
ax.set_ylabel("count")
ax.set_title("Residual distribution")
ax.legend()
ax.grid(alpha=0.3)

# Per-date RMS bars
ax = axes[1, 2]
date_labels = [d[0] for d in DATES]
rms_global_perdate = []
rms_perdet_perdate = []
rms_perdate_perdet = []
for date in date_labels:
    sub = big[big["date"] == date]
    rms_global_perdate.append(np.sqrt(np.mean(sub["resid1"] ** 2)))
    rms_perdet_perdate.append(np.sqrt(np.mean(sub["resid2"] ** 2)))
    rms_perdate_perdet.append(np.sqrt(np.mean(sub["resid3"] ** 2)))
xs = np.arange(len(date_labels))
ax.bar(xs - 0.27, rms_global_perdate, 0.27, color="C3", label="Global")
ax.bar(xs, rms_perdet_perdate, 0.27, color="C0", label="Per-det fixed")
ax.bar(xs + 0.27, rms_perdate_perdet, 0.27, color="C2", label="Per-(det, date)")
ax.set_xticks(xs)
ax.set_xticklabels(date_labels, rotation=15, fontsize=9)
ax.set_ylabel("RMS [cnt/s]")
ax.set_title("Per-date RMS")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = "plots/per_det_global_calib.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# === Print per-detector coefficients (the 18 calibration sets) ===
print(f"\n=== Per-detector calibration constants (18 sets, fixed forever) ===")
print(f"{'Det':>3s}  {'b':>7s}  {'a₁(PHO)':>9s}  {'a₂(W)':>8s}  {'a₃(L)':>8s}")
for k in sorted(det_coefs.keys()):
    c = det_coefs[k]
    print(f"  {k}  {c[0]:>+7.1f}  {c[1]:>+9.4f}  {c[2]:>+8.4f}  {c[3]:>+8.4f}")
