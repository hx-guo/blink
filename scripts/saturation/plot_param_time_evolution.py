#!/usr/bin/env python3
"""Per-(date, det) coefficients vs time — find smooth drift?

If parameters drift smoothly with MET, we can fit a time-evolution function
and predict (b, a₁, a₂, a₃) for any new date without explicit recalibration.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from datetime import datetime
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


def date_to_year(date_str):
    """Convert YYYY-MM-DD to decimal year."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = datetime(dt.year, 1, 1)
    end = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()


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
                                 "Sci": sci[i],
                                 "year": date_to_year(date_label)})
        print(f"  {date_label} Box {box} done")
big = pd.DataFrame(all_rows)
print(f"Total bins: {len(big)}")

# Per-(date, det) fit and store coefficients
coef_table = []
for (date, box, det), grp in big.groupby(["date", "box", "det"]):
    A = np.column_stack([np.ones(len(grp)), grp["PHO"], grp["Wide"], grp["Large"]])
    c, *_ = np.linalg.lstsq(A, grp["Sci"], rcond=None)
    coef_table.append({
        "date": date, "box": box, "det": det,
        "year": date_to_year(date),
        "b": c[0], "a1": c[1], "a2": c[2], "a3": c[3],
        "PHO_med": grp["PHO"].median(),
        "Wide_med": grp["Wide"].median(),
        "Large_med": grp["Large"].median(),
        "Sci_med": grp["Sci"].median(),
    })
coefs = pd.DataFrame(coef_table)
print(f"\nTotal (date, box, det) slots: {len(coefs)}")

# === Plot 1: each parameter vs time, per detector ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

box_colors = {"A": "C0", "B": "C1", "C": "C2"}
box_markers = {"A": "o", "B": "s", "C": "^"}

for ax, param, label in zip(axes.flat,
                             ["b", "a1", "a2", "a3"],
                             ["b (intercept) [cnt/s]", "a₁ (PHO coef)",
                              "a₂ (Wide coef)", "a₃ (Large coef)"]):
    for box in "ABC":
        for det in range(6):
            sub = coefs[(coefs["box"] == box) & (coefs["det"] == det)]
            sub = sub.sort_values("year")
            ax.plot(sub["year"], sub[param], "-",
                    color=box_colors[box], alpha=0.4, lw=0.7)
            ax.scatter(sub["year"], sub[param], marker=box_markers[box],
                        color=box_colors[box], s=30, edgecolor="k", linewidth=0.4,
                        alpha=0.7)
    # Per-box mean line
    for box in "ABC":
        sub = coefs[coefs["box"] == box]
        means = sub.groupby("year")[param].mean()
        ax.plot(means.index, means.values, "-", color=box_colors[box],
                lw=2.5, label=f"Box {box} mean")
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle("Per-detector fit coefficients vs MET (5 dates × 18 detectors)",
             fontsize=12)
fig.tight_layout()
out1 = "plots/coef_vs_time.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"Saved: {out1}")

# === Plot 2: parameters vs DAILY mean PHO rate (state-dependence?) ===
# Maybe params drift with operating state, not with absolute time
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for ax, param, label in zip(axes.flat,
                             ["b", "a1", "a2", "a3"],
                             ["b [cnt/s]", "a₁ (PHO coef)",
                              "a₂ (Wide coef)", "a₃ (Large coef)"]):
    for box in "ABC":
        for det in range(6):
            sub = coefs[(coefs["box"] == box) & (coefs["det"] == det)]
            ax.scatter(sub["PHO_med"], sub[param], marker=box_markers[box],
                        color=box_colors[box], s=40, edgecolor="k", linewidth=0.4,
                        alpha=0.7,
                        label=f"Box {box}" if det == 0 else None)
    # Linear fit per box
    for box in "ABC":
        sub = coefs[coefs["box"] == box]
        x = sub["PHO_med"].values; y = sub[param].values
        A_lin = np.column_stack([np.ones_like(x), x])
        c, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
        rho = np.corrcoef(x, y)[0, 1]
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, c[0] + c[1] * xs, "-", color=box_colors[box], lw=1.4,
                alpha=0.85)
    ax.set_xlabel("Daily median PHO [cnt/s/det]")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs daily PHO rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

fig.suptitle("Coefficients vs daily operating state (median PHO rate)\n"
             "If params follow PHO state, can predict from observable",
             fontsize=12)
fig.tight_layout()
out2 = "plots/coef_vs_state.png"
fig.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")

# === Per-detector consistency: how stable IS each detector across dates? ===
print(f"\n=== Per-detector parameter drift across 5 dates ===")
print(f"{'Det':>3s}  {'b std':>7s}  {'a₁ std':>7s}  {'a₂ std':>7s}  {'a₃ std':>7s}")
for box in "ABC":
    for det in range(6):
        sub = coefs[(coefs["box"] == box) & (coefs["det"] == det)]
        if len(sub) >= 2:
            print(f"  {box}{det}  {sub['b'].std():>7.1f}  "
                  f"{sub['a1'].std():>7.4f}  "
                  f"{sub['a2'].std():>7.4f}  "
                  f"{sub['a3'].std():>7.4f}")

# === Test: predict params from daily PHO_med, see if RMS improves ===
print(f"\n=== Can a + b·PHO_med model predict each parameter? ===")
print(f"{'Param':>5s}  {'slope':>8s}  {'intercept':>9s}  R²    range explained")
for param in ["b", "a1", "a2", "a3"]:
    x = coefs["PHO_med"].values; y = coefs[param].values
    A_lin = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
    pred = A_lin @ c
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  {param}  {c[1]:>+8.5f}  {c[0]:>+9.3f}  {r2:.3f}")
