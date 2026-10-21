#!/usr/bin/env python3
"""Try ALL 2nd-order interaction terms and see which residual structure remains.
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
    Dt = np.column_stack([d[f"DeadTime_PHODet_{i}"].astype(float) for i in det_ids]) * 16e-6
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
    return {"PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci, "Dt": Dt,
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
            dt = D["Dt"][v, det] / D["length"][v]
            for i in range(len(pho)):
                all_rows.append({"date": date_label, "PHO": pho[i], "Wide": wide[i],
                                 "Large": large[i], "Sci": sci[i], "Dt": dt[i]})
        print(f"  {date_label} Box {box} done")
big = pd.DataFrame(all_rows)
y = big["Sci"].values
print(f"Total bins: {len(big)}")

# Try multiple model variants
def fit_model(features, name):
    A = np.column_stack([np.ones(len(big))] + features)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ c
    resid = y - pred
    rms = np.sqrt(np.mean(resid ** 2))
    return c, pred, resid, rms

# Baseline: linear
c1, p1, r1, rms1 = fit_model([big["PHO"], big["Wide"], big["Large"]], "linear")
print(f"\n  Linear:               RMS = {rms1:.2f}")

# +PHO² + PHO·L (current)
c2, p2, r2, rms2 = fit_model([big["PHO"], big["Wide"], big["Large"],
                              big["PHO"]**2, big["PHO"]*big["Large"]], "quad-2")
print(f"  + PHO² + PHO·L:       RMS = {rms2:.2f}")

# All 2nd-order
c3, p3, r3, rms3 = fit_model([big["PHO"], big["Wide"], big["Large"],
                              big["PHO"]**2, big["Wide"]**2, big["Large"]**2,
                              big["PHO"]*big["Wide"], big["PHO"]*big["Large"],
                              big["Wide"]*big["Large"]], "all-2nd")
print(f"  All 2nd-order:        RMS = {rms3:.2f}")
labels = ["1", "PHO", "Wide", "Large", "PHO²", "Wide²", "Large²", "PHO·W", "PHO·L", "W·L"]
print(f"    coeffs: " + " ".join(f"{l}={v:.2e}" for l, v in zip(labels, c3)))

# +cubic in PHO
c4, p4, r4, rms4 = fit_model([big["PHO"], big["Wide"], big["Large"],
                              big["PHO"]**2, big["PHO"]*big["Large"],
                              big["PHO"]**3], "+cubic")
print(f"  + PHO² + PHO·L + PHO³: RMS = {rms4:.2f}")

# +cubic + PHO²·L
c5, p5, r5, rms5 = fit_model([big["PHO"], big["Wide"], big["Large"],
                              big["PHO"]**2, big["PHO"]*big["Large"],
                              big["PHO"]**3, big["PHO"]**2 * big["Large"]], "+more")
print(f"  + ... + PHO²·L:        RMS = {rms5:.2f}")


# === Plot residual vs Sci for each model ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

models = [(r1, "Linear (4)", rms1),
          (r2, "+PHO² +PHO·L (6)", rms2),
          (r3, "All 2nd-order (10)", rms3),
          (r4, "+PHO³ (7)", rms4),
          (r5, "+PHO²·L (8)", rms5)]

def plot_resid(ax, resid, label, rms):
    sample = np.random.choice(len(big), min(50000, len(big)), replace=False)
    sci_sample = big["Sci"].values[sample]
    res_sample = resid[sample]
    ax.scatter(sci_sample, res_sample, s=1, alpha=0.05, color="C0", rasterized=True)
    ax.axhline(0, color="r", ls="--", lw=0.8)
    bins = np.linspace(np.percentile(big["Sci"], 1), np.percentile(big["Sci"], 99), 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []; p16 = []; p84 = []
    for i in range(len(bins) - 1):
        m = (big["Sci"].values >= bins[i]) & (big["Sci"].values < bins[i + 1])
        if m.sum() > 30:
            med.append(np.median(resid[m]))
            p16.append(np.percentile(resid[m], 16))
            p84.append(np.percentile(resid[m], 84))
        else:
            med.append(np.nan); p16.append(np.nan); p84.append(np.nan)
    ax.fill_between(bc, p16, p84, color="k", alpha=0.18)
    ax.plot(bc, med, "k-", lw=2, label="median")
    ax.set_xlabel("Sci_obs [cnt/s]")
    ax.set_ylabel("residual [cnt/s]")
    ax.set_title(f"{label}\nRMS = {rms:.1f}")
    ax.set_ylim(-200, 200)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


for k, (resid, label, rms) in enumerate(models):
    ax = axes[k // 3, k % 3]
    plot_resid(ax, resid, label, rms)

# Last panel: zoom on median curves to compare bending
ax = axes[1, 2]
sci_arr = big["Sci"].values
bins = np.linspace(np.percentile(sci_arr, 1), np.percentile(sci_arr, 99), 30)
bc = 0.5 * (bins[:-1] + bins[1:])
for resid, label, rms in models:
    med = []
    for i in range(len(bins) - 1):
        m = (sci_arr >= bins[i]) & (sci_arr < bins[i + 1])
        med.append(np.median(resid[m]) if m.sum() > 30 else np.nan)
    ax.plot(bc, med, "-", lw=2, label=f"{label}  RMS={rms:.0f}")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Sci_obs [cnt/s]")
ax.set_ylabel("median residual [cnt/s]")
ax.set_title("Median residual vs Sci — model comparison")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
ax.set_ylim(-50, 50)

fig.tight_layout()
out = "plots/full_quadratic_comparison.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

print(f"\n=== 8-coefficient model coefficients ===")
print(f"  Sci = {c5[0]:+.2f}")
print(f"        {c5[1]:+.4f}·PHO")
print(f"        {c5[2]:+.4f}·Wide")
print(f"        {c5[3]:+.4f}·Large")
print(f"        {c5[4]:+.3e}·PHO²")
print(f"        {c5[5]:+.3e}·PHO·Large")
print(f"        {c5[6]:+.3e}·PHO³")
print(f"        {c5[7]:+.3e}·PHO²·Large")
