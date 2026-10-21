#!/usr/bin/env python3
"""Diagnose Sci_predicted - Sci_observed bending.

Check residual vs:
  - Sci itself  (does linear model break at high rate?)
  - PHO         (any rate-dependent saturation?)
  - Hardness    (Large/Sci or Wide/Sci)
  - Each counter individually
  - Try adding quadratic terms

Look for the missing physics that causes the downward bend.
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
    if not fits_files:
        return None
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
        fe.close()
        return None
    p5, p95 = np.percentile(sci_total[valid], [5, 95])
    valid &= (sci_total >= p5) & (sci_total <= p95)
    fe.close()
    return {"PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci, "Dt": Dt,
            "length": length_s, "valid": valid}


# Load
print("Loading 5 dates × 3 boxes × 6 dets ...")
all_rows = []
for date_label, fits_glob_tpl, sci_csv_tpl in DATES:
    for box in "ABC":
        D = load(date_label, fits_glob_tpl, sci_csv_tpl, box)
        if D is None:
            continue
        v = D["valid"]
        for det in range(6):
            sci = D["Sci"][v, det] / D["length"][v]
            wide = D["Wide"][v, det] / D["length"][v]
            large = D["Large"][v, det] / D["length"][v]
            pho = D["PHO"][v, det] / D["length"][v]
            dt = D["Dt"][v, det] / D["length"][v]
            for i in range(len(pho)):
                all_rows.append({
                    "date": date_label, "box": box, "det": det,
                    "PHO": pho[i], "Wide": wide[i], "Large": large[i],
                    "Sci": sci[i], "Dt": dt[i],
                })
        print(f"  {date_label} Box {box} done")
big = pd.DataFrame(all_rows)
print(f"\nTotal bins: {len(big)}")

# === Apply linear model and compute residuals ===
A = np.column_stack([np.ones(len(big)), big["PHO"], big["Wide"], big["Large"]])
y = big["Sci"].values
c, *_ = np.linalg.lstsq(A, y, rcond=None)
big["Sci_pred_lin"] = A @ c
big["resid_lin"] = big["Sci"] - big["Sci_pred_lin"]
print(f"\nLinear: Sci = {c[0]:.1f} + {c[1]:.4f}·PHO + {c[2]:.4f}·Wide + {c[3]:.4f}·Large")
print(f"  RMS = {np.sqrt(np.mean(big['resid_lin']**2)):.1f} cnt/s")

# === Try quadratic in PHO (rate-dependent) ===
A2 = np.column_stack([np.ones(len(big)), big["PHO"], big["Wide"], big["Large"],
                      big["PHO"]**2, big["PHO"] * big["Large"]])
c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
big["Sci_pred_quad"] = A2 @ c2
big["resid_quad"] = big["Sci"] - big["Sci_pred_quad"]
print(f"\nWith PHO² and PHO·Large terms:")
print(f"  Sci = {c2[0]:.1f} + {c2[1]:.4f}·PHO + {c2[2]:.4f}·Wide + {c2[3]:.4f}·Large")
print(f"        + {c2[4]:.6e}·PHO² + {c2[5]:.6e}·PHO·Large")
print(f"  RMS = {np.sqrt(np.mean(big['resid_quad']**2)):.1f} cnt/s")

# === Try with DeadTime ===
A_dt = np.column_stack([np.ones(len(big)), big["PHO"], big["Wide"], big["Large"], big["Dt"]])
c_dt, *_ = np.linalg.lstsq(A_dt, y, rcond=None)
big["Sci_pred_dt"] = A_dt @ c_dt
big["resid_dt"] = big["Sci"] - big["Sci_pred_dt"]
print(f"\nWith DeadTime:")
print(f"  Sci = {c_dt[0]:.1f} + {c_dt[1]:.4f}·PHO + {c_dt[2]:.4f}·Wide + {c_dt[3]:.4f}·Large + {c_dt[4]:.4f}·Dt")
print(f"  RMS = {np.sqrt(np.mean(big['resid_dt']**2)):.1f} cnt/s")

# === Plot ===
fig, axes = plt.subplots(3, 3, figsize=(16, 13))

# --- Row 1: residual vs Sci, PHO, Large ---
def panel(ax, x, y, xlabel, n_bins=30):
    sample = np.random.choice(len(x), min(50000, len(x)), replace=False)
    ax.scatter(x.values[sample] if hasattr(x, "values") else x[sample],
               y.values[sample] if hasattr(y, "values") else y[sample],
               s=1, alpha=0.05, color="C0", rasterized=True)
    bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        med.append(np.median(y[m]) if m.sum() > 30 else np.nan)
    ax.plot(bc, med, "k-", lw=2)
    ax.axhline(0, color="r", ls="--", lw=0.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel)

panel(axes[0, 0], big["Sci"], big["resid_lin"], "Sci_observed [cnt/s]")
axes[0, 0].set_ylabel("Residual = Sci_obs − Sci_pred [cnt/s]")
axes[0, 0].set_title("Linear model — residual vs Sci")

panel(axes[0, 1], big["PHO"], big["resid_lin"], "PHO [cnt/s]")
axes[0, 1].set_title("vs PHO (rate)")

panel(axes[0, 2], big["Large"], big["resid_lin"], "Large [cnt/s]")
axes[0, 2].set_title("vs Large")

# --- Row 2: hardness, dead fraction, time ---
big["hardness"] = big["Large"] / np.maximum(big["Sci"], 1)
big["dead_frac"] = big["Dt"]  # already fraction since Dt/length
panel(axes[1, 0], big["hardness"], big["resid_lin"], "Hardness = Large/Sci")
axes[1, 0].set_title("vs Hardness")

panel(axes[1, 1], big["dead_frac"], big["resid_lin"], "Dead fraction = Dt/Length")
axes[1, 1].set_title("vs Dead fraction")

panel(axes[1, 2], big["Wide"], big["resid_lin"], "Wide [cnt/s]")
axes[1, 2].set_title("vs Wide")

# --- Row 3: predicted-vs-observed for each model ---
def pred_obs(ax, pred, obs, title):
    sample = np.random.choice(len(pred), 30000, replace=False)
    ax.scatter(obs.values[sample], pred.values[sample], s=1, alpha=0.06,
               color="C0", rasterized=True)
    lo, hi = obs.min(), obs.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x")
    # Median curve
    bins = np.linspace(lo, hi, 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med_pred = []
    for i in range(len(bins) - 1):
        m = (obs >= bins[i]) & (obs < bins[i + 1])
        med_pred.append(np.median(pred[m]) if m.sum() > 30 else np.nan)
    ax.plot(bc, med_pred, "k-", lw=2, label="median pred")
    ax.set_xlabel("Sci_observed [cnt/s]")
    ax.set_ylabel("Sci_predicted [cnt/s]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

pred_obs(axes[2, 0], big["Sci_pred_lin"], big["Sci"],
         f"Linear model  RMS={np.sqrt(np.mean(big['resid_lin']**2)):.1f}")
pred_obs(axes[2, 1], big["Sci_pred_quad"], big["Sci"],
         f"+ PHO² + PHO·L  RMS={np.sqrt(np.mean(big['resid_quad']**2)):.1f}")
pred_obs(axes[2, 2], big["Sci_pred_dt"], big["Sci"],
         f"+ DeadTime  RMS={np.sqrt(np.mean(big['resid_dt']**2)):.1f}")

fig.suptitle("Residual diagnostics: where does the predicted-vs-observed bend come from?",
             fontsize=12)
fig.tight_layout()
out = "plots/residual_diagnostics.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# === Test if the bend correlates strongly with anything ===
print("\n=== Pearson ρ of residual vs candidate factors ===")
for factor in ["Sci", "PHO", "Wide", "Large", "Dt", "hardness", "dead_frac"]:
    rho = big[factor].corr(big["resid_lin"])
    print(f"  ρ(resid_lin, {factor:>10s}) = {rho:+.3f}")
