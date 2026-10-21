#!/usr/bin/env python3
"""Direct linear regression: Sci = a₀ + a₁·PHO + a₂·Wide + a₃·Large
Test if (a₀, a₁, a₂, a₃) are stable global constants across 90 detector slots.
This avoids the (β, γ, b, α) parameter redundancy.
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
    return {"PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci,
            "length": length_s, "valid": valid}


# Pool all 90 slots, keep per-bin info
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
            for i in range(len(pho)):
                all_rows.append({
                    "date": date_label, "box": box, "det": det,
                    "PHO": pho[i], "Wide": wide[i], "Large": large[i], "Sci": sci[i],
                })
        print(f"  {date_label} Box {box} done")
big = pd.DataFrame(all_rows)
print(f"\nTotal bins: {len(big)}")


# === Fit Sci = a₀ + a₁·PHO + a₂·Wide + a₃·Large ===
# Per-detector and globally
def fit4(df):
    X = np.column_stack([np.ones(len(df)), df["PHO"], df["Wide"], df["Large"]])
    y = df["Sci"].values
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ c
    return c, np.sqrt(np.mean(res ** 2))


# Global fit (all 90 slots pooled)
c_glob, rms_glob = fit4(big)
print(f"\n=== GLOBAL fit (all 90 detector slots, {len(big)} bins pooled) ===")
print(f"  Sci = {c_glob[0]:+.1f} + {c_glob[1]:+.4f}·PHO + {c_glob[2]:+.4f}·Wide + {c_glob[3]:+.4f}·Large")
print(f"  RMS = {rms_glob:.1f} cnt/s")

# Per-detector
print(f"\n=== Per (date, box, det) fit ===")
per_det = []
for (date, box, det), grp in big.groupby(["date", "box", "det"]):
    c_pd, rms_pd = fit4(grp)
    per_det.append({"date": date, "box": box, "det": det,
                    "a0": c_pd[0], "a1": c_pd[1], "a2": c_pd[2], "a3": c_pd[3],
                    "rms": rms_pd, "n": len(grp)})

# Stats
a0s = np.array([r["a0"] for r in per_det])
a1s = np.array([r["a1"] for r in per_det])
a2s = np.array([r["a2"] for r in per_det])
a3s = np.array([r["a3"] for r in per_det])
rmss = np.array([r["rms"] for r in per_det])

print(f"  a0    median = {np.median(a0s):>+8.1f}  std = {np.std(a0s):>5.1f}  range {a0s.min():+.0f}–{a0s.max():+.0f}")
print(f"  a1    median = {np.median(a1s):>+8.4f}  std = {np.std(a1s):>5.4f}  range {a1s.min():+.4f}–{a1s.max():+.4f}")
print(f"  a2    median = {np.median(a2s):>+8.4f}  std = {np.std(a2s):>5.4f}  range {a2s.min():+.4f}–{a2s.max():+.4f}")
print(f"  a3    median = {np.median(a3s):>+8.4f}  std = {np.std(a3s):>5.4f}  range {a3s.min():+.4f}–{a3s.max():+.4f}")
print(f"  RMS   median = {np.median(rmss):>5.1f} cnt/s")

# === Test global constants vs per-det fit RMS ===
print(f"\n=== RMS comparison: GLOBAL vs PER-DET coefficients ===")
rms_using_global = []
for (date, box, det), grp in big.groupby(["date", "box", "det"]):
    pred = c_glob[0] + c_glob[1] * grp["PHO"] + c_glob[2] * grp["Wide"] + c_glob[3] * grp["Large"]
    rms_using_global.append(np.sqrt(np.mean((grp["Sci"] - pred) ** 2)))
rms_using_global = np.array(rms_using_global)
print(f"  Using GLOBAL  (a0, a1, a2, a3):  mean RMS = {rms_using_global.mean():.1f}, median = {np.median(rms_using_global):.1f}")
print(f"  Using per-DET (a0, a1, a2, a3):  mean RMS = {rmss.mean():.1f}, median = {np.median(rmss):.1f}")

# === Plot ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Histograms of each coefficient
for k, (a, name, true_val) in enumerate([(a0s, "a₀ [cnt/s]", c_glob[0]),
                                          (a1s, "a₁ (PHO coef)", c_glob[1]),
                                          (a2s, "a₂ (Wide coef)", c_glob[2])]):
    ax = axes[0, k]
    ax.hist(a, bins=25, color=f"C{k}", edgecolor="k", alpha=0.7)
    ax.axvline(true_val, color="r", lw=2, label=f"global = {true_val:.4g}")
    ax.axvline(np.median(a), color="g", ls="--", lw=2, label=f"median = {np.median(a):.4g}")
    ax.set_xlabel(name)
    ax.set_ylabel("count")
    ax.set_title(f"{name} across 90 slots\nstd = {np.std(a):.4g}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# a3 histogram
ax = axes[1, 0]
ax.hist(a3s, bins=25, color="C3", edgecolor="k", alpha=0.7)
ax.axvline(c_glob[3], color="r", lw=2, label=f"global = {c_glob[3]:.4f}")
ax.axvline(np.median(a3s), color="g", ls="--", lw=2, label=f"median = {np.median(a3s):.4f}")
ax.set_xlabel("a₃ (Large coef)")
ax.set_ylabel("count")
ax.set_title(f"a₃ across 90 slots\nstd = {np.std(a3s):.4f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# RMS comparison
ax = axes[1, 1]
ax.hist(rms_using_global, bins=25, color="C3", edgecolor="k", alpha=0.6,
        label=f"GLOBAL coef (mean {rms_using_global.mean():.1f})")
ax.hist(rmss, bins=25, color="C0", edgecolor="k", alpha=0.6,
        label=f"per-det coef (mean {rmss.mean():.1f})")
ax.set_xlabel("RMS [cnt/s]")
ax.set_ylabel("count")
ax.set_title("RMS distribution per slot")
ax.legend()
ax.grid(alpha=0.3)

# Scatter: predicted vs actual Sci
ax = axes[1, 2]
sample = big.sample(n=min(20000, len(big)), random_state=0)
pred_glob = c_glob[0] + c_glob[1] * sample["PHO"] + c_glob[2] * sample["Wide"] + c_glob[3] * sample["Large"]
ax.scatter(sample["Sci"], pred_glob, s=1, alpha=0.1, color="C0", rasterized=True)
ax.plot([sample["Sci"].min(), sample["Sci"].max()],
        [sample["Sci"].min(), sample["Sci"].max()], "r--", lw=1.5, label="y=x")
ax.set_xlabel("Sci_observed [cnt/s]")
ax.set_ylabel("Sci_predicted (global) [cnt/s]")
ax.set_title("Predicted vs observed (sample 20k bins)")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
out = "plots/sci_linear_predictor.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# Equivalent (β, γ, b, α) interpretation
a0, a1, a2, a3 = c_glob
# Sci = a0 + a1·PHO + a2·Wide + a3·Large
# Equivalent to: Sci = (PHO - β·W - γ·L - b) / (1+α)
# Compare:
#   coef of PHO:   1/(1+α) = a1   →  α = 1/a1 - 1
#   coef of Wide:  -β/(1+α) = a2  →  β = -a2 · (1+α) = -a2/a1
#   coef of Large: -γ/(1+α) = a3  →  γ = -a3/a1
#   constant:      -b/(1+α) = a0  →  b = -a0/a1 · (1+α) = -a0/a1
alpha_eq = 1.0 / a1 - 1
beta_eq = -a2 / a1
gamma_eq = -a3 / a1
b_eq = -a0 / a1
print(f"\n=== Equivalent (β, γ, b, α) from global fit ===")
print(f"  α = 1/a1 - 1     = {alpha_eq:.4f}")
print(f"  β = -a2/a1       = {beta_eq:.3f}")
print(f"  γ = -a3/a1       = {gamma_eq:.3f}")
print(f"  b = -a0/a1       = {b_eq:.1f} cnt/s")
print(f"\n=== FINAL UNIVERSAL FORMULA ===")
print(f"  Sci_predicted = {a0:+.1f} + {a1:.4f}·PHO + {a2:+.4f}·Wide + {a3:+.4f}·Large")
print(f"  RMS = {rms_using_global.mean():.1f} cnt/s/det")
