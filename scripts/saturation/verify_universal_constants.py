#!/usr/bin/env python3
"""Verify (β=1.9, γ=1.19) as universal hardware constants.

Apply N_below = PHO − 1.9·Wide − 1.19·Large − Sci across all 5 dates × 6 dets,
fit (b + α·Sci) per-detector, compare RMS to per-date best (β, γ) fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import csv
import glob
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
BETA_FIXED = 1.9
GAMMA_FIXED = 1.19

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


def load_date(label, fits_glob, sci_csv):
    fits_files = sorted(glob.glob(fits_glob))
    if not fits_files:
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

    valid = (L_cycles > 50000) & (Sci.sum(axis=1) > 100)
    sci_total = Sci.sum(axis=1)
    p5, p95 = np.percentile(sci_total[valid], [5, 95])
    valid &= (sci_total >= p5) & (sci_total <= p95)
    fe.close()

    return {
        "label": label, "valid": valid,
        "PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci, "length": length_s,
    }


print(f"Universal constants: β = {BETA_FIXED}, γ = {GAMMA_FIXED}\n")
print(f"{'Date':>11s} {'D':>2s} | {'RMS_fix':>7s} {'RMS_fit':>7s} {'Δ':>5s} | "
      f"{'<N_b>':>5s} {'b_fix':>5s} {'α_fix':>6s}")

results = []
for label, fits_glob, sci_csv in DATES:
    D = load_date(label, fits_glob, sci_csv)
    if D is None:
        continue
    v = D["valid"]
    for det in range(6):
        sci = D["Sci"][v, det] / D["length"][v]
        wide = D["Wide"][v, det] / D["length"][v]
        large = D["Large"][v, det] / D["length"][v]
        pho = D["PHO"][v, det] / D["length"][v]

        # 1) Universal constants
        nb_fix = pho - BETA_FIXED * wide - GAMMA_FIXED * large - sci
        b_fix, a_fix, r_fix = fit_linear(nb_fix, sci)
        rms_fix = np.sqrt(np.mean(r_fix ** 2))

        # 2) Per-date best (β, γ) — for reference
        nb_base = pho - wide - large - sci
        X = np.column_stack([np.ones_like(sci), sci, wide, large])
        c, *_ = np.linalg.lstsq(X, nb_base, rcond=None)
        beta = 1.0 + c[2]; gamma = 1.0 + c[3]
        nb_fit = pho - beta * wide - gamma * large - sci
        _, _, r_fit = fit_linear(nb_fit, sci)
        rms_fit = np.sqrt(np.mean(r_fit ** 2))

        delta = rms_fix - rms_fit
        print(f"{label:>11s} {det:>2d} | {rms_fix:>7.1f} {rms_fit:>7.1f} {delta:>+5.1f} | "
              f"{np.mean(nb_fix):>5.0f} {b_fix:>5.0f} {a_fix:>+6.3f}")
        results.append({
            "date": label, "det": det,
            "rms_fix": rms_fix, "rms_fit": rms_fit, "delta": delta,
            "b_fix": b_fix, "a_fix": a_fix, "nb_fix_mean": np.mean(nb_fix),
            "beta_fitted": beta, "gamma_fitted": gamma,
        })

# Stats
print("\n=== Per-date RMS comparison ===")
print(f"{'Date':>11s}  RMS_fix(mean)  RMS_fit(mean)  Δ(mean)  Δ(max)")
for label in [d[0] for d in DATES]:
    rs = [r for r in results if r["date"] == label]
    rfix = np.array([r["rms_fix"] for r in rs])
    rfit = np.array([r["rms_fit"] for r in rs])
    print(f"  {label}   {rfix.mean():>10.1f}    {rfit.mean():>10.1f}   "
          f"{(rfix-rfit).mean():>+5.1f}    {(rfix-rfit).max():>+5.1f}")

all_fix = [r["rms_fix"] for r in results]
all_fit = [r["rms_fit"] for r in results]
print(f"\nGRAND TOTAL: RMS_fix = {np.mean(all_fix):.1f} cnt/s, "
      f"RMS_fit = {np.mean(all_fit):.1f} cnt/s, "
      f"Δ = +{np.mean(all_fix)-np.mean(all_fit):.1f} cnt/s "
      f"({(np.mean(all_fix)/np.mean(all_fit)-1)*100:+.1f}%)")

# === Plot: RMS comparison per date × det ===
fig, ax = plt.subplots(figsize=(13, 5))
labels = [f"{r['date'][5:]}-A{r['det']}" for r in results]
xs = np.arange(len(results))
ax.bar(xs - 0.2, [r["rms_fix"] for r in results], width=0.4,
       color="C3", label=f"Fixed (β=1.9, γ=1.19)  mean={np.mean(all_fix):.1f}")
ax.bar(xs + 0.2, [r["rms_fit"] for r in results], width=0.4,
       color="C0", label=f"Per-date fitted (β, γ)  mean={np.mean(all_fit):.1f}")
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_ylabel("Absolute RMS [cnt/s]")
ax.set_title("Fixed universal (β, γ) vs per-date fitted — same RMS = constants are universal")
ax.legend()
ax.grid(alpha=0.3, axis="y")
# Vertical separators between dates
for i in range(1, 5):
    ax.axvline(i * 6 - 0.5, color="gray", lw=0.6, alpha=0.5)
fig.tight_layout()
out = "plots/universal_constants_validation.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")
