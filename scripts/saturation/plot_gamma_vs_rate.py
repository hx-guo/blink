#!/usr/bin/env python3
"""γ as function of count rate — find universal γ(R) curve.

Approach: fit γ in narrow rate bins of the SAME data, check if the relationship
is universal across dates / boxes / detectors.

If yes, we have:  γ(R) = γ0 - κ·R  or  γ0 · (1 - f_pileup(R))
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


def fit_betagamma(pho, wide, large, sci):
    nb_base = pho - wide - large - sci
    X = np.column_stack([np.ones_like(sci), sci, wide, large])
    c, *_ = np.linalg.lstsq(X, nb_base, rcond=None)
    return 1.0 + c[2], 1.0 + c[3]


def load(date_label, fits_glob_tpl, sci_csv_tpl, box):
    fits_glob = fits_glob_tpl.format(box=BOX_FITS_CODES[box])
    sci_csv = sci_csv_tpl.format(B=box)
    fits_files = sorted(glob.glob(fits_glob))
    if not fits_files:
        return None
    fe = fits.open(fits_files[0], memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
    L_cycles = d["Length_Time_Cycle"].astype(float)
    length_s = L_cycles * 16e-6
    det_off = BOX_OFFSETS[box]
    det_ids = [det_off + i for i in range(6)]
    PHO = np.column_stack([d[f"Cnt_PHODet_{i}"].astype(float) for i in det_ids])
    Wide = np.column_stack([d[f"Cnt_CsI_PHODet_{i}"].astype(float) for i in det_ids])
    Large_raw = np.column_stack([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in det_ids])
    Large = np.column_stack([unwrap_large(PHO[:, i], Large_raw[:, i]) for i in range(6)])

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
    p1, p99 = np.percentile(sci_total[valid], [1, 99])
    valid &= (sci_total >= p1) & (sci_total <= p99)
    fe.close()
    return {"PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci,
            "length": length_s, "valid": valid}


# Pool data per (date, box) but keep PHO_rate to bin
print("Loading 5 dates × 3 boxes ...\n")
all_data = []
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
            all_data.append({
                "date": date_label, "box": box, "det": det,
                "sci": sci, "wide": wide, "large": large, "pho": pho,
            })
        print(f"  {date_label} Box {box} loaded")


# === Per-detector: bin by PHO rate, compute γ in each bin ===
print("\nFitting γ in PHO-rate bins per detector...")
RATE_BINS = np.linspace(400, 2200, 13)  # cnt/s/det
bin_centers = 0.5 * (RATE_BINS[:-1] + RATE_BINS[1:])
n_bins = len(bin_centers)

per_det_curves = []  # one entry per (date, box, det)
for d in all_data:
    pho = d["pho"]
    gammas_in_bin = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = (pho >= RATE_BINS[k]) & (pho < RATE_BINS[k + 1])
        if m.sum() >= 30:  # need enough points to fit
            try:
                _, gamma = fit_betagamma(pho[m], d["wide"][m], d["large"][m], d["sci"][m])
                gammas_in_bin[k] = gamma
            except Exception:
                pass
    per_det_curves.append({
        "date": d["date"], "box": d["box"], "det": d["det"],
        "rate_bin_centers": bin_centers,
        "gamma_in_bin": gammas_in_bin,
    })


# Aggregate: for each rate bin, take median γ across all detectors with measurement
median_gamma = np.full(n_bins, np.nan)
p16_gamma = np.full(n_bins, np.nan)
p84_gamma = np.full(n_bins, np.nan)
counts = np.zeros(n_bins, dtype=int)
for k in range(n_bins):
    vals = [c["gamma_in_bin"][k] for c in per_det_curves
            if not np.isnan(c["gamma_in_bin"][k])]
    if len(vals) >= 5:
        median_gamma[k] = np.median(vals)
        p16_gamma[k] = np.percentile(vals, 16)
        p84_gamma[k] = np.percentile(vals, 84)
        counts[k] = len(vals)

print(f"\n=== γ(rate) median curve ===")
print(f"{'Rate':>6s}  {'<γ>':>5s}  {'p16':>5s}  {'p84':>5s}  N")
for k in range(n_bins):
    if counts[k] > 0:
        print(f"{bin_centers[k]:>6.0f}  {median_gamma[k]:.3f}  {p16_gamma[k]:.3f}  {p84_gamma[k]:.3f}  {counts[k]:>3d}")

# Fit γ(R) = γ0 - κ·R
mask = ~np.isnan(median_gamma)
A = np.column_stack([np.ones_like(bin_centers[mask]), bin_centers[mask]])
clin, *_ = np.linalg.lstsq(A, median_gamma[mask], rcond=None)
gamma0_lin = clin[0]; kappa = -clin[1]

# Try γ(R) = γ0 · (1 - f0 · R²/R_sat²)  pile-up like
# Or γ(R) = γ0 · exp(-R / R0)
from scipy.optimize import curve_fit
def expmodel(R, g0, R0):
    return g0 * np.exp(-R / R0)
def linmodel(R, g0, k):
    return g0 - k * R
try:
    popt_e, _ = curve_fit(expmodel, bin_centers[mask], median_gamma[mask],
                           p0=[1.25, 5000], sigma=(p84_gamma[mask] - p16_gamma[mask]) + 0.01)
except Exception as e:
    print("exp fit failed:", e)
    popt_e = None

# === Plot ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
# Per-detector faint lines
for c in per_det_curves:
    box_alpha = {"A": "C0", "B": "C1", "C": "C2"}[c["box"]]
    ax.plot(c["rate_bin_centers"], c["gamma_in_bin"], "-", color=box_alpha,
            alpha=0.15, lw=0.5)
# Median curve with error band
ax.fill_between(bin_centers[mask], p16_gamma[mask], p84_gamma[mask],
                color="k", alpha=0.2, label="16-84% across 90 dets")
ax.plot(bin_centers[mask], median_gamma[mask], "ko-", lw=2, markersize=7,
        label="Median")
# Fits
xs = np.linspace(bin_centers[mask].min(), bin_centers[mask].max(), 200)
ax.plot(xs, gamma0_lin - kappa * xs, "r--", lw=1.5,
        label=f"Linear: γ₀={gamma0_lin:.3f}, κ={kappa*1000:.3f}/1000 cnt/s")
if popt_e is not None:
    ax.plot(xs, expmodel(xs, *popt_e), "b--", lw=1.5,
            label=f"Exp: γ₀={popt_e[0]:.3f}, R₀={popt_e[1]:.0f} cnt/s")
ax.set_xlabel("PHO rate [cnt/s/det]")
ax.set_ylabel("γ (Large multiplier)")
ax.set_title("γ vs PHO rate — universal curve across all dates × boxes × dets?")
ax.grid(alpha=0.3)
ax.legend()

# Right: same but residual after extrapolating to R=0
ax = axes[1]
gamma0_extrap = popt_e[0] if popt_e is not None else gamma0_lin
ax.set_title(f"Extrapolated γ at R=0 (zero-rate limit) per-detector\n"
             f"= true hardware γ?  Pool median: {gamma0_extrap:.3f}")

# Per-det: linear extrapolation γ vs rate -> intercept
extrap_g0 = []
for c in per_det_curves:
    rs = c["rate_bin_centers"]
    gs = c["gamma_in_bin"]
    m = ~np.isnan(gs)
    if m.sum() >= 4:
        a = np.column_stack([np.ones_like(rs[m]), rs[m]])
        cc, *_ = np.linalg.lstsq(a, gs[m], rcond=None)
        extrap_g0.append({"date": c["date"], "box": c["box"], "det": c["det"],
                          "g0": cc[0], "k": -cc[1]})

# Plot extrapolated γ_0 vs date
date_x = {d[0]: i for i, d in enumerate(DATES)}
for r in extrap_g0:
    box_marker = {"A": "o", "B": "s", "C": "^"}[r["box"]]
    ax.scatter(date_x[r["date"]] + (ord(r["box"]) - ord("B")) * 0.1, r["g0"],
               marker=box_marker, color="C0", s=40, alpha=0.6,
               edgecolor="k", linewidth=0.4)
date_g0 = {}
for r in extrap_g0:
    date_g0.setdefault(r["date"], []).append(r["g0"])
medians_g0 = []
for dt in [d[0] for d in DATES]:
    vals = date_g0.get(dt, [])
    if vals:
        medians_g0.append(np.median(vals))
        ax.plot(date_x[dt], np.median(vals), "kD", markersize=12, zorder=10)
        ax.text(date_x[dt], np.median(vals) + 0.02, f"{np.median(vals):.3f}",
                ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(DATES)))
ax.set_xticklabels([d[0] for d in DATES], rotation=15)
ax.set_ylabel("γ at zero-rate limit (extrapolated)")
ax.grid(alpha=0.3)
ax.axhline(np.mean(medians_g0), color="r", ls="--",
           label=f"Mean = {np.mean(medians_g0):.3f}")
ax.legend()

fig.tight_layout()
out = "plots/gamma_vs_rate.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

print(f"\n=== Universal γ(R) fit ===")
print(f"  Linear:  γ(R) = {gamma0_lin:.3f} - {kappa*1000:.3f}/1000 · R")
if popt_e is not None:
    print(f"  Exp:     γ(R) = {popt_e[0]:.3f} · exp(-R / {popt_e[1]:.0f})")
print(f"\n=== Extrapolated γ₀ per date (median across boxes/dets) ===")
for dt in [d[0] for d in DATES]:
    vals = date_g0.get(dt, [])
    if vals:
        print(f"  {dt}:  γ₀ = {np.median(vals):.3f} ± {np.std(vals):.3f}  (N={len(vals)})")
print(f"  ALL:        γ₀ = {np.median(medians_g0):.3f} ± {np.std(medians_g0):.3f}")
