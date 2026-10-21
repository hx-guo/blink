#!/usr/bin/env python3
"""γ vs MET (time) — fit decay model + test rate/spectrum dependence.

Tests:
  1. γ vs MET (date midpoint) — linear / exponential fit
  2. Rate dependence: split each date into low/high count-rate halves, refit γ
     If γ_low ≠ γ_high, γ depends on rate/spectrum — not a pure hardware constant
  3. γ vs spectral hardness (median <Large>/<Sci> per date)
  4. γ vs <PHO> rate per date
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
    ("2020-04-15", 81739844, {
        "A": ("data/1B/2020/20200415/0766/*.fits", "/tmp/200415_boxA.csv", 0),
        "B": ("data/1B/2020/20200415/1009/*.fits", "/tmp/200415_boxB.csv", 6),
        "C": ("data/1B/2020/20200415/1781/*.fits", "/tmp/200415_boxC.csv", 12),
    }),
    ("2020-04-28", 82884646, {
        "A": ("data/1B/2020/20200428/0766/*.fits", "/tmp/200428_boxA.csv", 0),
        "B": ("data/1B/2020/20200428/1009/*.fits", "/tmp/200428_boxB.csv", 6),
        "C": ("data/1B/2020/20200428/1781/*.fits", "/tmp/200428_boxC.csv", 12),
    }),
    ("2022-10-09", 160123395, {
        "A": ("data/1B/2022/20221009/0766/*.fits", "/tmp/221009_boxA.csv", 0),
        "B": ("data/1B/2022/20221009/1009/*.fits", "/tmp/221009_boxB.csv", 6),
        "C": ("data/1B/2022/20221009/1781/*.fits", "/tmp/221009_boxC.csv", 12),
    }),
    ("2026-02-26", 446724000, {
        "A": ("data/1B/2026/20260226/0766/*.fits", "/tmp/260226_boxA_full.csv", 0),
        "B": ("data/1B/2026/20260226/1009/*.fits", "/tmp/260226_boxB_full.csv", 6),
        "C": ("data/1B/2026/20260226/1781/*.fits", "/tmp/260226_boxC_full.csv", 12),
    }),
    ("2026-04-10", 35081270, {
        "A": ("data/1B/2026/20260410/0766/*.fits", "/tmp/260410_boxA.csv", 0),
        "B": ("data/1B/2026/20260410/1009/*.fits", "/tmp/260410_boxB.csv", 6),
        "C": ("data/1B/2026/20260410/1781/*.fits", "/tmp/260410_boxC.csv", 12),
    }),
]


def fit_linear(y, x):
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[0], c[1], y - X @ c


def fit_betagamma(pho, wide, large, sci):
    """Return (β, γ) by regression of (pho-wide-large-sci) on (1, sci, wide, large)."""
    nb_base = pho - wide - large - sci
    X = np.column_stack([np.ones_like(sci), sci, wide, large])
    c, *_ = np.linalg.lstsq(X, nb_base, rcond=None)
    return 1.0 + c[2], 1.0 + c[3]


def load_one(label, fits_glob, sci_csv, det_off):
    fits_files = sorted(glob.glob(fits_glob))
    if not fits_files:
        return None
    fe = fits.open(fits_files[0], memmap=True)
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

    return {
        "PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci,
        "length": length_s, "valid": valid, "met": met_eng,
    }


# === Run analysis ===
print("Computing per-detector γ + rate-split γ for all 5 dates...\n")
all_per_det = []   # one row per (date, box, det)
date_summary = {}  # date -> dict with median rates etc.
for date_label, met_start, boxes in DATES:
    print(f"--- {date_label} ---")
    for box_name in "ABC":
        fits_glob, sci_csv, det_off = boxes[box_name]
        D = load_one(date_label, fits_glob, sci_csv, det_off)
        if D is None:
            continue
        v = D["valid"]
        for det in range(6):
            sci = D["Sci"][v, det] / D["length"][v]
            wide = D["Wide"][v, det] / D["length"][v]
            large = D["Large"][v, det] / D["length"][v]
            pho = D["PHO"][v, det] / D["length"][v]

            beta, gamma = fit_betagamma(pho, wide, large, sci)

            # Rate split: low/high half of total (Sci+Large+Wide)
            rate = sci + wide + large
            cut = np.median(rate)
            lo = rate < cut
            hi = rate >= cut
            try:
                _, gamma_lo = fit_betagamma(pho[lo], wide[lo], large[lo], sci[lo])
                _, gamma_hi = fit_betagamma(pho[hi], wide[hi], large[hi], sci[hi])
            except Exception:
                gamma_lo, gamma_hi = np.nan, np.nan

            all_per_det.append({
                "date": date_label, "met_start": met_start,
                "box": box_name, "det": det,
                "beta": beta, "gamma": gamma,
                "gamma_lo": gamma_lo, "gamma_hi": gamma_hi,
                "sci": np.median(sci), "wide": np.median(wide),
                "large": np.median(large), "pho": np.median(pho),
            })
        rs = [r for r in all_per_det if r["date"] == date_label and r["box"] == box_name]
        gs = [r["gamma"] for r in rs]
        gls = [r["gamma_lo"] for r in rs]
        ghs = [r["gamma_hi"] for r in rs]
        print(f"  Box {box_name}: γ={np.mean(gs):.3f}  γ_low_rate={np.nanmean(gls):.3f}  γ_high_rate={np.nanmean(ghs):.3f}")

# Collect per-date averages
print("\n=== Per-date global γ + rate split ===")
print(f"{'Date':>11s}  {'<γ>':>5s}  {'γ_lo':>5s}  {'γ_hi':>5s}  Δ(hi-lo)  <PHO>  <Sci>  <Wide>  <Large>")
date_records = []
for date_label, met_start, _ in DATES:
    rs = [r for r in all_per_det if r["date"] == date_label]
    g_mean = np.mean([r["gamma"] for r in rs])
    g_std = np.std([r["gamma"] for r in rs])
    gl_mean = np.nanmean([r["gamma_lo"] for r in rs])
    gh_mean = np.nanmean([r["gamma_hi"] for r in rs])
    pho_mean = np.mean([r["pho"] for r in rs])
    sci_mean = np.mean([r["sci"] for r in rs])
    wide_mean = np.mean([r["wide"] for r in rs])
    large_mean = np.mean([r["large"] for r in rs])
    delta = gh_mean - gl_mean
    print(f"  {date_label}  {g_mean:.3f}  {gl_mean:.3f}  {gh_mean:.3f}  {delta:+.3f}  "
          f"{pho_mean:>5.0f}  {sci_mean:>5.0f}  {wide_mean:>6.0f}  {large_mean:>7.0f}")
    date_records.append({
        "date": date_label, "met": met_start,
        "g_mean": g_mean, "g_std": g_std,
        "g_lo": gl_mean, "g_hi": gh_mean,
        "pho": pho_mean, "sci": sci_mean, "wide": wide_mean, "large": large_mean,
    })

# === Plot 1: γ vs MET with linear / exp fit ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: γ vs MET
ax = axes[0, 0]
mets = np.array([r["met"] for r in date_records])
gs = np.array([r["g_mean"] for r in date_records])
gserr = np.array([r["g_std"] for r in date_records])
date_labels = [r["date"] for r in date_records]

# Per-detector points (faint)
for r in all_per_det:
    box_marker = {"A": "o", "B": "s", "C": "^"}[r["box"]]
    ax.scatter(r["met_start"], r["gamma"], marker=box_marker, alpha=0.3,
               color="C0", s=15)
# Date averages with error bars
ax.errorbar(mets, gs, yerr=gserr, fmt="ko", markersize=10, capsize=4,
            ecolor="k", label="Date mean ± std (18 dets)")

# Linear fit
A = np.column_stack([np.ones_like(mets, dtype=float), mets.astype(float)])
clin, *_ = np.linalg.lstsq(A, gs, rcond=None)
ts = np.linspace(mets.min(), mets.max(), 200)
ax.plot(ts, clin[0] + clin[1] * ts, "r--", lw=1.5,
        label=f"Linear: γ = {clin[0]:.3f} + {clin[1]*1e9:.3f}·1e-9·MET")

# Convert MET coefficient to per-year rate (HXMT MET starts 2012-01-01, but seconds are seconds)
gamma_per_year = clin[1] * 365.25 * 86400
ax.set_xlabel("MET [s]")
ax.set_ylabel("γ (Large multiplier)")
ax.set_title(f"γ vs MET — linear drift = {gamma_per_year:+.4f}/year")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# Add date labels at top
for r in date_records:
    ax.annotate(r["date"], (r["met"], r["g_mean"]), fontsize=8,
                xytext=(5, 12), textcoords="offset points", rotation=15)

# Panel 2: γ split by rate (low vs high half, per date)
ax = axes[0, 1]
glo = np.array([r["g_lo"] for r in date_records])
ghi = np.array([r["g_hi"] for r in date_records])
xs = np.arange(len(date_records))
ax.bar(xs - 0.2, glo, width=0.35, color="C0", label="Low-rate half (γ)")
ax.bar(xs + 0.2, ghi, width=0.35, color="C3", label="High-rate half (γ)")
for i, (l, h) in enumerate(zip(glo, ghi)):
    ax.annotate(f"Δ={h-l:+.3f}", (i, max(l, h) + 0.01), ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(date_labels, rotation=15)
ax.set_ylabel("γ")
ax.set_title("Within-date rate split: does γ depend on count rate?")
ax.set_ylim(0.95, 1.35)
ax.grid(alpha=0.3, axis="y")
ax.legend()

# Panel 3: γ vs <Large/Sci> (hardness) per date
ax = axes[1, 0]
hardness = np.array([r["large"] / r["sci"] for r in date_records])
ax.errorbar(hardness, gs, yerr=gserr, fmt="ko", markersize=10, capsize=4)
for r in date_records:
    ax.annotate(r["date"][5:], (r["large"]/r["sci"], r["g_mean"]),
                fontsize=8, xytext=(5, 5), textcoords="offset points")
rho = np.corrcoef(hardness, gs)[0, 1]
ax.set_xlabel("Hardness <Large> / <Sci>")
ax.set_ylabel("γ")
ax.set_title(f"γ vs spectral hardness  (ρ = {rho:+.3f})")
ax.grid(alpha=0.3)

# Panel 4: γ vs <PHO> rate per date
ax = axes[1, 1]
phos = np.array([r["pho"] for r in date_records])
ax.errorbar(phos, gs, yerr=gserr, fmt="ko", markersize=10, capsize=4)
for r in date_records:
    ax.annotate(r["date"][5:], (r["pho"], r["g_mean"]),
                fontsize=8, xytext=(5, 5), textcoords="offset points")
rho = np.corrcoef(phos, gs)[0, 1]
ax.set_xlabel("<PHO> rate [cnt/s/det]")
ax.set_ylabel("γ")
ax.set_title(f"γ vs PHO rate  (ρ = {rho:+.3f})")
ax.grid(alpha=0.3)

fig.tight_layout()
out = "plots/gamma_vs_time.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# Print numeric details
print("\n=== Drift fit ===")
print(f"  Linear:   γ(t) = {clin[0]:.3f} + {clin[1]*1e9:.3f}e-9 · MET  →  drift = {gamma_per_year*100:+.3f}% per year")

# Rate-split summary
print("\n=== Rate split: γ_high - γ_low per date ===")
for r in date_records:
    print(f"  {r['date']}:  γ_lo={r['g_lo']:.3f}  γ_hi={r['g_hi']:.3f}  Δ={r['g_hi']-r['g_lo']:+.3f}")
