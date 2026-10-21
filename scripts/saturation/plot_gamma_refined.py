#!/usr/bin/env python3
"""Refine γ model: (a) apply γ(R) and check RMS, (b) filter SAA edges,
(c) test if γ also depends on Wide / Large independently.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from scipy.optimize import curve_fit
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
    fe.close()
    return {"PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci, "length": length_s}


# === Pool ALL bins from 90 detector slots into one big table ===
print("Pooling bins from 5 dates × 3 boxes × 6 detectors = 90 detector slots...\n")
rows = []
for date_label, fits_glob_tpl, sci_csv_tpl in DATES:
    for box in "ABC":
        D = load(date_label, fits_glob_tpl, sci_csv_tpl, box)
        if D is None:
            continue
        for det in range(6):
            sci = D["Sci"][:, det] / D["length"]
            wide = D["Wide"][:, det] / D["length"]
            large = D["Large"][:, det] / D["length"]
            pho = D["PHO"][:, det] / D["length"]
            # Filter (b): strict — exclude SAA edges and abnormal bins
            ok = (D["length"] > 0.8) & (sci > 100) & (pho > 700) & (pho < 3000)
            rows.append(pd.DataFrame({
                "date": date_label, "box": box, "det": det,
                "sci": sci[ok], "wide": wide[ok], "large": large[ok], "pho": pho[ok],
            }))
        print(f"  {date_label} Box {box} done")

big = pd.concat(rows, ignore_index=True)
big["nb"] = big["pho"] - big["wide"] - big["large"] - big["sci"]
print(f"\nTotal bins (after SAA filter): {len(big)}")
print(f"  PHO range: {big['pho'].min():.0f}–{big['pho'].max():.0f}")
print(f"  Sci range: {big['sci'].min():.0f}–{big['sci'].max():.0f}")
print(f"  Large range: {big['large'].min():.0f}–{big['large'].max():.0f}")
print(f"  Wide range: {big['wide'].min():.0f}–{big['wide'].max():.0f}")


# === (b) Fit γ(R) on cleaner pooled data, finer rate bins ===
print("\n=== (b) Refined γ(R) fit on filtered data ===")
RATE_BINS = np.linspace(700, 2400, 18)  # finer
bin_centers = 0.5 * (RATE_BINS[:-1] + RATE_BINS[1:])
gammas, gamma_err, counts = [], [], []
for k in range(len(bin_centers)):
    m = (big["pho"].values >= RATE_BINS[k]) & (big["pho"].values < RATE_BINS[k + 1])
    if m.sum() >= 200:
        # 4-term regression: nb = b + α·Sci + (β-1)·Wide + (γ-1)·Large
        X = np.column_stack([np.ones(m.sum()), big["sci"].values[m],
                             big["wide"].values[m], big["large"].values[m]])
        c, res, *_ = np.linalg.lstsq(X, big["nb"].values[m], rcond=None)
        gamma = 1.0 + c[3]
        # error from covariance
        sigma2 = np.mean((big["nb"].values[m] - X @ c) ** 2)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        gerr = np.sqrt(cov[3, 3])
        gammas.append(gamma); gamma_err.append(gerr); counts.append(m.sum())
    else:
        gammas.append(np.nan); gamma_err.append(np.nan); counts.append(m.sum())

gammas = np.array(gammas); gamma_err = np.array(gamma_err)
mask = ~np.isnan(gammas)
print(f"{'R':>6s}  {'γ':>5s}  {'±err':>5s}  N")
for k in range(len(bin_centers)):
    if mask[k]:
        print(f"  {bin_centers[k]:>5.0f}  {gammas[k]:.3f}  {gamma_err[k]:.3f}  {counts[k]:>5d}")

# Fit γ(R) with multiple models
def expmod(R, g0, R0):
    return g0 * np.exp(-R / R0)

def linmod(R, g0, k):
    return g0 - k * R

def parmod(R, g0, A, R0):
    """Saturated paralyzable-like:  γ0 · (1 - A·R/(1+R/R0))"""
    return g0 * (1 - A * R / (1 + R / R0))

# Linear
A_lin = np.column_stack([np.ones_like(bin_centers[mask]), bin_centers[mask]])
c_lin, *_ = np.linalg.lstsq(A_lin, gammas[mask], rcond=None)
g0_lin, k_lin = c_lin[0], -c_lin[1]
rms_lin = np.sqrt(np.mean((gammas[mask] - (g0_lin - k_lin * bin_centers[mask])) ** 2))

# Exp
popt_e, _ = curve_fit(expmod, bin_centers[mask], gammas[mask],
                       p0=[1.25, 8000], sigma=gamma_err[mask])
g0_e, R0_e = popt_e
rms_e = np.sqrt(np.mean((gammas[mask] - expmod(bin_centers[mask], *popt_e)) ** 2))

print(f"\n  Linear:  γ(R) = {g0_lin:.3f} − {k_lin*1000:.4f}/1000·R   (RMS_fit = {rms_lin:.4f})")
print(f"  Exp:     γ(R) = {g0_e:.3f} · exp(−R/{R0_e:.0f})         (RMS_fit = {rms_e:.4f})")


# === (c) Add Wide rate as additional axis: γ(R, W) ===
# Test if Wide rate also drives γ independently
print("\n=== (c) Does γ depend on Wide rate independently of PHO? ===")
# Strategy: 2D bin in (PHO, Wide), fit γ in each cell, see if γ varies with W at fixed R
print("Splitting (PHO, Wide) into 2D cells...")

# Use only data where Wide > 0 (some dates have Wide=0)
m_wide = big["wide"].values > 1
big_w = big.loc[m_wide]
print(f"  bins with Wide > 1: {len(big_w)} / {len(big)}")

# 2D bins
PHO_BINS = np.array([700, 900, 1100, 1300, 1500, 1700, 2000, 2400])
WIDE_BINS = np.array([0, 10, 30, 60, 120, 250, 600, 2500])
gamma_2d = np.full((len(PHO_BINS) - 1, len(WIDE_BINS) - 1), np.nan)
n_2d = np.zeros_like(gamma_2d, dtype=int)
for i in range(len(PHO_BINS) - 1):
    for j in range(len(WIDE_BINS) - 1):
        m = ((big_w["pho"] >= PHO_BINS[i]) & (big_w["pho"] < PHO_BINS[i + 1])
             & (big_w["wide"] >= WIDE_BINS[j]) & (big_w["wide"] < WIDE_BINS[j + 1]))
        if m.sum() >= 80:
            X = np.column_stack([np.ones(m.sum()), big_w["sci"].values[m],
                                 big_w["wide"].values[m], big_w["large"].values[m]])
            try:
                c, *_ = np.linalg.lstsq(X, big_w["nb"].values[m], rcond=None)
                gamma_2d[i, j] = 1.0 + c[3]
                n_2d[i, j] = m.sum()
            except Exception:
                pass

print("\nγ in 2D (PHO bin × Wide bin):")
header = "PHO|Wide"
print(f"{header:>10s}", *[f"{WIDE_BINS[j]:>3d}-{WIDE_BINS[j+1]:>4d}" for j in range(len(WIDE_BINS)-1)])
for i in range(len(PHO_BINS) - 1):
    line = f"{PHO_BINS[i]:>4d}–{PHO_BINS[i+1]:>4d}  "
    for j in range(len(WIDE_BINS) - 1):
        if not np.isnan(gamma_2d[i, j]):
            line += f"  {gamma_2d[i, j]:.3f} "
        else:
            line += "    --   "
    print(line)


# === (a) Apply γ(R) model and compute residual RMS per detector ===
print("\n=== (a) Apply γ(R) model: residual RMS per (date, box, det) ===")

def gamma_fn(R):
    return g0_e * np.exp(-R / R0_e)

per_det_rms = []
for (date, box, det), grp in big.groupby(["date", "box", "det"], observed=True):
    sci = grp["sci"].values; wide = grp["wide"].values
    large = grp["large"].values; pho = grp["pho"].values
    nb = pho - 1.9 * wide - gamma_fn(pho) * large - sci  # γ(R)·Large
    # Linear-in-Sci fit
    X = np.column_stack([np.ones_like(sci), sci])
    c, *_ = np.linalg.lstsq(X, nb, rcond=None)
    rms_dyn = np.sqrt(np.mean((nb - X @ c) ** 2))

    # Compare: fixed γ=1.22 (low-rate limit)
    nb_fix = pho - 1.9 * wide - 1.22 * large - sci
    c2, *_ = np.linalg.lstsq(X, nb_fix, rcond=None)
    rms_fix = np.sqrt(np.mean((nb_fix - X @ c2) ** 2))

    # Per-detector best-fit γ (no rate dependence)
    nb_base = pho - wide - large - sci
    X4 = np.column_stack([np.ones_like(sci), sci, wide, large])
    c4, *_ = np.linalg.lstsq(X4, nb_base, rcond=None)
    rms_perdet = np.sqrt(np.mean((nb_base - X4 @ c4) ** 2))

    # Poisson estimate (from the count itself)
    sigma_poiss = np.sqrt(pho.mean() + 4 * wide.mean() + 1.5 * large.mean() + sci.mean())

    per_det_rms.append({
        "date": date, "box": box, "det": det,
        "rms_dyn": rms_dyn, "rms_fix": rms_fix, "rms_perdet": rms_perdet,
        "poisson": sigma_poiss, "n": len(grp),
    })

print(f"\n{'Date':>11s} {'B':>2s} {'D':>2s}  {'γ(R)':>7s} {'γ=1.22':>7s} {'perdet':>7s}  {'Poisson':>7s}")
for r in per_det_rms[:30]:
    print(f"  {r['date']}  {r['box']:>1s} {r['det']:>2d}  {r['rms_dyn']:>7.1f} {r['rms_fix']:>7.1f} "
          f"{r['rms_perdet']:>7.1f}  {r['poisson']:>7.1f}")
print(f"  ... ({len(per_det_rms)-30} more rows)")

mean_dyn = np.mean([r["rms_dyn"] for r in per_det_rms])
mean_fix = np.mean([r["rms_fix"] for r in per_det_rms])
mean_pd = np.mean([r["rms_perdet"] for r in per_det_rms])
mean_pois = np.mean([r["poisson"] for r in per_det_rms])
print(f"\n=== Mean RMS over {len(per_det_rms)} (date, box, det) ===")
print(f"  γ(R) model         : {mean_dyn:.1f} cnt/s")
print(f"  Fixed γ=1.22       : {mean_fix:.1f} cnt/s")
print(f"  Per-det best (β,γ) : {mean_pd:.1f} cnt/s")
print(f"  Poisson limit      : {mean_pois:.1f} cnt/s")


# === Plot ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) γ(R) curve with fits
ax = axes[0, 0]
ax.errorbar(bin_centers[mask], gammas[mask], yerr=gamma_err[mask], fmt="ko", capsize=3,
            label=f"Pooled fit (N≥200/bin)")
xs = np.linspace(700, 2400, 200)
ax.plot(xs, g0_lin - k_lin * xs, "r--", lw=1.4,
        label=f"Linear: γ₀={g0_lin:.3f}, slope={-k_lin*1000:.4f}/1000")
ax.plot(xs, expmod(xs, *popt_e), "b-", lw=1.5,
        label=f"Exp: γ₀={g0_e:.3f}, R₀={R0_e:.0f}")
ax.set_xlabel("PHO rate [cnt/s/det]")
ax.set_ylabel("γ")
ax.set_title("(b) γ(R) on filtered (R>700) pooled data")
ax.grid(alpha=0.3)
ax.legend()

# 2D γ(PHO, Wide)
ax = axes[0, 1]
im = ax.imshow(gamma_2d.T, aspect="auto", origin="lower", cmap="RdBu_r",
               vmin=0.9, vmax=1.3)
ax.set_xticks(np.arange(len(PHO_BINS) - 1))
ax.set_xticklabels([f"{PHO_BINS[i]}–{PHO_BINS[i+1]}" for i in range(len(PHO_BINS)-1)],
                    rotation=30, fontsize=8)
ax.set_yticks(np.arange(len(WIDE_BINS) - 1))
ax.set_yticklabels([f"{WIDE_BINS[j]}–{WIDE_BINS[j+1]}" for j in range(len(WIDE_BINS)-1)],
                    fontsize=8)
ax.set_xlabel("PHO rate bin")
ax.set_ylabel("Wide rate bin")
ax.set_title("(c) γ in 2D (PHO, Wide) cells")
# Annotate
for i in range(len(PHO_BINS) - 1):
    for j in range(len(WIDE_BINS) - 1):
        if not np.isnan(gamma_2d[i, j]):
            ax.text(i, j, f"{gamma_2d[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="γ")

# (a) RMS comparison
ax = axes[1, 0]
methods = ["γ(R)\nrate-dep", "γ=1.22\n固定", "Per-det\n最优 (β,γ)", "Poisson"]
means = [mean_dyn, mean_fix, mean_pd, mean_pois]
colors = ["C0", "C3", "C2", "k"]
ax.bar(methods, means, color=colors)
for i, v in enumerate(means):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)
ax.set_ylabel("Mean residual RMS [cnt/s]")
ax.set_title("(a) RMS comparison: γ(R) vs fixed vs Poisson")
ax.grid(alpha=0.3, axis="y")

# Distribution of RMS per detector
ax = axes[1, 1]
ax.hist([r["rms_dyn"] for r in per_det_rms], bins=20, alpha=0.6,
        color="C0", label="γ(R)")
ax.hist([r["rms_fix"] for r in per_det_rms], bins=20, alpha=0.6,
        color="C3", label="γ=1.22")
ax.hist([r["rms_perdet"] for r in per_det_rms], bins=20, alpha=0.6,
        color="C2", label="Per-det best")
ax.set_xlabel("RMS [cnt/s]")
ax.set_ylabel("count")
ax.set_title("Distribution of per-detector RMS")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
out = "plots/gamma_refined.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")
