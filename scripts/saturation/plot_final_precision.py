#!/usr/bin/env python3
"""Final precision test:
  Apply γ(R) = 1.22·exp(−R/7800) per bin (from earlier per-det analysis)
  vs fixed γ = 1.19
  vs per-detector best (β, γ)

Compare residual RMS to actual Poisson estimate (computed properly).
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


# Five γ models to compare
MODELS = {
    "γ=1.0": lambda pho: np.full_like(pho, 1.0),
    "γ=1.19": lambda pho: np.full_like(pho, 1.19),
    "γ=1.22": lambda pho: np.full_like(pho, 1.22),
    "γ(R)=1.22·exp(-R/7800)": lambda pho: 1.22 * np.exp(-pho / 7800.0),
    "γ(R)=1.30-0.0001·R": lambda pho: 1.30 - 0.0001 * pho,  # alternative simple model
}
BETA = 1.9


def fit_lin(y, x):
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c, y - X @ c


def per_det_best(pho, wide, large, sci):
    nb_base = pho - wide - large - sci
    X = np.column_stack([np.ones_like(sci), sci, wide, large])
    c, *_ = np.linalg.lstsq(X, nb_base, rcond=None)
    return c[0], c[1], 1.0 + c[2], 1.0 + c[3]


# === Process all dates ===
print("Loading 5 dates × 3 boxes ...\n")
results = []
for date_label, fits_glob_tpl, sci_csv_tpl in DATES:
    for box in "ABC":
        D = load(date_label, fits_glob_tpl, sci_csv_tpl, box)
        if D is None:
            continue
        # SAA filter
        v = (D["length"] > 0.8) & (D["Sci"].sum(axis=1) > 600) & (D["Sci"].sum(axis=1) < 12000)
        if v.sum() < 100:
            continue
        for det in range(6):
            sci = D["Sci"][v, det] / D["length"][v]
            wide = D["Wide"][v, det] / D["length"][v]
            large = D["Large"][v, det] / D["length"][v]
            pho = D["PHO"][v, det] / D["length"][v]

            row = {"date": date_label, "box": box, "det": det, "n_bins": len(pho),
                   "<PHO>": np.mean(pho), "<Wide>": np.mean(wide),
                   "<Large>": np.mean(large), "<Sci>": np.mean(sci)}

            # Apply each γ model + linear-in-Sci fit
            for name, gfunc in MODELS.items():
                gamma_R = gfunc(pho)
                nb = pho - BETA * wide - gamma_R * large - sci
                _, resid = fit_lin(nb, sci)
                row[f"RMS_{name}"] = np.sqrt(np.mean(resid ** 2))

            # Per-detector fully fitted (best possible per-det)
            b_pd, a_pd, beta_pd, gamma_pd = per_det_best(pho, wide, large, sci)
            nb_pd = pho - beta_pd * wide - gamma_pd * large - sci
            _, resid_pd = fit_lin(nb_pd, sci)
            row["RMS_perdet"] = np.sqrt(np.mean(resid_pd ** 2))

            # Poisson-only: residual variance for independent Poisson on each counter
            # nb_obs = Po(PHO) - Po(2W) - Po(γL) - Po(Sci) - (b + α·Sci)
            # The fit absorbs variance correlated with Sci. The independent residual is:
            #   Var = Var(PHO) + 4·Var(W) + γ²·Var(L) + Var(Sci)·(1-r²) + corrections
            # For approximation, use:
            #   sqrt(<PHO> + 4·<W> + γ²·<L> + (1-α²)·<Sci>) ≈ sqrt(<PHO> + 4W + γ²L)  if α ~ 1
            # Empirical estimate via Var(nb_pd - residual):
            row["poisson"] = np.sqrt(row["<PHO>"] + 4 * row["<Wide>"] +
                                      gamma_pd ** 2 * row["<Large>"])
            row["beta_pd"] = beta_pd
            row["gamma_pd"] = gamma_pd

            results.append(row)
        print(f"  {date_label} Box {box} done ({v.sum()} valid bins)")

print(f"\nTotal (date, box, det) slots: {len(results)}")

# === Summary ===
print(f"\n=== Mean RMS across {len(results)} detector slots ===")
model_names = list(MODELS.keys()) + ["perdet"]
for name in model_names:
    key = f"RMS_{name}"
    rmss = np.array([r[key] for r in results])
    print(f"  {name:>30s}:  mean RMS = {rmss.mean():>5.1f} cnt/s   median = {np.median(rmss):.1f}")

mean_pois = np.mean([r["poisson"] for r in results])
print(f"  {'Poisson sum (sqrt)':>30s}:  mean = {mean_pois:.1f} cnt/s")

# Per-date
print(f"\n=== Per-date mean RMS (γ=1.19 vs γ(R) vs perdet) ===")
print(f"{'Date':>11s}  {'γ=1.0':>7s} {'γ=1.19':>7s} {'γ=1.22':>7s} {'γ(R)':>7s} {'perdet':>7s}  Poisson")
for date in [d[0] for d in DATES]:
    rs = [r for r in results if r["date"] == date]
    if not rs:
        continue
    print(f"  {date}  "
          f"{np.mean([r['RMS_γ=1.0'] for r in rs]):>7.1f} "
          f"{np.mean([r['RMS_γ=1.19'] for r in rs]):>7.1f} "
          f"{np.mean([r['RMS_γ=1.22'] for r in rs]):>7.1f} "
          f"{np.mean([r['RMS_γ(R)=1.22·exp(-R/7800)'] for r in rs]):>7.1f} "
          f"{np.mean([r['RMS_perdet'] for r in rs]):>7.1f}  "
          f"{np.mean([r['poisson'] for r in rs]):>5.1f}")

# === Plot: histogram of RMS values for each method ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
colors = {"γ=1.0": "C2", "γ=1.19": "C3", "γ=1.22": "C1",
          "γ(R)=1.22·exp(-R/7800)": "C0", "γ(R)=1.30-0.0001·R": "C4",
          "perdet": "k"}
for name in model_names:
    key = f"RMS_{name}"
    vals = [r[key] for r in results]
    ax.hist(vals, bins=np.arange(0, 100, 2.5), alpha=0.5, label=f"{name} ({np.mean(vals):.1f})",
            color=colors.get(name, "gray"))
ax.set_xlabel("RMS [cnt/s]")
ax.set_ylabel("count")
ax.set_title(f"Per-detector RMS distribution (N={len(results)} slots)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Bar chart: mean RMS per method
ax = axes[1]
mean_rmss = []
labels = []
for name in model_names:
    key = f"RMS_{name}"
    mean_rmss.append(np.mean([r[key] for r in results]))
    labels.append(name)
mean_rmss.append(mean_pois)
labels.append("Poisson sum")
bars = ax.bar(labels, mean_rmss, color=[colors.get(n, "gray") for n in model_names] + ["lightgray"])
for bar, val in zip(bars, mean_rmss):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}",
            ha="center", fontsize=9)
ax.set_ylabel("Mean RMS [cnt/s]")
ax.set_title("Mean residual RMS comparison")
ax.tick_params(axis="x", rotation=15)
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = "plots/gamma_final_precision.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")
