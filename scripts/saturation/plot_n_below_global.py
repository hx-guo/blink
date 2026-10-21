#!/usr/bin/env python3
"""Verify N_below = b + α·Sci linearity across 5 dates × 18 detectors,
extract global constants (b, α) once and for all.

N_below := PHO - 2·Wide - 1.2·Large - Sci  (using the refined β=2, γ=1.2)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
BETA = 2.0
GAMMA = 1.0  # don't pre-correct Large; absorb 0.2·L into N_below model

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

    # Filter: reasonable Length, total Sci 5-95% percentile
    valid = (L_cycles > 50000) & (Sci.sum(axis=1) > 100)
    sci_total = Sci.sum(axis=1)
    if valid.sum() < 50:
        fe.close()
        return None
    p5, p95 = np.percentile(sci_total[valid], [5, 95])
    valid &= (sci_total >= p5) & (sci_total <= p95)
    fe.close()
    return {"PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci,
            "length": length_s, "valid": valid, "date": date_label, "box": box}


def fit_lin(y, x):
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[0], c[1], y - X @ c


# Load all 90 detector slots
print("Loading data...")
all_slots = []
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
            nb = pho - BETA * wide - GAMMA * large - sci  # Refined formula

            b, alpha, resid = fit_lin(nb, sci)
            all_slots.append({
                "date": date_label, "box": box, "det": det,
                "sci": sci, "nb": nb,
                "b": b, "alpha": alpha,
                "rms": np.sqrt(np.mean(resid ** 2)),
                "n": len(sci),
                "sci_med": np.median(sci),
                "nb_med": np.median(nb),
            })
        print(f"  {date_label} Box {box} done")

print(f"\nTotal slots: {len(all_slots)}")


# === Plot 1: 18 panels (one per detector identity, all 5 dates overlaid) ===
fig, axes = plt.subplots(3, 6, figsize=(20, 11), sharex=True, sharey=True)
date_colors = {d[0]: c for d, c in zip(DATES, plt.cm.viridis(np.linspace(0, 1, 5)))}
for slot in all_slots:
    box_idx = "ABC".index(slot["box"])
    det = slot["det"]
    ax = axes[box_idx, det]
    color = date_colors[slot["date"]]
    ax.scatter(slot["sci"], slot["nb"], s=1.5, alpha=0.15, color=color, rasterized=True)
    # Fit line
    xs = np.linspace(slot["sci"].min(), slot["sci"].max(), 50)
    ax.plot(xs, slot["b"] + slot["alpha"] * xs, color=color, lw=1.0, alpha=0.7)

# Add legend
for ax in axes.flat:
    ax.grid(alpha=0.2)
for box_idx, box in enumerate("ABC"):
    for det in range(6):
        ax = axes[box_idx, det]
        ax.set_title(f"{box}{det}", fontsize=9)
        if det == 0:
            ax.set_ylabel(f"Box {box}\nN_below [cnt/s]", fontsize=9)
        if box_idx == 2:
            ax.set_xlabel("Sci [cnt/s]", fontsize=9)

# Make a separate legend axis
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=date_colors[d[0]],
                   markersize=6, label=d[0]) for d in DATES]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=9)

fig.suptitle(f"N_below = (PHO − 2·Wide − 1.2·Large) − Sci  vs  Sci  per (box, det)\n"
             f"Each color = one of 5 dates;  18 panels = 18 detectors", fontsize=11)
fig.tight_layout()
out1 = "plots/n_below_per_det_5dates_gamma1.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"Saved: {out1}")


# === Plot 2: histogram of (b, α) across 90 slots ===
fig2, axes = plt.subplots(2, 2, figsize=(14, 9))
bs = np.array([s["b"] for s in all_slots])
alphas = np.array([s["alpha"] for s in all_slots])
rmss = np.array([s["rms"] for s in all_slots])

# Top-left: b histogram
ax = axes[0, 0]
ax.hist(bs, bins=25, color="C0", edgecolor="k", alpha=0.7)
ax.axvline(np.median(bs), color="r", lw=2,
           label=f"median = {np.median(bs):.1f}")
ax.axvline(np.mean(bs), color="g", lw=2, ls="--",
           label=f"mean = {np.mean(bs):.1f} ± {np.std(bs):.1f}")
ax.set_xlabel("b [cnt/s/det]  (constant background)")
ax.set_ylabel("count")
ax.set_title(f"b distribution (90 slots, range {bs.min():.0f}–{bs.max():.0f})")
ax.legend()
ax.grid(alpha=0.3)

# Top-right: α histogram
ax = axes[0, 1]
ax.hist(alphas, bins=25, color="C1", edgecolor="k", alpha=0.7)
ax.axvline(np.median(alphas), color="r", lw=2,
           label=f"median = {np.median(alphas):.3f}")
ax.axvline(np.mean(alphas), color="g", lw=2, ls="--",
           label=f"mean = {np.mean(alphas):.3f} ± {np.std(alphas):.3f}")
ax.set_xlabel("α  (Sci-correlated fraction)")
ax.set_ylabel("count")
ax.set_title(f"α distribution (range {alphas.min():.3f}–{alphas.max():.3f})")
ax.legend()
ax.grid(alpha=0.3)

# Bottom-left: (b, α) scatter, color by detector_id
ax = axes[1, 0]
for slot in all_slots:
    color = date_colors[slot["date"]]
    marker = {"A": "o", "B": "s", "C": "^"}[slot["box"]]
    ax.scatter(slot["b"], slot["alpha"], color=color, marker=marker, s=40,
               edgecolor="k", linewidth=0.4, alpha=0.8)
ax.set_xlabel("b [cnt/s]")
ax.set_ylabel("α")
ax.set_title("(b, α) scatter\n(color = date, marker = box)")
ax.grid(alpha=0.3)
# Annotate clusters
ax.scatter(np.median(bs), np.median(alphas), marker="*", s=300, color="red",
           edgecolor="k", linewidth=1, label=f"median ({np.median(bs):.1f}, {np.median(alphas):.3f})",
           zorder=10)
ax.legend(fontsize=9)

# Bottom-right: per-detector_id consistency: same det across 5 dates
ax = axes[1, 1]
det_labels = []
det_b_means = []; det_b_stds = []
det_a_means = []; det_a_stds = []
for box in "ABC":
    for det in range(6):
        slots = [s for s in all_slots if s["box"] == box and s["det"] == det]
        if len(slots) == 0:
            continue
        det_labels.append(f"{box}{det}")
        det_b_means.append(np.mean([s["b"] for s in slots]))
        det_b_stds.append(np.std([s["b"] for s in slots]))
        det_a_means.append(np.mean([s["alpha"] for s in slots]))
        det_a_stds.append(np.std([s["alpha"] for s in slots]))
xs = np.arange(len(det_labels))
ax2 = ax.twinx()
ax.errorbar(xs - 0.15, det_b_means, yerr=det_b_stds, fmt="o", color="C0", capsize=3, label="b")
ax2.errorbar(xs + 0.15, det_a_means, yerr=det_a_stds, fmt="s", color="C1", capsize=3, label="α")
ax.set_xticks(xs)
ax.set_xticklabels(det_labels, rotation=90, fontsize=8)
ax.set_ylabel("b [cnt/s]", color="C0")
ax2.set_ylabel("α", color="C1")
ax.set_title("Per-detector consistency across 5 dates\n(error bar = std across dates)")
ax.grid(alpha=0.3)

fig2.tight_layout()
out2 = "plots/n_below_constants_summary_gamma1.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")


# === Numerical summary ===
print(f"\n{'='*60}")
print(f"N_below = b + α·Sci  fit on 90 (date × box × det) slots")
print(f"  Using: N_below ≡ PHO − {BETA}·Wide − {GAMMA}·Large − Sci")
print(f"{'='*60}")
print(f"  b      mean = {np.mean(bs):>5.1f}  median = {np.median(bs):>5.1f}  std = {np.std(bs):>4.1f}  range {bs.min():.1f}–{bs.max():.1f}")
print(f"  α      mean = {np.mean(alphas):.3f}  median = {np.median(alphas):.3f}  std = {np.std(alphas):.3f}  range {alphas.min():.3f}–{alphas.max():.3f}")
print(f"  RMS    mean = {np.mean(rmss):.1f} cnt/s   median = {np.median(rmss):.1f} cnt/s")

# Per-box stats
print(f"\nPer-box:")
for box in "ABC":
    bs_b = [s["b"] for s in all_slots if s["box"] == box]
    as_b = [s["alpha"] for s in all_slots if s["box"] == box]
    print(f"  Box {box}: b = {np.mean(bs_b):.1f} ± {np.std(bs_b):.1f},  "
          f"α = {np.mean(as_b):.3f} ± {np.std(as_b):.3f}  (N={len(bs_b)})")

# Per-detector_id consistency (same box × det across 5 dates)
print(f"\nPer-detector_id consistency (std across 5 dates within same det):")
det_b_var = []
det_a_var = []
for box in "ABC":
    for det in range(6):
        slots = [s for s in all_slots if s["box"] == box and s["det"] == det]
        if len(slots) >= 2:
            det_b_var.append(np.std([s["b"] for s in slots]))
            det_a_var.append(np.std([s["alpha"] for s in slots]))
print(f"  Within-detector std of b:  mean {np.mean(det_b_var):.2f} cnt/s   max {np.max(det_b_var):.2f}")
print(f"  Within-detector std of α:  mean {np.mean(det_a_var):.4f}        max {np.max(det_a_var):.4f}")

# Suggested global constants
print(f"\n{'='*60}")
print("RECOMMENDED GLOBAL CONSTANTS")
print(f"{'='*60}")
print(f"  β = {BETA}    (Wide multiplier; CsI tail re-trigger)")
print(f"  γ = {GAMMA}    (Large multiplier)")
print(f"  b = {np.median(bs):.1f} cnt/s/det    (constant low-energy background)")
print(f"  α = {np.median(alphas):.3f}                  (Sci-correlated low-energy fraction)")
print(f"")
print(f"  Sci_predicted = (PHO - {BETA}·Wide - {GAMMA}·Large - {np.median(bs):.1f}) / {1+np.median(alphas):.3f}")
