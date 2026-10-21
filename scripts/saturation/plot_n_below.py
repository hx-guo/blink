#!/usr/bin/env python3
"""N_below = (PHO - Wide - Large) - Sci on non-saturated bins.

Model:
    PHO = N_normal + N_wide + N_large + N_below
    Sci = N_normal (Am-241 OOC events follow normal path -> already in Sci)
    => N_below = (PHO - Wide - Large) - Sci

Question: is N_below stable?
  (a) constant absolute rate?
  (b) constant ratio to Sci?
  (c) two-component: N_below = b + alpha * Sci ?
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import csv
import os
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
BOXES = [
    ("A", "0766", "/tmp/260226_boxA_full.csv", 0),
    ("B", "1009", "/tmp/260226_boxB_full.csv", 6),
    ("C", "1781", "/tmp/260226_boxC_full.csv", 12),
]

sat_intervals = {"A": [], "B": [], "C": []}
with open("/tmp/detect_260226a.csv") as f:
    for r in csv.DictReader(f):
        sat_intervals[r["box"]].append((float(r["start_met"]), float(r["stop_met"])))
for k in sat_intervals:
    sat_intervals[k].sort()


def overlaps_saturation(t0, t1, intervals):
    for s, e in intervals:
        if s < t1 and e > t0:
            return True
    return False


data = {}
for box_name, eng_code, sci_csv, det_off in BOXES:
    print(f"Loading Box {box_name}...")
    eng_file = f"data/1B/2026/20260226/{eng_code}/HXMT_1B_{eng_code}_20260226T100000_G076262_000_004.fits"
    fe = fits.open(eng_file, memmap=True)
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

    det_evts = {i: [] for i in range(6)}
    with open(sci_csv) as f:
        for r in csv.DictReader(f):
            if r["type"] == "EVT":
                det_evts[int(r["det_id"])].append(float(r["met"]))
    for k in det_evts:
        det_evts[k] = np.sort(np.array(det_evts[k]))

    Sci = np.zeros((len(met_eng), 6))
    for i in range(len(met_eng)):
        t0 = met_eng[i]
        t1 = t0 + length_s[i]
        for det in range(6):
            Sci[i, det] = np.searchsorted(det_evts[det], t1) - np.searchsorted(det_evts[det], t0)

    valid = np.ones(len(met_eng), dtype=bool)
    for i in range(len(met_eng)):
        t0 = met_eng[i]
        t1 = t0 + length_s[i]
        if overlaps_saturation(t0, t1, sat_intervals[box_name]):
            valid[i] = False
    valid &= (L_cycles > 50000) & (Sci.sum(axis=1) > 100)

    N_below = (PHO - Wide - Large) - Sci

    data[box_name] = {
        "sci": Sci, "pho": PHO, "wide": Wide, "large": Large,
        "n_below": N_below, "length": length_s, "valid": valid, "met": met_eng,
    }
    print(f"  bins: {len(met_eng)} total, {valid.sum()} non-saturated")
    fe.close()


# --- Plot: 3 cols (boxes), 3 rows (per-det scatter, ratio vs rate, time series) ---
fig, axes = plt.subplots(3, 3, figsize=(16, 11))
det_colors = plt.cm.tab10(np.arange(6))

for col, (box_name, _, _, _) in enumerate(BOXES):
    ax_top = axes[0, col]
    ax_mid = axes[1, col]
    ax_bot = axes[2, col]
    D = data[box_name]
    valid = D["valid"]
    length = D["length"]

    sci_all = []
    nb_all = []
    for det in range(6):
        sci_rate = D["sci"][valid, det] / length[valid]
        nb_rate = D["n_below"][valid, det] / length[valid]
        ratio = D["n_below"][valid, det] / np.maximum(D["sci"][valid, det], 1)
        c = det_colors[det]
        ax_top.scatter(sci_rate, nb_rate, s=2, alpha=0.25, color=c,
                       label=f"det {det}", rasterized=True)
        ax_mid.scatter(sci_rate, ratio, s=2, alpha=0.25, color=c, rasterized=True)
        sci_all.append(sci_rate)
        nb_all.append(nb_rate)
    sci_all = np.concatenate(sci_all)
    nb_all = np.concatenate(nb_all)

    # 分箱中位数
    bins = np.logspace(np.log10(max(sci_all.min(), 50)), np.log10(sci_all.max() + 1), 25)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    med_nb, med_ratio = [], []
    for i in range(len(bins) - 1):
        m = (sci_all >= bins[i]) & (sci_all < bins[i + 1])
        if m.sum() > 5:
            med_nb.append(np.median(nb_all[m]))
            med_ratio.append(np.median(nb_all[m] / np.maximum(sci_all[m], 1)))
        else:
            med_nb.append(np.nan)
            med_ratio.append(np.nan)
    med_nb = np.array(med_nb)
    med_ratio = np.array(med_ratio)
    ax_top.plot(bin_centers, med_nb, "k-", lw=1.8, label="binned median", zorder=5)
    ax_mid.plot(bin_centers, med_ratio, "k-", lw=1.8, zorder=5)

    # 拟合 N_below = b + alpha*Sci  (linear, no log)
    X = np.column_stack([np.ones_like(sci_all), sci_all])
    coef, *_ = np.linalg.lstsq(X, nb_all, rcond=None)
    b, alpha = coef
    xs = np.logspace(np.log10(sci_all.min()), np.log10(sci_all.max()), 100)
    ax_top.plot(xs, b + alpha * xs, "r--", lw=1.2,
                label=f"N_b = {b:.0f} + {alpha:.3f}·Sci", zorder=4)
    # 在 ratio panel 上对应曲线
    ax_mid.plot(xs, (b + alpha * xs) / xs, "r--", lw=1.2, zorder=4,
                label=f"({b:.0f}/Sci)+{alpha:.3f}")

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_xlabel("Sci rate [cnt/s/det]")
    ax_top.set_ylabel("$N_\\mathrm{below}$ rate [cnt/s/det]")
    ax_top.set_title(f"Box {box_name}  (b={b:.0f} cnt/s, α={alpha:.3f})")
    ax_top.grid(alpha=0.3, which="both")
    if col == 0:
        ax_top.legend(loc="upper left", fontsize=7, ncol=2, markerscale=3)
    else:
        ax_top.legend(loc="upper left", fontsize=7)

    ax_mid.set_xscale("log")
    ax_mid.set_xlabel("Sci rate [cnt/s/det]")
    ax_mid.set_ylabel("$N_\\mathrm{below}$ / Sci")
    ax_mid.grid(alpha=0.3, which="both")
    ax_mid.set_ylim(0, 0.7)
    ax_mid.legend(loc="upper right", fontsize=7)

    # 时序: per-detector N_below 随 MET
    t0_global = D["met"][valid].min()
    for det in range(6):
        nb_rate = D["n_below"][valid, det] / length[valid]
        ax_bot.plot(D["met"][valid] - t0_global, nb_rate, lw=0.4,
                    color=det_colors[det], alpha=0.7)
    ax_bot.set_xlabel("MET - start [s]")
    ax_bot.set_ylabel("$N_\\mathrm{below}$ rate [cnt/s/det]")
    ax_bot.grid(alpha=0.3)
    ax_bot.set_yscale("log")
    ax_bot.set_ylim(20, 5000)

fig.suptitle("$N_\\mathrm{below} = (\\mathrm{PHO} - \\mathrm{Wide} - \\mathrm{Large}) - \\mathrm{Sci}$  on non-saturated bins (260226A, ±30 min around trigger)",
             fontsize=11)
fig.tight_layout()
out = "plots/n_below_vs_rate_260226.png"
os.makedirs("plots", exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved: {out}")

# --- 数值汇总: per-det 拟合 ---
print("\n=== Per-detector fit: N_below = b + alpha * Sci  (linear, all valid bins) ===")
print(f"{'Box':>3s} {'Det':>3s} {'b[cnt/s]':>9s} {'alpha':>7s} {'<Sci>':>7s} {'<N_b>':>7s} {'<ratio>':>8s}")
for box_name, _, _, _ in BOXES:
    D = data[box_name]
    valid = D["valid"]
    length = D["length"]
    for det in range(6):
        sci_rate = D["sci"][valid, det] / length[valid]
        nb_rate = D["n_below"][valid, det] / length[valid]
        X = np.column_stack([np.ones_like(sci_rate), sci_rate])
        coef, *_ = np.linalg.lstsq(X, nb_rate, rcond=None)
        b, alpha = coef
        ratio_med = np.median(nb_rate / np.maximum(sci_rate, 1))
        print(f"{box_name:>3s} {det:>3d} {b:>9.1f} {alpha:>7.3f} "
              f"{np.median(sci_rate):>7.0f} {np.median(nb_rate):>7.0f} {ratio_med:>8.4f}")
