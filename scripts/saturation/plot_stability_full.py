#!/usr/bin/env python3
"""Full (β, γ) stability test: 5 dates × 18 detectors = 90 measurements.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
BETA_FIXED = 1.9
GAMMA_FIXED = 1.19

# (date, eng_dirs by box, sci_csvs by box)
DATES = [
    ("2020-04-15", {
        "A": ("data/1B/2020/20200415/0766/*.fits", "/tmp/200415_boxA.csv", 0),
        "B": ("data/1B/2020/20200415/1009/*.fits", "/tmp/200415_boxB.csv", 6),
        "C": ("data/1B/2020/20200415/1781/*.fits", "/tmp/200415_boxC.csv", 12),
    }),
    ("2020-04-28", {
        "A": ("data/1B/2020/20200428/0766/*.fits", "/tmp/200428_boxA.csv", 0),
        "B": ("data/1B/2020/20200428/1009/*.fits", "/tmp/200428_boxB.csv", 6),
        "C": ("data/1B/2020/20200428/1781/*.fits", "/tmp/200428_boxC.csv", 12),
    }),
    ("2022-10-09", {
        "A": ("data/1B/2022/20221009/0766/*.fits", "/tmp/221009_boxA.csv", 0),
        "B": ("data/1B/2022/20221009/1009/*.fits", "/tmp/221009_boxB.csv", 6),
        "C": ("data/1B/2022/20221009/1781/*.fits", "/tmp/221009_boxC.csv", 12),
    }),
    ("2026-02-26", {
        "A": ("data/1B/2026/20260226/0766/*.fits", "/tmp/260226_boxA_full.csv", 0),
        "B": ("data/1B/2026/20260226/1009/*.fits", "/tmp/260226_boxB_full.csv", 6),
        "C": ("data/1B/2026/20260226/1781/*.fits", "/tmp/260226_boxC_full.csv", 12),
    }),
    ("2026-04-10", {
        "A": ("data/1B/2026/20260410/0766/*.fits", "/tmp/260410_boxA.csv", 0),
        "B": ("data/1B/2026/20260410/1009/*.fits", "/tmp/260410_boxB.csv", 6),
        "C": ("data/1B/2026/20260410/1781/*.fits", "/tmp/260410_boxC.csv", 12),
    }),
]


def fit_linear(y, x):
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[0], c[1], y - X @ c


def analyze_one(label, box, fits_glob, sci_csv, det_off):
    fits_files = sorted(glob.glob(fits_glob))
    if not fits_files:
        return []
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

    # Fast pandas read; only EVT rows; only need met + det_id
    df = pd.read_csv(sci_csv, usecols=["type", "met", "det_id"],
                     dtype={"type": "category", "met": "float64", "det_id": "int8"})
    df = df[df["type"] == "EVT"]
    det_evts = {}
    for det in range(6):
        m = df["det_id"].values == det
        det_evts[det] = np.sort(df["met"].values[m])
    del df

    Sci = np.zeros((len(met_eng), 6))
    for i in range(len(met_eng)):
        t0 = met_eng[i]; t1 = t0 + length_s[i]
        for det in range(6):
            Sci[i, det] = np.searchsorted(det_evts[det], t1) - np.searchsorted(det_evts[det], t0)

    valid = (L_cycles > 50000) & (Sci.sum(axis=1) > 100)
    sci_total = Sci.sum(axis=1)
    if valid.sum() < 50:
        fe.close()
        return []
    p5, p95 = np.percentile(sci_total[valid], [5, 95])
    valid &= (sci_total >= p5) & (sci_total <= p95)
    fe.close()

    out = []
    for det in range(6):
        v = valid
        sci = Sci[v, det] / length_s[v]
        wide = Wide[v, det] / length_s[v]
        large = Large[v, det] / length_s[v]
        pho = PHO[v, det] / length_s[v]

        # Per-date best (β, γ)
        nb_base = pho - wide - large - sci
        X = np.column_stack([np.ones_like(sci), sci, wide, large])
        c, *_ = np.linalg.lstsq(X, nb_base, rcond=None)
        beta_fit = 1.0 + c[2]; gamma_fit = 1.0 + c[3]

        nb_fit = pho - beta_fit * wide - gamma_fit * large - sci
        _, _, r_fit = fit_linear(nb_fit, sci)
        rms_fit = np.sqrt(np.mean(r_fit ** 2))

        # Universal constants
        nb_fix = pho - BETA_FIXED * wide - GAMMA_FIXED * large - sci
        _, _, r_fix = fit_linear(nb_fix, sci)
        rms_fix = np.sqrt(np.mean(r_fix ** 2))

        out.append({
            "date": label, "box": box, "det_local": det,
            "det_global": det_ids[det],
            "beta": beta_fit, "gamma": gamma_fit,
            "rms_fit": rms_fit, "rms_fix": rms_fix,
            "wide_med": np.median(wide), "large_med": np.median(large),
            "sci_med": np.median(sci),
        })
    return out


print("Analyzing 5 dates × 3 boxes × 6 detectors = 90 measurements...\n")
all_results = []
for date_label, boxes in DATES:
    print(f"--- {date_label} ---")
    for box_name in "ABC":
        fits_glob, sci_csv, det_off = boxes[box_name]
        try:
            res = analyze_one(date_label, box_name, fits_glob, sci_csv, det_off)
            all_results.extend(res)
            if res:
                gs = [r["gamma"] for r in res]
                bs = [r["beta"] for r in res]
                print(f"  Box {box_name}: γ = {np.mean(gs):.3f} ± {np.std(gs):.3f},  "
                      f"β = {np.mean(bs):.2f} ± {np.std(bs):.2f},  "
                      f"<Wide> = {np.mean([r['wide_med'] for r in res]):.0f}")
        except Exception as e:
            print(f"  Box {box_name}: ERROR {e}")

# === Summary tables ===
date_labels = [d[0] for d in DATES]
print(f"\n=== γ summary per date × box ===")
print(f"{'Date':>11s} | {'Box A':>15s} | {'Box B':>15s} | {'Box C':>15s} | overall")
for label in date_labels:
    line = f"  {label} |"
    for box in "ABC":
        rs = [r for r in all_results if r["date"] == label and r["box"] == box]
        if rs:
            gs = [r["gamma"] for r in rs]
            line += f" {np.mean(gs):.3f} ± {np.std(gs):.3f}  |"
        else:
            line += "       --        |"
    rs_all = [r for r in all_results if r["date"] == label]
    gs_all = [r["gamma"] for r in rs_all]
    if gs_all:
        line += f" {np.mean(gs_all):.3f} ± {np.std(gs_all):.3f}"
    print(line)

# === Plot 1: γ across all 90 measurements ===
fig, axes = plt.subplots(2, 1, figsize=(15, 8))
box_markers = {"A": "o", "B": "s", "C": "^"}
date_x = {d: i for i, d in enumerate(date_labels)}

# Top: γ
for box in "ABC":
    for det in range(6):
        rs = [r for r in all_results if r["box"] == box and r["det_local"] == det]
        rs.sort(key=lambda x: date_x[x["date"]])
        xs = [date_x[r["date"]] + (ord(box) - ord("A") - 1) * 0.06 for r in rs]
        ys = [r["gamma"] for r in rs]
        c = plt.cm.tab10(det)
        axes[0].plot(xs, ys, "-", color=c, alpha=0.4, lw=0.6)
        axes[0].scatter(xs, ys, marker=box_markers[box], color=c, s=40,
                        edgecolor="k", linewidth=0.4,
                        label=f"{box}{det}" if box == "A" else None)

axes[0].axhline(GAMMA_FIXED, color="r", ls="--", label=f"γ={GAMMA_FIXED}")
axes[0].set_xticks(range(len(date_labels)))
axes[0].set_xticklabels(date_labels)
axes[0].set_ylabel("γ (Large multiplier)")
axes[0].set_title("γ across 5 dates × 18 detectors  (○=Box A, □=Box B, △=Box C)")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="upper right", fontsize=7, ncol=3, title="(Box A only — Box B/C share colors)")

# Bottom: β
for box in "ABC":
    for det in range(6):
        rs = [r for r in all_results if r["box"] == box and r["det_local"] == det]
        rs.sort(key=lambda x: date_x[x["date"]])
        xs = [date_x[r["date"]] + (ord(box) - ord("A") - 1) * 0.06 for r in rs]
        ys = [r["beta"] for r in rs]
        c = plt.cm.tab10(det)
        axes[1].plot(xs, ys, "-", color=c, alpha=0.4, lw=0.6)
        axes[1].scatter(xs, ys, marker=box_markers[box], color=c, s=40,
                        edgecolor="k", linewidth=0.4)

axes[1].axhline(BETA_FIXED, color="r", ls="--", label=f"β={BETA_FIXED}")
axes[1].set_xticks(range(len(date_labels)))
axes[1].set_xticklabels(date_labels)
axes[1].set_ylabel("β (Wide multiplier)")
axes[1].set_title("β across 5 dates × 18 detectors")
axes[1].grid(alpha=0.3)
axes[1].legend()

fig.tight_layout()
out = "plots/stability_full_18det.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# === Histogram of γ across all 90 measurements ===
fig2, ax = plt.subplots(figsize=(9, 5))
all_g = [r["gamma"] for r in all_results]
all_b = [r["beta"] for r in all_results]
ax.hist(all_g, bins=30, color="C0", alpha=0.7, edgecolor="k")
ax.axvline(GAMMA_FIXED, color="r", ls="--", lw=2, label=f"Fixed γ={GAMMA_FIXED}")
ax.axvline(np.mean(all_g), color="C2", ls="-", lw=2, label=f"Mean = {np.mean(all_g):.3f}")
ax.set_xlabel("γ (Large multiplier)")
ax.set_ylabel("count")
ax.set_title(f"γ histogram: {len(all_g)} measurements\n"
             f"Mean = {np.mean(all_g):.3f},  std = {np.std(all_g):.3f},  "
             f"range = {min(all_g):.3f}–{max(all_g):.3f}")
ax.legend()
ax.grid(alpha=0.3)
fig2.tight_layout()
out2 = "plots/gamma_histogram_full.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")

# Final summary
print(f"\n=== GRAND TOTAL ({len(all_g)} measurements) ===")
print(f"  γ = {np.mean(all_g):.3f} ± {np.std(all_g):.3f}  range {min(all_g):.3f}–{max(all_g):.3f}")
print(f"  β = {np.mean(all_b):.2f} ± {np.std(all_b):.2f}  range {min(all_b):.2f}–{max(all_b):.2f}")
print(f"\n=== RMS comparison ===")
all_fix = [r["rms_fix"] for r in all_results]
all_fitr = [r["rms_fit"] for r in all_results]
print(f"  Fixed (β=1.9, γ=1.19):  mean RMS = {np.mean(all_fix):.1f} cnt/s")
print(f"  Per-date fitted:        mean RMS = {np.mean(all_fitr):.1f} cnt/s")
print(f"  Penalty for using fixed: +{np.mean(all_fix)-np.mean(all_fitr):.1f} cnt/s "
      f"({(np.mean(all_fix)/np.mean(all_fitr)-1)*100:+.1f}%)")
