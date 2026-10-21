#!/usr/bin/env python3
"""Demo: apply universal formula on 2026-02-26 (1 hour, 3 boxes, 18 detectors).

Sci_pred = (PHO - 2·Wide - 1.2·Large - 25) / 1.20
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import csv
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
# Universal coefficients per box from 274-date × 6-detectors × 1-hour median fit
# Sci = a0 + a1·PHO + a2·Wide + a3·Large
COEFS_BY_BOX = {
    "A": {"a0": +107.0, "a1": 0.676, "a2": -1.318, "a3": -0.633},
    "B": {"a0": +116.0, "a1": 0.646, "a2": -1.247, "a3": -0.579},
    "C": {"a0": +128.0, "a1": 0.599, "a2": -1.236, "a3": -0.509},
}

BOXES = [
    ("A", "0766", "/tmp/260226_boxA.csv", 0),
    ("B", "1009", "/tmp/260226_boxB.csv", 6),
    ("C", "1781", "/tmp/260226_boxC.csv", 12),
]


def predict_sci(box, pho, wide, large):
    c = COEFS_BY_BOX[box]
    return c["a0"] + c["a1"] * pho + c["a2"] * wide + c["a3"] * large


def load_box(box_name, box_code, csv_path, det_off):
    fe = fits.open(f"data/1B/2026/20260226/{box_code}/HXMT_1B_{box_code}_20260226T100000_G076262_000_004.fits", memmap=True)
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

    # Read Sci events per detector
    det_evts = {i: [] for i in range(6)}
    with open(csv_path) as f:
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

    fe.close()
    return {"box": box_name, "met": met_eng, "length": length_s,
            "PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci, "det_ids": det_ids}


print("Loading 3 boxes...")
data = [load_box(*b) for b in BOXES]

# === Apply universal formula ===
# Convert all to per-second rates, then predict
fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex='col')
det_colors = plt.cm.tab10(np.arange(6))

for col, D in enumerate(data):
    box = D["box"]
    length_s = D["length"]
    # Per-detector rates
    for row in range(3):
        ax = axes[row, col]
        if row == 0:
            # Light curves: observed vs predicted (sum of 6 detectors)
            t = D["met"] - D["met"][0]
            sci_total = D["Sci"].sum(axis=1) / length_s
            pho_total = D["PHO"].sum(axis=1) / length_s
            wide_total = D["Wide"].sum(axis=1) / length_s
            large_total = D["Large"].sum(axis=1) / length_s
            # Universal predictor sums per detector × 6 dets
            sci_pred_total = np.zeros_like(sci_total)
            for det in range(6):
                pho_d = D["PHO"][:, det] / length_s
                wide_d = D["Wide"][:, det] / length_s
                large_d = D["Large"][:, det] / length_s
                sci_pred_total += predict_sci(box, pho_d, wide_d, large_d)
            ax.plot(t, sci_total, "C0-", lw=0.7, label="Sci (observed)", alpha=0.9)
            ax.plot(t, sci_pred_total, "C3-", lw=0.7, label="Sci (predicted)", alpha=0.7)
            ax.set_ylabel(f"Box {box}\nSci [cnt/s/box]")
            ax.set_title(f"Box {box}: light curve (sum 6 dets)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_yscale("log")
        elif row == 1:
            # Predicted vs observed scatter, per detector
            for det in range(6):
                sci_d = D["Sci"][:, det] / length_s
                pho_d = D["PHO"][:, det] / length_s
                wide_d = D["Wide"][:, det] / length_s
                large_d = D["Large"][:, det] / length_s
                sci_pred_d = predict_sci(box, pho_d, wide_d, large_d)
                ax.scatter(sci_d, sci_pred_d, s=2, alpha=0.3, color=det_colors[det],
                           label=f"det {det}" if box == "A" else None, rasterized=True)
            lo, hi = 0, 2500
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6, label="y=x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Sci observed [cnt/s/det]")
            ax.set_ylabel("Sci predicted [cnt/s/det]")
            ax.set_title(f"Box {box}: per-second per-det")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.legend(fontsize=7, loc="lower right")
        else:
            # Residual vs Sci (per-det)
            for det in range(6):
                sci_d = D["Sci"][:, det] / length_s
                pho_d = D["PHO"][:, det] / length_s
                wide_d = D["Wide"][:, det] / length_s
                large_d = D["Large"][:, det] / length_s
                sci_pred_d = predict_sci(box, pho_d, wide_d, large_d)
                resid = sci_d - sci_pred_d
                ax.scatter(sci_d, resid, s=2, alpha=0.3, color=det_colors[det], rasterized=True)
            ax.axhline(0, color="r", ls="--", lw=1)
            ax.set_xlabel("Sci observed [cnt/s/det]")
            ax.set_ylabel("Sci_obs − Sci_pred [cnt/s/det]")
            ax.set_title(f"Box {box}: residual")
            ax.grid(alpha=0.3)
            ax.set_ylim(-300, 300)
            ax.set_xlim(0, 2500)

# Overall RMS computation
all_sci = []; all_pred = []
for D in data:
    length_s = D["length"]
    for det in range(6):
        sci_d = D["Sci"][:, det] / length_s
        pho_d = D["PHO"][:, det] / length_s
        wide_d = D["Wide"][:, det] / length_s
        large_d = D["Large"][:, det] / length_s
        sci_pred_d = predict_sci(D["box"], pho_d, wide_d, large_d)
        all_sci.append(sci_d)
        all_pred.append(sci_pred_d)
all_sci = np.concatenate(all_sci); all_pred = np.concatenate(all_pred)
# Filter to non-saturated reasonable bins
mask = (all_sci > 100) & (all_sci < np.percentile(all_sci, 95))
rms = np.sqrt(np.mean((all_sci[mask] - all_pred[mask]) ** 2))
print(f"\nOverall RMS (non-saturated 5-95% bins, 1s × 18 dets): {rms:.1f} cnt/s/det")
print(f"Mean Sci: {np.mean(all_sci[mask]):.1f}, mean Sci_pred: {np.mean(all_pred[mask]):.1f}")

fig.suptitle(f"Universal formula demo on 2026-02-26 T10 (1 hour, 3 boxes, 18 detectors)\n"
             f"Sci = a₀ + a₁·PHO + a₂·Wide + a₃·Large  (per-box constants)\n"
             f"Non-saturated bin RMS = {rms:.1f} cnt/s/det",
             fontsize=11)
fig.tight_layout()
out = "plots/universal_demo_260226.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")
