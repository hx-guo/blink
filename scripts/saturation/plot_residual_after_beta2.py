#!/usr/bin/env python3
"""Diagnose remaining ~40 cnt/s residual after β=2 correction.

Per detector:
  N_below_β2 = PHO - 2·Wide - Large - Sci
  Fit (b + α·Sci); residual = N_below_β2 - b - α·Sci

Plot residual vs candidate factors (time, DeadFrac, hardness, etc.) to see
which factor still has structure.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import csv
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


det_data = []
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
    Dt = np.column_stack([d[f"DeadTime_PHODet_{i}"].astype(float) for i in det_ids]) * 16e-6

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

    valid_box = np.ones(len(met_eng), dtype=bool)
    for i in range(len(met_eng)):
        t0 = met_eng[i]
        t1 = t0 + length_s[i]
        if overlaps_saturation(t0, t1, sat_intervals[box_name]):
            valid_box[i] = False
    valid_box &= (L_cycles > 50000) & (Sci.sum(axis=1) > 100)

    for det_local in range(6):
        v = valid_box
        det_data.append({
            "box": box_name,
            "det_local": det_local,
            "det_global": det_ids[det_local],
            "sci": Sci[v, det_local] / length_s[v],
            "wide": Wide[v, det_local] / length_s[v],
            "large": Large[v, det_local] / length_s[v],
            "pho": PHO[v, det_local] / length_s[v],
            "deadfrac": Dt[v, det_local] / length_s[v],
            "t_rel": met_eng[v] - met_eng[v].min(),
        })
    fe.close()


# Fit (b + α·Sci) on N_below_β2 per detector, store residuals
all_resid = []
all_factors = {"sci": [], "wide": [], "large": [], "deadfrac": [], "t_rel": [],
               "hardness": [], "pho": [], "nb2_rate": []}
det_colors = plt.cm.tab10(np.arange(6))
det_box_idx = []  # for color coding by box
det_local_idx = []
for dd in det_data:
    nb2 = dd["pho"] - 2 * dd["wide"] - dd["large"] - dd["sci"]
    X = np.column_stack([np.ones_like(dd["sci"]), dd["sci"]])
    c, *_ = np.linalg.lstsq(X, nb2, rcond=None)
    resid = nb2 - X @ c
    all_resid.append(resid)
    all_factors["sci"].append(dd["sci"])
    all_factors["wide"].append(dd["wide"])
    all_factors["large"].append(dd["large"])
    all_factors["deadfrac"].append(dd["deadfrac"])
    all_factors["t_rel"].append(dd["t_rel"])
    all_factors["hardness"].append(dd["large"] / np.maximum(dd["sci"], 1))
    all_factors["pho"].append(dd["pho"])
    all_factors["nb2_rate"].append(nb2)
    det_box_idx.append("ABC".index(dd["box"]))
    det_local_idx.append(dd["det_local"])

# Stack
resid_all = np.concatenate(all_resid)
sci_all = np.concatenate(all_factors["sci"])
wide_all = np.concatenate(all_factors["wide"])
large_all = np.concatenate(all_factors["large"])
df_all = np.concatenate(all_factors["deadfrac"])
t_all = np.concatenate(all_factors["t_rel"])
h_all = np.concatenate(all_factors["hardness"])
pho_all = np.concatenate(all_factors["pho"])
nb2_all = np.concatenate(all_factors["nb2_rate"])

# === Plot 1: 2x3 diagnostic — residual vs each factor (all 18 detectors overlaid) ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

def panel(ax, x, y, xlabel, n_bins=25, log_x=False):
    if log_x:
        x_pos = np.maximum(x, 1e-6)
        ax.scatter(x_pos, y, s=1.5, alpha=0.12, color="C0", rasterized=True)
        ax.set_xscale("log")
        bins = np.logspace(np.log10(np.percentile(x_pos, 1)), np.log10(np.percentile(x_pos, 99)), n_bins)
    else:
        ax.scatter(x, y, s=1.5, alpha=0.12, color="C0", rasterized=True)
        bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    p16 = []
    p84 = []
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() > 5:
            med.append(np.median(y[m]))
            p16.append(np.percentile(y[m], 16))
            p84.append(np.percentile(y[m], 84))
        else:
            med.append(np.nan); p16.append(np.nan); p84.append(np.nan)
    ax.plot(bc, med, "k-", lw=1.7, label="median")
    ax.fill_between(bc, p16, p84, color="k", alpha=0.18, label="16/84%")
    rho = np.corrcoef(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)])[0, 1]
    ax.axhline(0, color="r", ls="--", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (cnt/s)")
    ax.set_title(f"{xlabel}    ρ={rho:+.3f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    return rho

panel(axes[0, 0], sci_all, resid_all, "Sci rate [cnt/s]")
panel(axes[0, 1], wide_all, resid_all, "Wide rate [cnt/s]")
panel(axes[0, 2], df_all, resid_all, "Dead fraction = Dt/Length")
panel(axes[1, 0], h_all, resid_all, "Hardness = Large/Sci")
panel(axes[1, 1], t_all, resid_all, "MET - start [s]")
panel(axes[1, 2], pho_all, resid_all, "PHO rate [cnt/s]")

fig.suptitle("Residual = N_below_β=2 − (b + α·Sci)  vs candidate factors\n"
             "All 18 detectors overlaid;  if β=2 + linear-in-Sci is complete, all panels should be flat",
             fontsize=11)
fig.tight_layout()
out1 = "plots/residual_after_beta2_diagnostics_260226.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"Saved: {out1}")

# === Plot 2: residual vs time per detector (3x6 grid), to see if time structure is the same across detectors ===
fig2, axes2 = plt.subplots(3, 6, figsize=(20, 9), sharey=True, sharex=True)
for k, dd in enumerate(det_data):
    box_idx = "ABC".index(dd["box"])
    ax = axes2[box_idx, dd["det_local"]]
    t = dd["t_rel"]
    r = all_resid[k]
    ax.plot(t, r, lw=0.4, color="C0")
    # Smooth (binned median)
    bins = np.linspace(0, t.max(), 60)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (t >= bins[i]) & (t < bins[i + 1])
        med.append(np.median(r[m]) if m.sum() > 0 else np.nan)
    ax.plot(bc, med, "r-", lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{dd['box']}{dd['det_local']}", fontsize=9)
    ax.grid(alpha=0.3)
    if box_idx == 2:
        ax.set_xlabel("t [s]")
    if dd["det_local"] == 0:
        ax.set_ylabel(f"Box {dd['box']}\nResid (cnt/s)")
fig2.suptitle("Residual vs time per detector (β=2, after b+α·Sci subtracted)\n"
              "Common slow drift = orbital background;  detector-specific = local issues",
              fontsize=11)
fig2.tight_layout()
out2 = "plots/residual_after_beta2_vs_time_per_det_260226.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")
