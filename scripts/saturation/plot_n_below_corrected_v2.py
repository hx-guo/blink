#!/usr/bin/env python3
"""β=2 hypothesis: detailed residual diagnostics.

For each detector:
  - β=1:   nb1 = PHO - 1·Wide - Large - Sci
  - β=2:   nb2 = PHO - 2·Wide - Large - Sci
  - β=fit: nbf = PHO - β_fit·Wide - Large - Sci

Fit (b + α·Sci) per detector. Compare:
  - Absolute RMS (cnt/s)  — apples-to-apples
  - Residual vs Wide       — should flatten if β is right
  - Residual vs DeadFrac   — independent check
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


def fit_linear(y, x):
    """Fit y = b + α·x. Return (b, α, residual array)."""
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[0], c[1], y - X @ c


# === Per-detector analysis with multiple β values ===
print("\n=== Per-detector RMS in absolute cnt/s (and ρ residual-vs-Wide) ===")
print(f"{'Box':>3s} {'D':>2s}  {'β_fit':>5s}  "
      f"{'RMS_β1':>7s} {'ρ_β1':>6s}  "
      f"{'RMS_β2':>7s} {'ρ_β2':>6s}  "
      f"{'RMS_βfit':>9s} {'ρ_βfit':>7s}  "
      f"{'<N_b>1':>7s} {'<N_b>2':>7s}")

results = []
for dd in det_data:
    pho, wide, large, sci = dd["pho"], dd["wide"], dd["large"], dd["sci"]
    nb1 = pho - 1 * wide - large - sci
    nb2 = pho - 2 * wide - large - sci

    # Find β_fit via 3-term regression
    X = np.column_stack([np.ones_like(sci), sci, wide])
    c, *_ = np.linalg.lstsq(X, nb1, rcond=None)
    beta_fit = 1.0 + c[2]
    nbf = pho - beta_fit * wide - large - sci

    # Fit each
    b1, a1, r1 = fit_linear(nb1, sci)
    b2, a2, r2 = fit_linear(nb2, sci)
    bf, af, rf = fit_linear(nbf, sci)

    rms1 = np.sqrt(np.mean(r1 ** 2))
    rms2 = np.sqrt(np.mean(r2 ** 2))
    rmsf = np.sqrt(np.mean(rf ** 2))
    rho1 = np.corrcoef(r1, wide)[0, 1]
    rho2 = np.corrcoef(r2, wide)[0, 1]
    rhof = np.corrcoef(rf, wide)[0, 1]

    results.append({
        "dd": dd, "beta": beta_fit,
        "b1": b1, "a1": a1, "r1": r1, "rms1": rms1, "rho1": rho1, "nb1": nb1,
        "b2": b2, "a2": a2, "r2": r2, "rms2": rms2, "rho2": rho2, "nb2": nb2,
        "bf": bf, "af": af, "rf": rf, "rmsf": rmsf, "rhof": rhof, "nbf": nbf,
    })
    print(f"{dd['box']:>3s} {dd['det_local']:>2d}  {beta_fit:>5.2f}  "
          f"{rms1:>7.1f} {rho1:>+6.2f}  "
          f"{rms2:>7.1f} {rho2:>+6.2f}  "
          f"{rmsf:>9.1f} {rhof:>+7.2f}  "
          f"{np.mean(nb1):>7.0f} {np.mean(nb2):>7.0f}")

print("\n=== Box-level summary (mean) ===")
for box in "ABC":
    rs = [r for r in results if r["dd"]["box"] == box]
    print(f"Box {box}: <β>={np.mean([r['beta'] for r in rs]):.2f}  "
          f"RMS β=1: {np.mean([r['rms1'] for r in rs]):.1f}  "
          f"β=2: {np.mean([r['rms2'] for r in rs]):.1f}  "
          f"β=fit: {np.mean([r['rmsf'] for r in rs]):.1f}  cnt/s")

# === Plot: residual vs Wide for β=1 vs β=2, all 18 detectors ===
fig, axes = plt.subplots(3, 6, figsize=(20, 11), sharey=True)
for r in results:
    dd = r["dd"]
    box_idx = "ABC".index(dd["box"])
    ax = axes[box_idx, dd["det_local"]]

    ax.scatter(dd["wide"], r["r1"], s=2, alpha=0.18, color="C3",
               label=f"β=1 (ρ={r['rho1']:+.2f})", rasterized=True)
    ax.scatter(dd["wide"], r["r2"], s=2, alpha=0.18, color="C0",
               label=f"β=2 (ρ={r['rho2']:+.2f})", rasterized=True)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{dd['box']}{dd['det_local']}  β_fit={r['beta']:.2f}\n"
                 f"RMS {r['rms1']:.0f}→{r['rms2']:.0f} cnt/s", fontsize=8)
    ax.grid(alpha=0.3)
    if box_idx == 2:
        ax.set_xlabel("Wide rate [cnt/s]")
    if dd["det_local"] == 0:
        ax.set_ylabel(f"Box {dd['box']}\nResidual (cnt/s)")
    if dd["det_local"] == 0 and box_idx == 0:
        ax.legend(fontsize=7, loc="upper right", markerscale=4)

fig.suptitle("Residual of (b + α·Sci) fit vs Wide rate.\n"
             "Red: β=1 (original), should slope ↑ with Wide. Blue: β=2 (hypothesis), should be flat.",
             fontsize=11)
fig.tight_layout()
out1 = "plots/n_below_residual_vs_wide_260226.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out1}")

# === Summary plot: ρ_residual_vs_Wide for β=1 vs β=2 vs β=fit ===
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
xs = np.arange(len(results))
labels = [f"{r['dd']['box']}{r['dd']['det_local']}" for r in results]
rho1s = [r["rho1"] for r in results]
rho2s = [r["rho2"] for r in results]
rhofs = [r["rhof"] for r in results]
rms1s = [r["rms1"] for r in results]
rms2s = [r["rms2"] for r in results]
rmsfs = [r["rmsf"] for r in results]

axL.bar(xs - 0.27, rho1s, width=0.27, color="C3", label="β=1")
axL.bar(xs, rho2s, width=0.27, color="C2", label="β=2")
axL.bar(xs + 0.27, rhofs, width=0.27, color="C0", label="β=fit")
axL.axhline(0, color="k", lw=0.7)
axL.set_xticks(xs)
axL.set_xticklabels(labels, rotation=90, fontsize=8)
axL.set_ylabel("Pearson ρ(residual, Wide)")
axL.set_title("Residual−Wide correlation: β=2 should give ≈ 0")
axL.legend()
axL.grid(alpha=0.3, axis="y")

axR.plot(xs, rms1s, "ro-", label=f"β=1 (mean {np.mean(rms1s):.0f})")
axR.plot(xs, rms2s, "g^-", label=f"β=2 (mean {np.mean(rms2s):.0f})")
axR.plot(xs, rmsfs, "C0s-", label=f"β=fit (mean {np.mean(rmsfs):.0f})")
axR.set_xticks(xs)
axR.set_xticklabels(labels, rotation=90, fontsize=8)
axR.set_ylabel("Absolute RMS [cnt/s]")
axR.set_title("Absolute RMS of (b+α·Sci) fit")
axR.legend()
axR.grid(alpha=0.3)

fig2.tight_layout()
out2 = "plots/n_below_beta_summary_v2_260226.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")
