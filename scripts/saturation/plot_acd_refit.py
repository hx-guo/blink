#!/usr/bin/env python3
"""Re-fit universal predictor on Sci_photon (Sci minus ≥2-bit ACD events).
   Compare residual structure to original (Sci_all) fit.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0


def popcount(x):
    return bin(int(x)).count("1")


BOXES = [
    ("A", "0766", "/tmp/260226_boxA_acd.csv", 0),
    ("B", "1009", "/tmp/260226_boxB_acd.csv", 6),
    ("C", "1781", "/tmp/260226_boxC_acd.csv", 12),
]


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
    fe.close()

    df = pd.read_csv(csv_path, usecols=["type", "met", "det_id", "aminfo"],
                     dtype={"type": "category", "met": "float64", "det_id": "int8", "aminfo": "uint32"})
    df = df[df["type"] == "EVT"].copy()
    df["bitcount"] = df["aminfo"].apply(popcount).astype("int8")

    Sci_all = np.zeros((len(met_eng), 6))
    Sci_photon = np.zeros((len(met_eng), 6))     # bitcount < 2
    Sci_particle = np.zeros((len(met_eng), 6))   # bitcount >= 2
    for det in range(6):
        e = df[df["det_id"] == det]
        m_all = np.sort(e["met"].values)
        m_phot = np.sort(e.loc[e["bitcount"] < 2, "met"].values)
        m_part = np.sort(e.loc[e["bitcount"] >= 2, "met"].values)
        for i in range(len(met_eng)):
            t0 = met_eng[i]; t1 = t0 + length_s[i]
            Sci_all[i, det] = np.searchsorted(m_all, t1) - np.searchsorted(m_all, t0)
            Sci_photon[i, det] = np.searchsorted(m_phot, t1) - np.searchsorted(m_phot, t0)
            Sci_particle[i, det] = np.searchsorted(m_part, t1) - np.searchsorted(m_part, t0)
    return {"box": box_name, "met": met_eng, "length": length_s,
            "PHO": PHO, "Wide": Wide, "Large": Large,
            "Sci_all": Sci_all, "Sci_photon": Sci_photon, "Sci_particle": Sci_particle}


print("Loading 3 boxes...")
data = [load_box(*b) for b in BOXES]


# === Per-box refit ===
print(f"\n=== Refit results ===")
print(f"{'Box':>3s}  {'a0':>8s}  {'a1':>6s}  {'a2':>7s}  {'a3':>7s}  {'RMS_phot':>9s}  {'RMS_all':>8s}")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fits_by_box = {}
for col, D in enumerate(data):
    box = D["box"]
    length = D["length"]
    sci_all = (D["Sci_all"] / length[:, None]).flatten()
    sci_phot = (D["Sci_photon"] / length[:, None]).flatten()
    sci_part = (D["Sci_particle"] / length[:, None]).flatten()
    pho = (D["PHO"] / length[:, None]).flatten()
    wide = (D["Wide"] / length[:, None]).flatten()
    large = (D["Large"] / length[:, None]).flatten()
    # Filter out outliers / saturation
    mask = (sci_all > 100) & (sci_all < np.percentile(sci_all, 95)) & (pho > 100)

    # Fit Sci_photon = a0 + a1·PHO + a2·W + a3·L
    A = np.column_stack([np.ones(mask.sum()), pho[mask], wide[mask], large[mask]])
    c, *_ = np.linalg.lstsq(A, sci_phot[mask], rcond=None)
    pred_phot = c[0] + c[1] * pho + c[2] * wide + c[3] * large
    rms_phot = np.sqrt(np.mean((sci_phot[mask] - pred_phot[mask]) ** 2))
    fits_by_box[box] = c

    # Total Sci prediction = pred_phot + observed Sci_particle
    pred_total = pred_phot + sci_part
    rms_all_combined = np.sqrt(np.mean((sci_all[mask] - pred_total[mask]) ** 2))

    print(f"  {box}  {c[0]:>+8.1f}  {c[1]:>6.4f}  {c[2]:>+7.4f}  {c[3]:>+7.4f}  {rms_phot:>9.1f}  {rms_all_combined:>8.1f}")

    # Top: Sci_photon vs pred_photon
    ax = axes[0, col]
    ax.scatter(sci_phot[mask], pred_phot[mask], s=2, alpha=0.2, color="C2", rasterized=True)
    lo, hi = max(sci_phot[mask].min() * 0.9, 100), sci_phot[mask].max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci_photon obs [cnt/s/det]")
    ax.set_ylabel("Sci_photon pred")
    ax.set_title(f"Box {box}: photon-only model  RMS={rms_phot:.1f}")
    ax.grid(alpha=0.3)

    # Bottom: Sci_all vs (pred_photon + Sci_particle obs)
    ax = axes[1, col]
    ax.scatter(sci_all[mask], pred_total[mask], s=2, alpha=0.2, color="C0", rasterized=True)
    lo, hi = max(sci_all[mask].min() * 0.9, 100), sci_all[mask].max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci_all obs [cnt/s/det]")
    ax.set_ylabel("Sci_pred = Sci_photon_pred + Sci_particle_obs")
    ax.set_title(f"Box {box}: full Sci  RMS={rms_all_combined:.1f}")
    ax.grid(alpha=0.3)

fig.suptitle("Refit on Sci_photon (≥2-bit ACD removed): does the bending disappear?\n"
             "Top: photon-only model. Bottom: photon-pred + ACD-particle-obs = full Sci.",
             fontsize=11)
fig.tight_layout()
out = "plots/acd_refit_260226.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# Compare to original (with ACD-blind Sci_all)
print(f"\n=== Original universal fit on Sci_all (for comparison) ===")
COEFS_ORIG = {
    "A": {"a0": +107.0, "a1": 0.676, "a2": -1.318, "a3": -0.633},
    "B": {"a0": +116.0, "a1": 0.646, "a2": -1.247, "a3": -0.579},
    "C": {"a0": +128.0, "a1": 0.599, "a2": -1.236, "a3": -0.509},
}
for D in data:
    box = D["box"]
    length = D["length"]
    sci_all = (D["Sci_all"] / length[:, None]).flatten()
    pho = (D["PHO"] / length[:, None]).flatten()
    wide = (D["Wide"] / length[:, None]).flatten()
    large = (D["Large"] / length[:, None]).flatten()
    co = COEFS_ORIG[box]
    pred_orig = co["a0"] + co["a1"] * pho + co["a2"] * wide + co["a3"] * large
    mask = (sci_all > 100) & (sci_all < np.percentile(sci_all, 95)) & (pho > 100)
    rms_orig = np.sqrt(np.mean((sci_all[mask] - pred_orig[mask]) ** 2))
    print(f"  Box {box}: original universal RMS = {rms_orig:.1f}")
