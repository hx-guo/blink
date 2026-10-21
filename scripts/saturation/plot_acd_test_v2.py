#!/usr/bin/env python3
"""ACD test v2: filter only ≥2-bit (charged particles).
   If hypothesis correct: residual at high rate should be explained by
   the ≥2-bit (charged particle) Sci component.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
COEFS = {
    "A": {"a0": +107.0, "a1": 0.676, "a2": -1.318, "a3": -0.633},
    "B": {"a0": +116.0, "a1": 0.646, "a2": -1.247, "a3": -0.579},
    "C": {"a0": +128.0, "a1": 0.599, "a2": -1.236, "a3": -0.509},
}

BOXES = [
    ("A", "0766", "/tmp/260226_boxA_acd.csv", 0),
    ("B", "1009", "/tmp/260226_boxB_acd.csv", 6),
    ("C", "1781", "/tmp/260226_boxC_acd.csv", 12),
]


def popcount(x):
    return bin(int(x)).count("1")


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
    Sci_clean = np.zeros((len(met_eng), 6))   # bitcount == 0
    Sci_1bit = np.zeros((len(met_eng), 6))    # bitcount == 1
    Sci_part = np.zeros((len(met_eng), 6))    # bitcount >= 2 (charged particle suspect)
    for det in range(6):
        e = df[df["det_id"] == det]
        m_all = np.sort(e["met"].values)
        m_clean = np.sort(e.loc[e["bitcount"] == 0, "met"].values)
        m_1 = np.sort(e.loc[e["bitcount"] == 1, "met"].values)
        m_part = np.sort(e.loc[e["bitcount"] >= 2, "met"].values)
        for i in range(len(met_eng)):
            t0 = met_eng[i]; t1 = t0 + length_s[i]
            Sci_all[i, det] = np.searchsorted(m_all, t1) - np.searchsorted(m_all, t0)
            Sci_clean[i, det] = np.searchsorted(m_clean, t1) - np.searchsorted(m_clean, t0)
            Sci_1bit[i, det] = np.searchsorted(m_1, t1) - np.searchsorted(m_1, t0)
            Sci_part[i, det] = np.searchsorted(m_part, t1) - np.searchsorted(m_part, t0)

    return {"box": box_name, "met": met_eng, "length": length_s,
            "PHO": PHO, "Wide": Wide, "Large": Large,
            "Sci_all": Sci_all, "Sci_clean": Sci_clean,
            "Sci_1bit": Sci_1bit, "Sci_part": Sci_part}


def predict_sci(box, pho, wide, large):
    c = COEFS[box]
    return c["a0"] + c["a1"] * pho + c["a2"] * wide + c["a3"] * large


print("Loading 3 boxes...")
data = [load_box(*b) for b in BOXES]


# === Plot ===
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col, D in enumerate(data):
    box = D["box"]
    length = D["length"]
    sci_all = (D["Sci_all"] / length[:, None]).flatten()
    sci_clean = (D["Sci_clean"] / length[:, None]).flatten()
    sci_part = (D["Sci_part"] / length[:, None]).flatten()
    pho_rate = (D["PHO"] / length[:, None]).flatten()
    wide_rate = (D["Wide"] / length[:, None]).flatten()
    large_rate = (D["Large"] / length[:, None]).flatten()
    sci_pred = predict_sci(box, pho_rate, wide_rate, large_rate)
    mask = (sci_all > 100) & (sci_all < np.percentile(sci_all, 99))

    # Top: residual (Sci_all - Sci_pred) vs Sci_part rate
    ax = axes[0, col]
    resid = sci_all - sci_pred
    ax.scatter(sci_part[mask], resid[mask], s=2, alpha=0.2, color="C3", rasterized=True)
    rho = np.corrcoef(sci_part[mask], resid[mask])[0, 1]
    ax.axhline(0, color="k", ls="--", lw=1)
    # binned median
    bins = np.linspace(0, sci_part[mask].quantile(0.99) if hasattr(sci_part[mask], "quantile") else np.percentile(sci_part[mask], 99), 25)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (sci_part[mask] >= bins[i]) & (sci_part[mask] < bins[i + 1])
        med.append(np.median(resid[mask][m]) if m.sum() > 5 else np.nan)
    ax.plot(bc, med, "k-", lw=2)
    ax.set_xlabel("Sci_particle (≥2 bits) [cnt/s/det]")
    ax.set_ylabel("Sci_all − Sci_pred [cnt/s/det]")
    ax.set_title(f"Box {box}: residual vs charged-particle rate\nρ = {rho:+.3f}")
    ax.grid(alpha=0.3)
    ax.set_ylim(-300, 1500)

    # Bottom: pred vs Sci_clean+1bit (i.e., minus particles)
    ax = axes[1, col]
    sci_photon = sci_all - sci_part  # remove ≥2-bit (charged particles)
    ax.scatter(sci_photon[mask], sci_pred[mask], s=2, alpha=0.2, color="C2", rasterized=True)
    lo, hi = 100, sci_all[mask].max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci_photon (Sci − particles) [cnt/s/det]")
    ax.set_ylabel("Sci predicted")
    rms_photon = np.sqrt(np.mean((sci_photon[mask] - sci_pred[mask]) ** 2))
    rms_all = np.sqrt(np.mean((sci_all[mask] - sci_pred[mask]) ** 2))
    ax.set_title(f"Box {box}: minus particles  RMS={rms_photon:.0f} (was {rms_all:.0f})")
    ax.grid(alpha=0.3)

fig.suptitle("ACD bitcount-based filter: do ≥2-bit ACD events drive the non-linearity?\n"
             "Top: residual vs particle rate (positive ρ = particles cause excess Sci_obs)\n"
             "Bottom: pred vs (Sci − particles) — should be cleaner than including particles",
             fontsize=11)
fig.tight_layout()
out = "plots/acd_test_v2.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")

print(f"\n=== Summary ===")
for D in data:
    box = D["box"]
    length = D["length"]
    sci_all = (D["Sci_all"] / length[:, None]).flatten()
    sci_part = (D["Sci_part"] / length[:, None]).flatten()
    pho_rate = (D["PHO"] / length[:, None]).flatten()
    wide_rate = (D["Wide"] / length[:, None]).flatten()
    large_rate = (D["Large"] / length[:, None]).flatten()
    sci_pred = predict_sci(box, pho_rate, wide_rate, large_rate)
    mask = (sci_all > 100) & (sci_all < np.percentile(sci_all, 95))
    rms_all = np.sqrt(np.mean((sci_all[mask] - sci_pred[mask]) ** 2))
    sci_photon = sci_all - sci_part
    rms_photon = np.sqrt(np.mean((sci_photon[mask] - sci_pred[mask]) ** 2))
    rho = np.corrcoef(sci_part[mask], (sci_all - sci_pred)[mask])[0, 1]
    print(f"  Box {box}: RMS(Sci_all)={rms_all:.1f}, RMS(Sci_minus_particles)={rms_photon:.1f}, ρ(particles, residual)={rho:+.3f}")
