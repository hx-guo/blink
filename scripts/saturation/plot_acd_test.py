#!/usr/bin/env python3
"""Test ACD hypothesis: do ACD-flagged (charged-particle) events explain
   the non-linearity at high rate?

Compare:
  - Sci_all: all events (current behavior)
  - Sci_no_acd: events with aminfo == 0 (no ACD veto, pure photons)
  - Sci_acd: events with aminfo > 0 (ACD-flagged, suspect particles)

Apply universal predictor and check residual structure.
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

    # Read EVT events with aminfo
    df = pd.read_csv(csv_path, usecols=["type", "met", "det_id", "aminfo"],
                     dtype={"type": "category", "met": "float64", "det_id": "int8", "aminfo": "uint32"})
    df = df[df["type"] == "EVT"]
    print(f"  {box_name}: {len(df):,} events, {(df['aminfo']>0).sum():,} ACD-flagged ({(df['aminfo']>0).mean()*100:.1f}%)", flush=True)

    Sci_all = np.zeros((len(met_eng), 6))
    Sci_no_acd = np.zeros((len(met_eng), 6))
    Sci_acd = np.zeros((len(met_eng), 6))
    for det in range(6):
        evts_all = df.loc[df["det_id"] == det].sort_values("met")
        m_all = evts_all["met"].values
        m_no = evts_all.loc[evts_all["aminfo"] == 0, "met"].values
        m_yes = evts_all.loc[evts_all["aminfo"] > 0, "met"].values
        for i in range(len(met_eng)):
            t0 = met_eng[i]; t1 = t0 + length_s[i]
            Sci_all[i, det] = np.searchsorted(m_all, t1) - np.searchsorted(m_all, t0)
            Sci_no_acd[i, det] = np.searchsorted(m_no, t1) - np.searchsorted(m_no, t0)
            Sci_acd[i, det] = np.searchsorted(m_yes, t1) - np.searchsorted(m_yes, t0)

    return {"box": box_name, "met": met_eng, "length": length_s,
            "PHO": PHO, "Wide": Wide, "Large": Large,
            "Sci_all": Sci_all, "Sci_no_acd": Sci_no_acd, "Sci_acd": Sci_acd}


def predict_sci(box, pho, wide, large):
    c = COEFS[box]
    return c["a0"] + c["a1"] * pho + c["a2"] * wide + c["a3"] * large


print("Loading 3 boxes...")
data = [load_box(*b) for b in BOXES]


# === Plot: Sci comparison ===
fig, axes = plt.subplots(3, 3, figsize=(16, 11))

for col, D in enumerate(data):
    box = D["box"]
    length = D["length"]
    # Per-second per-det rates
    sci_all_rate = (D["Sci_all"] / length[:, None]).flatten()
    sci_no_rate = (D["Sci_no_acd"] / length[:, None]).flatten()
    sci_acd_rate = (D["Sci_acd"] / length[:, None]).flatten()
    pho_rate = (D["PHO"] / length[:, None]).flatten()
    wide_rate = (D["Wide"] / length[:, None]).flatten()
    large_rate = (D["Large"] / length[:, None]).flatten()
    sci_pred = predict_sci(box, pho_rate, wide_rate, large_rate)
    # filter
    mask = (sci_all_rate > 100) & (pho_rate > 100) & (sci_all_rate < np.percentile(sci_all_rate, 99))

    # Top: Sci_all vs predicted
    ax = axes[0, col]
    ax.scatter(sci_all_rate[mask], sci_pred[mask], s=2, alpha=0.2, color="C0", rasterized=True)
    lo, hi = 100, sci_all_rate[mask].max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci_all observed [cnt/s/det]")
    ax.set_ylabel("Sci predicted (universal)")
    rms_all = np.sqrt(np.mean((sci_all_rate[mask] - sci_pred[mask]) ** 2))
    ax.set_title(f"Box {box} — Sci_all  RMS={rms_all:.0f}")
    ax.grid(alpha=0.3)

    # Middle: Sci_no_acd vs predicted (this is the test)
    ax = axes[1, col]
    ax.scatter(sci_no_rate[mask], sci_pred[mask], s=2, alpha=0.2, color="C2", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Sci_no_ACD observed [cnt/s/det]")
    ax.set_ylabel("Sci predicted")
    rms_no = np.sqrt(np.mean((sci_no_rate[mask] - sci_pred[mask]) ** 2))
    ax.set_title(f"Box {box} — Sci_no_ACD  RMS={rms_no:.0f}")
    ax.grid(alpha=0.3)

    # Bottom: residual vs Sci_all (showing which subset drives the curve)
    ax = axes[2, col]
    ax.scatter(sci_all_rate[mask], (sci_all_rate - sci_pred)[mask], s=2, alpha=0.2,
               color="C0", label="Sci_all − pred", rasterized=True)
    ax.scatter(sci_acd_rate[mask], (sci_acd_rate - sci_pred * 0)[mask], s=2, alpha=0.05,
               color="C3", rasterized=True)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Sci_all observed [cnt/s/det]")
    ax.set_ylabel("residual & Sci_acd")
    ax.set_title(f"Box {box} — Sci_acd alone (red, alpha)\n"
                  f"Mean ACD fraction = {sci_acd_rate[mask].mean()/sci_all_rate[mask].mean()*100:.1f}%")
    ax.grid(alpha=0.3)
    ax.set_ylim(-300, 1500)
    ax.set_xlim(0, 2500)

fig.suptitle("ACD hypothesis test: does removing ACD-flagged events fix the non-linearity?\n"
             "Top: Sci_all vs pred (shows the bend). Middle: Sci_no_ACD vs pred. Bottom: residual + Sci_acd cloud.",
             fontsize=11)
fig.tight_layout()
out = "plots/acd_test_260226.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# Numeric summary
print(f"\n=== Per-box RMS comparison ===")
for D in data:
    box = D["box"]
    length = D["length"]
    sci_all = (D["Sci_all"] / length[:, None]).flatten()
    sci_no = (D["Sci_no_acd"] / length[:, None]).flatten()
    pho_rate = (D["PHO"] / length[:, None]).flatten()
    wide_rate = (D["Wide"] / length[:, None]).flatten()
    large_rate = (D["Large"] / length[:, None]).flatten()
    sci_pred = predict_sci(box, pho_rate, wide_rate, large_rate)
    mask = (sci_all > 100) & (pho_rate > 100) & (sci_all < np.percentile(sci_all, 95))
    rms_all = np.sqrt(np.mean((sci_all[mask] - sci_pred[mask]) ** 2))
    rms_no = np.sqrt(np.mean((sci_no[mask] - sci_pred[mask]) ** 2))
    print(f"  Box {box}: RMS(Sci_all) = {rms_all:.1f}, RMS(Sci_no_ACD) = {rms_no:.1f}, ratio = {rms_no/rms_all:.2f}")
