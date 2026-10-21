#!/usr/bin/env python3
"""Scan ACD bitcount threshold: which gives cleanest photon-only fit?
   For each threshold k:
     Sci_photon = events with bitcount < k
     Sci_particle = events with bitcount >= k
   Refit Sci_photon = a0 + a1·PHO + a2·W + a3·L, compute RMS.
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

    # Pre-binned per (det, threshold)
    return {"box": box_name, "met": met_eng, "length": length_s,
            "PHO": PHO, "Wide": Wide, "Large": Large,
            "events": df, "det_off": det_off}


print("Loading 3 boxes...")
data = [load_box(*b) for b in BOXES]


def bin_sci(D, thresh):
    """thresh = bitcount; events with bitcount < thresh are 'photon'."""
    met_eng = D["met"]; length_s = D["length"]
    Sci_phot = np.zeros((len(met_eng), 6))
    Sci_part = np.zeros((len(met_eng), 6))
    for det in range(6):
        e = D["events"][D["events"]["det_id"] == det]
        m_phot = np.sort(e.loc[e["bitcount"] < thresh, "met"].values)
        m_part = np.sort(e.loc[e["bitcount"] >= thresh, "met"].values)
        for i in range(len(met_eng)):
            t0 = met_eng[i]; t1 = t0 + length_s[i]
            Sci_phot[i, det] = np.searchsorted(m_phot, t1) - np.searchsorted(m_phot, t0)
            Sci_part[i, det] = np.searchsorted(m_part, t1) - np.searchsorted(m_part, t0)
    return Sci_phot, Sci_part


# Scan thresholds
thresholds = [1, 2, 3, 4, 5, 6, 100]  # 100 = no filter (all are "photon")
print(f"\n{'Box':>3s}  thresh  fraction_kept  RMS_photon_fit  a0    a1    a2    a3   |β    γ    α    b")

results = []
for D in data:
    box = D["box"]
    for thresh in thresholds:
        Sci_phot, Sci_part = bin_sci(D, thresh)
        length = D["length"]
        sci_phot = (Sci_phot / length[:, None]).flatten()
        sci_part = (Sci_part / length[:, None]).flatten()
        sci_all = sci_phot + sci_part
        pho = (D["PHO"] / length[:, None]).flatten()
        wide = (D["Wide"] / length[:, None]).flatten()
        large = (D["Large"] / length[:, None]).flatten()
        mask = (sci_all > 100) & (sci_all < np.percentile(sci_all, 95)) & (pho > 100)
        # Fit
        A = np.column_stack([np.ones(mask.sum()), pho[mask], wide[mask], large[mask]])
        c, *_ = np.linalg.lstsq(A, sci_phot[mask], rcond=None)
        pred = c[0] + c[1] * pho + c[2] * wide + c[3] * large
        rms = np.sqrt(np.mean((sci_phot[mask] - pred[mask]) ** 2))
        # Implied physical
        alpha = 1.0 / c[1] - 1 if c[1] != 0 else np.nan
        beta = -c[2] / c[1] if c[1] != 0 else np.nan
        gamma = -c[3] / c[1] if c[1] != 0 else np.nan
        b_phys = -c[0] / c[1] if c[1] != 0 else np.nan
        frac_kept = (sci_phot[mask].sum()) / (sci_all[mask].sum())
        results.append({"box": box, "thresh": thresh, "frac_kept": frac_kept,
                         "rms": rms, "a0": c[0], "a1": c[1], "a2": c[2], "a3": c[3],
                         "beta": beta, "gamma": gamma, "alpha": alpha, "b": b_phys})
        thresh_label = "all" if thresh == 100 else f"≥{thresh}"
        print(f"  {box}  bit{thresh_label:>5s}  {frac_kept:>11.3f}  {rms:>14.1f}  "
              f"{c[0]:>+5.0f} {c[1]:>5.3f} {c[2]:>+5.3f} {c[3]:>+5.3f}  | "
              f"{beta:>4.2f} {gamma:>4.2f} {alpha:>4.2f} {b_phys:>+4.0f}")

res_df = pd.DataFrame(results)

# === Plot RMS vs threshold ===
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Top-left: RMS vs threshold (per box)
ax = axes[0, 0]
for box in "ABC":
    sub = res_df[res_df["box"] == box]
    # x-axis: threshold (use ≥1, ≥2, ..., "all" → x=99)
    xs = sub["thresh"].values
    xs = np.where(xs == 100, 8, xs)
    ax.plot(xs, sub["rms"], "o-", lw=2, markersize=8, label=f"Box {box}")
ax.set_xlabel("Bitcount threshold (events with bitcount < thresh kept as photons)")
ax.set_ylabel("RMS [cnt/s/det]")
ax.set_title("Photon-only fit RMS vs ACD threshold")
ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
ax.set_xticklabels(["≥1\n(any ACD)", "≥2", "≥3", "≥4", "≥5", "≥6", "no\nfilter"])
ax.grid(alpha=0.3)
ax.legend()

# Top-right: β vs threshold (should be 2)
ax = axes[0, 1]
for box in "ABC":
    sub = res_df[res_df["box"] == box]
    xs = sub["thresh"].values
    xs = np.where(xs == 100, 8, xs)
    ax.plot(xs, sub["beta"], "o-", lw=2, markersize=8, label=f"Box {box}")
ax.axhline(2.0, color="r", ls="--", lw=1, label="β=2 (expected)")
ax.set_xlabel("Bitcount threshold")
ax.set_ylabel("β (Wide multiplier)")
ax.set_title("β vs ACD threshold")
ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
ax.set_xticklabels(["≥1", "≥2", "≥3", "≥4", "≥5", "≥6", "all"])
ax.grid(alpha=0.3)
ax.legend()

# Bottom-left: γ vs threshold (should be 1.2)
ax = axes[1, 0]
for box in "ABC":
    sub = res_df[res_df["box"] == box]
    xs = sub["thresh"].values
    xs = np.where(xs == 100, 8, xs)
    ax.plot(xs, sub["gamma"], "o-", lw=2, markersize=8, label=f"Box {box}")
ax.axhline(1.2, color="r", ls="--", lw=1, label="γ=1.2 (expected)")
ax.set_xlabel("Bitcount threshold")
ax.set_ylabel("γ (Large multiplier)")
ax.set_title("γ vs ACD threshold")
ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
ax.set_xticklabels(["≥1", "≥2", "≥3", "≥4", "≥5", "≥6", "all"])
ax.grid(alpha=0.3)
ax.legend()

# Bottom-right: fraction_kept (photon fraction) vs threshold
ax = axes[1, 1]
for box in "ABC":
    sub = res_df[res_df["box"] == box]
    xs = sub["thresh"].values
    xs = np.where(xs == 100, 8, xs)
    ax.plot(xs, sub["frac_kept"], "o-", lw=2, markersize=8, label=f"Box {box}")
ax.set_xlabel("Bitcount threshold")
ax.set_ylabel("Fraction of events kept as 'photons'")
ax.set_title("Photon fraction vs threshold")
ax.set_xticks([1, 2, 3, 4, 5, 6, 8])
ax.set_xticklabels(["≥1", "≥2", "≥3", "≥4", "≥5", "≥6", "all"])
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
out = "plots/acd_threshold_scan.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")
