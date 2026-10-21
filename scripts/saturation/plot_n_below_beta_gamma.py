#!/usr/bin/env python3
"""Generalize: PHO = N_n + β·Wide + γ·Large + N_below
=> N_below = PHO - β·Wide - γ·Large - Sci

Find optimal (β, γ) per detector by regressing
nb1 = pho - 1*wide - 1*large - sci
on (1, sci, wide, large). Coefficients give (β-1, γ-1).
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

    det_evts = {i: [] for i in range(6)}
    with open(sci_csv) as f:
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

    valid_box = np.ones(len(met_eng), dtype=bool)
    for i in range(len(met_eng)):
        t0 = met_eng[i]; t1 = t0 + length_s[i]
        if overlaps_saturation(t0, t1, sat_intervals[box_name]):
            valid_box[i] = False
    valid_box &= (L_cycles > 50000) & (Sci.sum(axis=1) > 100)

    for det_local in range(6):
        v = valid_box
        det_data.append({
            "box": box_name, "det_local": det_local,
            "sci": Sci[v, det_local] / length_s[v],
            "wide": Wide[v, det_local] / length_s[v],
            "large": Large[v, det_local] / length_s[v],
            "pho": PHO[v, det_local] / length_s[v],
        })
    fe.close()


def fit_linear(y, x):
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[0], c[1], y - X @ c


print("\n=== Per-detector fit: N_below = PHO - β·Wide - γ·Large - Sci  with optimal (β, γ) ===")
print(f"{'Box':>3s} {'D':>2s}  {'β':>5s} {'γ':>5s}  "
      f"{'RMS_β1γ1':>9s} {'RMS_β2γ1':>9s} {'RMS_β2γ2':>9s} {'RMS_βγfit':>10s}")

results = []
for dd in det_data:
    pho, wide, large, sci = dd["pho"], dd["wide"], dd["large"], dd["sci"]
    nb_base = pho - wide - large - sci  # β=γ=1

    # Find optimal (β, γ) via regression of nb_base on (1, sci, wide, large)
    X = np.column_stack([np.ones_like(sci), sci, wide, large])
    c, *_ = np.linalg.lstsq(X, nb_base, rcond=None)
    beta_fit = 1.0 + c[2]
    gamma_fit = 1.0 + c[3]

    # Compute residuals at different (β, γ)
    def rms(b_mult, g_mult):
        nb = pho - b_mult * wide - g_mult * large - sci
        _, _, r = fit_linear(nb, sci)
        return np.sqrt(np.mean(r ** 2))

    rms_11 = rms(1, 1)
    rms_21 = rms(2, 1)
    rms_22 = rms(2, 2)
    rms_fit = rms(beta_fit, gamma_fit)

    # Also Pearson with hardness (after best fit)
    nb_best = pho - beta_fit * wide - gamma_fit * large - sci
    _, _, r_best = fit_linear(nb_best, sci)
    hardness = large / np.maximum(sci, 1)
    rho_h_best = np.corrcoef(r_best, hardness)[0, 1]

    nb_22 = pho - 2 * wide - 2 * large - sci
    _, _, r_22 = fit_linear(nb_22, sci)
    rho_h_22 = np.corrcoef(r_22, hardness)[0, 1]

    nb_21 = pho - 2 * wide - 1 * large - sci
    _, _, r_21 = fit_linear(nb_21, sci)
    rho_h_21 = np.corrcoef(r_21, hardness)[0, 1]

    results.append({
        "dd": dd, "beta": beta_fit, "gamma": gamma_fit,
        "rms_11": rms_11, "rms_21": rms_21, "rms_22": rms_22, "rms_fit": rms_fit,
        "rho_h_21": rho_h_21, "rho_h_22": rho_h_22, "rho_h_best": rho_h_best,
    })
    print(f"{dd['box']:>3s} {dd['det_local']:>2d}  {beta_fit:>5.2f} {gamma_fit:>5.2f}  "
          f"{rms_11:>9.1f} {rms_21:>9.1f} {rms_22:>9.1f} {rms_fit:>10.1f}")

print("\n=== Box-level (β, γ) medians ===")
for box in "ABC":
    rs = [r for r in results if r["dd"]["box"] == box]
    bs = [r["beta"] for r in rs]
    gs = [r["gamma"] for r in rs]
    print(f"Box {box}: β = {np.median(bs):.2f} ± {np.std(bs):.2f}, "
          f"γ = {np.median(gs):.2f} ± {np.std(gs):.2f}")
all_b = [r["beta"] for r in results]; all_g = [r["gamma"] for r in results]
print(f"All:    β = {np.median(all_b):.2f} ± {np.std(all_b):.2f}, "
      f"γ = {np.median(all_g):.2f} ± {np.std(all_g):.2f}")

# === Plot 1: (β, γ) scatter ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}
labels_added = set()
for r in results:
    box = r["dd"]["box"]
    lab = f"Box {box}" if box not in labels_added else None
    labels_added.add(box)
    axes[0].scatter(r["beta"], r["gamma"], color=box_colors[box], s=80,
                    edgecolor="k", linewidth=0.6, label=lab)
    axes[0].annotate(f"{box}{r['dd']['det_local']}", (r["beta"], r["gamma"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")

axes[0].axvline(2, color="r", ls="--", lw=0.8, label="β=2")
axes[0].axhline(1, color="b", ls="--", lw=0.8, label="γ=1")
axes[0].axhline(2, color="g", ls="--", lw=0.8, label="γ=2")
axes[0].set_xlabel("β (Wide multiplier)")
axes[0].set_ylabel("γ (Large multiplier)")
axes[0].set_title("Per-detector (β, γ) — Wide and Large multipliers in PHO")
axes[0].legend()
axes[0].grid(alpha=0.3)

# RMS comparison
xs = np.arange(len(results))
labels = [f"{r['dd']['box']}{r['dd']['det_local']}" for r in results]
axes[1].plot(xs, [r["rms_11"] for r in results], "ro-", label=f"β=1,γ=1 (mean {np.mean([r['rms_11'] for r in results]):.0f})")
axes[1].plot(xs, [r["rms_21"] for r in results], "g^-", label=f"β=2,γ=1 (mean {np.mean([r['rms_21'] for r in results]):.0f})")
axes[1].plot(xs, [r["rms_22"] for r in results], "ms-", label=f"β=2,γ=2 (mean {np.mean([r['rms_22'] for r in results]):.0f})")
axes[1].plot(xs, [r["rms_fit"] for r in results], "C0d-", label=f"(β,γ) fitted (mean {np.mean([r['rms_fit'] for r in results]):.0f})")
axes[1].set_xticks(xs)
axes[1].set_xticklabels(labels, rotation=90, fontsize=8)
axes[1].set_ylabel("Absolute RMS [cnt/s]")
axes[1].set_title("RMS comparison")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
out1 = "plots/n_below_beta_gamma_summary_260226.png"
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out1}")

# === Plot 2: residual vs hardness, comparing β=2,γ=1 vs β=2,γ=2 vs (β,γ)_fit ===
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

# Aggregate across all dets
def collect_resid(b_mult, g_mult):
    rs, hs = [], []
    for dd in det_data:
        pho = dd["pho"]; wide = dd["wide"]; large = dd["large"]; sci = dd["sci"]
        nb = pho - b_mult * wide - g_mult * large - sci
        _, _, r = fit_linear(nb, sci)
        h = large / np.maximum(sci, 1)
        rs.append(r); hs.append(h)
    return np.concatenate(rs), np.concatenate(hs)

for k, (bm, gm, title) in enumerate([(2, 1, "β=2, γ=1 (Wide-only)"),
                                      (2, 2, "β=2, γ=2 (both)"),
                                      (None, None, "(β, γ) fitted per det")]):
    if bm is None:
        rs, hs = [], []
        for r in results:
            dd = r["dd"]
            pho = dd["pho"]; wide = dd["wide"]; large = dd["large"]; sci = dd["sci"]
            nb = pho - r["beta"] * wide - r["gamma"] * large - sci
            _, _, resid = fit_linear(nb, sci)
            rs.append(resid)
            hs.append(large / np.maximum(sci, 1))
        rs = np.concatenate(rs); hs = np.concatenate(hs)
    else:
        rs, hs = collect_resid(bm, gm)

    ax = axes2[k]
    ax.scatter(hs, rs, s=1.5, alpha=0.12, color="C0", rasterized=True)
    bins = np.linspace(np.percentile(hs, 1), np.percentile(hs, 99), 25)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []
    for i in range(len(bins) - 1):
        m = (hs >= bins[i]) & (hs < bins[i + 1])
        med.append(np.median(rs[m]) if m.sum() > 5 else np.nan)
    ax.plot(bc, med, "k-", lw=1.7)
    rho = np.corrcoef(hs, rs)[0, 1]
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("Hardness = Large/Sci")
    if k == 0:
        ax.set_ylabel("Residual [cnt/s]")
    ax.set_title(f"{title}   ρ={rho:+.3f}")
    ax.grid(alpha=0.3)

fig2.suptitle("Residual vs Hardness — does adding γ·Large flatten it?", fontsize=11)
fig2.tight_layout()
out2 = "plots/residual_vs_hardness_compare_260226.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"Saved: {out2}")
