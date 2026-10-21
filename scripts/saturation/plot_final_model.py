#!/usr/bin/env python3
"""Final model visualization: linear vs quadratic, focused plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from unwrap_large import unwrap_large

MET_CORRECTION = 4.0
DATES = [
    ("2020-04-15", "data/1B/2020/20200415/{box}/*.fits", "/tmp/200415_box{B}.csv"),
    ("2020-04-28", "data/1B/2020/20200428/{box}/*.fits", "/tmp/200428_box{B}.csv"),
    ("2022-10-09", "data/1B/2022/20221009/{box}/*.fits", "/tmp/221009_box{B}.csv"),
    ("2026-02-26", "data/1B/2026/20260226/{box}/*.fits", "/tmp/260226_box{B}_full.csv"),
    ("2026-04-10", "data/1B/2026/20260410/{box}/*.fits", "/tmp/260410_box{B}.csv"),
]
BOX_FITS_CODES = {"A": "0766", "B": "1009", "C": "1781"}
BOX_OFFSETS = {"A": 0, "B": 6, "C": 12}


def load(date_label, fits_glob_tpl, sci_csv_tpl, box):
    fits_glob = fits_glob_tpl.format(box=BOX_FITS_CODES[box])
    sci_csv = sci_csv_tpl.format(B=box)
    fits_files = sorted(glob.glob(fits_glob))
    if not fits_files: return None
    fe = fits.open(fits_files[0], memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    L_cycles = d["Length_Time_Cycle"].astype(float)
    length_s = L_cycles * 16e-6
    det_off = BOX_OFFSETS[box]
    det_ids = [det_off + i for i in range(6)]
    PHO = np.column_stack([d[f"Cnt_PHODet_{i}"].astype(float) for i in det_ids])
    Wide = np.column_stack([d[f"Cnt_CsI_PHODet_{i}"].astype(float) for i in det_ids])
    Large_raw = np.column_stack([d[f"Cnt_LargeEvt_{i}"].astype(float) for i in det_ids])
    Large = np.column_stack([unwrap_large(PHO[:, i], Large_raw[:, i]) for i in range(6)])
    met_eng = d["Time"].astype(float) + offset + MET_CORRECTION

    df = pd.read_csv(sci_csv, usecols=["type", "met", "det_id"],
                     dtype={"type": "category", "met": "float64", "det_id": "int8"})
    df = df[df["type"] == "EVT"]
    Sci = np.zeros((len(met_eng), 6))
    for det in range(6):
        evts = np.sort(df["met"].values[df["det_id"].values == det])
        for i in range(len(met_eng)):
            t0 = met_eng[i]; t1 = t0 + length_s[i]
            Sci[i, det] = np.searchsorted(evts, t1) - np.searchsorted(evts, t0)
    del df
    valid = (L_cycles > 50000) & (Sci.sum(axis=1) > 100)
    sci_total = Sci.sum(axis=1)
    if valid.sum() < 50:
        fe.close(); return None
    p5, p95 = np.percentile(sci_total[valid], [5, 95])
    valid &= (sci_total >= p5) & (sci_total <= p95)
    fe.close()
    return {"PHO": PHO, "Wide": Wide, "Large": Large, "Sci": Sci,
            "length": length_s, "valid": valid}


print("Loading...")
all_rows = []
for date_label, fits_glob_tpl, sci_csv_tpl in DATES:
    for box in "ABC":
        D = load(date_label, fits_glob_tpl, sci_csv_tpl, box)
        if D is None: continue
        v = D["valid"]
        for det in range(6):
            sci = D["Sci"][v, det] / D["length"][v]
            wide = D["Wide"][v, det] / D["length"][v]
            large = D["Large"][v, det] / D["length"][v]
            pho = D["PHO"][v, det] / D["length"][v]
            for i in range(len(pho)):
                all_rows.append({"date": date_label, "PHO": pho[i], "Wide": wide[i],
                                 "Large": large[i], "Sci": sci[i]})
        print(f"  {date_label} Box {box} done")
big = pd.DataFrame(all_rows)
y = big["Sci"].values
print(f"Total bins: {len(big)}")

# Linear model
A_lin = np.column_stack([np.ones(len(big)), big["PHO"], big["Wide"], big["Large"]])
c_lin, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
big["pred_lin"] = A_lin @ c_lin
big["resid_lin"] = big["Sci"] - big["pred_lin"]

# Quadratic model (PHO^2 + PHO·Large)
A_q = np.column_stack([np.ones(len(big)), big["PHO"], big["Wide"], big["Large"],
                       big["PHO"] ** 2, big["PHO"] * big["Large"]])
c_q, *_ = np.linalg.lstsq(A_q, y, rcond=None)
big["pred_q"] = A_q @ c_q
big["resid_q"] = big["Sci"] - big["pred_q"]

rms_lin = np.sqrt(np.mean(big["resid_lin"] ** 2))
rms_q = np.sqrt(np.mean(big["resid_q"] ** 2))
print(f"\nLinear RMS: {rms_lin:.1f}")
print(f"Quadratic RMS: {rms_q:.1f}")
print(f"\nLinear: Sci = {c_lin[0]:.1f} + {c_lin[1]:.4f}·PHO + {c_lin[2]:.4f}·Wide + {c_lin[3]:.4f}·Large")
print(f"\nQuadratic:")
print(f"  Sci = {c_q[0]:.1f} + {c_q[1]:.4f}·PHO + {c_q[2]:.4f}·Wide + {c_q[3]:.4f}·Large")
print(f"        + {c_q[4]:.3e}·PHO² + {c_q[5]:.3e}·PHO·Large")


# === Plot ===
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

# Top row: pred vs obs scatter, side by side, with residual on top
def hexbin_with_median(ax, x, y, title, sub_xlabel="Sci_observed [cnt/s]",
                       sub_ylabel="Sci_predicted [cnt/s]"):
    sample = np.random.choice(len(x), min(80000, len(x)), replace=False)
    xs = x.values[sample] if hasattr(x, "values") else x[sample]
    ys = y.values[sample] if hasattr(y, "values") else y[sample]
    hb = ax.hexbin(xs, ys, gridsize=70, cmap="viridis", mincnt=1, bins="log")
    lo, hi = max(xs.min(), 300), min(xs.max(), 2000)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x", zorder=10)
    bins = np.linspace(lo, hi, 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med_y = []
    for i in range(len(bins) - 1):
        m = (xs >= bins[i]) & (xs < bins[i + 1])
        med_y.append(np.median(ys[m]) if m.sum() > 30 else np.nan)
    ax.plot(bc, med_y, "w-", lw=2.5, label="median pred", zorder=10)
    ax.plot(bc, med_y, "k-", lw=1.2, zorder=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(sub_xlabel)
    ax.set_ylabel(sub_ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

ax_lin = fig.add_subplot(gs[0, 0])
hexbin_with_median(ax_lin, big["Sci"], big["pred_lin"],
                    f"Linear (4 coeffs)\nRMS = {rms_lin:.1f} cnt/s")

ax_q = fig.add_subplot(gs[0, 1])
hexbin_with_median(ax_q, big["Sci"], big["pred_q"],
                    f"Quadratic (6 coeffs, +PHO² +PHO·Large)\nRMS = {rms_q:.1f} cnt/s")

# Right: histogram of residuals
ax_h = fig.add_subplot(gs[0, 2])
ax_h.hist(big["resid_lin"], bins=np.linspace(-200, 200, 80), color="C3", alpha=0.6,
          label=f"Linear (σ={rms_lin:.1f})")
ax_h.hist(big["resid_q"], bins=np.linspace(-200, 200, 80), color="C0", alpha=0.6,
          label=f"Quadratic (σ={rms_q:.1f})")
ax_h.set_xlabel("Residual = Sci_obs − Sci_pred [cnt/s]")
ax_h.set_ylabel("count")
ax_h.set_title("Residual distribution")
ax_h.legend()
ax_h.grid(alpha=0.3)

# Bottom row: residual vs Sci, both models
def resid_panel(ax, sci, resid, title, color):
    sample = np.random.choice(len(sci), min(60000, len(sci)), replace=False)
    xs = sci.values[sample]; ys = resid.values[sample]
    ax.scatter(xs, ys, s=1.5, alpha=0.06, color=color, rasterized=True)
    ax.axhline(0, color="r", ls="--", lw=1)
    bins = np.linspace(np.percentile(sci, 1), np.percentile(sci, 99), 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = []; p16 = []; p84 = []
    for i in range(len(bins) - 1):
        m = (sci.values >= bins[i]) & (sci.values < bins[i + 1])
        if m.sum() > 30:
            med.append(np.median(resid.values[m]))
            p16.append(np.percentile(resid.values[m], 16))
            p84.append(np.percentile(resid.values[m], 84))
        else:
            med.append(np.nan); p16.append(np.nan); p84.append(np.nan)
    ax.fill_between(bc, p16, p84, color="k", alpha=0.18, label="16-84%")
    ax.plot(bc, med, "k-", lw=2, label="median")
    ax.set_xlabel("Sci_observed [cnt/s]")
    ax.set_ylabel("Residual [cnt/s]")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-300, 300)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

ax_rl = fig.add_subplot(gs[1, 0])
resid_panel(ax_rl, big["Sci"], big["resid_lin"],
             "Linear residual (still bends slightly)", "C3")

ax_rq = fig.add_subplot(gs[1, 1])
resid_panel(ax_rq, big["Sci"], big["resid_q"],
             "Quadratic residual (flatter)", "C0")

# Bottom-right: bar of model improvement
ax_b = fig.add_subplot(gs[1, 2])
ax_b.bar(["Linear\n(4 coeffs)", "Quadratic\n(6 coeffs)", "Per-det\n(refit each)", "Poisson\n(estimated)"],
         [rms_lin, rms_q, 20.5, 50],
         color=["C3", "C0", "k", "gray"])
for i, v in enumerate([rms_lin, rms_q, 20.5, 50]):
    ax_b.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax_b.set_ylabel("Mean RMS [cnt/s]")
ax_b.set_title("Model precision summary")
ax_b.grid(alpha=0.3, axis="y")

# Title
formula = (f"Sci = {c_q[0]:+.0f} {c_q[1]:+.3f}·PHO {c_q[2]:+.3f}·Wide "
           f"{c_q[3]:+.3f}·Large {c_q[4]:+.2e}·PHO² {c_q[5]:+.2e}·PHO·Large")
fig.suptitle(f"Final Sci predictor — 6-coefficient global model\n{formula}",
             fontsize=11)
out = "plots/final_quadratic_model.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# Numeric summary
print(f"\n=== FINAL FORMULA ===")
print(f"  Sci = {c_q[0]:+.1f}")
print(f"        {c_q[1]:+.4f}·PHO")
print(f"        {c_q[2]:+.4f}·Wide")
print(f"        {c_q[3]:+.4f}·Large")
print(f"        {c_q[4]:+.3e}·PHO²")
print(f"        {c_q[5]:+.3e}·PHO·Large")
print(f"  RMS = {rms_q:.1f} cnt/s/det  ({rms_q/650*100:.1f}% of typical Sci)")
