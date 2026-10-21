#!/usr/bin/env python3
"""Diagnose source of +18% over-prediction at GRB 260226A burst peak.

Three hypotheses:
  H1. Model extrapolation: model fails at Sci > 2500/det (burst regime).
      → Residual should correlate with rate, present even WITHOUT FIFO drops.
  H2. Recovery over-fill: Sci_recov is too high.
      → Residual should correlate with recovery factor f = Sci_recov/Sci_obs.
  H3. Wide/Large non-linearity at burst: β, γ coefficients break at high rate.
      → Residual should correlate with Wide/Sci or Large/Sci ratios.

Test: look at "burst onset" bins (t = +18 to +21) where rate is rising but
FIFO drops haven't started yet. If model already over-predicts THERE, it's
H1 or H3, not H2.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
MET_CORRECTION = 4.0
TRIGGER_MET = 446726273.0
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

# Same coefficients as validate_sci_recov.py
M1_HIGH = {
    "A": dict(b=-103.5, alpha1=1.3705, beta=3.2589, gamma=0.7627),
    "B": dict(b=-127.2, alpha1=1.4565, beta=3.0297, gamma=0.6954),
    "C": dict(b=-129.2, alpha1=1.4284, beta=3.2752, gamma=0.7334),
}
M1_ALL = {
    "A": dict(b=-410.8, alpha1=2.3131, beta=2.9998, gamma=0.0359),
    "B": dict(b=-449.7, alpha1=2.3997, beta=2.6847, gamma=0.0237),
    "C": dict(b=-458.6, alpha1=2.3857, beta=2.9554, gamma=0.0509),
}


def pred_m1(box, sci, wide, large, coefs):
    c = coefs[box]
    return c["b"] + c["alpha1"]*sci + c["beta"]*wide + c["gamma"]*large


def load_data():
    # Engineering FITS
    eng_dfs = []
    for box, code in [("A","0766"),("B","1009"),("C","1781")]:
        eng_file = f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits"
        fe = fits.open(eng_file, memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
        length_cyc = d["Length_Time_Cycle"].astype(float)
        mask = (met_eng >= TRIGGER_MET - 30) & (met_eng <= TRIGGER_MET + 70)
        met_eng = met_eng[mask]; length_cyc = length_cyc[mask]
        for det_local in range(6):
            det_g = BOX_OFFSET[box] + det_local
            pho = d[f"Cnt_PHODet_{det_g}"].astype(float)[mask]
            csi = d[f"Cnt_CsI_PHODet_{det_g}"].astype(float)[mask]
            dead = d[f"DeadTime_PHODet_{det_g}"].astype(float)[mask]
            large_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)[mask]
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                eng_dfs.append({
                    "box": box, "det": det_local,
                    "met_sec": int(met_eng[i]),
                    "length_cyc": length_cyc[i],
                    "PHO": pho[i], "Wide": csi[i], "Large": large[i], "Dt": dead[i],
                })
        fe.close()
    eng = pd.DataFrame(eng_dfs)

    # Sci observed
    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    sci_obs = sci_obs[sci_obs["type"]=="EVT"]
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    sci_obs = sci_obs.groupby(["box","det_id","met_sec"]).size().rename("Sci_obs").reset_index()
    sci_obs = sci_obs.rename(columns={"det_id":"det"})

    # Sci recovered (box-level)
    sci_recov = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    sci_recov = sci_recov[sci_recov["type"].isin(["EVT","FILL_GAP"])].copy()
    sci_recov["box"] = sci_recov["box"].astype(str)
    sci_recov["met_sec"] = sci_recov["met"].astype("int64")
    sci_recov_box = sci_recov.groupby(["box","met_sec"]).size().rename("Sci_recov_box").reset_index()

    df = eng.merge(sci_obs, on=["box","det","met_sec"], how="left")
    df["Sci_obs"] = df["Sci_obs"].fillna(0)
    df = df.merge(sci_recov_box, on=["box","met_sec"], how="left")
    df["Sci_recov_box"] = df["Sci_recov_box"].fillna(0)
    box_obs_sum = df.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    df["Sci_recov"] = np.where(box_obs_sum > 0,
                                df["Sci_recov_box"] * df["Sci_obs"]/box_obs_sum.clip(lower=1),
                                df["Sci_recov_box"] / 6)
    df["t_rel"] = df["met_sec"] - TRIGGER_MET
    return df


def main():
    df = load_data()
    box_agg = df.groupby(["box","t_rel"]).agg(
        PHO=("PHO","sum"), Wide=("Wide","sum"), Large=("Large","sum"),
        Sci_obs=("Sci_obs","sum"), Sci_recov=("Sci_recov","sum"),
        Dt=("Dt","sum"), length_cyc=("length_cyc","sum"),
    ).reset_index()

    # Apply M1_HIGH and M1_ALL with both Sci_obs and Sci_recov
    for sci_kind, sci_col in [("obs","Sci_obs"),("recov","Sci_recov")]:
        for tag, coefs in [("M1_HIGH",M1_HIGH),("M1_ALL",M1_ALL)]:
            preds = []
            for _, r in box_agg.iterrows():
                preds.append(pred_m1(r.box, r[sci_col], r.Wide, r.Large, coefs))
            box_agg[f"pred_{tag}_{sci_kind}"] = preds

    # Compute residuals
    for tag in ["M1_HIGH","M1_ALL"]:
        for kind in ["obs","recov"]:
            box_agg[f"rel_{tag}_{kind}"] = 100*(box_agg[f"pred_{tag}_{kind}"]-box_agg["PHO"])/box_agg["PHO"]
    box_agg["f_recov"] = box_agg["Sci_recov"] / box_agg["Sci_obs"].clip(lower=1)

    # ============ Print per-second per-box ============
    print(f"\n=== Per-second analysis, Box A burst window ===")
    cols_show = ["t_rel","Sci_obs","Sci_recov","f_recov","PHO","Wide","Large",
                 "pred_M1_HIGH_obs","rel_M1_HIGH_obs","pred_M1_HIGH_recov","rel_M1_HIGH_recov"]
    sub_a = box_agg[(box_agg["box"]=="A") & (box_agg["t_rel"]>=10) & (box_agg["t_rel"]<=35)]
    print(sub_a[cols_show].to_string(index=False, float_format="%.1f"))

    # ============ Pre-burst onset (no FIFO drop) ============
    print(f"\n=== Burst ONSET (rising rate, no FIFO drops yet, t=+18~+21) ===")
    print(f"  If model OVER-predicts here → model extrapolation issue (not recovery)")
    onset = box_agg[(box_agg["t_rel"]>=18) & (box_agg["t_rel"]<=21)]
    for box in "ABC":
        sub = onset[onset["box"]==box]
        if len(sub) == 0: continue
        f_max = sub["f_recov"].max()
        rel_high = sub["rel_M1_HIGH_obs"].median()
        rel_all = sub["rel_M1_ALL_obs"].median()
        print(f"  Box {box}: f_recov max={f_max:.2f}, med rel resid = "
              f"M1_HIGH {rel_high:+.1f}%, M1_ALL {rel_all:+.1f}%")

    # ============ Deep burst (Sci_recov >> Sci_obs) ============
    print(f"\n=== Deep burst (heavily saturated, t=+22~+29) ===")
    deep = box_agg[(box_agg["t_rel"]>=22) & (box_agg["t_rel"]<=29)]
    for box in "ABC":
        sub = deep[deep["box"]==box]
        if len(sub) == 0: continue
        rel_obs = sub["rel_M1_HIGH_obs"].median()
        rel_recov = sub["rel_M1_HIGH_recov"].median()
        f_med = sub["f_recov"].median()
        print(f"  Box {box}: f_recov med={f_med:.2f}, "
              f"M1_HIGH med rel resid: Sci_obs {rel_obs:+.1f}%, Sci_recov {rel_recov:+.1f}%")

    # ============ Plot: residual vs rate, separated by FIFO status ============
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: M1_HIGH relative residual vs Sci rate (use observed, no FIFO drops)
    ax = axes[0]
    # Filter: high-rate non-saturated bins (f_recov ≈ 1)
    clean = box_agg[(box_agg["f_recov"] < 1.05) & (box_agg["Sci_obs"] > 500)]
    sat = box_agg[box_agg["f_recov"] > 1.05]
    for box, color in zip("ABC", ["C0","C1","C2"]):
        c_sub = clean[clean["box"]==box]
        ax.scatter(c_sub["Sci_obs"], c_sub["rel_M1_HIGH_obs"], s=15, color=color,
                   alpha=0.5, label=f"Box {box} (no FIFO drop)")
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Sci rate [cnt/s/box]")
    ax.set_ylabel("M1_HIGH residual (%)")
    ax.set_title("Residual vs rate, ONLY non-saturated bins\n(if grows → model extrapolation issue)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # Panel 2: M1_HIGH residual vs recovery factor f for saturated bins
    ax = axes[1]
    for box, color in zip("ABC", ["C0","C1","C2"]):
        s_sub = sat[sat["box"]==box]
        ax.scatter(s_sub["f_recov"], s_sub["rel_M1_HIGH_recov"], s=30, color=color,
                   label=f"Box {box}", edgecolor="black", lw=0.3)
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.axvline(1, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Sci_recov / Sci_obs (recovery factor f)")
    ax.set_ylabel("M1_HIGH residual % (with Sci_recov)")
    ax.set_title("Residual vs recovery factor, saturated bins\n(if grows with f → recovery over-fill)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: Time series of f and rel residual
    ax = axes[2]
    ax2 = ax.twinx()
    sub_a = box_agg[box_agg["box"]=="A"]
    ax.plot(sub_a["t_rel"], sub_a["f_recov"], "-", color="C3", lw=2, label="f = Sci_recov/Sci_obs")
    ax2.plot(sub_a["t_rel"], sub_a["rel_M1_HIGH_recov"], "-", color="C0", lw=2,
             label="M1_HIGH residual (%)")
    ax2.plot(sub_a["t_rel"], sub_a["rel_M1_HIGH_obs"], "--", color="C0", lw=1.5,
             label="M1_HIGH residual w/Sci_obs (%)", alpha=0.6)
    ax.set_xlabel("t-T0 [s]")
    ax.set_ylabel("f_recov", color="C3")
    ax2.set_ylabel("residual %", color="C0")
    ax.set_title("Box A: f and residual time series")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(1, color="gray", ls=":", lw=0.5)
    ax2.axhline(0, color="C0", ls=":", lw=0.5, alpha=0.5)

    fig.suptitle("GRB 260226A: diagnosing the +18% over-prediction in burst", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "analyze_260226A_residual.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
