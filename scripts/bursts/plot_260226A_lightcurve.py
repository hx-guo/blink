#!/usr/bin/env python3
"""260226A 1-hour light curve: Sci_obs / Sci_recov / Sci_pred per box.

All rates normalized by Length_Time_Cycle × 16μs (~0.94s per second).

Sci_pred uses M7-merged with assumed ACD ratio (22%) baked into c_eff:
  c_eff = c_pure + α_ACD × (c_ACD - c_pure)
  Sci_pred = (PHO - β·Wide - γ·Large - b) / c_eff
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
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
TRIGGER_260 = 446726273.0
MET_CORRECTION = 4.0

# M7-merged-ACD coefficients (CLEAN band, per-box)
COEFS = {
    "A": dict(b=25.0, c_pure=0.931, c_ACD=1.425, beta=2.888, gamma=1.084),
    "B": dict(b=30.0, c_pure=0.958, c_ACD=1.790, beta=2.849, gamma=0.952),
    "C": dict(b=35.9, c_pure=0.905, c_ACD=1.394, beta=2.961, gamma=1.102),
}
ALPHA_ACD = 0.22  # assumed Sci_ACD / Sci ratio (avg from quiet observations)


def load_eng():
    rows = []
    for box, code in [("A","0766"),("B","1009"),("C","1781")]:
        eng_file = f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits"
        fe = fits.open(eng_file, memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
        length_cyc = d["Length_Time_Cycle"].astype(float)
        length_s = length_cyc * 16e-6
        for det_local in range(6):
            det_g = BOX_OFFSET[box] + det_local
            pho = d[f"Cnt_PHODet_{det_g}"].astype(float)
            csi = d[f"Cnt_CsI_PHODet_{det_g}"].astype(float)
            large_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                rows.append({
                    "box": box, "det": det_local,
                    "met_sec": int(met_eng[i]),
                    "length_s": length_s[i],
                    "PHO": pho[i], "Wide": csi[i], "Large": large[i],
                })
        fe.close()
    return pd.DataFrame(rows)


def load_sci():
    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    sci_obs = sci_obs[sci_obs["type"]=="EVT"]
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    obs_per_det = sci_obs.groupby(["box","det_id","met_sec"]).size().rename("Sci_obs_count").reset_index()
    obs_per_det = obs_per_det.rename(columns={"det_id":"det"})

    sci_rec = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    sci_rec["box"] = sci_rec["box"].astype(str)
    sci_rec["met_sec"] = sci_rec["met"].astype("int64")
    fill_box = sci_rec[sci_rec["type"]=="FILL_GAP"].groupby(["box","met_sec"]).size().rename("Sci_fill_count").reset_index()
    return obs_per_det, fill_box


def main():
    print("Loading engineering...")
    eng = load_eng()
    print(f"  eng rows: {len(eng):,}")

    print("Loading Sci obs and fills...")
    obs_per_det, fill_box = load_sci()

    # Merge
    df = eng.merge(obs_per_det, on=["box","det","met_sec"], how="left")
    df["Sci_obs_count"] = df["Sci_obs_count"].fillna(0)
    df = df.merge(fill_box, on=["box","met_sec"], how="left")
    df["Sci_fill_count"] = df["Sci_fill_count"].fillna(0)

    # Apportion fill events to dets proportional to observed
    box_obs_count = df.groupby(["box","met_sec"])["Sci_obs_count"].transform("sum")
    df["Sci_obs_count_share"] = np.where(box_obs_count > 0,
                                          df["Sci_obs_count"] / box_obs_count.clip(lower=1),
                                          1.0/6)
    df["Sci_recov_count"] = df["Sci_obs_count"] + df["Sci_fill_count"] * df["Sci_obs_count_share"]

    # Rates per det (normalized by actual length_s ≈ 0.94s)
    df["sci_obs_rate"]   = df["Sci_obs_count"]   / df["length_s"]
    df["sci_recov_rate"] = df["Sci_recov_count"] / df["length_s"]
    df["pho_rate"]   = df["PHO"]   / df["length_s"]
    df["wide_rate"]  = df["Wide"]  / df["length_s"]
    df["large_rate"] = df["Large"] / df["length_s"]
    df["t_rel"] = df["met_sec"] - TRIGGER_260

    # Predicted Sci rate using c_eff approach:
    #   Assume Sci_ACD = α·Sci_total, so PHO ≈ c_eff·Sci + β·Wide + γ·Large + b
    #   c_eff = c_pure + α·(c_ACD - c_pure)
    df["box_str"] = df["box"].astype(str)
    c_pure = df["box_str"].map(lambda b: COEFS[b]["c_pure"]).values
    c_ACD  = df["box_str"].map(lambda b: COEFS[b]["c_ACD"]).values
    beta   = df["box_str"].map(lambda b: COEFS[b]["beta"]).values
    gamma  = df["box_str"].map(lambda b: COEFS[b]["gamma"]).values
    b      = df["box_str"].map(lambda b: COEFS[b]["b"]).values
    c_eff  = c_pure + ALPHA_ACD * (c_ACD - c_pure)
    df["sci_pred_rate"] = (df["pho_rate"].values - beta*df["wide_rate"].values
                            - gamma*df["large_rate"].values - b) / c_eff

    # Aggregate per box per second (sum 6 dets)
    box_agg = df.groupby(["box","met_sec","t_rel"]).agg(
        sci_obs   = ("sci_obs_rate",   "sum"),
        sci_recov = ("sci_recov_rate", "sum"),
        sci_pred  = ("sci_pred_rate",  "sum"),
    ).reset_index()

    # ============ Plot: 3 boxes × (light curve + residual) ============
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(6, 1, height_ratios=[3, 1, 3, 1, 3, 1], hspace=0.08)

    for row_pair, box in enumerate("ABC"):
        sub = box_agg[box_agg["box"]==box].sort_values("t_rel")
        ax_lc = fig.add_subplot(gs[2*row_pair])
        ax_res = fig.add_subplot(gs[2*row_pair + 1], sharex=ax_lc)

        # Light curve panel
        ax_lc.step(sub["t_rel"], sub["sci_pred"], where="post",
                    color="green", lw=0.7, alpha=0.8, label="Sci predicted (M7 from PHO,W,L)")
        ax_lc.step(sub["t_rel"], sub["sci_obs"], where="post",
                    color="black", lw=0.9, label="Sci observed (1B raw)")
        ax_lc.step(sub["t_rel"], sub["sci_recov"], where="post",
                    color="red", lw=0.7, alpha=0.8, label="Sci recovered (1B + fill)")
        ax_lc.set_ylabel(f"Box {box}\n[cnt/s/box]")
        ax_lc.legend(fontsize=8, loc="upper right")
        ax_lc.grid(alpha=0.3)
        ax_lc.set_xlim(-2200, 1300)
        ax_lc.tick_params(axis="x", labelbottom=False)

        # Residual panel: both relative to prediction
        res_pred_obs   = sub["sci_pred"] - sub["sci_obs"]
        res_pred_recov = sub["sci_pred"] - sub["sci_recov"]
        ax_res.step(sub["t_rel"], res_pred_obs, where="post",
                    color="black", lw=0.7, alpha=0.8, label="pred − obs")
        ax_res.step(sub["t_rel"], res_pred_recov, where="post",
                    color="blue", lw=0.7, alpha=0.8, label="pred − recov")
        ax_res.axhline(0, color="k", ls=":", lw=0.6)
        ax_res.set_ylabel("Δ [cnt/s/box]")
        ax_res.legend(fontsize=7, loc="upper right", ncol=2)
        ax_res.grid(alpha=0.3)
        ax_res.set_xlim(-2200, 1300)
        if row_pair < 2:
            ax_res.tick_params(axis="x", labelbottom=False)
        else:
            ax_res.set_xlabel(f"Time since trigger T0 [s]  (T0 = 2026-02-26T10:37:53 UTC)")

    fig.suptitle("GRB 260226A: Sci light curves + residuals — observed vs recovered vs M7-predicted",
                 fontsize=13, y=0.995)
    out = OUT_DIR / "260226A_lightcurve_3way.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
