#!/usr/bin/env python3
"""Zoom into the −500 to 0s post-SAA recovery window of 260226A.

3×3 panel layout:
  rows = Box A / B / C
  col 1: Sci light curves (obs, recov, V8 pred) — zoom −500 to +30s
  col 2: per-second ACD ratio (Sci_ACD_obs / Sci_obs) with 0.22 reference
  col 3: residual (V8_pred − Sci_recov) — highlight spikes
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
T_MIN, T_MAX = -550.0, +30.0          # zoom window in seconds relative to T0

# V8 per-box (from training; sufficient for box-level light curve)
COEFS = {
    "A": dict(b=25.0, c_pure=0.931, c_ACD=1.425, beta=2.888, gamma=1.084),
    "B": dict(b=30.0, c_pure=0.958, c_ACD=1.790, beta=2.849, gamma=0.952),
    "C": dict(b=35.9, c_pure=0.905, c_ACD=1.394, beta=2.961, gamma=1.102),
}


def load_eng_box(box):
    code = {"A":"0766","B":"1009","C":"1781"}[box]
    eng_file = f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits"
    fe = fits.open(eng_file, memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met = (d["Time"].astype(float) + offset + MET_CORRECTION).astype(np.int64)
    length_s = d["Length_Time_Cycle"].astype(float) * 16e-6
    pho = np.zeros(len(met)); csi = np.zeros(len(met)); large = np.zeros(len(met))
    for det in range(6):
        det_g = BOX_OFFSET[box] + det
        pho += d[f"Cnt_PHODet_{det_g}"]
        csi += d[f"Cnt_CsI_PHODet_{det_g}"]
        l_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)
        l_pho = d[f"Cnt_PHODet_{det_g}"].astype(float)
        large += unwrap_large(l_pho, l_raw)
    fe.close()
    df = pd.DataFrame({"met": met, "length_s": length_s,
                        "PHO": pho, "Wide": csi, "Large": large})
    df["t_rel"] = df["met"] - TRIGGER_260
    return df


def load_sci_obs_with_acd():
    """Per-(box, sec) Sci_obs + ACD-tagged count from solved.csv."""
    df = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    df = df[df["type"]=="EVT"].copy()
    df["box"] = df["box"].astype(str)
    df["met"] = df["met"].astype(np.int64)
    aminfo = df["aminfo"].values.astype(np.int64)
    popcount = np.zeros(len(aminfo), dtype=np.int32)
    for bit in range(18):
        popcount += ((aminfo >> bit) & 1).astype(np.int32)
    df["is_acd"] = (popcount > 0).astype("int32")
    agg = df.groupby(["box", "met"]).agg(
        Sci_obs=("is_acd", "size"),
        Sci_ACD=("is_acd", "sum"),
    ).reset_index()
    return agg


def load_fill_box():
    df = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    df = df[df["type"]=="FILL_GAP"]
    return df.groupby(["box", "met"]).size().rename("fill").reset_index()


def main():
    print("Loading data...")
    sci_agg = load_sci_obs_with_acd()
    fill = load_fill_box()

    fig, axes = plt.subplots(3, 3, figsize=(20, 12),
                              gridspec_kw={"width_ratios":[2.5, 1.5, 1.5]})
    boxes = ["A", "B", "C"]
    box_color = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}

    for row, box in enumerate(boxes):
        eng = load_eng_box(box)
        sci_b = sci_agg[sci_agg["box"]==box].rename(columns={"met":"met"})
        fill_b = fill[fill["box"]==box]

        # Merge per (met)
        df = eng.merge(sci_b, on="met", how="left")
        df["Sci_obs"]  = df["Sci_obs"].fillna(0).astype(int)
        df["Sci_ACD"]  = df["Sci_ACD"].fillna(0).astype(int)
        df = df.merge(fill_b, on="met", how="left")
        df["fill"]    = df["fill"].fillna(0).astype(int)
        df["Sci_recov"] = df["Sci_obs"] + df["fill"]

        # V8 box-level prediction
        c = COEFS[box]
        df["PHO_rate"]  = df["PHO"]   / df["length_s"]
        df["Wide_rate"] = df["Wide"]  / df["length_s"]
        df["Lrg_rate"]  = df["Large"] / df["length_s"]
        df["Sci_obs_rate"]   = df["Sci_obs"]   / df["length_s"]
        df["Sci_recov_rate"] = df["Sci_recov"] / df["length_s"]
        df["Sci_ACD_rate"]   = df["Sci_ACD"]   / df["length_s"]
        # Self-consistent inversion with local r
        df["r_local"] = np.where(df["Sci_obs"] > 0,
                                  df["Sci_ACD"] / df["Sci_obs"].clip(lower=1),
                                  0.22)
        denom = c["c_pure"]*(1.0 - df["r_local"]) + c["c_ACD"]*df["r_local"]
        df["Sci_pred_rate"] = (df["PHO_rate"]
                                - c["beta"]*df["Wide_rate"]
                                - c["gamma"]*df["Lrg_rate"]
                                - c["b"]) / denom

        # Restrict to box level (6 dets summed already in eng load)
        # Note: Sci_obs is summed across all 6 dets in this box

        # Zoom mask
        m = (df["t_rel"] >= T_MIN) & (df["t_rel"] <= T_MAX)
        d_zoom = df[m].sort_values("t_rel")

        # ---- col 1: light curves ----
        ax_lc = axes[row, 0]
        ax_lc.plot(d_zoom["t_rel"], d_zoom["Sci_obs_rate"], color="black",
                    lw=0.7, alpha=0.85, label="Sci_obs (1B raw)")
        ax_lc.plot(d_zoom["t_rel"], d_zoom["Sci_recov_rate"], color="red",
                    lw=0.7, alpha=0.7, label="Sci_recov (1B + fill)")
        ax_lc.plot(d_zoom["t_rel"], d_zoom["Sci_pred_rate"], color="green",
                    lw=0.7, alpha=0.7, label="Sci_pred (V8 from PHO/W/L)")
        ax_lc.axvline(0, color="orange", ls="--", lw=1, alpha=0.6, label="T₀ trigger")
        ax_lc.axvline(-500, color="purple", ls=":", lw=1, alpha=0.6, label="SAA exit")
        ax_lc.set_xlim(T_MIN, T_MAX)
        ax_lc.set_ylim(0, max(d_zoom["Sci_recov_rate"].max(), 1) * 1.05)
        ax_lc.set_ylabel(f"Box {box}\nrate [cnt/s/box]")
        ax_lc.set_title(f"Box {box}: Sci light curve (post-SAA recovery)" if row == 0 else "")
        if row == 2: ax_lc.set_xlabel("t − T₀ [s]")
        ax_lc.legend(loc="upper left", fontsize=8)
        ax_lc.grid(alpha=0.3)

        # ---- col 2: ACD ratio per second ----
        ax_ratio = axes[row, 1]
        valid = d_zoom["Sci_obs"] > 50    # need enough events for ratio to be meaningful
        ratio = d_zoom["Sci_ACD"] / d_zoom["Sci_obs"].clip(lower=1)
        ax_ratio.scatter(d_zoom.loc[valid, "t_rel"], ratio[valid],
                          s=2, color=box_color[box], alpha=0.5,
                          label="Sci_ACD / Sci_obs (per sec)")
        # Running median
        win = 5
        if valid.sum() > 2*win:
            rolling = ratio.where(valid).rolling(win, center=True, min_periods=1).median()
            ax_ratio.plot(d_zoom["t_rel"], rolling, color="black", lw=1.2,
                           label=f"{win}s median")
        ax_ratio.axhline(0.22, color="gray", ls="--", lw=1.0,
                          label="0.22 (trained avg)")
        ax_ratio.axvline(0, color="orange", ls="--", lw=1, alpha=0.6)
        ax_ratio.axvline(-500, color="purple", ls=":", lw=1, alpha=0.6)
        ax_ratio.set_xlim(T_MIN, T_MAX)
        ax_ratio.set_ylim(0.0, 0.5)
        ax_ratio.set_ylabel("ACD fraction")
        ax_ratio.set_title(f"Box {box}: ACD ratio  (deviation from 0.22 → activation)" if row == 0 else "")
        if row == 2: ax_ratio.set_xlabel("t − T₀ [s]")
        ax_ratio.legend(loc="upper right", fontsize=8)
        ax_ratio.grid(alpha=0.3)

        # ---- col 3: residual w/ spike highlights ----
        ax_res = axes[row, 2]
        d_zoom = d_zoom.copy()
        d_zoom["res_obs"]   = d_zoom["Sci_pred_rate"] - d_zoom["Sci_obs_rate"]
        d_zoom["res_recov"] = d_zoom["Sci_pred_rate"] - d_zoom["Sci_recov_rate"]
        ax_res.plot(d_zoom["t_rel"], d_zoom["res_obs"], color="black",
                     lw=0.6, alpha=0.55, label="pred − obs")
        ax_res.plot(d_zoom["t_rel"], d_zoom["res_recov"], color="blue",
                     lw=0.6, alpha=0.55, label="pred − recov")
        # Highlight spikes (|residual| > 3σ of recov residual in this window)
        sigma = d_zoom.loc[d_zoom["Sci_obs_rate"] > 1000, "res_recov"].std()
        spike_mask = np.abs(d_zoom["res_recov"]) > 3 * sigma
        ax_res.scatter(d_zoom.loc[spike_mask, "t_rel"],
                        d_zoom.loc[spike_mask, "res_recov"],
                        s=10, color="red", alpha=0.8, zorder=5,
                        label=f">3σ spikes (n={int(spike_mask.sum())})")
        ax_res.axhline(0, color="gray", lw=0.5)
        ax_res.axvline(0, color="orange", ls="--", lw=1, alpha=0.6)
        ax_res.axvline(-500, color="purple", ls=":", lw=1, alpha=0.6)
        ax_res.set_xlim(T_MIN, T_MAX)
        ax_res.set_ylabel("Δ [cnt/s/box]")
        ax_res.set_title(f"Box {box}: residual + spike detection" if row == 0 else "")
        if row == 2: ax_res.set_xlabel("t − T₀ [s]")
        ax_res.legend(loc="upper left", fontsize=8)
        ax_res.grid(alpha=0.3)

    fig.suptitle(
        f"GRB 260226A — zoom into post-SAA recovery period ({int(T_MIN)} ≤ t−T₀ ≤ {int(T_MAX)} s)",
        fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "260226A_zoom_SAA_recovery.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    desktop = Path.home() / "Desktop" / out.name
    fig.savefig(desktop, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}\n       {desktop}")


if __name__ == "__main__":
    main()
