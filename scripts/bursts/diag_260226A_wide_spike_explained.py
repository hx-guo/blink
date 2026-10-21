#!/usr/bin/env python3
"""Visual explanation of why under-prediction spikes happen:
The β·Wide subtraction in the inversion formula overcorrects when Wide spikes briefly.

3-panel diagnostic:
  (a) Time series around C +12s spike — show Wide spike and resulting Sci_pred dip
  (b) Prediction budget bar chart — decompose what makes Sci_pred vs Sci_recov
  (c) Wide_rate vs residual scatter — show the population-wide trend
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
TRIGGER_260 = 446726273.0
MET_CORRECTION = 4.0

COEFS = {
    "A": dict(b=25.0, c_pure=0.931, c_ACD=1.425, beta=2.888, gamma=1.084),
    "B": dict(b=30.0, c_pure=0.958, c_ACD=1.790, beta=2.849, gamma=0.952),
    "C": dict(b=35.9, c_pure=0.905, c_ACD=1.394, beta=2.961, gamma=1.102),
}


def load_box(box):
    code = {"A":"0766","B":"1009","C":"1781"}[box]
    eng_file = f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits"
    fe = fits.open(eng_file, memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met = (d["Time"].astype(float) + offset + MET_CORRECTION).astype(np.int64)
    L_cyc = d["Length_Time_Cycle"].astype(float)
    length_s = L_cyc * 16e-6
    pho = np.zeros(len(met)); csi = np.zeros(len(met))
    large = np.zeros(len(met)); dt = np.zeros(len(met))
    for det in range(6):
        det_g = BOX_OFFSET[box] + det
        pho += d[f"Cnt_PHODet_{det_g}"]
        csi += d[f"Cnt_CsI_PHODet_{det_g}"]
        l_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)
        l_pho = d[f"Cnt_PHODet_{det_g}"].astype(float)
        large += unwrap_large(l_pho, l_raw)
        dt += d[f"DeadTime_PHODet_{det_g}"]
    fe.close()
    return pd.DataFrame({"met": met, "length_s": length_s,
                          "PHO": pho, "Wide": csi, "Large": large, "Dt": dt,
                          "L_cyc": L_cyc,
                          "t_rel": met - TRIGGER_260})


def load_sci():
    df = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    df = df[df["type"]=="EVT"].copy()
    df["box"] = df["box"].astype(str); df["met"] = df["met"].astype(np.int64)
    aminfo = df["aminfo"].values.astype(np.int64)
    pc = np.zeros(len(aminfo), dtype=np.int32)
    for bit in range(18): pc += ((aminfo >> bit) & 1).astype(np.int32)
    df["is_acd"] = (pc > 0).astype("int32")
    return df.groupby(["box", "met"]).agg(
        Sci_obs=("is_acd", "size"),
        Sci_ACD=("is_acd", "sum"),
    ).reset_index()


def load_fill():
    df = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    df = df[df["type"]=="FILL_GAP"]
    return df.groupby(["box", "met"]).size().rename("fill").reset_index()


def build_df(box, sci_agg, fill):
    eng = load_box(box)
    df = eng.merge(sci_agg[sci_agg["box"]==box], on="met", how="left")
    df["Sci_obs"] = df["Sci_obs"].fillna(0).astype(int)
    df["Sci_ACD"] = df["Sci_ACD"].fillna(0).astype(int)
    df = df.merge(fill[fill["box"]==box], on="met", how="left")
    df["fill"] = df["fill"].fillna(0).astype(int)
    df["Sci_recov"] = df["Sci_obs"] + df["fill"]
    for col in ["PHO","Wide","Large","Sci_obs","Sci_recov","Sci_ACD"]:
        df[col+"_rate"] = df[col] / df["length_s"]
    df["r"] = np.where(df["Sci_obs"]>0,
                        df["Sci_ACD"]/df["Sci_obs"].clip(lower=1), 0.22)
    c = COEFS[box]
    df["denom"] = c["c_pure"]*(1-df["r"]) + c["c_ACD"]*df["r"]
    df["beta_Wide"]  = c["beta"]  * df["Wide_rate"]
    df["gamma_Large"] = c["gamma"] * df["Large_rate"]
    df["Sci_pred"] = (df["PHO_rate"] - df["beta_Wide"]
                       - df["gamma_Large"] - c["b"]) / df["denom"]
    df["res"] = df["Sci_pred"] - df["Sci_recov_rate"]
    return df


def main():
    print("Loading...")
    sci_agg = load_sci(); fill = load_fill()
    dfs = {b: build_df(b, sci_agg, fill) for b in "ABC"}

    # Single-panel: time-series zoom around C +12s spike
    fig, ax = plt.subplots(figsize=(12, 8))
    box, t_spike = "C", 12
    df = dfs[box]
    m = (df["t_rel"] >= t_spike - 10) & (df["t_rel"] <= t_spike + 10)
    d = df[m].sort_values("t_rel")
    ax.plot(d["t_rel"], d["PHO_rate"],       color="purple", lw=1.5, marker="o", ms=6, label="PHO rate")
    ax.plot(d["t_rel"], d["Sci_recov_rate"], color="red",    lw=1.5, marker="o", ms=6, label="Sci_recov (truth)")
    ax.plot(d["t_rel"], d["Sci_pred"],       color="green",  lw=1.5, marker="o", ms=6, label="Sci_pred (V8)")
    ax.plot(d["t_rel"], d["Wide_rate"]*5,    color="cyan",   lw=1.5, marker="s", ms=6, label="Wide × 5")
    ax.plot(d["t_rel"], d["Large_rate"],     color="orange", lw=1.2, alpha=0.7, marker="d", ms=5, label="Large")
    ax.axvline(t_spike, color="red", ls="--", lw=1.5, alpha=0.6, label=f"t = +{t_spike}s spike")
    ax.set_xlabel("t − T₀ [s]")
    ax.set_ylabel("rate [cnt/s/box]")
    ax.set_title(f"Box {box} — 10-second zoom around the worst under-prediction spike")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    # Annotation
    row = d[d["t_rel"]==t_spike].iloc[0]
    ax.annotate(
        f"At t = +{t_spike}s:\n"
        f"  Wide jumps {int(d.loc[d['t_rel']==t_spike-1, 'Wide_rate'].iloc[0])} → {int(row['Wide_rate'])} cnt/s/box\n"
        f"  β·Wide subtraction = {int(row['beta_Wide'])}\n"
        f"  ⇒ Sci_pred crashes to {int(row['Sci_pred'])}\n"
        f"     while Sci_recov stays at {int(row['Sci_recov_rate'])}",
        xy=(t_spike, row["Sci_pred"]), xytext=(t_spike - 9, 12000),
        fontsize=11, ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                   ec="orange", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))
    out = OUT_DIR / "260226A_wide_spike_explained.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    desktop = Path.home() / "Desktop" / out.name
    fig.savefig(desktop, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}\n       {desktop}")


if __name__ == "__main__":
    main()
