#!/usr/bin/env python3
"""Spike forensics: for each box, plot ALL engineering+event time series
side-by-side in the −500 to 0s post-SAA window, and table-print the rates
at every >3σ residual spike.

Goal: see which channel (PHO / Wide / Large / dt / fill / obs) is anomalous
at the moment of model under-prediction.
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
T_MIN, T_MAX = -550.0, +30.0

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
    return pd.DataFrame({"met": met, "L_cyc": L_cyc, "length_s": length_s,
                          "PHO": pho, "Wide": csi, "Large": large, "Dt": dt,
                          "t_rel": met - TRIGGER_260})


def load_sci():
    df = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    df = df[df["type"]=="EVT"].copy()
    df["box"] = df["box"].astype(str)
    df["met"] = df["met"].astype(np.int64)
    aminfo = df["aminfo"].values.astype(np.int64)
    pc = np.zeros(len(aminfo), dtype=np.int32)
    for bit in range(18):
        pc += ((aminfo >> bit) & 1).astype(np.int32)
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


def main():
    print("Loading sci_obs + fill...")
    sci_agg = load_sci()
    fill = load_fill()

    fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.10})
    box_colors = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}

    all_spike_records = []
    for box_i, box in enumerate("ABC"):
        eng = load_eng_box(box)
        sci_b = sci_agg[sci_agg["box"]==box]
        fill_b = fill[fill["box"]==box]
        df = eng.merge(sci_b, on="met", how="left")
        df["Sci_obs"] = df["Sci_obs"].fillna(0).astype(int)
        df["Sci_ACD"] = df["Sci_ACD"].fillna(0).astype(int)
        df = df.merge(fill_b, on="met", how="left")
        df["fill"] = df["fill"].fillna(0).astype(int)
        df["Sci_recov"] = df["Sci_obs"] + df["fill"]
        for col in ["PHO","Wide","Large","Sci_obs","Sci_recov","fill","Sci_ACD"]:
            df[col+"_rate"] = df[col] / df["length_s"]
        df["dt_frac"] = df["Dt"] / df["L_cyc"] / 6   # box-avg, 6 dets summed
        df["r_local"] = np.where(df["Sci_obs"] > 0,
                                  df["Sci_ACD"] / df["Sci_obs"].clip(lower=1),
                                  0.22)
        c = COEFS[box]
        denom = c["c_pure"]*(1 - df["r_local"]) + c["c_ACD"]*df["r_local"]
        df["Sci_pred_rate"] = ((df["PHO_rate"]
                                  - c["beta"]*df["Wide_rate"]
                                  - c["gamma"]*df["Large_rate"]
                                  - c["b"]) / denom)
        df["res"] = df["Sci_pred_rate"] - df["Sci_recov_rate"]

        m = (df["t_rel"] >= T_MIN) & (df["t_rel"] <= T_MAX)
        d_zoom = df[m].sort_values("t_rel").copy()

        # Identify spikes (>3σ residual, calibrated on the active part)
        active = (d_zoom["Sci_obs_rate"] > 1000) & (d_zoom["t_rel"] > -490)
        sigma = d_zoom.loc[active, "res"].std()
        spike_mask = active & (np.abs(d_zoom["res"]) > 3 * sigma)
        print(f"\n=== Box {box}: σ_residual = {sigma:.1f} cnt/s/box, "
              f"{int(spike_mask.sum())} spikes >3σ ===")

        # Forensic records
        for _, r in d_zoom[spike_mask].iterrows():
            all_spike_records.append({
                "box": box, "t_rel": r["t_rel"],
                "PHO": r["PHO_rate"], "Wide": r["Wide_rate"],
                "Large": r["Large_rate"], "dt_frac": r["dt_frac"],
                "Sci_obs": r["Sci_obs_rate"], "fill": r["fill_rate"],
                "Sci_recov": r["Sci_recov_rate"],
                "Sci_pred": r["Sci_pred_rate"], "res": r["res"],
                "r_local": r["r_local"],
            })

        # ===== Plot for this box =====
        ax = axes[box_i]
        # primary y: rates
        ax.plot(d_zoom["t_rel"], d_zoom["PHO_rate"], color="purple",
                 lw=0.7, alpha=0.7, label="PHO rate")
        ax.plot(d_zoom["t_rel"], d_zoom["Sci_obs_rate"], color="black",
                 lw=0.7, alpha=0.7, label="Sci_obs")
        ax.plot(d_zoom["t_rel"], d_zoom["Sci_recov_rate"], color="red",
                 lw=0.7, alpha=0.5, label="Sci_recov")
        ax.plot(d_zoom["t_rel"], d_zoom["Sci_pred_rate"], color="green",
                 lw=0.7, alpha=0.7, label="Sci_pred (V8)")
        ax.plot(d_zoom["t_rel"], d_zoom["Large_rate"]*10, color="orange",
                 lw=0.5, alpha=0.4, label="Large × 10")
        ax.plot(d_zoom["t_rel"], d_zoom["Wide_rate"]*100, color="cyan",
                 lw=0.5, alpha=0.4, label="Wide × 100")
        ax.plot(d_zoom["t_rel"], d_zoom["fill_rate"]*5, color="magenta",
                 lw=0.5, alpha=0.6, label="fill × 5")
        # spikes
        ax.scatter(d_zoom.loc[spike_mask, "t_rel"],
                    d_zoom.loc[spike_mask, "Sci_recov_rate"],
                    s=50, color="red", marker="v", zorder=10,
                    edgecolor="black", linewidth=0.4,
                    label=f">3σ underpred spikes ({int(spike_mask.sum())})")
        # secondary y for dt
        ax_dt = ax.twinx()
        ax_dt.plot(d_zoom["t_rel"], d_zoom["dt_frac"]*100, color="brown",
                    lw=0.6, alpha=0.7, ls="--", label="dt/L %")
        ax_dt.set_ylabel("dt/L [%]", color="brown", fontsize=9)
        ax_dt.tick_params(axis="y", labelcolor="brown")

        ax.axvline(0, color="orange", ls="--", lw=1, alpha=0.5)
        ax.axvline(-500, color="purple", ls=":", lw=1, alpha=0.5)
        ax.set_xlim(T_MIN, T_MAX)
        ax.set_ylim(0, 14000)
        ax.set_ylabel(f"Box {box}\nrate [cnt/s/box]")
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
        if box_i == 0:
            ax.set_title(f"GRB 260226A spike forensics — engineering vs Sci channels in -500 ~ +30 s\n"
                          f"(red ▼ = >3σ under-prediction; dashed brown line = dt/L %; "
                          f"channels scaled to fit common y axis)",
                          fontsize=10)
        if box_i == 2:
            ax.set_xlabel("t − T₀ [s]")

    out = OUT_DIR / "260226A_spike_forensics.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    desktop = Path.home() / "Desktop" / out.name
    fig.savefig(desktop, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out}\n       {desktop}")

    # Forensic table of biggest spikes
    spike_df = pd.DataFrame(all_spike_records).sort_values("res").head(20)
    print("\n" + "="*120)
    print("Top 20 most-negative spikes (pred << recov):")
    print("="*120)
    print(f"  {'box':>3s} {'t_rel':>8s} {'PHO':>7s} {'Wide':>5s} {'Large':>6s} "
          f"{'dt%':>5s} {'Sci_obs':>8s} {'fill':>6s} {'Sci_rec':>8s} "
          f"{'Sci_pred':>9s} {'res':>7s} {'r_loc':>5s}")
    for _, r in spike_df.iterrows():
        print(f"  {r['box']:>3s} {r['t_rel']:>+8.0f} {r['PHO']:>7.0f} "
              f"{r['Wide']:>5.0f} {r['Large']:>6.0f} {r['dt_frac']*100:>4.2f}% "
              f"{r['Sci_obs']:>8.0f} {r['fill']:>6.0f} {r['Sci_recov']:>8.0f} "
              f"{r['Sci_pred']:>9.0f} {r['res']:>+7.0f} {r['r_local']:>5.3f}")


if __name__ == "__main__":
    main()
