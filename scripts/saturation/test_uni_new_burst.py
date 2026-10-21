#!/usr/bin/env python3
"""Test if M_uni_M1 NEW (fit on tightened clean band [400,1000]/det,
group_rate < 6000) extrapolates better to burst peak than plain M1 NEW.

NEW band coefficients (from test_fifo_contamination.py):
  M1 NEW:    b, 1+α, β, γ
  M_uni NEW: b, 1+α, β, γ_0, γ_1   (γ_1·Large²/Sci interaction)

Apply both to 260226A with Sci_recov. If M_uni NEW has smaller residual at
burst peak than M1 NEW, the Large²/Sci term carries real extrapolation
information. If both have same residual, the interaction term doesn't help.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
TRIGGER_MET = 446726273.0
MET_CORRECTION = 4.0
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

# NEW (clean band) coefficients
M1_NEW = {
    "A": dict(b=+25.0, alpha1=0.884, beta=3.019, gamma=1.263),
    "B": dict(b=+29.1, alpha1=0.878, beta=2.897, gamma=1.263),
    "C": dict(b=+34.7, alpha1=0.860, beta=3.110, gamma=1.278),
}

M_UNI_NEW = {  # (b, 1+α, β, γ_0, γ_1)
    "A": (27.2, 0.946, 3.017, 1.071, 0.135),
    "B": (35.2, 1.024, 2.895, 0.813, 0.314),
    "C": (37.2, 0.917, 3.107, 1.099, 0.127),
}


def pred_m1(coefs, sci, wide, large):
    return coefs["b"] + coefs["alpha1"]*sci + coefs["beta"]*wide + coefs["gamma"]*large


def pred_uni(coefs, sci, wide, large):
    b, c1plus, beta, g0, g1 = coefs
    return b + c1plus*sci + beta*wide + g0*large + g1*large**2 / np.maximum(sci, 1)


def load_data():
    rows = []
    for box, code in [("A","0766"),("B","1009"),("C","1781")]:
        eng_file = f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits"
        fe = fits.open(eng_file, memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
        mask = (met_eng >= TRIGGER_MET - 30) & (met_eng <= TRIGGER_MET + 70)
        met_eng = met_eng[mask]
        for det_local in range(6):
            det_g = BOX_OFFSET[box] + det_local
            pho = d[f"Cnt_PHODet_{det_g}"].astype(float)[mask]
            csi = d[f"Cnt_CsI_PHODet_{det_g}"].astype(float)[mask]
            large_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)[mask]
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                rows.append({"box":box,"det":det_local,"met_sec":int(met_eng[i]),
                             "PHO":pho[i],"Wide":csi[i],"Large":large[i]})
        fe.close()
    eng = pd.DataFrame(rows)

    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    sci_obs = sci_obs[sci_obs["type"]=="EVT"]
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    sci_obs = sci_obs.groupby(["box","det_id","met_sec"]).size().rename("Sci_obs").reset_index()
    sci_obs = sci_obs.rename(columns={"det_id":"det"})

    sci_rec = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    sci_rec = sci_rec[sci_rec["type"].isin(["EVT","FILL_GAP"])].copy()
    sci_rec["box"] = sci_rec["box"].astype(str)
    sci_rec["met_sec"] = sci_rec["met"].astype("int64")
    sci_rec_box = sci_rec.groupby(["box","met_sec"]).size().rename("Sci_recov_box").reset_index()

    df = eng.merge(sci_obs, on=["box","det","met_sec"], how="left")
    df["Sci_obs"] = df["Sci_obs"].fillna(0)
    df = df.merge(sci_rec_box, on=["box","met_sec"], how="left")
    df["Sci_recov_box"] = df["Sci_recov_box"].fillna(0)
    box_obs_sum = df.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    df["Sci_recov"] = np.where(box_obs_sum > 0,
                                df["Sci_recov_box"] * df["Sci_obs"]/box_obs_sum.clip(lower=1),
                                df["Sci_recov_box"]/6)
    df["t_rel"] = df["met_sec"] - TRIGGER_MET
    return df


def main():
    df = load_data()
    box_agg = df.groupby(["box","t_rel"]).agg(
        PHO=("PHO","sum"), Wide=("Wide","sum"), Large=("Large","sum"),
        Sci_obs=("Sci_obs","sum"), Sci_recov=("Sci_recov","sum"),
    ).reset_index()
    box_agg["f"] = box_agg["Sci_recov"] / box_agg["Sci_obs"].clip(lower=1)
    box_agg["Large_over_Sci"] = box_agg["Large"] / box_agg["Sci_obs"].clip(lower=1)
    box_agg["Large_over_Sci_recov"] = box_agg["Large"] / box_agg["Sci_recov"].clip(lower=1)

    # Apply both models with Sci_obs and Sci_recov
    for label_m1, label_uni, sci_col in [
        ("m1_obs","uni_obs","Sci_obs"),
        ("m1_recov","uni_recov","Sci_recov"),
    ]:
        m1_preds, uni_preds = [], []
        for _, r in box_agg.iterrows():
            m1_preds.append(pred_m1(M1_NEW[r.box], r[sci_col], r.Wide, r.Large))
            uni_preds.append(pred_uni(M_UNI_NEW[r.box], r[sci_col], r.Wide, r.Large))
        box_agg[f"pred_{label_m1}"] = m1_preds
        box_agg[f"pred_{label_uni}"] = uni_preds
        box_agg[f"rel_{label_m1}"] = 100*(box_agg[f"pred_{label_m1}"]-box_agg["PHO"])/box_agg["PHO"]
        box_agg[f"rel_{label_uni}"] = 100*(box_agg[f"pred_{label_uni}"]-box_agg["PHO"])/box_agg["PHO"]

    print(f"\n=== Burst window comparison: M1 NEW vs M_uni NEW ===")
    print(f"  Both use Sci_recov, both fit on Sci ∈ [400, 1000] per det (clean training).")
    print(f"  {'t':>4s} {'Sci_obs':>8s} {'Sci_recov':>10s} {'f':>5s} {'L/S_rec':>8s} "
          f"{'M1 obs':>8s} {'M1 rec':>8s} {'Uni obs':>9s} {'Uni rec':>9s}")
    for box in "ABC":
        sub_b = box_agg[box_agg["box"]==box]
        for t in range(18, 36):
            row = sub_b[sub_b["t_rel"]==t]
            if len(row) == 0: continue
            r = row.iloc[0]
            print(f"  Box {box} {r.t_rel:>+4.0f} {r.Sci_obs:>8.0f} {r.Sci_recov:>10.0f} "
                  f"{r.f:>5.2f} {r.Large_over_Sci_recov:>8.3f} "
                  f"{r.rel_m1_obs:>+7.1f}% {r.rel_m1_recov:>+7.1f}% "
                  f"{r.rel_uni_obs:>+8.1f}% {r.rel_uni_recov:>+8.1f}%")

    # Summary in burst saturated bins
    print(f"\n=== Saturated bins (f > 1.05, t ∈ [22, 29]) ===")
    sat = box_agg[(box_agg["f"]>1.05) & (box_agg["t_rel"]>=22) & (box_agg["t_rel"]<=29)]
    for box in "ABC":
        sub = sat[sat["box"]==box]
        if len(sub) == 0: continue
        print(f"\n  Box {box} ({len(sub)} bins):")
        for tag in ["m1_recov", "uni_recov"]:
            med = sub[f"rel_{tag}"].median()
            rms = np.sqrt((sub[f"rel_{tag}"]**2).mean())
            print(f"    {tag:>12s}: median {med:+.2f}%, RMS {rms:.2f}%")

    # Quiet high-rate bins
    print(f"\n=== Quiet high-rate bins (Sci > 7000, no FIFO drop) ===")
    quiet = box_agg[(box_agg["Sci_obs"]>7000) & (box_agg["f"]<1.05)]
    for box in "ABC":
        sub = quiet[quiet["box"]==box]
        if len(sub) == 0: continue
        print(f"\n  Box {box} ({len(sub)} bins):")
        for tag in ["m1_obs", "uni_obs"]:
            med = sub[f"rel_{tag}"].median()
            rms = np.sqrt((sub[f"rel_{tag}"]**2).mean())
            print(f"    {tag:>12s}: median {med:+.2f}%, RMS {rms:.2f}%")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey="row")
    for col, box in enumerate("ABC"):
        sub = box_agg[box_agg["box"]==box].sort_values("t_rel")
        # Top: with Sci_recov
        ax = axes[0, col]
        ax.plot(sub["t_rel"], sub["rel_m1_recov"], "-", color="C0", lw=1.8,
                label="M1 NEW (Sci_recov)")
        ax.plot(sub["t_rel"], sub["rel_uni_recov"], "-", color="C3", lw=1.8,
                label="M_uni NEW (Sci_recov)")
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.axvspan(22, 29, alpha=0.1, color="orange")
        ax.set_ylabel("relative residual (%)")
        ax.set_title(f"Box {box}: prediction with Sci_recov")
        ax.set_ylim(-15, 30)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        # Bottom: with Sci_obs
        ax = axes[1, col]
        ax.plot(sub["t_rel"], sub["rel_m1_obs"], "-", color="C0", lw=1.8,
                label="M1 NEW (Sci_obs)")
        ax.plot(sub["t_rel"], sub["rel_uni_obs"], "-", color="C3", lw=1.8,
                label="M_uni NEW (Sci_obs)")
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.axvspan(22, 29, alpha=0.1, color="orange")
        ax.set_xlabel("t-T0 [s]")
        ax.set_ylabel("relative residual (%)")
        ax.set_title(f"Box {box}: prediction with Sci_obs (saturated)")
        ax.set_ylim(-30, 15)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("260226A: M1 NEW vs M_uni NEW (γ_1·Large²/Sci) extrapolation to burst",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "test_uni_new_burst.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
