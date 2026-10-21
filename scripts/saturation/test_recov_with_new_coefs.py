#!/usr/bin/env python3
"""Apply NEW (clean-band) coefficients to 260226A burst region with both
Sci_obs and Sci_recov. If recovery is correct: M1_NEW(Sci_recov) ≈ PHO_obs.

Uses NEW coefficients from test_fifo_contamination.py:
  Box A: b=+25.0, 1+α=0.884, β=3.019, γ=1.263
  Box B: b=+29.1, 1+α=0.878, β=2.897, γ=1.263
  Box C: b=+34.7, 1+α=0.860, β=3.110, γ=1.278
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

# NEW coefficients (fit on tightened HIGH-mode band [400,1000] + box rate cap)
M1_NEW = {
    "A": dict(b=+25.0, alpha1=0.884, beta=3.019, gamma=1.263),
    "B": dict(b=+29.1, alpha1=0.878, beta=2.897, gamma=1.263),
    "C": dict(b=+34.7, alpha1=0.860, beta=3.110, gamma=1.278),
}
M1_OLD = {
    "A": dict(b=-103.5, alpha1=1.3705, beta=3.2589, gamma=0.7627),
    "B": dict(b=-127.2, alpha1=1.4565, beta=3.0297, gamma=0.6954),
    "C": dict(b=-129.2, alpha1=1.4284, beta=3.2752, gamma=0.7334),
}


def pred_m1(coefs, sci, wide, large):
    return coefs["b"] + coefs["alpha1"]*sci + coefs["beta"]*wide + coefs["gamma"]*large


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

    # Apply OLD and NEW coefficients with both Sci_obs and Sci_recov
    for label, coefs in [("OLD", M1_OLD), ("NEW", M1_NEW)]:
        for sci_kind, sci_col in [("obs","Sci_obs"),("recov","Sci_recov")]:
            preds = []
            for _, r in box_agg.iterrows():
                preds.append(pred_m1(coefs[r.box], r[sci_col], r.Wide, r.Large))
            box_agg[f"pred_{label}_{sci_kind}"] = preds
            box_agg[f"rel_{label}_{sci_kind}"] = 100*(box_agg[f"pred_{label}_{sci_kind}"]
                                                       - box_agg["PHO"]) / box_agg["PHO"]

    # ============ Print burst window comparisons ============
    print(f"\n=== Burst window comparison (Box A, M1 OLD vs NEW × Sci_obs vs Sci_recov) ===")
    print(f"  {'t':>4s} {'Sci_obs':>8s} {'Sci_recov':>10s} {'f':>5s} {'PHO_obs':>9s}  "
          f"{'OLD_obs':>8s} {'OLD_rec':>8s}  {'NEW_obs':>8s} {'NEW_rec':>8s}")
    sub_a = box_agg[(box_agg["box"]=="A") & (box_agg["t_rel"]>=18) & (box_agg["t_rel"]<=35)]
    for _, r in sub_a.iterrows():
        print(f"  {r.t_rel:>+4.0f} {r.Sci_obs:>8.0f} {r.Sci_recov:>10.0f} {r.f:>5.2f} "
              f"{r.PHO:>9.0f}  {r.rel_OLD_obs:>+7.1f}% {r.rel_OLD_recov:>+7.1f}%  "
              f"{r.rel_NEW_obs:>+7.1f}% {r.rel_NEW_recov:>+7.1f}%")

    # ============ Summary statistics ============
    print(f"\n=== Saturated bins (f > 1.05, t ∈ [22, 29]) ===")
    sat = box_agg[(box_agg["f"]>1.05) & (box_agg["t_rel"]>=22) & (box_agg["t_rel"]<=29)]
    for box in "ABC":
        sub = sat[sat["box"]==box]
        if len(sub) == 0: continue
        print(f"\n  Box {box} ({len(sub)} bins):")
        for tag in ["OLD_obs","OLD_recov","NEW_obs","NEW_recov"]:
            med = sub[f"rel_{tag}"].median()
            print(f"    M1 {tag:>10s}: median rel = {med:+.1f}%")

    print(f"\n=== Quiet high-rate bins (Sci > 7000, no FIFO drop) ===")
    quiet = box_agg[(box_agg["Sci_obs"]>7000) & (box_agg["f"]<1.05)]
    for box in "ABC":
        sub = quiet[quiet["box"]==box]
        if len(sub) == 0: continue
        print(f"\n  Box {box} ({len(sub)} bins):")
        for tag in ["OLD_obs","NEW_obs"]:
            med = sub[f"rel_{tag}"].median()
            print(f"    M1 {tag:>10s}: median rel = {med:+.1f}%")

    # ============ Plot ============
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey="row")
    for col, box in enumerate("ABC"):
        sub = box_agg[box_agg["box"]==box].sort_values("t_rel")
        # Top: OLD coefficients
        ax = axes[0, col]
        ax.plot(sub["t_rel"], sub["rel_OLD_obs"], "-", color="C0", lw=1.5,
                label="M1 OLD (Sci_obs)")
        ax.plot(sub["t_rel"], sub["rel_OLD_recov"], "-", color="C3", lw=1.5,
                label="M1 OLD (Sci_recov)")
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.axvspan(22, 29, alpha=0.1, color="orange")
        ax.set_ylabel("relative residual (%)")
        ax.set_title(f"Box {box}: M1 OLD coefficients")
        ax.set_ylim(-30, 40)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        # Bottom: NEW coefficients
        ax = axes[1, col]
        ax.plot(sub["t_rel"], sub["rel_NEW_obs"], "-", color="C0", lw=1.5,
                label="M1 NEW (Sci_obs)")
        ax.plot(sub["t_rel"], sub["rel_NEW_recov"], "-", color="C3", lw=1.5,
                label="M1 NEW (Sci_recov)")
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.axvspan(22, 29, alpha=0.1, color="orange")
        ax.set_xlabel("t-T0 [s]")
        ax.set_ylabel("relative residual (%)")
        ax.set_title(f"Box {box}: M1 NEW coefficients (clean band fit)")
        ax.set_ylim(-30, 40)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("260226A: M1 OLD vs NEW × Sci_obs vs Sci_recov\n"
                 "If recovery is correct: NEW + Sci_recov should give residual ≈ 0",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "test_recov_with_new_coefs.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
