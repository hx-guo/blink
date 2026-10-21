#!/usr/bin/env python3
"""Validate M_uni_M7 (and simpler variants) on GRB 260226A.

Workflow:
  1. Read 1B engineering FITS (PHO, CsI=Wide, Large, Dt per det per second)
  2. Read solve output (1B EVT times per det per second → Sci_obs)
  3. Read reconstruct output (gap-filled events at box level → Sci_recovered)
  4. Apply 3 models, plot:
     M_obs    : prediction using OBSERVED Sci (should match observed PHO if
                saturation hits Sci and PHO equally)
     M_recov  : prediction using RECOVERED Sci (should EXCEED observed PHO
                during saturation, matching the 'true' rate)
     M_diff   : (M_recov - M_obs) shows the saturation impact magnitude

Models tested (per Box, fit on quiet 2017-2019 main band):
  - Multiplicative: PHO·live = k·Sci + Wide + Large
  - M_uni_M1:       PHO = (1+α)Sci + β·Wide + γ_0·Large + γ_1·Large²/Sci + b
  - M_uni_M7 needs Sci_ACD breakdown (not done here; would need 1K data)
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

TRIGGER_MET = 446726273.0  # CLI-reported trigger (header T0)
MET_CORRECTION = 4.0       # from existing script

# Window: -30 to +70 s around trigger
T_LO = TRIGGER_MET - 30
T_HI = TRIGGER_MET + 70

BOXES = [
    ("A", "0766", 0),
    ("B", "1009", 6),
    ("C", "1781", 12),
]

# Multiplicative model: k_mul ≈ 1.36 (Box A/B/C consistent from m0_original_models.py)
K_MUL = {"A": 1.358, "B": 1.381, "C": 1.380}

# M_uni_M1 coefficients from m13_mode_aware.py (fit on clean band Sci ∈ [400, 1500])
# Note these were per-det rate fits, so each det has its own. Box-level avg:
M_UNI_M1 = {
    # Box: (b, 1+α, β, γ_0, γ_1)
    "A": (-245.2, 2.6764, 2.8533, -2.3310, 2.1252),
    "B": (-279.8, 2.7438, 2.6459, -2.2915, 2.0585),
    "C": (-277.0, 2.7019, 2.9011, -2.2357, 2.0431),
}


def load_eng_fits(box_letter, code):
    """Load 1B engineering FITS for one box. Returns DataFrame indexed by met_eng."""
    eng_file = (Path("data/1B/2026/20260226") / code
                / f"HXMT_1B_{code}_20260226T100000_G076262_000_004.fits")
    fe = fits.open(eng_file, memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
    length_cyc = d["Length_Time_Cycle"].astype(float)
    length_s = length_cyc * 16e-6

    # Window selection
    mask = (met_eng >= T_LO) & (met_eng <= T_HI)
    met_eng = met_eng[mask]
    length_s = length_s[mask]
    length_cyc = length_cyc[mask]

    # Per-det counters
    rows = []
    for det_local in range(6):
        det_global = BOX_OFFSET[box_letter] + det_local
        pho   = d[f"Cnt_PHODet_{det_global}"].astype(float)[mask]
        csi   = d[f"Cnt_CsI_PHODet_{det_global}"].astype(float)[mask]
        dead  = d[f"DeadTime_PHODet_{det_global}"].astype(float)[mask]
        large_raw = d[f"Cnt_LargeEvt_{det_global}"].astype(float)[mask]
        large = unwrap_large(pho, large_raw)
        for i in range(len(met_eng)):
            rows.append({
                "box": box_letter,
                "det": det_local,
                "met_int": int(met_eng[i]),  # round to 1s bin
                "met_eng": met_eng[i],
                "length_s": length_s[i],
                "length_cyc": length_cyc[i],
                "PHO": pho[i],
                "Wide": csi[i],   # CsI = Wide
                "Large": large[i],
                "Dt": dead[i],
            })
    df = pd.DataFrame(rows)
    fe.close()
    return df


BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


def load_solved_events():
    """Load solve CSV (has det_id), per-second per-det event counts."""
    df = pd.read_csv("/tmp/260226_validate/solved_clean.csv",
                     names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"],
                     dtype={"box":"category","type":"string","met":"float64",
                            "det_id":"int8"})
    df = df[df["type"] == "EVT"]
    # Round to 1-second bin
    df["met_sec"] = df["met"].astype("int64")
    # Group by box, det, met_sec → count events
    counts = df.groupby(["box","det_id","met_sec"], observed=True).size().rename("Sci_obs").reset_index()
    return counts


def load_reconstructed_events():
    """Load reconstruct CSV (no det_id, only box). Per-second per-box.

    type can be 'EVT' (original 1B event) or 'FILL_GAP' (reconstructed gap-fill).
    Both contribute to Sci_recov.
    """
    df = pd.read_csv("/tmp/260226_validate/reconstructed_clean.csv",
                     names=["box","type","met","channel","pkt_idx","evt_idx"],
                     dtype={"box":"category","type":"string","met":"float64"})
    df = df[df["type"].isin(["EVT", "FILL_GAP"])].copy()
    df["box"] = df["box"].astype(str)  # avoid categorical fillna issue
    df["met_sec"] = df["met"].astype("int64")
    counts = df.groupby(["box","met_sec"]).size().rename("Sci_recov_box").reset_index()
    fills = df[df["type"] == "FILL_GAP"].groupby(["box","met_sec"]).size().rename("Sci_fill_box").reset_index()
    out = counts.merge(fills, on=["box","met_sec"], how="left")
    out["Sci_fill_box"] = out["Sci_fill_box"].fillna(0)
    return out


def main():
    print("Loading 1B engineering FITS...")
    eng_dfs = []
    for box, code, _ in BOXES:
        eng_dfs.append(load_eng_fits(box, code))
    eng = pd.concat(eng_dfs, ignore_index=True)
    print(f"  Engineering rows: {len(eng):,}")

    print("Loading 1B solved events (observed Sci)...")
    sci_obs = load_solved_events()
    print(f"  Solved EVT rows: {len(sci_obs):,}")

    print("Loading reconstructed events (gap-filled, box-level)...")
    sci_recov = load_reconstructed_events()
    print(f"  Reconstructed EVT rows: {len(sci_recov):,}")

    # Merge engineering with observed Sci (by box, det, met_sec)
    eng["met_sec"] = eng["met_int"]
    sci_obs_renamed = sci_obs.rename(columns={"det_id":"det"})
    df = eng.merge(sci_obs_renamed, on=["box","det","met_sec"], how="left")
    df["Sci_obs"] = df["Sci_obs"].fillna(0)
    print(f"  Merged rows: {len(df):,}")

    # Box-level recovery: total recovered Sci per box per second
    df = df.merge(sci_recov, on=["box","met_sec"], how="left")
    df["Sci_recov_box"] = df["Sci_recov_box"].fillna(0)
    df["Sci_fill_box"] = df["Sci_fill_box"].fillna(0)

    # Per-det recovery: assume fill events distributed proportionally
    box_obs_sum = df.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    df["Sci_recov"] = df["Sci_recov_box"] * (df["Sci_obs"] / box_obs_sum.clip(lower=1))
    # When obs is 0 (FIFO totally blocked), distribute equally
    fully_lost = box_obs_sum == 0
    df.loc[fully_lost, "Sci_recov"] = df.loc[fully_lost, "Sci_recov_box"] / 6

    # ============ Apply models ============
    df["live_frac"] = 1.0 - df["Dt"] / df["length_cyc"]

    # Multiplicative: PHO·live = k·Sci + Wide + Large
    # → PHO_pred = (k·Sci + Wide + Large) / live
    for sci_var, suffix in [("Sci_obs", "obs"), ("Sci_recov", "recov")]:
        # Multiplicative
        df[f"PHO_pred_mul_{suffix}"] = (
            df["box"].map(K_MUL) * df[sci_var] + df["Wide"] + df["Large"]
        ) / df["live_frac"].clip(lower=0.01)

        # M_uni_M1: PHO = (1+α)Sci + β·Wide + γ_0·Large + γ_1·Large²/Sci + b
        bs = df["box"].map(lambda b: M_UNI_M1[b][0])
        c1pluss = df["box"].map(lambda b: M_UNI_M1[b][1])
        betas = df["box"].map(lambda b: M_UNI_M1[b][2])
        g0s = df["box"].map(lambda b: M_UNI_M1[b][3])
        g1s = df["box"].map(lambda b: M_UNI_M1[b][4])
        df[f"PHO_pred_uni_{suffix}"] = (
            bs + c1pluss * df[sci_var] + betas * df["Wide"]
            + g0s * df["Large"] + g1s * df["Large"]**2 / df[sci_var].clip(lower=1)
        )

    # ============ Box-level aggregation for plotting ============
    box_agg = df.groupby(["box","met_sec"]).agg(
        PHO_obs=("PHO", "sum"),
        Wide=("Wide", "sum"),
        Large=("Large", "sum"),
        Sci_obs=("Sci_obs", "sum"),
        Sci_recov=("Sci_recov", "sum"),
        PHO_pred_mul_obs=("PHO_pred_mul_obs", "sum"),
        PHO_pred_mul_recov=("PHO_pred_mul_recov", "sum"),
        PHO_pred_uni_obs=("PHO_pred_uni_obs", "sum"),
        PHO_pred_uni_recov=("PHO_pred_uni_recov", "sum"),
        Dt=("Dt", "sum"),
        length_cyc=("length_cyc", "sum"),
    ).reset_index()
    box_agg["live_box"] = 1.0 - box_agg["Dt"] / box_agg["length_cyc"]
    box_agg["t_rel"] = box_agg["met_sec"] - TRIGGER_MET

    # ============ Plot ============
    # Filter out edge bins where t > 60 (incomplete data, model breaks)
    box_agg_plot = box_agg[(box_agg["t_rel"] >= -28) & (box_agg["t_rel"] <= 60)].copy()

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    for col, (box, _, _) in enumerate(BOXES):
        sub = box_agg_plot[box_agg_plot["box"] == box].sort_values("t_rel")

        # Row 0: Sci observed vs recovered
        ax = axes[0, col]
        ax.plot(sub["t_rel"], sub["Sci_obs"], "-", color="C0", lw=1.5, label="Sci observed (1B)")
        ax.plot(sub["t_rel"], sub["Sci_recov"], "-", color="C3", lw=1.5,
                label="Sci recovered (1B + fill)")
        ax.set_ylabel(f"Box {box}\nSci [cnt/s/box]")
        ax.set_title(f"Box {box}: 1B Sci, observed vs recovered")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 1: PHO observed vs predicted (linear scale)
        ax = axes[1, col]
        ax.plot(sub["t_rel"], sub["PHO_obs"], "-", color="black", lw=2,
                label="PHO observed (1K)", alpha=0.9)
        ax.plot(sub["t_rel"], sub["PHO_pred_uni_obs"], "--", color="C2", lw=1.5,
                label="M_uni_M1 pred (Sci_obs)", alpha=0.7)
        ax.plot(sub["t_rel"], sub["PHO_pred_uni_recov"], "-", color="C3", lw=1.5,
                label="M_uni_M1 pred (Sci_recov)")
        ax.set_ylabel("PHO [cnt/s/box]")
        ax.set_title(f"Box {box}: M_uni_M1 prediction vs observed PHO")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 2: Multiplicative model prediction
        ax = axes[2, col]
        ax.plot(sub["t_rel"], sub["PHO_obs"], "-", color="black", lw=2,
                label="PHO observed (1K)", alpha=0.9)
        ax.plot(sub["t_rel"], sub["PHO_pred_mul_obs"], "--", color="C2", lw=1.5,
                label="Multiplicative pred (Sci_obs)", alpha=0.7)
        ax.plot(sub["t_rel"], sub["PHO_pred_mul_recov"], "-", color="C3", lw=1.5,
                label="Multiplicative pred (Sci_recov)")
        ax.set_ylabel("PHO [cnt/s/box]")
        ax.set_title(f"Box {box}: Multiplicative pred vs observed")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 3: Fractional residual (pred-obs)/obs for both models
        ax = axes[3, col]
        ax.plot(sub["t_rel"], (sub["PHO_pred_uni_obs"]-sub["PHO_obs"]) / sub["PHO_obs"].clip(lower=1) * 100,
                "--", color="C2", lw=1.2, label="M_uni_M1 (Sci_obs)")
        ax.plot(sub["t_rel"], (sub["PHO_pred_uni_recov"]-sub["PHO_obs"]) / sub["PHO_obs"].clip(lower=1) * 100,
                "-", color="C3", lw=1.5, label="M_uni_M1 (Sci_recov)")
        ax.plot(sub["t_rel"], (sub["PHO_pred_mul_obs"]-sub["PHO_obs"]) / sub["PHO_obs"].clip(lower=1) * 100,
                "--", color="C4", lw=1.0, label="Mul (Sci_obs)", alpha=0.6)
        ax.plot(sub["t_rel"], (sub["PHO_pred_mul_recov"]-sub["PHO_obs"]) / sub["PHO_obs"].clip(lower=1) * 100,
                "-", color="C5", lw=1.2, label="Mul (Sci_recov)", alpha=0.7)
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.set_ylabel("relative residual (%)")
        ax.set_xlabel("Time since trigger [s]")
        ax.set_title(f"Box {box}: (pred - obs) / obs × 100%")
        ax.set_ylim(-60, 80)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("GRB 260226A validation: PHO prediction vs observed (M_uni_M1 + Multiplicative)\n"
                 "Recovered Sci should give HIGHER PHO prediction in saturated bins → matches what we'd see if FIFO didn't drop events",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "validate_260226a.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # ============ Numerical summary ============
    print(f"\n=== Saturation detection ===")
    for box, _, _ in BOXES:
        sub = box_agg[box_agg["box"] == box]
        saturated = sub["Sci_recov_box" if False else "Sci_recov"] > sub["Sci_obs"] * 1.05
        sat_times = sub.loc[saturated, "t_rel"].values
        if len(sat_times):
            print(f"  Box {box}: saturated bins at t = {sat_times.min():.1f} to {sat_times.max():.1f}s")
            print(f"           {len(sat_times)} seconds with > 5% recovery")
            sat_sub = sub[saturated]
            recov_total = (sat_sub["Sci_recov"] - sat_sub["Sci_obs"]).sum()
            print(f"           total recovered events: {recov_total:.0f}")
        else:
            print(f"  Box {box}: no saturation detected in window")

    print(f"\n=== Validation: residuals in QUIET time (|t|>5, excluding burst & edges) ===")
    for box, _, _ in BOXES:
        sub = box_agg[box_agg["box"] == box]
        # Quiet: before burst (t < 15) or well after (t in [40, 60])
        quiet = ((sub["t_rel"] >= -25) & (sub["t_rel"] < 15)) | \
                ((sub["t_rel"] >= 40) & (sub["t_rel"] <= 55))
        sub_q = sub[quiet].copy()
        if len(sub_q) == 0:
            continue
        for model_name, col in [("M_uni_M1", "PHO_pred_uni_obs"),
                                  ("Multiplicative", "PHO_pred_mul_obs")]:
            resid = sub_q[col] - sub_q["PHO_obs"]
            rel = 100 * resid / sub_q["PHO_obs"].clip(lower=1)
            med_pho = sub_q["PHO_obs"].median()
            print(f"  Box {box} {model_name:>15s}: N={len(sub_q)}, "
                  f"med PHO_obs={med_pho:.0f}, "
                  f"med residual={resid.median():+.1f} ({rel.median():+.2f}%), "
                  f"|residual| 90%-tile={np.quantile(np.abs(rel), 0.9):.2f}%")

    print(f"\n=== Saturation consistency check ===")
    print(f"  Hypothesis: FIFO drops events uniformly across all bands (Sci, PHO, Wide, Large)")
    print(f"  Then: M_uni_M1(Sci_obs, Wide_obs, Large_obs) should ≈ PHO_obs (saturated relation preserved)")
    print(f"  And:  True PHO = PHO_obs × (Sci_recov/Sci_obs)")
    print(f"  ")
    print(f"  {'box':>3s} {'t':>4s} {'Sci_obs':>8s} {'Sci_recov':>10s} {'f=ratio':>8s}  "
          f"{'PHO_obs':>8s} {'PHO_pred_obs':>13s} {'rel%':>6s}  "
          f"{'PHO_pred_recov':>15s} {'True (scale)':>13s}")
    for box, _, _ in BOXES:
        sub = box_agg[box_agg["box"] == box]
        sat = sub["Sci_recov"] > sub["Sci_obs"] * 1.05
        sub_s = sub[sat & (sub["t_rel"] >= 18) & (sub["t_rel"] <= 35)]
        for _, row in sub_s.iterrows():
            f = row["Sci_recov"] / row["Sci_obs"]
            rel = 100 * (row["PHO_pred_uni_obs"] - row["PHO_obs"]) / row["PHO_obs"]
            true_pho_scaled = row["PHO_obs"] * f
            print(f"  {box:>3s} {row['t_rel']:>+4.0f} {row['Sci_obs']:>8.0f} "
                  f"{row['Sci_recov']:>10.0f} {f:>8.2f}  "
                  f"{row['PHO_obs']:>8.0f} {row['PHO_pred_uni_obs']:>13.0f} {rel:>+6.1f}  "
                  f"{row['PHO_pred_uni_recov']:>15.0f} {true_pho_scaled:>13.0f}")


if __name__ == "__main__":
    main()
