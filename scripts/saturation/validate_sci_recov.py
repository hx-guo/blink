#!/usr/bin/env python3
"""Validate 1B Sci recovery using ENGINEERING PHO as ground truth.

User's key insight: engineering counters (PHO, Wide, Large, Dt) are NOT
affected by FIFO saturation. They are accurate at all times.

Validation logic:
  - Engineering PHO_obs = true PHO rate (always)
  - Model M(Sci, Wide, Large) should predict PHO_obs
  - During saturation: Sci_obs is undercount → M(Sci_obs) < PHO_obs
  - After recovery: Sci_recov ≈ Sci_true → M(Sci_recov) ≈ PHO_obs
  - Therefore: M(Sci_recov, Wide, Large) tracking PHO_obs validates recovery

Two test cases: GRB 260226A and GRB 221009A tail (both well-recovered).

Multiple models compared:
  M1_ALL    : fit on all 2017-2019 main band
  M1_HIGH   : fit on HIGH-Large date subset (hard source mode, matches GRBs)
  M_mul     : multiplicative: PHO·live = k·Sci + Wide + Large
  M_uni_M1  : non-linear γ_1·Large²/Sci interaction

Best model = one whose M(Sci_recov, Wide, Large) traces PHO_obs throughout
(including burst peak) without bias.
"""
from pathlib import Path
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
MET_CORRECTION = 4.0

BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

# Model coefficients (per Box, from M13 fit on clean band Sci ∈ [400, 1500])
M1_ALL = {
    "A": dict(b=-410.8, alpha1=2.3131, beta=2.9998, gamma=0.0359),
    "B": dict(b=-449.7, alpha1=2.3997, beta=2.6847, gamma=0.0237),
    "C": dict(b=-458.6, alpha1=2.3857, beta=2.9554, gamma=0.0509),
}
M1_HIGH = {  # HIGH-Large mode (hard source) — closest to GRB regime
    "A": dict(b=-103.5, alpha1=1.3705, beta=3.2589, gamma=0.7627),
    "B": dict(b=-127.2, alpha1=1.4565, beta=3.0297, gamma=0.6954),
    "C": dict(b=-129.2, alpha1=1.4284, beta=3.2752, gamma=0.7334),
}
K_MUL = {"A": 1.358, "B": 1.381, "C": 1.380}
M_UNI_M1 = {  # b, 1+α, β, γ_0, γ_1
    "A": (-245.2, 2.6764, 2.8533, -2.3310, 2.1252),
    "B": (-279.8, 2.7438, 2.6459, -2.2915, 2.0585),
    "C": (-277.0, 2.7019, 2.9011, -2.2357, 2.0431),
}


def pred_m1(box, sci, wide, large, coefs):
    c = coefs[box]
    return c["b"] + c["alpha1"]*sci + c["beta"]*wide + c["gamma"]*large


def pred_mul(box, sci, wide, large, live_frac):
    return (K_MUL[box] * sci + wide + large) / live_frac.clip(lower=0.01)


def pred_uni(box, sci, wide, large):
    b, c1plus, beta, g0, g1 = M_UNI_M1[box]
    return b + c1plus*sci + beta*wide + g0*large + g1*large**2 / np.maximum(sci, 1)


def load_eng_fits(box_letter, code, year_month_day_hour, t_lo_met, t_hi_met):
    """Load 1B engineering FITS for one box within MET window."""
    y, m, d, h = year_month_day_hour
    folder = Path("data/1B") / f"{y}" / f"{y}{m:02d}{d:02d}" / code
    file_prefix = f"HXMT_1B_{code}_{y}{m:02d}{d:02d}T{h:02d}"
    matches = sorted(folder.glob(f"{file_prefix}*.fits"))
    if not matches:
        raise FileNotFoundError(f"No FITS at {folder} matching {file_prefix}")
    fe = fits.open(matches[0], memmap=True)
    d_arr = fe["HE_Eng"].data
    offset = d_arr["UTC_Last_Bdc"][0] - d_arr["sTime_Last_Bdc"][0]
    met_eng = d_arr["Time"].astype(float) + offset + MET_CORRECTION
    length_cyc = d_arr["Length_Time_Cycle"].astype(float)
    length_s = length_cyc * 16e-6
    mask = (met_eng >= t_lo_met) & (met_eng <= t_hi_met)
    met_eng = met_eng[mask]; length_s = length_s[mask]; length_cyc = length_cyc[mask]

    rows = []
    for det_local in range(6):
        det_global = BOX_OFFSET[box_letter] + det_local
        pho   = d_arr[f"Cnt_PHODet_{det_global}"].astype(float)[mask]
        csi   = d_arr[f"Cnt_CsI_PHODet_{det_global}"].astype(float)[mask]
        dead  = d_arr[f"DeadTime_PHODet_{det_global}"].astype(float)[mask]
        large_raw = d_arr[f"Cnt_LargeEvt_{det_global}"].astype(float)[mask]
        large = unwrap_large(pho, large_raw)
        for i in range(len(met_eng)):
            rows.append({
                "box": box_letter, "det": det_local,
                "met_sec": int(met_eng[i]),
                "length_cyc": length_cyc[i], "length_s": length_s[i],
                "PHO": pho[i], "Wide": csi[i], "Large": large[i], "Dt": dead[i],
            })
    fe.close()
    return pd.DataFrame(rows)


def aggregate_events(csv_path, has_det_id=True, types=("EVT",)):
    """Group events per (box, [det,] met_sec)."""
    if has_det_id:
        names = ["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"]
        df = pd.read_csv(csv_path, names=names)
        df = df[df["type"].isin(types)].copy()
        df["box"] = df["box"].astype(str)
        df["met_sec"] = df["met"].astype("int64")
        return df.groupby(["box","det_id","met_sec"]).size().rename("n").reset_index()
    else:
        names = ["box","type","met","channel","pkt_idx","evt_idx"]
        df = pd.read_csv(csv_path, names=names)
        df = df[df["type"].isin(types)].copy()
        df["box"] = df["box"].astype(str)
        df["met_sec"] = df["met"].astype("int64")
        return df.groupby(["box","met_sec"]).size().rename("n").reset_index()


def run_solve_reconstruct(epoch_hour, trigger_utc, before, after, work_dir):
    """Run blink solve and reconstruct, return CSV paths."""
    work_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HXMT_1B_DIR": "/Users/skyair/Developer/ihep/blink/data/1B"}

    # CLI requires positive --before. If user wants window starting AFTER trigger,
    # they pass before=-N to mean trigger+N is window start. We must use a different
    # approach: use a forward-shifted "trigger" for CLI.
    if before < 0:
        # Effective window [trigger - before, trigger + after]
        # CLI doesn't accept negative before; pass a window centered differently
        cli_before = 0
        cli_after = after + (-before)  # extend after by |before|
    else:
        cli_before = before
        cli_after = after

    solved_csv = work_dir / "solved.csv"
    if not solved_csv.exists():
        print(f"  Running solve... (before={cli_before}, after={cli_after})")
        cmd = ["./target/release/blink", "sat", "extract", trigger_utc,
               "--source", "1b", "--before", str(cli_before), "--after", str(cli_after)]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        with open(solved_csv, "w") as f:
            for line in result.stdout.split("\n"):
                if line.startswith(("A,","B,","C,")):
                    f.write(line + "\n")
    else:
        print(f"  Using cached solved.csv")

    recon_csv = work_dir / "reconstructed.csv"
    if not recon_csv.exists():
        print(f"  Running reconstruct... (before={cli_before}, after={cli_after})")
        cmd = ["./target/release/blink", "sat", "reconstruct", trigger_utc,
               "--before", str(cli_before), "--after", str(cli_after), "--bin", "1.0"]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        with open(recon_csv, "w") as f:
            for line in result.stdout.split("\n"):
                if line.startswith(("A,","B,","C,")):
                    f.write(line + "\n")
    else:
        print(f"  Using cached reconstructed.csv")

    return solved_csv, recon_csv


def prepare_data(grb_label, epoch_hour, trigger_utc, trigger_met,
                  before, after, ymdh):
    """Prepare combined DataFrame for one GRB."""
    work_dir = Path(f"/tmp/{grb_label}_validate")
    solved_csv, recon_csv = run_solve_reconstruct(
        epoch_hour, trigger_utc, before, after, work_dir
    )

    print(f"  Loading engineering FITS...")
    t_lo = trigger_met - before
    t_hi = trigger_met + after
    eng_dfs = []
    box_codes = [("A","0766"), ("B","1009"), ("C","1781")]
    for box, code in box_codes:
        eng_dfs.append(load_eng_fits(box, code, ymdh, t_lo, t_hi))
    eng = pd.concat(eng_dfs, ignore_index=True)
    print(f"  Engineering rows: {len(eng):,}")

    print(f"  Loading events...")
    sci_obs = aggregate_events(solved_csv, has_det_id=True, types=("EVT",))
    sci_obs = sci_obs.rename(columns={"det_id":"det","n":"Sci_obs"})
    sci_recov = aggregate_events(recon_csv, has_det_id=False,
                                  types=("EVT","FILL_GAP"))
    sci_recov = sci_recov.rename(columns={"n":"Sci_recov_box"})

    # Merge
    df = eng.merge(sci_obs, on=["box","det","met_sec"], how="left")
    df["Sci_obs"] = df["Sci_obs"].fillna(0)
    df = df.merge(sci_recov, on=["box","met_sec"], how="left")
    df["Sci_recov_box"] = df["Sci_recov_box"].fillna(0)

    # Distribute box-level recovery to dets proportionally to obs
    box_obs_sum = df.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    df["Sci_recov"] = np.where(
        box_obs_sum > 0,
        df["Sci_recov_box"] * df["Sci_obs"] / box_obs_sum.clip(lower=1),
        df["Sci_recov_box"] / 6
    )
    df["live_frac"] = 1.0 - df["Dt"] / df["length_cyc"]

    # Apply each model per-row (vectorized via box mapping)
    box_str = df["box"].astype(str)
    for sci_kind, sci_col in [("obs", "Sci_obs"), ("recov", "Sci_recov")]:
        sci_v = df[sci_col].values
        wide_v = df["Wide"].values
        large_v = df["Large"].values
        live_v = df["live_frac"].values

        # M1_ALL
        b_all = box_str.map(lambda b: M1_ALL[b]["b"]).values
        a_all = box_str.map(lambda b: M1_ALL[b]["alpha1"]).values
        be_all = box_str.map(lambda b: M1_ALL[b]["beta"]).values
        g_all = box_str.map(lambda b: M1_ALL[b]["gamma"]).values
        df[f"PHO_pred_M1_ALL_{sci_kind}"] = b_all + a_all*sci_v + be_all*wide_v + g_all*large_v

        # M1_HIGH
        b_h = box_str.map(lambda b: M1_HIGH[b]["b"]).values
        a_h = box_str.map(lambda b: M1_HIGH[b]["alpha1"]).values
        be_h = box_str.map(lambda b: M1_HIGH[b]["beta"]).values
        g_h = box_str.map(lambda b: M1_HIGH[b]["gamma"]).values
        df[f"PHO_pred_M1_HIGH_{sci_kind}"] = b_h + a_h*sci_v + be_h*wide_v + g_h*large_v

        # Multiplicative
        k_v = box_str.map(K_MUL).values
        df[f"PHO_pred_Mul_{sci_kind}"] = (k_v*sci_v + wide_v + large_v) / np.maximum(live_v, 0.01)

        # M_uni_M1
        b_u = box_str.map(lambda b: M_UNI_M1[b][0]).values
        c1_u = box_str.map(lambda b: M_UNI_M1[b][1]).values
        be_u = box_str.map(lambda b: M_UNI_M1[b][2]).values
        g0_u = box_str.map(lambda b: M_UNI_M1[b][3]).values
        g1_u = box_str.map(lambda b: M_UNI_M1[b][4]).values
        df[f"PHO_pred_Uni_{sci_kind}"] = (b_u + c1_u*sci_v + be_u*wide_v
                                           + g0_u*large_v + g1_u*large_v**2 / np.maximum(sci_v, 1))

    # Per-box aggregation
    agg_cols = ["PHO","Wide","Large","Dt","length_cyc","Sci_obs","Sci_recov"]
    pred_cols = [c for c in df.columns if c.startswith("PHO_pred_")]
    sums = {c: "sum" for c in agg_cols + pred_cols}
    box_agg = df.groupby(["box","met_sec"]).agg(sums).reset_index()
    box_agg["t_rel"] = box_agg["met_sec"] - trigger_met
    box_agg["live_box"] = 1.0 - box_agg["Dt"] / box_agg["length_cyc"]
    return df, box_agg


# ============ Two GRB configs ============
def main():
    grbs = {
        "260226A": dict(
            epoch_hour="2026-02-26T10",
            trigger_utc="2026-02-26T10:37:53",
            trigger_met=446726273.0,
            before=30, after=70,
            ymdh=(2026, 2, 26, 10),
            burst_lo=18, burst_hi=35,
        ),
        "221009A_tail": dict(
            # Well-recovered tail at T+500 to T+550 (per user); use local trigger at T+525
            epoch_hour="2022-10-09T13",
            trigger_utc="2022-10-09T13:25:47",  # T0 + 525s
            trigger_met=339945947.0,            # 339945422 + 525
            before=30, after=30,                # local [-30,+30] = T+495 to T+555
            ymdh=(2022, 10, 9, 13),
            burst_lo=-25, burst_hi=25,          # tail saturation in local time
        ),
    }

    for grb, cfg in grbs.items():
        print(f"\n========== GRB {grb} ==========")
        try:
            df, box_agg = prepare_data(grb, cfg["epoch_hour"], cfg["trigger_utc"],
                                        cfg["trigger_met"], cfg["before"], cfg["after"],
                                        cfg["ymdh"])
            plot_validation(df, box_agg, grb, cfg)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()


def plot_validation(df, box_agg, grb_label, cfg):
    """4 models × 3 Boxes plot showing prediction vs PHO_obs."""
    BOXES = list("ABC")
    models = [
        ("M1_ALL",  "M1 (ALL fit)",  "C0"),
        ("M1_HIGH", "M1 (HIGH mode)", "C3"),
        ("Mul",     "Multiplicative", "C2"),
        ("Uni",     "M_uni_M1",       "C4"),
    ]

    # Time window for plot: skip edges
    t_lo_plot = -cfg["before"] + 5
    t_hi_plot = cfg["after"] - 5

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    for col, box in enumerate(BOXES):
        sub = box_agg[(box_agg["box"] == box) &
                       (box_agg["t_rel"] >= t_lo_plot - 100) &
                       (box_agg["t_rel"] <= t_hi_plot)].sort_values("t_rel")
        burst_mask = (sub["t_rel"] >= cfg["burst_lo"]) & (sub["t_rel"] <= cfg["burst_hi"])

        # Row 0: Sci comparison
        ax = axes[0, col]
        ax.plot(sub["t_rel"], sub["Sci_obs"], "-", color="C0", lw=1.2,
                label="Sci_obs (1B)")
        ax.plot(sub["t_rel"], sub["Sci_recov"], "-", color="C3", lw=1.5,
                label="Sci_recov (1B+fill)")
        ax.set_ylabel(f"Box {box}\nSci [cnt/s/box]")
        ax.set_title(f"Box {box}: Sci recovery")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axvspan(cfg["burst_lo"], cfg["burst_hi"], alpha=0.1, color="orange")

        # Row 1: PHO_obs (engineering = TRUTH) vs all predictions with Sci_recov
        ax = axes[1, col]
        ax.plot(sub["t_rel"], sub["PHO"], "-", color="black", lw=2.5,
                label="PHO_obs (engineering, TRUTH)", alpha=0.9, zorder=10)
        for tag, name, color in models:
            ax.plot(sub["t_rel"], sub[f"PHO_pred_{tag}_recov"], "-", color=color,
                    lw=1.2, label=name, alpha=0.7)
        ax.set_ylabel("PHO [cnt/s/box]")
        ax.set_title(f"Box {box}: prediction with Sci_recov vs PHO_obs")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.axvspan(cfg["burst_lo"], cfg["burst_hi"], alpha=0.1, color="orange")

        # Row 2: Same but with Sci_obs (under-saturated)
        ax = axes[2, col]
        ax.plot(sub["t_rel"], sub["PHO"], "-", color="black", lw=2.5,
                label="PHO_obs (TRUTH)", alpha=0.9, zorder=10)
        for tag, name, color in models:
            ax.plot(sub["t_rel"], sub[f"PHO_pred_{tag}_obs"], "-", color=color,
                    lw=1.2, label=f"{name} (Sci_obs)", alpha=0.7)
        ax.set_ylabel("PHO [cnt/s/box]")
        ax.set_title(f"Box {box}: prediction with Sci_obs (should under-predict in burst)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.axvspan(cfg["burst_lo"], cfg["burst_hi"], alpha=0.1, color="orange")

        # Row 3: Relative residual % for recov prediction (validates recovery)
        ax = axes[3, col]
        for tag, name, color in models:
            rel = 100*(sub[f"PHO_pred_{tag}_recov"]-sub["PHO"]) / sub["PHO"].clip(lower=1)
            ax.plot(sub["t_rel"], rel, "-", color=color, lw=1.2, label=name)
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.set_ylabel("(pred-truth)/truth [%]")
        ax.set_xlabel("Time since trigger [s]")
        ax.set_title(f"Box {box}: recovery validation — close to 0% = good recovery")
        ax.set_ylim(-40, 80)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.axvspan(cfg["burst_lo"], cfg["burst_hi"], alpha=0.1, color="orange")

    fig.suptitle(f"GRB {grb_label}: Sci recovery validation against engineering PHO (TRUTH)\n"
                 f"orange band = saturated burst region. Best model = traces black PHO_obs throughout.",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"validate_sci_recov_{grb_label}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Numerical summary: relative residual in saturated bins
    print(f"\n=== Saturated bins: residual of each model with Sci_recov vs PHO_obs ===")
    print(f"  (smaller |residual| = better recovery validation)")
    for box in BOXES:
        sub_b = box_agg[(box_agg["box"] == box) &
                         (box_agg["t_rel"] >= cfg["burst_lo"]) &
                         (box_agg["t_rel"] <= cfg["burst_hi"])]
        sat = sub_b["Sci_recov"] > sub_b["Sci_obs"] * 1.05
        sub_s = sub_b[sat]
        if len(sub_s) == 0:
            continue
        print(f"\n  Box {box}: {len(sub_s)} saturated bins in burst window")
        for tag, name, _ in models:
            rel = 100*(sub_s[f"PHO_pred_{tag}_recov"]-sub_s["PHO"])/sub_s["PHO"].clip(lower=1)
            print(f"    {name:>20s}: median rel resid = {rel.median():+6.1f}%, "
                  f"|rel| 90%-tile = {np.quantile(np.abs(rel), 0.9):.1f}%")

    print(f"\n=== Quiet bins: each model with Sci_obs (=Sci_recov here) vs PHO_obs ===")
    for box in BOXES:
        sub_b = box_agg[(box_agg["box"] == box)]
        quiet = ((sub_b["t_rel"] >= -25) & (sub_b["t_rel"] < cfg["burst_lo"]-5)) | \
                ((sub_b["t_rel"] >= cfg["burst_hi"]+10) & (sub_b["t_rel"] <= cfg["after"]-5))
        sub_q = sub_b[quiet]
        if len(sub_q) == 0:
            continue
        print(f"\n  Box {box}: {len(sub_q)} quiet bins")
        for tag, name, _ in models:
            rel = 100*(sub_q[f"PHO_pred_{tag}_obs"]-sub_q["PHO"])/sub_q["PHO"].clip(lower=1)
            print(f"    {name:>20s}: median rel resid = {rel.median():+6.1f}%, "
                  f"|rel| 90%-tile = {np.quantile(np.abs(rel), 0.9):.1f}%")


if __name__ == "__main__":
    main()
