#!/usr/bin/env python3
"""Test hypothesis: training data has FIFO drop contamination, inflating
the (1+α), β, γ coefficients. Tightening the fit window away from FIFO
threshold should reduce the +11% over-prediction at 260226A burst onset.

Original training filter: Sci ∈ [400, 1500] per det, box rate up to 9000.
FIFO threshold: ~15,000 cnt/s/box.
Tightened filter:         Sci ∈ [400, 1000] per det, box rate < 6000.
                           (well below FIFO threshold even with bright bursts)

Workflow:
  1. Load 2017-2019 normal-mode data
  2. Classify HIGH/LOW Large modes (date-level, as before)
  3. Fit M1_HIGH and M_uni_M1 on TIGHTENED clean band
  4. Apply new coefs to 260226A engineering data
  5. Check residual at t=+21 (Sci=9658, no FIFO drop, was +11%)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

# Existing training band (for comparison)
SCI_LO_OLD, SCI_HI_OLD = 400.0, 1500.0
# Tightened: well below FIFO threshold
SCI_LO_NEW, SCI_HI_NEW = 400.0, 1000.0
BOX_RATE_CAP = 6000.0  # box total Sci cap
MET_CORRECTION = 4.0
TRIGGER_260 = 446726273.0


def load_training():
    """Load 2017-2019 training data."""
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32",
             "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs...")
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, usecols=list(dtype), dtype=dtype))
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH]
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN].copy()

    df["Sci_pure"]     = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["sci_rate"]     = df["Sci"]      / df["length"]
    df["scipure_rate"] = df["Sci_pure"] / df["length"]
    df["acd1_rate"]    = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]    = df["Sci_ACDN"] / df["length"]
    df["wide_rate"]    = df["Wide"]     / df["length"]
    df["large_rate"]   = df["Large"]    / df["length"]
    df["pho_rate"]     = df["PHO"]      / df["length"]
    df["group_rate"]   = df["sci_sec_total"] / df["length"]
    df["large_frac"]   = df["large_rate"] / df["sci_rate"].clip(lower=1)
    df["large2_over_sci"] = df["large_rate"]**2 / df["sci_rate"].clip(lower=1)
    df["det_global"]   = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

    hv = pd.read_csv(HV_TABLE,
                     dtype={"date":"string","met_sec":"int64",
                            **{f"hv{i}":"float32" for i in range(18)}})
    hv = hv.set_index(["date","met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    print(f"  normal-mode rows: {len(df):,}")
    return df


def classify_modes(df):
    main = df[(df["sci_rate"] >= 1000) & (df["sci_rate"] < 1500)].copy()
    by_date = main.groupby("date").agg(
        large_frac=("large_frac", "median"),
        N=("large_frac", "count"),
    )
    by_date = by_date[by_date["N"] > 200]
    high_dates = set(by_date[by_date["large_frac"] > 0.55].index)
    return high_dates


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def fit_uni(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values,
                         sub["large2_over_sci"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def load_260226A_eng():
    """Load 260226A engineering FITS data per-det."""
    rows = []
    for box, code in [("A","0766"),("B","1009"),("C","1781")]:
        eng_file = f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits"
        fe = fits.open(eng_file, memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
        mask = (met_eng >= TRIGGER_260 - 30) & (met_eng <= TRIGGER_260 + 70)
        met_eng = met_eng[mask]
        for det_local in range(6):
            det_g = BOX_OFFSET[box] + det_local
            pho = d[f"Cnt_PHODet_{det_g}"].astype(float)[mask]
            csi = d[f"Cnt_CsI_PHODet_{det_g}"].astype(float)[mask]
            large_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)[mask]
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                rows.append({
                    "box": box, "det": det_local,
                    "met_sec": int(met_eng[i]),
                    "PHO": pho[i], "Wide": csi[i], "Large": large[i],
                })
        fe.close()
    eng = pd.DataFrame(rows)
    # Load Sci_obs from cached solve
    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    sci_obs = sci_obs[sci_obs["type"]=="EVT"]
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    sci_obs = sci_obs.groupby(["box","det_id","met_sec"]).size().rename("Sci_obs").reset_index()
    sci_obs = sci_obs.rename(columns={"det_id":"det"})
    eng = eng.merge(sci_obs, on=["box","det","met_sec"], how="left")
    eng["Sci_obs"] = eng["Sci_obs"].fillna(0)
    eng["t_rel"] = eng["met_sec"] - TRIGGER_260
    return eng


def main():
    df = load_training()
    high_dates = classify_modes(df)
    df["mode"] = np.where(df["date"].isin(high_dates), "HIGH", "OTHER")

    # ============ Fit on OLD vs NEW band ============
    print(f"\n=== Refit M1 (HIGH mode) and M_uni_M1 on OLD vs NEW band ===")
    print(f"  OLD: Sci ∈ [{SCI_LO_OLD}, {SCI_HI_OLD}] per det")
    print(f"  NEW: Sci ∈ [{SCI_LO_NEW}, {SCI_HI_NEW}] per det, group_rate < {BOX_RATE_CAP}")

    fits = {}
    for box in "ABC":
        # OLD fit
        mask_old = ((df["box"] == box) & (df["mode"] == "HIGH")
                    & (df["sci_rate"] >= SCI_LO_OLD) & (df["sci_rate"] < SCI_HI_OLD))
        sub_old = df[mask_old]
        m1_old = fit_m1(sub_old)
        uni_old = fit_uni(sub_old)
        # NEW fit
        mask_new = ((df["box"] == box) & (df["mode"] == "HIGH")
                    & (df["sci_rate"] >= SCI_LO_NEW) & (df["sci_rate"] < SCI_HI_NEW)
                    & (df["group_rate"] < BOX_RATE_CAP))
        sub_new = df[mask_new]
        m1_new = fit_m1(sub_new)
        uni_new = fit_uni(sub_new)
        fits[box] = dict(m1_old=m1_old, m1_new=m1_new,
                         uni_old=uni_old, uni_new=uni_new,
                         n_old=len(sub_old), n_new=len(sub_new))
        print(f"\n  Box {box}:")
        print(f"    OLD (N={len(sub_old):,}): "
              f"M1 (b, 1+α, β, γ) = ({m1_old[0]:.1f}, {m1_old[1]:.3f}, "
              f"{m1_old[2]:.3f}, {m1_old[3]:.3f})")
        print(f"    NEW (N={len(sub_new):,}): "
              f"M1 (b, 1+α, β, γ) = ({m1_new[0]:.1f}, {m1_new[1]:.3f}, "
              f"{m1_new[2]:.3f}, {m1_new[3]:.3f})")
        d1plus = m1_new[1] - m1_old[1]
        dbeta = m1_new[2] - m1_old[2]
        dgamma = m1_new[3] - m1_old[3]
        print(f"    Δ:   d(1+α)={d1plus:+.3f}, dβ={dbeta:+.3f}, dγ={dgamma:+.3f}")
        print(f"    OLD: M_uni (b, 1+α, β, γ_0, γ_1) = ({uni_old[0]:.1f}, "
              f"{uni_old[1]:.3f}, {uni_old[2]:.3f}, {uni_old[3]:.3f}, {uni_old[4]:.3f})")
        print(f"    NEW: M_uni (b, 1+α, β, γ_0, γ_1) = ({uni_new[0]:.1f}, "
              f"{uni_new[1]:.3f}, {uni_new[2]:.3f}, {uni_new[3]:.3f}, {uni_new[4]:.3f})")

    # ============ Validation on 260226A ============
    print(f"\n=== Validation on 260226A using new coefficients ===")
    eng = load_260226A_eng()

    # Per-det predictions
    for tag in ["m1_old","m1_new","uni_old","uni_new"]:
        b_v = eng["box"].map(lambda b: fits[b][tag][0]).values
        a_v = eng["box"].map(lambda b: fits[b][tag][1]).values
        be_v = eng["box"].map(lambda b: fits[b][tag][2]).values
        if tag.startswith("m1"):
            g_v = eng["box"].map(lambda b: fits[b][tag][3]).values
            eng[f"pred_{tag}"] = (b_v + a_v*eng["Sci_obs"].values
                                  + be_v*eng["Wide"].values + g_v*eng["Large"].values)
        else:
            g0_v = eng["box"].map(lambda b: fits[b][tag][3]).values
            g1_v = eng["box"].map(lambda b: fits[b][tag][4]).values
            eng[f"pred_{tag}"] = (b_v + a_v*eng["Sci_obs"].values
                                  + be_v*eng["Wide"].values + g0_v*eng["Large"].values
                                  + g1_v*eng["Large"].values**2
                                  / np.maximum(eng["Sci_obs"].values, 1))

    # Per-box aggregation
    box_agg = eng.groupby(["box","t_rel"]).agg(
        PHO=("PHO","sum"), Wide=("Wide","sum"), Large=("Large","sum"),
        Sci_obs=("Sci_obs","sum"),
        pred_m1_old=("pred_m1_old","sum"), pred_m1_new=("pred_m1_new","sum"),
        pred_uni_old=("pred_uni_old","sum"), pred_uni_new=("pred_uni_new","sum"),
    ).reset_index()
    for tag in ["m1_old","m1_new","uni_old","uni_new"]:
        box_agg[f"rel_{tag}"] = 100*(box_agg[f"pred_{tag}"]-box_agg["PHO"])/box_agg["PHO"]

    # Key bin: t=+21 (no FIFO drop, high rate, Sci~9700 per box)
    print(f"\n  Critical test bins (no FIFO drops, high rate):")
    print(f"  {'t':>4s} {'Sci_obs':>8s}  {'M1_OLD':>9s} {'M1_NEW':>9s}  {'Uni_OLD':>10s} {'Uni_NEW':>10s}")
    for box in "ABC":
        sub_b = box_agg[box_agg["box"]==box]
        for t in [18, 19, 20, 21, 27, 29, 33, 35]:
            row = sub_b[sub_b["t_rel"]==t]
            if len(row) == 0: continue
            r = row.iloc[0]
            print(f"  Box {box} t={t:>+3.0f}: Sci={r.Sci_obs:>7.0f}  "
                  f"{r.rel_m1_old:>+8.1f}% {r.rel_m1_new:>+8.1f}%  "
                  f"{r.rel_uni_old:>+9.1f}% {r.rel_uni_new:>+9.1f}%")

    # Summary statistics: bins with Sci > 7000 per box, no FIFO drop
    print(f"\n=== Summary: 'high-rate no-FIFO' bins (Sci > 7000 per box, "
          f"|t-T0| < 35 or t > 25, no FIFO) ===")
    box_agg["is_burst_edge"] = ((box_agg["Sci_obs"] > 7000) &
                                ((box_agg["t_rel"] < 22) | (box_agg["t_rel"] > 28)))
    high_no_fifo = box_agg[box_agg["is_burst_edge"]]
    for tag, name in [("m1_old", "M1 (OLD band)"), ("m1_new", "M1 (NEW band)"),
                       ("uni_old", "M_uni (OLD)"), ("uni_new", "M_uni (NEW)")]:
        med = high_no_fifo[f"rel_{tag}"].median()
        rms = np.sqrt((high_no_fifo[f"rel_{tag}"]**2).mean())
        print(f"  {name:>20s}: median = {med:+.2f}%, RMS = {rms:.2f}%, N={len(high_no_fifo)}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for col, box in enumerate("ABC"):
        ax = axes[col]
        sub = box_agg[box_agg["box"]==box].sort_values("t_rel")
        for tag, name, color, ls in [
            ("m1_old","M1 (OLD band)","C0","--"),
            ("m1_new","M1 (NEW band)","C0","-"),
            ("uni_old","M_uni (OLD)","C3","--"),
            ("uni_new","M_uni (NEW)","C3","-"),
        ]:
            ax.plot(sub["t_rel"], sub[f"rel_{tag}"], ls, color=color,
                    lw=1.5, label=name, alpha=0.85)
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.axvspan(22, 29, alpha=0.1, color="orange", label="FIFO drop region")
        ax.set_xlabel("t-T0 [s]")
        ax.set_ylabel("relative residual (%)")
        ax.set_title(f"Box {box}: residual using Sci_obs (260226A)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-30, 30)
    fig.suptitle("Test: tighter training band reduces over-prediction in burst edges?",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "test_fifo_contamination.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
