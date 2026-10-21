#!/usr/bin/env python3
"""Test user's hypothesis: PHO × livetime_frac = c_pure·Sci_pure + c_ACD·Sci_ACD
                                                  + β·Wide + γ·Large + b

where livetime_frac (lf) = 1 - Dt/L_cycles.

Logic:
  Baseline A (current model):  PHO ~ RHS   (no lf correction)
  Hypothesis B:                PHO·lf ~ RHS

Three checks:
  (1) Direct fit of B → compare RMS (converted back to PHO units) vs A
  (2) Plot/fit baseline residual vs (PHO × dt/L). If hypothesis holds,
      residual ≈ PHO × dt/L → slope ≈ +1
  (3) Stratify residual_A by dt/L bin: residual should grow with dt/L
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
SCI_LO, SCI_HI, BOX_RATE_CAP = 400.0, 1000.0, 6000.0


def load():
    dtype = {"date":"string","box":"category","met_sec":"int64","det":"int8",
             "L_cycles":"int32","PHO":"int32","Wide":"int32","Large":"int32",
             "Dt":"int32","Sci":"int32","Sci_ACD1":"int32","Sci_ACDN":"int32"}
    parts = []
    for f in sorted(CSV_DIR.glob("*.csv")):
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
    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["Sci_ACD"] = df["Sci_ACD1"] + df["Sci_ACDN"]
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("acd_rate","Sci_ACD"),("wide_rate","Wide"),
                    ("large_rate","Large"),("pho_rate","PHO")]:
        df[c] = df[src] / df["length"]
    df["group_rate"] = df["sci_sec_total"] / df["length"]
    df["dt_frac"] = df["Dt"].astype("float32") / df["L_cycles"]
    df["lf"] = 1.0 - df["dt_frac"]
    df["pho_lf"] = df["pho_rate"] * df["lf"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    hv = pd.read_csv(HV_TABLE, dtype={"date":"string","met_sec":"int64",
        **{f"hv{i}":"float32" for i in range(18)}})
    hv = hv.set_index(["date","met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    return df


def fit(sub, target_col):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"], sub["large_rate"]])
    coef, *_ = np.linalg.lstsq(X, sub[target_col].values, rcond=None)
    return coef


def predict_RHS(coef, sub):
    return (coef[0] + coef[1]*sub["scipure_rate"] + coef[2]*sub["acd_rate"]
             + coef[3]*sub["wide_rate"] + coef[4]*sub["large_rate"])


def main():
    df = load()
    print(f"rows loaded: {len(df):,}")
    print(f"\nDt/L deadtime fraction:  mean={df['dt_frac'].mean()*100:.3f}%  "
          f"median={df['dt_frac'].median()*100:.3f}%  "
          f"95p={df['dt_frac'].quantile(0.95)*100:.3f}%  "
          f"99p={df['dt_frac'].quantile(0.99)*100:.3f}%  "
          f"max={df['dt_frac'].max()*100:.3f}%")

    clean = ((df["sci_rate"] >= SCI_LO) & (df["sci_rate"] < SCI_HI)
             & (df["group_rate"] < BOX_RATE_CAP))
    print(f"CLEAN-band rows: {int(clean.sum()):,}")

    # ============= Test 1: per-box fits, A baseline vs B hypothesis =============
    print(f"\n{'='*86}")
    print("Test 1  Per-box fits   A: PHO ~ RHS    B: PHO·lf ~ RHS  (rebuilt to PHO units)")
    print(f"{'='*86}")
    print(f"  {'box':>3s} {'model':>10s}  {'b':>7s}  {'c_pure':>7s}  {'c_ACD':>7s}  "
          f"{'beta':>6s}  {'gamma':>6s}  {'RMS_PHO':>9s}   Δ_vs_A")
    rms_summary = {}
    for box in "ABC":
        sub = df[clean & (df["box"]==box)]
        cA = fit(sub, "pho_rate")
        predA = predict_RHS(cA, sub)
        rmsA = float(np.sqrt(np.mean((sub["pho_rate"] - predA)**2)))
        print(f"  {box:>3s} {'A':>10s}  {cA[0]:>+7.2f}  {cA[1]:>7.4f}  {cA[2]:>7.4f}  "
              f"{cA[3]:>6.4f}  {cA[4]:>6.4f}  {rmsA:>9.3f}")
        # Model B: target = PHO·lf, then divide pred by lf to recover PHO
        cB = fit(sub, "pho_lf")
        predB = predict_RHS(cB, sub) / sub["lf"]
        rmsB = float(np.sqrt(np.mean((sub["pho_rate"] - predB)**2)))
        delta = 100.0 * (rmsB - rmsA) / rmsA
        print(f"  {box:>3s} {'B':>10s}  {cB[0]:>+7.2f}  {cB[1]:>7.4f}  {cB[2]:>7.4f}  "
              f"{cB[3]:>6.4f}  {cB[4]:>6.4f}  {rmsB:>9.3f}   {delta:+6.2f}%")
        rms_summary[box] = (rmsA, rmsB)

    # ============= Test 2: baseline residual vs PHO×dt/L =============
    print(f"\n{'='*86}")
    print("Test 2  Baseline residual vs (PHO × dt/L)")
    print("        hypothesis predicts: residual ≈ +1 · (PHO × dt/L)")
    print(f"{'='*86}")
    for box in "ABC":
        sub = df[clean & (df["box"]==box)].copy()
        cA = fit(sub, "pho_rate")
        sub["pho_pred"] = predict_RHS(cA, sub)
        sub["residual"] = sub["pho_rate"] - sub["pho_pred"]
        sub["pho_x_dtfrac"] = sub["pho_rate"] * sub["dt_frac"]
        # Linear fit residual = α + β · (PHO·dt/L)
        X = np.column_stack([np.ones(len(sub)), sub["pho_x_dtfrac"].values])
        c, *_ = np.linalg.lstsq(X, sub["residual"].values, rcond=None)
        rho = float(np.corrcoef(sub["pho_x_dtfrac"].values,
                                  sub["residual"].values)[0,1])
        print(f"  Box {box}:  residual = {c[0]:+.3f} + ({c[1]:+.4f}) × (PHO·dt/L)   "
              f"Pearson ρ = {rho:+.3f}")

    # ============= Test 3: stratify residual_A by dt/L bin =============
    print(f"\n{'='*86}")
    print("Test 3  Mean baseline residual by dt/L decile  (should grow if hypothesis correct)")
    print(f"{'='*86}")
    for box in "ABC":
        sub = df[clean & (df["box"]==box)].copy()
        cA = fit(sub, "pho_rate")
        sub["residual"] = sub["pho_rate"] - predict_RHS(cA, sub)
        sub["dt_decile"] = pd.qcut(sub["dt_frac"], 10, labels=False, duplicates='drop')
        agg = sub.groupby("dt_decile").agg(
            dt_mean=("dt_frac", "mean"),
            res_mean=("residual", "mean"),
            res_std=("residual", "std"),
            pho_mean=("pho_rate", "mean"),
            N=("pho_rate", "size"),
        )
        agg["res_norm"] = agg["res_mean"] / agg["pho_mean"]
        print(f"\n  Box {box}:")
        print(f"    {'decile':>6s}  {'dt%':>7s}  {'PHO̅':>8s}  {'residual̅':>11s}  "
              f"{'res/PHO̅':>10s}  {'expected res/PHO̅ ≈ dt%':>22s}")
        for d, row in agg.iterrows():
            print(f"    {int(d):>6d}  {row['dt_mean']*100:>6.2f}%  {row['pho_mean']:>8.1f}  "
                  f"{row['res_mean']:>+11.3f}  {row['res_norm']*100:>+9.3f}%  "
                  f"{row['dt_mean']*100:>+21.3f}%")

    # ============= Test 4: k-scan ============
    # PHO·(1 - k·dt/L) = RHS  → free k
    #   k=0 : baseline
    #   k=1 : full user hypothesis
    # Find k_opt that minimises RMS in PHO units
    print(f"\n{'='*86}")
    print("Test 4  Grid-scan k in   PHO · (1 − k·dt/L) = RHS")
    print(f"{'='*86}")
    k_grid = np.linspace(-1.0, 8.0, 181)

    def kscan(sub):
        Xmat = np.column_stack([np.ones(len(sub)), sub["scipure_rate"],
                                 sub["acd_rate"], sub["wide_rate"],
                                 sub["large_rate"]]).astype(np.float64)
        pho  = sub["pho_rate"].values.astype(np.float64)
        dtf  = sub["dt_frac"].values.astype(np.float64)
        out = np.empty_like(k_grid)
        for j, k in enumerate(k_grid):
            lf = 1.0 - k*dtf
            target = pho * lf
            coef, *_ = np.linalg.lstsq(Xmat, target, rcond=None)
            pred_rhs = Xmat @ coef
            pred_pho = pred_rhs / lf
            out[j] = float(np.sqrt(np.mean((pho - pred_pho)**2)))
        return out

    rms_k = {}
    print(f"  {'box':>3s}  {'k_opt':>6s}  {'RMS@k=0':>10s}  {'RMS@k=1':>10s}  "
          f"{'RMS@k_opt':>10s}  improvement vs baseline")
    for box in "ABC":
        sub = df[clean & (df["box"]==box)]
        rms_k[box] = kscan(sub)
        j_opt = int(np.argmin(rms_k[box]))
        k_opt = float(k_grid[j_opt])
        rms_base = float(rms_k[box][np.argmin(np.abs(k_grid))])
        rms_hyp  = float(rms_k[box][np.argmin(np.abs(k_grid-1))])
        rms_opt  = float(rms_k[box][j_opt])
        print(f"  {box:>3s}  {k_opt:>+6.2f}  {rms_base:>10.3f}  {rms_hyp:>10.3f}  "
              f"{rms_opt:>10.3f}    {100*(rms_opt-rms_base)/rms_base:+.2f}%")

    # Pooled fit across all 3 boxes (single global k)
    sub_all = df[clean]
    rms_pool = kscan(sub_all)
    j_pool = int(np.argmin(rms_pool))
    k_pool = float(k_grid[j_pool])
    print(f"\n  pooled (3-box):  k_opt = {k_pool:+.2f},  "
          f"RMS@k=0 = {rms_pool[np.argmin(np.abs(k_grid))]:.3f},  "
          f"RMS@k_opt = {rms_pool[j_pool]:.3f},  "
          f"improvement = {100*(rms_pool[j_pool]-rms_pool[np.argmin(np.abs(k_grid))])/rms_pool[np.argmin(np.abs(k_grid))]:+.2f}%")

    # k-scan plot
    fig_k, ax_k = plt.subplots(1, 1, figsize=(8, 5))
    colors = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    for box in "ABC":
        rms_norm = rms_k[box] / rms_k[box][np.argmin(np.abs(k_grid))]
        ax_k.plot(k_grid, rms_norm, '-', color=colors[box], lw=1.8, label=f"Box {box}")
        j_opt = int(np.argmin(rms_k[box]))
        ax_k.scatter([k_grid[j_opt]], [rms_norm[j_opt]],
                      color=colors[box], s=80, zorder=5,
                      edgecolor='black', linewidth=0.8)
    rms_pool_norm = rms_pool / rms_pool[np.argmin(np.abs(k_grid))]
    ax_k.plot(k_grid, rms_pool_norm, '--', color='black', lw=1.5, label="pooled")
    ax_k.scatter([k_pool], [rms_pool_norm[j_pool]], color='black', s=80, zorder=5)
    ax_k.axvline(0, color='gray', ls=':', lw=1, label="k=0 (baseline)")
    ax_k.axvline(1, color='orange', ls=':', lw=1, label="k=1 (user hypothesis)")
    ax_k.axhline(1.0, color='gray', ls=':', lw=0.5)
    ax_k.set_xlabel("k  in  PHO · (1 − k·dt/L) = RHS")
    ax_k.set_ylabel("RMS_PHO / RMS_PHO(k=0)")
    ax_k.set_title("Free-k scan: deadtime correction strength on PHO")
    ax_k.legend(loc="upper right", fontsize=9)
    ax_k.grid(alpha=0.3)
    out_k = OUT_DIR / "deadtime_kscan.png"
    fig_k.tight_layout()
    fig_k.savefig(out_k, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {out_k}")

    # ============= Plot =============
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, box in enumerate("ABC"):
        sub = df[clean & (df["box"]==box)].copy()
        cA = fit(sub, "pho_rate")
        sub["residual"] = sub["pho_rate"] - predict_RHS(cA, sub)

        # Top row: residual vs dt/L (2D density)
        ax = axes[0, col]
        H, xed, yed = np.histogram2d(sub["dt_frac"]*100, sub["residual"],
                                       bins=[60, 60])
        ax.imshow(H.T, origin='lower', aspect='auto',
                   extent=[xed[0], xed[-1], yed[0], yed[-1]],
                   cmap='viridis', norm='log')
        # Overlay decile means
        agg = sub.assign(d=pd.qcut(sub["dt_frac"], 20, labels=False,
                                     duplicates='drop')).groupby("d").agg(
            dt=("dt_frac","mean"), r=("residual","mean"))
        ax.plot(agg["dt"]*100, agg["r"], 'r-o', lw=1.5, ms=4, label='ventile mean')
        ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.7)
        ax.set_xlabel("Dt / L_cycles  [%]")
        ax.set_ylabel("residual = PHO_obs − PHO_pred [cnt/s/det]")
        ax.set_title(f"Box {box}: residual vs deadtime")
        ax.legend(loc='upper left', fontsize=9)

        # Bottom row: residual vs PHO×dt/L with linear fit
        ax = axes[1, col]
        sub["pho_x_dtfrac"] = sub["pho_rate"] * sub["dt_frac"]
        H, xed, yed = np.histogram2d(sub["pho_x_dtfrac"], sub["residual"],
                                       bins=[60, 60])
        ax.imshow(H.T, origin='lower', aspect='auto',
                   extent=[xed[0], xed[-1], yed[0], yed[-1]],
                   cmap='viridis', norm='log')
        X = np.column_stack([np.ones(len(sub)), sub["pho_x_dtfrac"].values])
        c, *_ = np.linalg.lstsq(X, sub["residual"].values, rcond=None)
        xline = np.array([sub["pho_x_dtfrac"].min(), sub["pho_x_dtfrac"].max()])
        ax.plot(xline, c[0] + c[1]*xline, 'r-', lw=1.5,
                label=f"fit: {c[0]:+.2f} + {c[1]:+.3f}·x")
        ax.plot(xline, xline, 'w--', lw=1.2, alpha=0.7,
                 label="hypothesis: slope=+1")
        ax.axhline(0, color='white', ls=':', lw=0.6, alpha=0.5)
        ax.set_xlabel("PHO × (Dt/L)  [cnt/s/det]")
        ax.set_ylabel("residual = PHO_obs − PHO_pred [cnt/s/det]")
        ax.set_title(f"Box {box}: residual vs PHO·dt/L  (slope ≈ 1 ⇒ hypothesis)")
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle("Deadtime hypothesis test:  PHO × (1 − Dt/L) = c_pure·Sci_pure + c_ACD·Sci_ACD + β·Wide + γ·Large + b",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "deadtime_hypothesis_test.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
