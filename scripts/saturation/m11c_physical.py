#!/usr/bin/env python3
"""M11c: physically-bounded non-linear fit.

Two physical model variants tested:

(A) F(G) = (1 + p·G)·exp(-τ·G)                                        [2 params]
    - (1+p·G): 2-event pile-up gain (PHO band picks up sub-threshold pairs)
    - exp(-τ·G): paralyzable ADC dead time
    - Bounds: p > 0, τ > 0 (enforces physical interpretation)
    - Peak at G* = 1/τ - 1/p (requires p > τ for a peak to exist)

(B) F(G) = exp(-τ·G)·(1 + a·τ·G - b·(τ·G)²)                          [3 params]
    - exp(-τ·G): dead time loss
    - (1 + a·τ·G - b·(τ·G)²): 2-/3-event pile-up redistribution
      a = 2-event PHO gain (positive)
      b = 3-event PHO loss (positive)
      x = τ·G is dimensionless pile-up probability
    - Bounds: τ > 0, a > 0, b > 0
    - One rate scale τ (combined ADC pile-up + dead-time window)

Each parameter has clear physical interpretation.

Strategy:
  Stage 1. Fit M7 linear coefs on FULL data (best linear baseline).
  Stage 2. Compute empirical F(G) per box.
  Stage 3. Fit F_A, F_B with bounds to box-pooled empirical F(G).
  Stage 4. Apply baseline × F(G) jointly; refit baseline if needed to absorb residual bias.
  Stage 5. Compare M1, M7, M11c-A, M11c-B residuals.

The KEY question: does a physically-constrained F(G) capture the data as well
as the unconstrained M11 fit?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
MAIN_BAND_LO = 300.0
G_NORM = 10000.0
SCI_REF = 2.24


def load():
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

    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]
    df["sci_rate"]     = df["Sci"]      / df["length"]
    df["scipure_rate"] = df["Sci_pure"] / df["length"]
    df["acd1_rate"]    = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]    = df["Sci_ACDN"] / df["length"]
    df["wide_rate"]    = df["Wide"]     / df["length"]
    df["large_rate"]   = df["Large"]    / df["length"]
    df["pho_rate"]     = df["PHO"]      / df["length"]
    df["group_rate"]   = df["sci_sec_total"] / df["length"]
    df["G"]            = df["group_rate"] / G_NORM
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
    print(f"normal-mode rows: {len(df):,}")
    return df


# ============ Physical F(G) forms ============
def F_A(G, p, tau):
    """Pile-up gain × paralyzable dead time. Two params, p>0, τ>0."""
    return (1.0 + p*G) * np.exp(-tau*G)


def F_B(G, tau, a, b):
    """Pile-up redistribution (2-/3-event) × paralyzable dead time."""
    x = tau * G
    return np.exp(-tau*G) * (1.0 + a*x - b*x*x)


# ============ Joint fit: baseline × F(G) ============
def joint_model_A(X, c0, c1, cN, beta, gamma, b, p, tau):
    Sp, S1, SN, W, L, G = X
    baseline = c0*Sp + c1*S1 + cN*SN + beta*W + gamma*L + b
    return baseline * F_A(G, p, tau)


def joint_model_B(X, c0, c1, cN, beta, gamma, b, tau, a, bp):
    Sp, S1, SN, W, L, G = X
    baseline = c0*Sp + c1*S1 + cN*SN + beta*W + gamma*L + b
    return baseline * F_B(G, tau, a, bp)


def fit_m7_linear(sub):
    X = np.column_stack([
        np.ones(len(sub)),
        sub["scipure_rate"].values,
        sub["acd1_rate"].values,
        sub["acdn_rate"].values,
        sub["wide_rate"].values,
        sub["large_rate"].values,
    ])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    b, c0, c1, cN, beta, gamma = coef
    return c0, c1, cN, beta, gamma, b


def fit_m1_linear(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def predict_m1(sub, coef):
    b, c1plus, beta, gamma = coef
    return b + c1plus*sub["sci_rate"].values + beta*sub["wide_rate"].values + gamma*sub["large_rate"].values


def predict_m7(sub, c0, c1, cN, beta, gamma, b):
    return (c0*sub["scipure_rate"].values + c1*sub["acd1_rate"].values
            + cN*sub["acdn_rate"].values + beta*sub["wide_rate"].values
            + gamma*sub["large_rate"].values + b)


def median_per_bin(x, y, bins, min_count=200):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    # ============ Stage 1: M7 linear on full data (best baseline) ============
    print(f"\n=== Stage 1: M7 linear baseline on full data ===")
    m7_params = {}
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        params = fit_m7_linear(df[mask_fit])
        m7_params[box] = params
        c0, c1, cN, beta, gamma, b = params
        print(f"  Box {box}: c0={c0:.3f}, c1={c1:.3f}, cN={cN:.3f}, "
              f"β={beta:.3f}, γ={gamma:.3f}, b={b:.1f}")

    # ============ Stage 2: empirical F(G) per box ============
    print(f"\n=== Stage 2: empirical F(G) ===")
    G_MIN, G_MAX = 0.18, 2.5
    bins_g = np.linspace(G_MIN, G_MAX, 30)
    bc_g = 0.5 * (bins_g[:-1] + bins_g[1:])

    F_emp = {}
    counts_in_bin = np.zeros(len(bc_g))
    for box in "ABC":
        sub = df[df["box"] == box]
        c0, c1, cN, beta, gamma, b = m7_params[box]
        pho_pred_m7 = predict_m7(sub, c0, c1, cN, beta, gamma, b)
        # Filter out near-zero baseline rows to avoid ratio blowup
        mask = pho_pred_m7 > 100
        ratio = sub["pho_rate"].values[mask] / pho_pred_m7[mask]
        G_vals = sub["G"].values[mask]
        med = median_per_bin(G_vals, ratio, bins_g)
        F_emp[box] = med
        if box == "A":
            for i in range(len(bins_g)-1):
                m = (G_vals >= bins_g[i]) & (G_vals < bins_g[i+1])
                counts_in_bin[i] += m.sum()
    print(f"  Box A F at G≈0.3: {F_emp['A'][np.searchsorted(bc_g, 0.3)]:.4f}  (should be ≈1)")
    print(f"  Box A F at G≈0.8: {F_emp['A'][np.searchsorted(bc_g, 0.8)]:.4f}  (peak)")
    print(f"  Box A F at G≈1.5: {F_emp['A'][np.searchsorted(bc_g, 1.5)]:.4f}  (drop)")

    # Pool across boxes
    F_combined = np.nanmean(np.stack([F_emp[box] for box in "ABC"]), axis=0)
    valid = np.isfinite(F_combined) & (counts_in_bin > 100)
    G_v = bc_g[valid]
    F_v = F_combined[valid]
    sigma = 1.0 / np.sqrt(np.maximum(counts_in_bin[valid], 1.0))
    sigma /= sigma.min()  # normalize

    # ============ Stage 3: fit F_A, F_B with PHYSICAL bounds ============
    print(f"\n=== Stage 3: parametric fit (bounded p>0, τ>0; a>0, b>0) ===")

    # F_A: try multiple starting points
    best_A = None
    for p0 in [[0.5, 0.5], [2.0, 1.0], [5.0, 2.0], [0.1, 0.05]]:
        try:
            popt, pcov = curve_fit(F_A, G_v, F_v, p0=p0,
                                    bounds=([0, 0], [50, 50]),
                                    sigma=sigma, absolute_sigma=False)
            ssr = np.sum(((F_A(G_v, *popt) - F_v) / sigma)**2)
            if best_A is None or ssr < best_A[1]:
                best_A = (popt, ssr)
        except Exception as e:
            pass

    if best_A is not None:
        popt_A, ssr_A = best_A
        p_A, tau_A = popt_A
        print(f"  F_A = (1 + {p_A:.4f}·G) · exp(-{tau_A:.4f}·G)   χ²/dof={ssr_A/(len(G_v)-2):.2f}")
        if p_A > tau_A:
            G_star_A = 1/tau_A - 1/p_A
            F_star_A = F_A(G_star_A, *popt_A)
            print(f"    peak at G*={G_star_A:.3f} (group_rate={G_star_A*G_NORM:.0f}), "
                  f"F(G*)={F_star_A:.3f}")
        else:
            print(f"    monotone decreasing (no peak — pile-up < dead-time)")

    # F_B with multiple starts
    best_B = None
    for p0 in [[1.0, 1.0, 0.5], [2.0, 2.0, 1.0], [0.5, 0.5, 0.2], [3.0, 1.5, 1.0]]:
        try:
            popt, pcov = curve_fit(F_B, G_v, F_v, p0=p0,
                                    bounds=([0, 0, 0], [20, 20, 20]),
                                    sigma=sigma, absolute_sigma=False)
            ssr = np.sum(((F_B(G_v, *popt) - F_v) / sigma)**2)
            if best_B is None or ssr < best_B[1]:
                best_B = (popt, ssr)
        except Exception as e:
            pass

    if best_B is not None:
        popt_B, ssr_B = best_B
        tau_B, a_B, b_B = popt_B
        print(f"  F_B = exp(-{tau_B:.4f}·G) · (1 + {a_B:.4f}·{tau_B:.4f}·G "
              f"- {b_B:.4f}·({tau_B:.4f}·G)²)   χ²/dof={ssr_B/(len(G_v)-3):.2f}")
        # Plot peak
        G_dense = np.linspace(0, 2.5, 200)
        F_vals = F_B(G_dense, *popt_B)
        idx_peak = np.argmax(F_vals)
        print(f"    peak at G={G_dense[idx_peak]:.3f} (group_rate={G_dense[idx_peak]*G_NORM:.0f}), "
              f"F={F_vals[idx_peak]:.3f}")

    # ============ Stage 4: joint fit of full M11c-A and M11c-B models ============
    print(f"\n=== Stage 4: joint fit baseline × F(G) with bounds ===")
    m11cA_params = {}
    m11cB_params = {}
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        sub = df[mask_fit]
        # Downsample for speed
        if len(sub) > 200_000:
            idx = np.random.RandomState(42).choice(len(sub), 200_000, replace=False)
            sub_fit = sub.iloc[idx]
        else:
            sub_fit = sub

        X = (sub_fit["scipure_rate"].values, sub_fit["acd1_rate"].values,
             sub_fit["acdn_rate"].values, sub_fit["wide_rate"].values,
             sub_fit["large_rate"].values, sub_fit["G"].values)
        y = sub_fit["pho_rate"].values

        # M11c-A: baseline × F_A, p>0, τ>0
        c0_i, c1_i, cN_i, beta_i, gamma_i, b_i = m7_params[box]
        p0_A = [c0_i, c1_i, cN_i, beta_i, gamma_i, b_i] + list(best_A[0])
        try:
            popt_A, _ = curve_fit(joint_model_A, X, y, p0=p0_A,
                                   bounds=([0,0,0,0,0,-500, 0, 0],
                                           [10,15,15,10,5,500, 50, 50]),
                                   maxfev=20000)
            m11cA_params[box] = popt_A
            c0,c1,cN,beta,gamma,b,p,tau = popt_A
            print(f"  Box {box} M11c-A: c0={c0:.3f}, c1={c1:.3f}, cN={cN:.3f}, "
                  f"β={beta:.3f}, γ={gamma:.3f}, b={b:.1f}, p={p:.3f}, τ={tau:.3f}")
        except Exception as e:
            print(f"  Box {box} M11c-A fit failed: {e}")
            m11cA_params[box] = None

        # M11c-B: baseline × F_B, τ>0, a>0, b>0
        p0_B = [c0_i, c1_i, cN_i, beta_i, gamma_i, b_i] + list(best_B[0])
        try:
            popt_B, _ = curve_fit(joint_model_B, X, y, p0=p0_B,
                                   bounds=([0,0,0,0,0,-500, 0,0,0],
                                           [10,15,15,10,5,500, 20,20,20]),
                                   maxfev=20000)
            m11cB_params[box] = popt_B
            c0,c1,cN,beta,gamma,b,tau,a,bp = popt_B
            print(f"  Box {box} M11c-B: c0={c0:.3f}, c1={c1:.3f}, cN={cN:.3f}, "
                  f"β={beta:.3f}, γ={gamma:.3f}, b={b:.1f}, τ={tau:.3f}, a={a:.3f}, b={bp:.3f}")
        except Exception as e:
            print(f"  Box {box} M11c-B fit failed: {e}")
            m11cB_params[box] = None

    # ============ Stage 5: residuals comparison ============
    for n in ["M1", "M7", "M11cA", "M11cB"]:
        df[f"resid_{n}"] = np.nan
    m1_params_by_box = {}
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        m1_params_by_box[box] = fit_m1_linear(df[mask_fit])

    for box in "ABC":
        mask_apply = df["box"] == box
        sub = df[mask_apply]
        pho_pred_m1 = predict_m1(sub, m1_params_by_box[box])
        df.loc[mask_apply, "resid_M1"] = (sub["pho_rate"].values - pho_pred_m1) / SCI_REF

        c0, c1, cN, beta, gamma, b = m7_params[box]
        pho_pred_m7 = predict_m7(sub, c0, c1, cN, beta, gamma, b)
        df.loc[mask_apply, "resid_M7"] = (sub["pho_rate"].values - pho_pred_m7) / SCI_REF

        X_all = (sub["scipure_rate"].values, sub["acd1_rate"].values,
                 sub["acdn_rate"].values, sub["wide_rate"].values,
                 sub["large_rate"].values, sub["G"].values)
        if m11cA_params[box] is not None:
            pho_A = joint_model_A(X_all, *m11cA_params[box])
            df.loc[mask_apply, "resid_M11cA"] = (sub["pho_rate"].values - pho_A) / SCI_REF
        if m11cB_params[box] is not None:
            pho_B = joint_model_B(X_all, *m11cB_params[box])
            df.loc[mask_apply, "resid_M11cB"] = (sub["pho_rate"].values - pho_B) / SCI_REF

    print(f"\n=== RMS by Sci bin ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'M1':>9s}  {'M7':>9s}  {'M11cA':>9s}  {'M11cB':>9s}")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1", "M7", "M11cA", "M11cB"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>9.1f}  {rmss[1]:>9.1f}  {rmss[2]:>9.1f}  {rmss[3]:>9.1f}")

    print(f"\n=== Median residual by Sci bin ===")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median()
                for n in ["M1", "M7", "M11cA", "M11cB"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{meds[0]:>+9.1f}  {meds[1]:>+9.1f}  {meds[2]:>+9.1f}  {meds[3]:>+9.1f}")

    print(f"\n=== RMS by group_rate bin ===")
    g_edges = [1800, 4000, 6000, 8000, 10000, 14000, 25000]
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1", "M7", "M11cA", "M11cB"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>9.1f}  {rmss[1]:>9.1f}  {rmss[2]:>9.1f}  {rmss[3]:>9.1f}")

    print(f"\n=== Median residual by group_rate bin ===")
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median()
                for n in ["M1", "M7", "M11cA", "M11cB"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{meds[0]:>+9.1f}  {meds[1]:>+9.1f}  {meds[2]:>+9.1f}  {meds[3]:>+9.1f}")

    # ============ Plots ============
    # Empirical F(G) + parametric fits
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for box, color in zip("ABC", ["C0","C1","C2"]):
        ax.plot(bc_g*G_NORM, F_emp[box], "o-", color=color, lw=1.5, ms=4,
                label=f"Box {box} empirical", alpha=0.7)
    G_dense = np.linspace(0.05, 2.5, 300)
    if best_A is not None:
        ax.plot(G_dense*G_NORM, F_A(G_dense, *best_A[0]), "-", color="red", lw=2,
                label=f"F_A = (1+{best_A[0][0]:.2f}G)·exp(-{best_A[0][1]:.2f}G)")
    if best_B is not None:
        ax.plot(G_dense*G_NORM, F_B(G_dense, *best_B[0]), "-", color="darkgreen", lw=2,
                label=f"F_B: τ={best_B[0][0]:.2f}, a={best_B[0][1]:.2f}, b={best_B[0][2]:.2f}")
    ax.axhline(1.0, color="k", ls=":", lw=0.7)
    ax.set_xlabel("group_rate [cnt/s/box]")
    ax.set_ylabel("F(G) = PHO_obs / PHO_M7_pred")
    ax.set_title("M11c: physical F(G) with bounded parameters (p>0, τ>0, a>0, b>0)")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "m11c_F_bounded.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Residual plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey="row")
    SCI_MIN, SCI_MAX = 300, 4500
    G_MIN_p, G_MAX_p = 1800, 25000
    bins_s = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bins_g_p = np.logspace(np.log10(G_MIN_p), np.log10(G_MAX_p), 40)
    bc_s = 0.5 * (bins_s[:-1] + bins_s[1:])
    bc_g_p = 0.5 * (bins_g_p[:-1] + bins_g_p[1:])

    for col_idx, name in enumerate(["M1", "M7", "M11cA", "M11cB"]):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub = df[df["box"] == box]
            med_s = median_per_bin(sub["sci_rate"].values,
                                   sub[f"resid_{name}"].values, bins_s)
            med_g = median_per_bin(sub["group_rate"].values,
                                   sub[f"resid_{name}"].values, bins_g_p)
            axes[0, col_idx].plot(bc_s, med_s, "-", color=color, lw=2, label=f"Box {box}")
            axes[1, col_idx].plot(bc_g_p, med_g, "-", color=color, lw=2, label=f"Box {box}")
        for ax, xlim, xlab in zip(axes[:, col_idx],
                                   [(SCI_MIN, SCI_MAX), (G_MIN_p, G_MAX_p)],
                                   ["Sci [cnt/s/det]", "group_rate [cnt/s/box]"]):
            ax.axhline(0, color="k", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(-700, 250)
            ax.set_xlabel(xlab)
            ax.grid(alpha=0.3, which="both")
        axes[0, col_idx].set_title(f"{name}", fontsize=11)
    axes[0, 0].set_ylabel("residual\n(vs per-det Sci)")
    axes[1, 0].set_ylabel("residual\n(vs group_rate)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("M11c: physically-bounded non-linear F(G) — interpretable params",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m11c_physical.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
