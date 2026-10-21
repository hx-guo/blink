#!/usr/bin/env python3
"""M11d: decoupled fit — M7 baseline fixed, then F(G) fitted on residual ratio.

Avoids the joint-fit degeneracy (baseline × F can swap variance, giving
unphysical coefficients).

Step 1: M7 linear fit on FULL data → m7_params (true physical yields).
Step 2: For each row, compute R = PHO_obs / PHO_M7_pred (row-level ratio).
Step 3: Fit F(G) to (G, R) pairs at ROW LEVEL (no binning), with physical
        bounded form.
Step 4: Apply M7 × F(G) as final prediction. Compute residuals.

Physical form chosen (3 params, all positive):

    F(G) = (1 + a·G − b·G²) · exp(−τ·G)

  exp(−τ·G):   paralyzable ADC dead-time loss (uniform across bands)
  1 + a·G:     2-event pile-up gain for PHO band (sub-thresh pairs)
  − b·G²:      3-event pile-up loss from PHO band (energy shifts to Wide)

Bounds: a > 0, b > 0, τ > 0. F(0) = 1 automatic.

Why this form?
  Pile-up rate of n-event clusters scales as G^(n-1)/n!. For n=2: linear in G.
  For n=3: quadratic in G. Each n-cluster has a probability of landing in
  the PHO band, captured by a and b. The exp(−τ·G) is the standard
  paralyzable dead-time factor for ADC dead time τ.

Alternative form for comparison (only 2 params):

    F(G) = (1 + a·G) · exp(−τ·G)         [no 3-event loss]
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


# ============ F(G) candidates ============
def F_2param(G, a, tau):
    """Pile-up gain × dead-time, 2 params (1 + a·G)·exp(-τ·G)."""
    return (1.0 + a*G) * np.exp(-tau*G)


def F_3param(G, a, b, tau):
    """Pile-up gain (2-event) − 3-event loss × dead-time, 3 params."""
    return (1.0 + a*G - b*G*G) * np.exp(-tau*G)


def F_powerlaw_dt(G, tau, n):
    """Non-linear paralyzable dead-time: F(G) = exp(-τ·G^n).
    n=1 is standard paralyzable; n>1 is "cascade" dead-time (events queue up).
    Physical: τ_dead(R) = τ_0 · R^(n-1), so PHO_obs/PHO_true = exp(-R·τ_dead) = exp(-τ_0·R^n)."""
    return np.exp(-tau * np.power(np.maximum(G, 1e-9), n))


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

    # ============ Step 1: M7 linear fit on full data ============
    print(f"\n=== Step 1: M7 linear baseline (full data) ===")
    m7_params = {}
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        params = fit_m7_linear(df[mask_fit])
        m7_params[box] = params
        c0, c1, cN, beta, gamma, b = params
        print(f"  Box {box}: c0={c0:.3f}, c1={c1:.3f}, cN={cN:.3f}, "
              f"β={beta:.3f}, γ={gamma:.3f}, b={b:.1f}")

    # ============ Step 2: row-level ratio (PHO_obs / PHO_M7_pred) ============
    print(f"\n=== Step 2: row-level ratio R = PHO_obs / PHO_M7_pred ===")
    for box in "ABC":
        mask = df["box"] == box
        sub = df[mask]
        c0, c1, cN, beta, gamma, b = m7_params[box]
        pho_pred = predict_m7(sub, c0, c1, cN, beta, gamma, b)
        df.loc[mask, "pho_m7_pred"] = pho_pred
        df.loc[mask, "R_ratio"] = sub["pho_rate"].values / np.maximum(pho_pred, 1e-3)

    # Filter rows for fit: PHO_M7_pred > 200 (avoid ratio blowup)
    fit_mask = (df["pho_m7_pred"] > 200) & (df["R_ratio"] > 0) & (df["R_ratio"] < 3)
    print(f"  Filtered rows: {fit_mask.sum():,} / {len(df):,}")

    # Bin into G slices; one data point per box-G bin → equal weight across G
    bins_g_fit = np.linspace(0.18, 2.5, 50)
    bc_g_fit = 0.5 * (bins_g_fit[:-1] + bins_g_fit[1:])
    G_pts, R_pts, N_pts = [], [], []
    for box in "ABC":
        mask_box = fit_mask & (df["box"] == box)
        G_vals = df.loc[mask_box, "G"].values
        R_vals = df.loc[mask_box, "R_ratio"].values
        for i in range(len(bins_g_fit) - 1):
            m = (G_vals >= bins_g_fit[i]) & (G_vals < bins_g_fit[i+1])
            if m.sum() < 300:
                continue
            G_pts.append(bc_g_fit[i])
            R_pts.append(np.median(R_vals[m]))
            N_pts.append(m.sum())
    G_fit = np.array(G_pts)
    R_fit = np.array(R_pts)
    N_arr = np.array(N_pts, dtype=float)
    # sigma per bin: 1/sqrt(N) of inter-quartile spread, normalized
    sigma_fit = 1.0 / np.sqrt(N_arr)
    sigma_fit /= sigma_fit.min()
    print(f"  Fit data: {len(G_fit)} (box, G-bin) median points across 3 boxes")
    print(f"  G range: {G_fit.min():.2f} to {G_fit.max():.2f}")
    print(f"  R range: {R_fit.min():.3f} to {R_fit.max():.3f}")

    # ============ Step 3: fit F(G) with physical bounds ============
    print(f"\n=== Step 3: fit F(G) on row-level (G, R) data ===")

    # 2-param fit via brute-force grid search (more reliable than curve_fit with bounds)
    print("  Grid search F_2param(a, τ) ...")
    a_grid = np.linspace(0.0, 8.0, 80)
    tau_grid = np.linspace(0.0, 4.0, 80)
    A, T = np.meshgrid(a_grid, tau_grid, indexing="ij")
    ssr_grid = np.zeros_like(A)
    for i in range(len(a_grid)):
        for j in range(len(tau_grid)):
            F_pred = F_2param(G_fit, A[i,j], T[i,j])
            ssr_grid[i,j] = np.sum((R_fit - F_pred)**2 / sigma_fit**2)
    i_min, j_min = np.unravel_index(np.argmin(ssr_grid), ssr_grid.shape)
    a_brute = a_grid[i_min]; tau_brute = tau_grid[j_min]
    print(f"  Brute-force F_2: a={a_brute:.3f}, τ={tau_brute:.3f}, χ²={ssr_grid[i_min,j_min]:.2f}")
    # Now polish with curve_fit starting at brute point
    best_2 = None
    for p0 in [[a_brute, tau_brute], [0.5, 0.5], [2.0, 1.0], [0.1, 0.05], [3.0, 2.0]]:
        try:
            popt, pcov = curve_fit(F_2param, G_fit, R_fit, p0=p0,
                                    bounds=([0, 0], [50, 50]),
                                    sigma=sigma_fit, absolute_sigma=False,
                                    maxfev=10000)
            res = R_fit - F_2param(G_fit, *popt)
            ssr = np.sum((res/sigma_fit)**2)
            if best_2 is None or ssr < best_2[1]:
                best_2 = (popt, ssr)
        except Exception as e:
            pass

    if best_2 is not None:
        popt_2, ssr_2 = best_2
        a_2, tau_2 = popt_2
        rms_2 = np.sqrt(ssr_2 / len(G_fit))
        print(f"  F_2param: a={a_2:.4f}, τ={tau_2:.4f}    row-RMS={rms_2:.4f}")
        if a_2 > tau_2:
            G_star = 1/tau_2 - 1/a_2
            print(f"    peak at G*={G_star:.3f} (group_rate={G_star*G_NORM:.0f}), "
                  f"F(G*)={F_2param(G_star, *popt_2):.4f}")
        else:
            print(f"    monotone decreasing")

    # 3-param fit via grid search + polish
    print("  Grid search F_3param(a, b, τ) ...")
    a_grid3 = np.linspace(0.0, 6.0, 31)
    b_grid3 = np.linspace(0.0, 5.0, 26)
    tau_grid3 = np.linspace(0.0, 3.0, 31)
    best_ssr = np.inf
    best_abt = None
    for a_v in a_grid3:
        for b_v in b_grid3:
            for t_v in tau_grid3:
                F_pred = F_3param(G_fit, a_v, b_v, t_v)
                ssr = np.sum((R_fit - F_pred)**2 / sigma_fit**2)
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_abt = (a_v, b_v, t_v)
    print(f"  Brute-force F_3: a={best_abt[0]:.3f}, b={best_abt[1]:.3f}, "
          f"τ={best_abt[2]:.3f}, χ²={best_ssr:.2f}")
    # Polish with curve_fit
    best_3 = None
    for p0 in [list(best_abt), [0.5, 0.2, 0.5], [2.0, 1.0, 1.0],
              [3.0, 2.0, 2.0], [1.0, 0.5, 1.5], [5.0, 3.0, 2.0]]:
        try:
            popt, pcov = curve_fit(F_3param, G_fit, R_fit, p0=p0,
                                    bounds=([0, 0, 0], [50, 50, 50]),
                                    sigma=sigma_fit, absolute_sigma=False,
                                    maxfev=10000)
            res = R_fit - F_3param(G_fit, *popt)
            ssr = np.sum((res/sigma_fit)**2)
            if best_3 is None or ssr < best_3[1]:
                best_3 = (popt, ssr)
        except Exception as e:
            pass

    if best_3 is not None:
        popt_3, ssr_3 = best_3
        a_3, b_3, tau_3 = popt_3
        rms_3 = np.sqrt(ssr_3 / len(G_fit))
        print(f"  F_3param: a={a_3:.4f}, b={b_3:.4f}, τ={tau_3:.4f}   row-RMS={rms_3:.4f}")
        # Find peak numerically
        G_dense = np.linspace(0.01, 2.5, 500)
        F_vals = F_3param(G_dense, *popt_3)
        idx_peak = np.argmax(F_vals)
        print(f"    peak at G*={G_dense[idx_peak]:.3f} "
              f"(group_rate={G_dense[idx_peak]*G_NORM:.0f}), "
              f"F(G*)={F_vals[idx_peak]:.4f}")
        # Where does F cross 1 going down?
        below_one = np.where(F_vals < 1)[0]
        if len(below_one) > 0 and below_one[0] > idx_peak:
            G_cross = G_dense[below_one[0]]
            print(f"    F=1 (descending) at G={G_cross:.3f} "
                  f"(group_rate={G_cross*G_NORM:.0f})")

    # NEW: power-law dead-time form (no bump, just non-linear decay)
    print("  Grid search F_powerlaw_dt(τ, n) — non-linear paralyzable dead-time ...")
    tau_grid_p = np.linspace(0.001, 2.0, 100)
    n_grid_p = np.linspace(0.5, 6.0, 56)
    best_ssr_p = np.inf
    best_tn = None
    for t_v in tau_grid_p:
        for n_v in n_grid_p:
            F_pred = F_powerlaw_dt(G_fit, t_v, n_v)
            ssr = np.sum((R_fit - F_pred)**2 / np.ones_like(sigma_fit)**2)  # uniform weight
            if ssr < best_ssr_p:
                best_ssr_p = ssr
                best_tn = (t_v, n_v)
    print(f"  Brute-force F_pl: τ={best_tn[0]:.4f}, n={best_tn[1]:.3f}, χ²={best_ssr_p:.4f}")
    # Polish
    best_pl = None
    for p0 in [list(best_tn), [0.1, 2.0], [0.5, 3.0], [0.05, 4.0]]:
        try:
            popt, _ = curve_fit(F_powerlaw_dt, G_fit, R_fit, p0=p0,
                                bounds=([0, 0.5], [10, 8]),
                                maxfev=10000)
            res = R_fit - F_powerlaw_dt(G_fit, *popt)
            ssr = np.sum(res**2)
            if best_pl is None or ssr < best_pl[1]:
                best_pl = (popt, ssr)
        except Exception:
            pass
    if best_pl is not None:
        tau_pl, n_pl = best_pl[0]
        print(f"  F_powerlaw_dt: τ={tau_pl:.4f}, n={n_pl:.3f}    unweighted RMS={np.sqrt(best_pl[1]/len(G_fit)):.4f}")
        # Where does F = 0.5?
        if tau_pl > 0:
            G_half = (np.log(2) / tau_pl) ** (1/n_pl)
            print(f"    F=0.5 at G={G_half:.3f} (group_rate={G_half*G_NORM:.0f})")
        print(f"\n  Physical interpretation:")
        print(f"    Effective dead-time per event ∝ R^({n_pl:.2f}-1) = R^{n_pl-1:.2f}")
        print(f"    n={n_pl:.2f} ≈ {'linear' if abs(n_pl-1) < 0.3 else ('quadratic cascade' if abs(n_pl-2) < 0.4 else 'higher-order cascade')}")

    # Physical interpretation:
    if best_3 is not None:
        a_3, b_3, tau_3 = popt_3
        print(f"\n  Physical interpretation of M11d-3param:")
        print(f"    2-event pile-up PHO gain coefficient a = {a_3:.3f}")
        print(f"      → each Δ(G/10000) of group_rate adds {a_3*100:.1f}% relative PHO gain")
        print(f"    3-event pile-up PHO loss coefficient b = {b_3:.3f}")
        print(f"      → quadratic loss term, dominates above G ≈ {a_3/b_3:.2f} "
              f"(group_rate ≈ {a_3/b_3*G_NORM:.0f})")
        print(f"    Paralyzable dead-time τ = {tau_3:.3f}")
        print(f"      → exp(-τ·G) decays by 50% at G = {np.log(2)/tau_3:.2f} "
              f"(group_rate = {np.log(2)/tau_3*G_NORM:.0f})")

    # ============ Step 4: apply M7 × F(G), compute residuals ============
    print(f"\n=== Step 4: residual comparison ===")
    for n in ["M1", "M7", "M11d2", "M11d3", "M11dPL"]:
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

        if best_2 is not None:
            F2 = F_2param(sub["G"].values, *best_2[0])
            pho_m11d2 = pho_pred_m7 * F2
            df.loc[mask_apply, "resid_M11d2"] = (sub["pho_rate"].values - pho_m11d2) / SCI_REF

        if best_3 is not None:
            F3 = F_3param(sub["G"].values, *best_3[0])
            pho_m11d3 = pho_pred_m7 * F3
            df.loc[mask_apply, "resid_M11d3"] = (sub["pho_rate"].values - pho_m11d3) / SCI_REF

        if best_pl is not None:
            FPL = F_powerlaw_dt(sub["G"].values, *best_pl[0])
            pho_m11dpl = pho_pred_m7 * FPL
            df.loc[mask_apply, "resid_M11dPL"] = (sub["pho_rate"].values - pho_m11dpl) / SCI_REF

    print(f"\n=== RMS by Sci bin ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    names = ["M1", "M7", "M11d2", "M11d3", "M11dPL"]
    print(f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>9s}" for n in names))
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in names]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>9.1f}" for r in rmss))

    print(f"\n=== Median residual by Sci bin ===")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in names]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+9.1f}" for m in meds))

    print(f"\n=== RMS by group_rate bin ===")
    g_edges = [1800, 4000, 6000, 8000, 10000, 14000, 25000]
    print(f"{'group_rate':>15s}  {'N':>10s}  " + "  ".join(f"{n:>9s}" for n in names))
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in names]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>9.1f}" for r in rmss))

    print(f"\n=== Median residual by group_rate bin ===")
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in names]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+9.1f}" for m in meds))

    # ============ Plot F(G) and residuals ============
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # F(G) fit overlay
    ax = axes[0]
    bins_g = np.linspace(0.18, 2.5, 40)
    bc_g = 0.5 * (bins_g[:-1] + bins_g[1:])
    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = df[df["box"] == box]
        med = median_per_bin(sub["G"].values, sub["R_ratio"].values, bins_g)
        ax.plot(bc_g*G_NORM, med, "o-", color=color, lw=1.5, ms=4,
                label=f"Box {box} empirical", alpha=0.7)
    G_dense = np.linspace(0.01, 2.5, 300)
    if best_2 is not None:
        ax.plot(G_dense*G_NORM, F_2param(G_dense, *best_2[0]), "--", color="red", lw=2,
                label=f"F_2: (1+{best_2[0][0]:.3f}G)·exp(-{best_2[0][1]:.3f}G)")
    if best_3 is not None:
        ax.plot(G_dense*G_NORM, F_3param(G_dense, *best_3[0]), "-", color="darkgreen", lw=2,
                label=f"F_3: (1+{best_3[0][0]:.3f}G−{best_3[0][1]:.3f}G²)·exp(−{best_3[0][2]:.3f}G)")
    if best_pl is not None:
        ax.plot(G_dense*G_NORM, F_powerlaw_dt(G_dense, *best_pl[0]), "-", color="purple", lw=2.5,
                label=f"F_PL: exp(−{best_pl[0][0]:.3f}·G^{best_pl[0][1]:.2f})")
    ax.axhline(1.0, color="k", ls=":", lw=0.7)
    ax.set_xlabel("group_rate [cnt/s/box]")
    ax.set_ylabel("F(G) = PHO_obs / PHO_M7_pred")
    ax.set_title("M11d: physically-bounded F(G) decoupled from baseline fit")
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim(0.4, 1.2)
    ax.grid(alpha=0.3)

    # Residual: M11dPL vs M1
    ax = axes[1]
    bins_s = np.logspace(np.log10(300), np.log10(4500), 40)
    bc_s = 0.5 * (bins_s[:-1] + bins_s[1:])
    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = df[df["box"] == box]
        med_m1 = median_per_bin(sub["sci_rate"].values, sub["resid_M1"].values, bins_s)
        med_pl = median_per_bin(sub["sci_rate"].values, sub["resid_M11dPL"].values, bins_s)
        ax.plot(bc_s, med_m1, "--", color=color, lw=1.5, alpha=0.5,
                label=f"M1 {box}" if box=="A" else None)
        ax.plot(bc_s, med_pl, "-", color=color, lw=2,
                label=f"M11dPL {box}")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(300, 4500)
    ax.set_ylim(-700, 250)
    ax.set_xlabel("Sci [cnt/s/det]")
    ax.set_ylabel("median residual [cnt/s/det]")
    ax.set_title("Residual comparison: M1 (dashed) vs M11dPL = M7 × exp(−τ·G^n) (solid)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = OUT_DIR / "m11d_decoupled.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
