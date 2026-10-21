#!/usr/bin/env python3
"""M11b: stage the non-linear fit to recover clean physical interpretation.

Issue with M11 (joint fit): scipy found a local minimum where baseline
coefficients drift (c0: 1.31→0.89) so that bigger F(G) compensates. Total
fit is good but individual params lose meaning.

Fix (staged fit):
  Stage 1. Fit M7 linear coefficients ONLY on low-G data (G < 0.6, i.e.,
           group_rate < 6000). In this region empirical F(G) ≈ 1, so M7
           coefficients reflect true linear physics.
  Stage 2. On ALL data, compute empirical F(G) = PHO_obs / PHO_M7_pred.
  Stage 3. Fit parametric form F(G) = (1 + p·G)·exp(-τ·G) with bounds
           p ≥ 0, τ ≥ 0 to enforce physical interpretation.

Compare M11b to:
  M1, M7, M11 (unconstrained joint fit)

Also: try a richer non-linear form to handle the SHARP transition observed:
  F(G) = 1 + p·G^a · exp(-q·G^b)
  or
  F(G) = exp(p·G) · exp(-q·G^n)  with n=3 or 4 for sharper falloff

The empirical F(G) shows:
  - F=1 below G=0.6
  - peak +5% at G=0.75
  - drops to 0.65 at G=1.5
  → very sharp transition, need n>2 in the cutoff term
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
G_LIN_MAX = 0.6   # use group_rate < 6000 for "linear" baseline fit
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


# ============ F(G) parametric candidates ============
def F_simple(G, p, tau):
    """(1 + p·G) · exp(-τ·G)  -- two params, physical pile-up × dead-time"""
    return (1.0 + p*G) * np.exp(-tau*G)


def F_sharp(G, p, tau, n):
    """(1 + p·G) · exp(-τ·G^n)  -- three params, sharper cutoff for large n"""
    return (1.0 + p*G) * np.exp(-tau*np.power(G, n))


def F_double_exp(G, p, q):
    """exp(p·G) · exp(-q·G^3)  -- pile-up exp gain and cubic cutoff"""
    return np.exp(p*G) * np.exp(-q*G**3)


# ============ Fits ============
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

    # ============ Stage 1: M7 linear baseline on LOW-G data only ============
    print(f"\n=== Stage 1: M7 linear baseline (group_rate < {G_LIN_MAX*G_NORM:.0f}) ===")
    m7_params = {}
    for box in "ABC":
        mask_fit = ((df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
                    & (df["G"] < G_LIN_MAX))
        params = fit_m7_linear(df[mask_fit])
        m7_params[box] = params
        c0, c1, cN, beta, gamma, b = params
        print(f"  Box {box} N={mask_fit.sum():,}: c0={c0:.3f}, c1={c1:.3f}, "
              f"cN={cN:.3f}, β={beta:.3f}, γ={gamma:.3f}, b={b:.1f}")

    # ============ Stage 2: empirical F(G) per Box ============
    print(f"\n=== Stage 2: empirical F(G), now using low-G-only baseline ===")
    G_MIN, G_MAX = 1800/G_NORM, 25000/G_NORM
    bins_g = np.linspace(G_MIN, G_MAX, 40)
    bc_g = 0.5 * (bins_g[:-1] + bins_g[1:])

    F_emp = {}
    for box in "ABC":
        sub = df[df["box"] == box]
        c0, c1, cN, beta, gamma, b = m7_params[box]
        pho_pred_m7 = predict_m7(sub, c0, c1, cN, beta, gamma, b)
        ratio = sub["pho_rate"].values / np.maximum(pho_pred_m7, 1e-3)
        med = median_per_bin(sub["G"].values, ratio, bins_g)
        F_emp[box] = med
        # Sanity at low G
        low_g_mask = (sub["G"] < 0.3) & (sub["pho_rate"] > 100)
        F_low = (sub.loc[low_g_mask, "pho_rate"].values
                 / pho_pred_m7[low_g_mask.values]).mean()
        print(f"  Box {box}: F(G≈0.3) median ≈ {F_low:.4f} (should be ~1)")

    # ============ Stage 3: fit parametric F(G) with PHYSICAL bounds ============
    print(f"\n=== Stage 3: fit F(G) on combined (box-pooled) data with bounds p≥0, τ≥0 ===")
    # Combine across boxes (F is supposed to be box-independent)
    F_combined = np.nanmean(np.stack([F_emp[box] for box in "ABC"]), axis=0)
    valid = np.isfinite(F_combined)
    G_v = bc_g[valid]
    F_v = F_combined[valid]
    weights = np.array([(df["G"].values >= bins_g[i]) & (df["G"].values < bins_g[i+1])
                        for i in range(len(bins_g)-1)])[valid].sum(axis=1).astype(float)
    weights /= weights.sum()
    sigma = 1.0 / np.sqrt(np.maximum(weights, 1e-4))  # weight by sample count

    # Try simple form first
    try:
        popt_s, pcov_s = curve_fit(F_simple, G_v, F_v, p0=[0.1, 0.5],
                                    bounds=([0, 0], [10, 10]), sigma=sigma,
                                    absolute_sigma=False)
        p_s, tau_s = popt_s
        print(f"  F_simple = (1+{p_s:.3f}·G)·exp(-{tau_s:.3f}·G)")
        if p_s > tau_s:
            G_star = 1/tau_s - 1/p_s
            print(f"    peak at G*={G_star:.2f}, F(G*)={F_simple(G_star, *popt_s):.3f}")
    except Exception as e:
        print(f"  F_simple fit failed: {e}")
        popt_s = None

    # Sharper form (third-power cutoff)
    try:
        popt_sh, pcov_sh = curve_fit(F_sharp, G_v, F_v, p0=[0.2, 0.1, 3.0],
                                      bounds=([0, 0, 1.5], [10, 10, 6]),
                                      sigma=sigma, absolute_sigma=False)
        p_sh, tau_sh, n_sh = popt_sh
        print(f"  F_sharp  = (1+{p_sh:.3f}·G)·exp(-{tau_sh:.3f}·G^{n_sh:.2f})")
    except Exception as e:
        print(f"  F_sharp fit failed: {e}")
        popt_sh = None

    # Double-exp form
    try:
        popt_de, pcov_de = curve_fit(F_double_exp, G_v, F_v, p0=[0.1, 0.1],
                                      bounds=([0, 0], [10, 10]),
                                      sigma=sigma, absolute_sigma=False)
        p_de, q_de = popt_de
        print(f"  F_de    = exp({p_de:.3f}·G)·exp(-{q_de:.3f}·G³)")
    except Exception as e:
        print(f"  F_de fit failed: {e}")
        popt_de = None

    # ============ Plot empirical F(G) and parametric fits ============
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for box, color in zip("ABC", ["C0","C1","C2"]):
        ax.plot(bc_g*G_NORM, F_emp[box], "o-", color=color, lw=1.5, ms=4,
                label=f"Box {box}", alpha=0.7)
    G_dense = np.linspace(G_MIN, G_MAX, 200)
    if popt_s is not None:
        ax.plot(G_dense*G_NORM, F_simple(G_dense, *popt_s), "--", color="red", lw=2,
                label=f"F_simple: (1+{popt_s[0]:.2f}G)·exp(-{popt_s[1]:.2f}G)")
    if popt_sh is not None:
        ax.plot(G_dense*G_NORM, F_sharp(G_dense, *popt_sh), "-.", color="darkgreen", lw=2,
                label=f"F_sharp: (1+{popt_sh[0]:.2f}G)·exp(-{popt_sh[1]:.2f}·G^{popt_sh[2]:.1f})")
    if popt_de is not None:
        ax.plot(G_dense*G_NORM, F_double_exp(G_dense, *popt_de), ":", color="purple", lw=2,
                label=f"F_de: exp({popt_de[0]:.2f}G)·exp(-{popt_de[1]:.2f}·G³)")
    ax.axhline(1.0, color="k", ls=":", lw=0.7)
    ax.axvline(G_LIN_MAX*G_NORM, color="gray", ls="--", lw=0.7,
               label=f"low-G fit boundary ({G_LIN_MAX*G_NORM:.0f})")
    ax.set_xlabel("group_rate [cnt/s/box]")
    ax.set_ylabel("F(G) = PHO_obs / PHO_M7_pred")
    ax.set_title("M11b: empirical correction factor F(G) and parametric fits\n"
                 "M7 baseline fit on low-G data only — clean physical interpretation")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "m11b_F_fits.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # ============ Stage 4: apply F to all data, compute residuals ============
    # Use the BEST F form (F_sharp if it succeeded, else F_simple)
    best_F = F_sharp if popt_sh is not None else (F_simple if popt_s is not None else None)
    best_popt = popt_sh if popt_sh is not None else popt_s
    best_name = "F_sharp" if popt_sh is not None else "F_simple"

    print(f"\n=== Stage 4: residual comparison (using {best_name}) ===")
    for n in ["M1", "M7", "M11b"]:
        df[f"resid_{n}"] = np.nan

    m1_params_by_box = {}
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        m1_params_by_box[box] = fit_m1_linear(df[mask_fit])

    for box in "ABC":
        mask_apply = df["box"] == box
        sub = df[mask_apply]
        # M1
        pho_pred_m1 = predict_m1(sub, m1_params_by_box[box])
        df.loc[mask_apply, "resid_M1"] = (sub["pho_rate"].values - pho_pred_m1) / SCI_REF
        # M7 (low-G calibrated)
        c0, c1, cN, beta, gamma, b = m7_params[box]
        pho_pred_m7 = predict_m7(sub, c0, c1, cN, beta, gamma, b)
        df.loc[mask_apply, "resid_M7"] = (sub["pho_rate"].values - pho_pred_m7) / SCI_REF
        # M11b: M7 baseline × F(G)
        F_vals = best_F(sub["G"].values, *best_popt)
        pho_pred_m11b = pho_pred_m7 * F_vals
        df.loc[mask_apply, "resid_M11b"] = (sub["pho_rate"].values - pho_pred_m11b) / SCI_REF

    print(f"\n=== RMS by Sci bin ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'M1':>9s}  {'M7':>9s}  {'M11b':>9s}   M11b-M1%  M11b-M7%")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1", "M7", "M11b"]]
        d_m1 = 100 * (rmss[2] - rmss[0]) / rmss[0]
        d_m7 = 100 * (rmss[2] - rmss[1]) / rmss[1]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>9.1f}  {rmss[1]:>9.1f}  {rmss[2]:>9.1f}   "
              f"{d_m1:>+7.1f}%  {d_m7:>+7.1f}%")

    print(f"\n=== Median residual by Sci bin ===")
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'M1':>9s}  {'M7':>9s}  {'M11b':>9s}")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in ["M1", "M7", "M11b"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{meds[0]:>+9.1f}  {meds[1]:>+9.1f}  {meds[2]:>+9.1f}")

    print(f"\n=== RMS by group_rate bin ===")
    g_edges = [1800, 4000, 6000, 8000, 10000, 14000, 25000]
    print(f"{'group_rate':>15s}  {'N':>10s}  {'M1':>9s}  {'M7':>9s}  {'M11b':>9s}")
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1", "M7", "M11b"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>9.1f}  {rmss[1]:>9.1f}  {rmss[2]:>9.1f}")

    print(f"\n=== Median residual by group_rate bin ===")
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in ["M1", "M7", "M11b"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{meds[0]:>+9.1f}  {meds[1]:>+9.1f}  {meds[2]:>+9.1f}")

    # ============ Final plot: M1 vs M7 vs M11b ============
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row")
    SCI_MIN, SCI_MAX = 300, 4500
    G_MIN_p, G_MAX_p = 1800, 25000
    bins_s = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bins_g_p = np.logspace(np.log10(G_MIN_p), np.log10(G_MAX_p), 40)
    bc_s = 0.5 * (bins_s[:-1] + bins_s[1:])
    bc_g_p = 0.5 * (bins_g_p[:-1] + bins_g_p[1:])

    for col_idx, name in enumerate(["M1", "M7", "M11b"]):
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
    fig.suptitle(f"M11b: M7 baseline (low-G fit) × {best_name}(G) — clean physics",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m11b_staged.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
