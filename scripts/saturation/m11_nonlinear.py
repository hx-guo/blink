#!/usr/bin/env python3
"""M11: physically motivated non-linear form.

Replaces M9/M10's polynomial-in-G terms with a multiplicative rate-correction
factor that has explicit physical meaning:

  PHO = baseline · F(G)

  baseline = c0·Sci_pure + c1·Sci_ACD1 + cN·Sci_ACDN + β·Wide + γ·Large + b
  F(G)     = (1 + p·G) · exp(-τ·G)
  G        = group_rate / 10000

Physical meaning of F(G):
  - (1 + p·G):  pile-up gain factor; p > 0 means at moderate group rate,
                sub-threshold event pairs combine into PHO band → extra PHO
  - exp(-τ·G):  paralyzable dead-time / saturation cutoff; τ > 0 means at
                high group rate, ADC processing time dominates → PHO loss

The product:
  - at low G: F → 1 + (p-τ)·G        (linear deviation from 1)
  - peak at G* = 1/τ - 1/p           (if p > τ)
  - at high G: F → 0                  (saturation)

Initial coefficients seeded from M7 fit. Non-linear (p, τ) are fitted jointly
via scipy.optimize.curve_fit (Levenberg-Marquardt).

Comparison:
  M1   : (1+α)·Sci + β·Wide + γ·Large + b                          [baseline]
  M7   : c0·Sp + c1·S1 + cN·SN + β·W + γ·L + b                     [ACD]
  M10  : M7 + δ·Other + ε·Other²                                    [polynomial pile-up]
  M11  : M7 baseline × (1 + p·G)·exp(-τ·G)                          [non-linear F(G)]
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

    df["sci_rate"]      = df["Sci"]      / df["length"]
    df["scipure_rate"]  = df["Sci_pure"] / df["length"]
    df["acd1_rate"]     = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]     = df["Sci_ACDN"] / df["length"]
    df["wide_rate"]     = df["Wide"]     / df["length"]
    df["large_rate"]    = df["Large"]    / df["length"]
    df["pho_rate"]      = df["PHO"]      / df["length"]
    df["group_rate"]    = df["sci_sec_total"] / df["length"]
    df["G"]             = df["group_rate"] / G_NORM
    df["det_global"]    = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

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


# ------------ Model: PHO = baseline(linear) × F(G; p, τ) ------------
def model_pho(X, c0, c1, cN, beta, gamma, b, p, tau):
    Sp, S1, SN, W, L, G = X
    baseline = c0*Sp + c1*S1 + cN*SN + beta*W + gamma*L + b
    F = (1.0 + p*G) * np.exp(-tau*G)
    return baseline * F


def fit_m7_linear(sub):
    """M7 linear baseline fit, used to seed initial guess."""
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

    # ============ Step 1: empirical F(G) shape using M7 baseline ============
    print(f"\n=== Step 1: empirical F(G) = PHO_obs / PHO_M7_pred ===")
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    G_MIN, G_MAX = 1800/G_NORM, 25000/G_NORM
    bins_g = np.linspace(G_MIN, G_MAX, 30)
    bc_g = 0.5 * (bins_g[:-1] + bins_g[1:])

    m7_params_by_box = {}
    for box, color in zip("ABC", ["C0","C1","C2"]):
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        c0, c1, cN, beta, gamma, b = fit_m7_linear(df[mask_fit])
        m7_params_by_box[box] = (c0, c1, cN, beta, gamma, b)
        print(f"  Box {box} M7 linear: c0={c0:.3f}, c1={c1:.3f}, cN={cN:.3f}, "
              f"β={beta:.3f}, γ={gamma:.3f}, b={b:.1f}")

        # Apply to all data
        sub = df[df["box"] == box]
        pho_pred = predict_m7(sub, c0, c1, cN, beta, gamma, b)
        ratio = sub["pho_rate"].values / np.maximum(pho_pred, 1e-3)
        # Empirical F(G)
        med = median_per_bin(sub["G"].values, ratio, bins_g)
        ax.plot(bc_g * G_NORM, med, "o-", color=color, lw=2, markersize=5,
                label=f"Box {box} empirical")

    # Overlay candidate parametric forms (with p=1.0, τ=0.5 — illustrative)
    G_curve = np.linspace(0.1, 2.5, 100)
    for p_cand, tau_cand, ls in [(1.0, 0.5, "--"), (1.5, 0.7, "-."), (0.5, 0.3, ":")]:
        F_cand = (1 + p_cand*G_curve) * np.exp(-tau_cand*G_curve)
        ax.plot(G_curve * G_NORM, F_cand, ls, color="gray", alpha=0.6,
                label=f"F: p={p_cand}, τ={tau_cand}")
    ax.axhline(1.0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("group_rate [cnt/s/box]")
    ax.set_ylabel("PHO_obs / PHO_M7_pred")
    ax.set_title("Empirical correction factor F(G)\n"
                 "(F=1 means M7 is exact; F>1 = extra PHO; F<1 = PHO loss)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "m11_empirical_F.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    # ============ Step 2: joint non-linear fit M11 ============
    print(f"\n=== Step 2: joint non-linear fit (curve_fit on full PHO) ===")
    m11_params_by_box = {}
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        sub = df[mask_fit]
        # Down-sample for speed (curve_fit on 1M+ rows is slow)
        if len(sub) > 200_000:
            idx = np.random.RandomState(42).choice(len(sub), 200_000, replace=False)
            sub_fit = sub.iloc[idx]
        else:
            sub_fit = sub

        X = (sub_fit["scipure_rate"].values,
             sub_fit["acd1_rate"].values,
             sub_fit["acdn_rate"].values,
             sub_fit["wide_rate"].values,
             sub_fit["large_rate"].values,
             sub_fit["G"].values)
        y = sub_fit["pho_rate"].values

        # Initial guess from M7 + small (p, τ)
        c0_init, c1_init, cN_init, beta_init, gamma_init, b_init = m7_params_by_box[box]
        p0 = [c0_init, c1_init, cN_init, beta_init, gamma_init, b_init, 0.1, 0.1]

        try:
            popt, pcov = curve_fit(model_pho, X, y, p0=p0, maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            m11_params_by_box[box] = popt
            c0, c1, cN, beta, gamma, b, p, tau = popt
            print(f"  Box {box} M11:")
            print(f"    c0={c0:.3f}±{perr[0]:.3f}, c1={c1:.3f}±{perr[1]:.3f}, "
                  f"cN={cN:.3f}±{perr[2]:.3f}")
            print(f"    β={beta:.3f}±{perr[3]:.3f}, γ={gamma:.3f}±{perr[4]:.3f}, "
                  f"b={b:.1f}±{perr[5]:.1f}")
            print(f"    p={p:.4f}±{perr[6]:.4f}, τ={tau:.4f}±{perr[7]:.4f}")
            if p > tau:
                G_star = 1/tau - 1/p
                F_star = (1 + p*G_star) * np.exp(-tau*G_star)
                print(f"    → F(G) peak at G*={G_star:.2f} (group_rate={G_star*G_NORM:.0f}), "
                      f"amplitude F(G*)={F_star:.3f}")
            else:
                print(f"    → F(G) monotone decreasing (no pile-up gain peak)")
        except Exception as e:
            print(f"  Box {box} M11 fit FAILED: {e}")
            m11_params_by_box[box] = None

    # ============ Step 3: compare M1, M7, M11 residuals ============
    print(f"\n=== Step 3: residual comparison ===")
    m1_params_by_box = {}
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        m1_params_by_box[box] = fit_m1_linear(df[mask_fit])

    for name, label in [("M1", "M1"), ("M7", "M7"), ("M11", "M11")]:
        col = f"resid_{name}"
        df[col] = np.nan
    for box in "ABC":
        mask_apply = df["box"] == box
        sub = df[mask_apply]
        # M1
        pho_pred_m1 = predict_m1(sub, m1_params_by_box[box])
        df.loc[mask_apply, "resid_M1"] = (sub["pho_rate"].values - pho_pred_m1) / SCI_REF
        # M7
        c0, c1, cN, beta, gamma, b = m7_params_by_box[box]
        pho_pred_m7 = predict_m7(sub, c0, c1, cN, beta, gamma, b)
        df.loc[mask_apply, "resid_M7"] = (sub["pho_rate"].values - pho_pred_m7) / SCI_REF
        # M11
        if m11_params_by_box[box] is not None:
            X_all = (sub["scipure_rate"].values, sub["acd1_rate"].values,
                     sub["acdn_rate"].values, sub["wide_rate"].values,
                     sub["large_rate"].values, sub["G"].values)
            pho_pred_m11 = model_pho(X_all, *m11_params_by_box[box])
            df.loc[mask_apply, "resid_M11"] = (sub["pho_rate"].values - pho_pred_m11) / SCI_REF

    # RMS table
    print(f"\n=== RMS by Sci bin ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'M1':>9s}  {'M7':>9s}  {'M11':>9s}   M11-M1%  M11-M7%")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1", "M7", "M11"]]
        d_m1 = 100 * (rmss[2] - rmss[0]) / rmss[0]
        d_m7 = 100 * (rmss[2] - rmss[1]) / rmss[1]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>9.1f}  {rmss[1]:>9.1f}  {rmss[2]:>9.1f}   "
              f"{d_m1:>+6.1f}%  {d_m7:>+6.1f}%")

    print(f"\n=== Median residual by Sci bin ===")
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'M1':>9s}  {'M7':>9s}  {'M11':>9s}")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in ["M1", "M7", "M11"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{meds[0]:>+9.1f}  {meds[1]:>+9.1f}  {meds[2]:>+9.1f}")

    print(f"\n=== RMS by group_rate bin ===")
    g_edges = [1800, 4000, 6000, 8000, 10000, 14000, 25000]
    print(f"{'group_rate':>15s}  {'N':>10s}  {'M1':>9s}  {'M7':>9s}  {'M11':>9s}")
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2))
                for n in ["M1", "M7", "M11"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{rmss[0]:>9.1f}  {rmss[1]:>9.1f}  {rmss[2]:>9.1f}")

    print(f"\n=== Median residual by group_rate bin ===")
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in ["M1", "M7", "M11"]]
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  "
              f"{meds[0]:>+9.1f}  {meds[1]:>+9.1f}  {meds[2]:>+9.1f}")

    # ============ Step 4: plot residual vs Sci and vs group_rate ============
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row")
    SCI_MIN, SCI_MAX = 300, 4500
    G_MIN_p, G_MAX_p = 1800, 25000
    bins_s = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bins_g_p = np.logspace(np.log10(G_MIN_p), np.log10(G_MAX_p), 40)
    bc_s = 0.5 * (bins_s[:-1] + bins_s[1:])
    bc_g_p = 0.5 * (bins_g_p[:-1] + bins_g_p[1:])

    for col_idx, name in enumerate(["M1", "M7", "M11"]):
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

    fig.suptitle("M11 = M7 baseline × F(G), F(G) = (1 + p·G)·exp(-τ·G)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m11_nonlinear.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
