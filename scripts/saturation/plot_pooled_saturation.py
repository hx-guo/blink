#!/usr/bin/env python3
"""Saturation-type (FIFO dead-time) bend model on full pooled dataset.

Models compared against the linear/quadratic baselines:

  Linear:        Sci = b + a·PHO + c·Wide + d·Large
  Quadratic:     + (10 coefs of 2nd order)
  Saturation 1:  Sci = (PHO − β·Wide − γ·Large − b) / (1 + τ·PHO)
  Saturation 2:  same but with total = PHO + Wide + Large
  Saturation 3:  same as 1 but per-year intercept b(year)  (handles drift)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

CSV_DIR = Path("n_below_study/per_sec_csvs")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

L_THRESH = 50_000
SCI_MIN = 100
PHO_MIN = 100


def load_aggregate():
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs ...")
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
    use = list(dtype)
    parts = []
    for i, f in enumerate(files):
        try:
            parts.append(pd.read_csv(f, usecols=use, dtype=dtype))
        except Exception as e:
            print(f"  ERR {f.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")
    df = pd.concat(parts, ignore_index=True)
    g = df.groupby(["date", "box", "met_sec"], observed=True).agg(
        L_cycles=("L_cycles", "max"),
        PHO=("PHO", "sum"),
        Wide=("Wide", "sum"),
        Large=("Large", "sum"),
        Sci=("Sci", "sum"),
    ).reset_index()
    return g


def main():
    g = load_aggregate()
    g["length"] = g["L_cycles"] * 16e-6
    g["sci_rate"] = g["Sci"] / g["length"]
    g["PHO_rate"] = g["PHO"] / g["length"]
    g["Wide_rate"] = g["Wide"] / g["length"]
    g["Large_rate"] = g["Large"] / g["length"]
    g["year"] = g["date"].str.slice(0, 4).astype(int)

    mask = (
        (g["L_cycles"] > L_THRESH)
        & (g["sci_rate"] > SCI_MIN)
        & (g["PHO_rate"] > PHO_MIN)
        & np.isfinite(g["sci_rate"])
    )
    big = g[mask].copy().reset_index(drop=True)
    print(f"After filter: {len(big):,} bins, dates: {big['date'].nunique()}, "
          f"Sci range {big['sci_rate'].min():.0f}–{big['sci_rate'].max():.0f}")

    sci = big["sci_rate"].values
    pho = big["PHO_rate"].values
    wide = big["Wide_rate"].values
    large = big["Large_rate"].values
    year = big["year"].values

    # === Model 1: Linear ===
    A1 = np.column_stack([np.ones(len(sci)), pho, wide, large])
    c1, *_ = np.linalg.lstsq(A1, sci, rcond=None)
    pred1 = A1 @ c1
    rms1 = float(np.sqrt(np.mean((sci - pred1) ** 2)))

    # === Model 2: All 2nd order ===
    A2 = np.column_stack([np.ones(len(sci)), pho, wide, large,
                          pho ** 2, wide ** 2, large ** 2,
                          pho * wide, pho * large, wide * large])
    c2, *_ = np.linalg.lstsq(A2, sci, rcond=None)
    pred2 = A2 @ c2
    rms2 = float(np.sqrt(np.mean((sci - pred2) ** 2)))

    # === Model 3: Saturation (5 params: b, β, γ, τ) ===
    # Sci = (PHO − β·Wide − γ·Large − b) / (1 + τ·PHO)
    def sat_model(X, b, beta, gamma, tau):
        pho_, wide_, large_ = X
        return (pho_ - beta * wide_ - gamma * large_ - b) / (1 + tau * pho_)

    def sat_resid(params, X, y):
        return sat_model(X, *params) - y

    p0 = [0.0, 2.0, 1.0, 1e-5]
    Xfull = (pho, wide, large)
    print("\nFitting saturation model (4 params)...")
    sol = least_squares(sat_resid, p0, args=(Xfull, sci),
                        method="lm", max_nfev=5000)
    b3, beta3, gamma3, tau3 = sol.x
    pred3 = sat_model(Xfull, *sol.x)
    rms3 = float(np.sqrt(np.mean((sci - pred3) ** 2)))
    print(f"  b={b3:.1f}  β={beta3:.3f}  γ={gamma3:.3f}  τ={tau3:.3e} (s/cnt)")
    print(f"  1/(1+τ·PHO_max) at PHO=45000: {1/(1+tau3*45000):.3f}")

    # === Model 4: Saturation w/ total rate (PHO+Wide+Large) in denominator ===
    def sat_model_tot(X, b, beta, gamma, tau):
        pho_, wide_, large_ = X
        return (pho_ - beta * wide_ - gamma * large_ - b) / (1 + tau * (pho_ + wide_ + large_))

    def sat_resid_tot(params, X, y):
        return sat_model_tot(X, *params) - y

    print("Fitting saturation w/ total in denom...")
    sol4 = least_squares(sat_resid_tot, [0.0, 2.0, 1.0, 1e-5],
                         args=(Xfull, sci), method="lm", max_nfev=5000)
    b4, beta4, gamma4, tau4 = sol4.x
    pred4 = sat_model_tot(Xfull, *sol4.x)
    rms4 = float(np.sqrt(np.mean((sci - pred4) ** 2)))
    print(f"  b={b4:.1f}  β={beta4:.3f}  γ={gamma4:.3f}  τ={tau4:.3e}")

    print(f"\n{'Model':>30s}  RMS  vs linear")
    print(f"  {'Linear (4 coefs)':>30s}  {rms1:.0f}")
    print(f"  {'Quadratic (10)':>30s}  {rms2:.0f}    {(1-rms2/rms1)*100:+.0f}%")
    print(f"  {'Saturation (PHO in den)':>30s}  {rms3:.0f}    {(1-rms3/rms1)*100:+.0f}%")
    print(f"  {'Saturation (total in den)':>30s}  {rms4:.0f}    {(1-rms4/rms1)*100:+.0f}%")

    # === Plot ===
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    panels = [
        (pred1, sci - pred1, rms1, "Linear (4)", "C0"),
        (pred2, sci - pred2, rms2, "Quadratic (10)", "C2"),
        (pred3, sci - pred3, rms3,
         f"Saturation/PHO\nβ={beta3:.2f} γ={gamma3:.2f} τ={tau3:.2e}", "C1"),
        (pred4, sci - pred4, rms4,
         f"Saturation/total\nβ={beta4:.2f} γ={gamma4:.2f} τ={tau4:.2e}", "C3"),
    ]

    year_uniq = sorted(np.unique(year))
    cmap = plt.cm.viridis
    yr2col = {y: cmap((i + 0.5) / len(year_uniq)) for i, y in enumerate(year_uniq)}

    for col, (pred, resid, rms, label, color) in enumerate(panels):
        ax = axes[0, col]
        for y in year_uniq:
            m = year == y
            ax.scatter(sci[m], pred[m], s=0.8, alpha=0.12, color=yr2col[y],
                       label=str(y) if col == 0 else None, rasterized=True)
        lo, hi = sci.min() * 0.95, sci.max() * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Sci observed [cnt/s/box]")
        ax.set_ylabel("Sci predicted")
        ax.set_title(f"{label}\nRMS = {rms:.0f}")
        if col == 0:
            ax.legend(fontsize=7, markerscale=5, loc="upper left", ncol=2)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.scatter(sci, resid, s=0.6, alpha=0.025, color=color, rasterized=True)
        ax.axhline(0, color="r", ls="--", lw=1)
        bins = np.linspace(sci.min(), np.percentile(sci, 99.5), 40)
        bc = 0.5 * (bins[:-1] + bins[1:])
        med, qlo, qhi = [], [], []
        for i in range(len(bins) - 1):
            m = (sci >= bins[i]) & (sci < bins[i + 1])
            if m.sum() > 100:
                med.append(np.median(resid[m]))
                qlo.append(np.percentile(resid[m], 16))
                qhi.append(np.percentile(resid[m], 84))
            else:
                med.append(np.nan); qlo.append(np.nan); qhi.append(np.nan)
        med = np.array(med); qlo = np.array(qlo); qhi = np.array(qhi)
        ax.fill_between(bc, qlo, qhi, alpha=0.25, color="k")
        ax.plot(bc, med, "k-", lw=2)
        ax.set_xlabel("Sci observed [cnt/s/box]")
        ax.set_ylabel("Residual [cnt/s/box]")
        ax.set_title("Residual median ± 1σ")
        ax.set_ylim(-2000, 2000)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Saturation vs poly bend: {big['date'].nunique()} dates × 3 boxes ({len(big):,} bins)",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "pooled_saturation_full.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
