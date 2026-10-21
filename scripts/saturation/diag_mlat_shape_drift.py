#!/usr/bin/env python3
"""Per-year fit of mlat sigmoid shape — does it drift over time?

For each year y, fit C(|mlat|) = a + b · σ((|mlat| − μ_m)/k_m) on the
year-averaged marginal. Look at μ_m(y), k_m(y), b(y) trends.

Independent fit per year, no time-decay assumption — pure time-slice view.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_2D_heatmap.npz"


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def model_mlat_only(p, mlat):
    a, b, mu_m, k_m = p
    return a + b * sigm((mlat - mu_m) / k_m)


def main():
    z = np.load(NPZ)
    C_data = z["C_med"]; n_data = z["C_n"]
    months = z["months"]; edges = z["mlat_edges"]
    mlat_centers = 0.5 * (edges[:-1] + edges[1:])
    mask = n_data > 200

    # Group months by year
    year_groups = {}
    for i, m in enumerate(months):
        y = int(m[:4])
        year_groups.setdefault(y, []).append(i)

    years = sorted(year_groups.keys())
    results = {}
    print(f"{'year':<5s} {'a (low-mlat)':>12s} {'b (mlat rise)':>14s} {'μ_m':>7s} {'k_m':>7s} {'n_months':>9s}")
    for y in years:
        idx = year_groups[y]
        with np.errstate(invalid='ignore'):
            n_y = n_data[:, idx].sum(axis=1).astype(float)
            sum_y = (n_data[:, idx] * np.where(mask[:, idx], C_data[:, idx], 0)).sum(axis=1)
            C_y = np.where(n_y > 500, sum_y / np.where(n_y > 0, n_y, 1), np.nan)
        valid = np.isfinite(C_y)
        if valid.sum() < 30: continue

        def resid(p):
            return (model_mlat_only(p, mlat_centers[valid]) - C_y[valid])

        p0 = [115.0, 230.0, 44.0, 6.0]   # a, b, μ_m, k_m
        lo = [50.0,   50.0, 30.0, 0.5]
        hi = [300.0, 500.0, 60.0, 15.0]
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), method='trf', max_nfev=2000)
            a, b, mu_m, k_m = r.x
            results[y] = {"a": a, "b": b, "mu_m": mu_m, "k_m": k_m,
                          "C_y": C_y, "valid": valid, "n_months": len(idx)}
            print(f"{y:<5d} {a:>12.2f} {b:>14.2f} {mu_m:>7.2f} {k_m:>7.2f} {len(idx):>9d}")
        except Exception as e:
            print(f"{y}: fit failed: {e}")

    # ─── Plot ───
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Per-year mlat sigmoid fit — is the shape time-stationary?",
                 fontsize=13, fontweight='bold')

    # Panel A: μ_m vs year
    ax = axes[0, 0]
    ys = np.array(sorted(results.keys()), dtype=float)
    mus = np.array([results[int(y)]["mu_m"] for y in ys])
    kms = np.array([results[int(y)]["k_m"] for y in ys])
    bs  = np.array([results[int(y)]["b"]  for y in ys])
    as_ = np.array([results[int(y)]["a"]  for y in ys])
    ax.plot(ys, mus, 'o-', lw=1.5, ms=6, color='C3')
    ax.axhline(44.46, ls='--', color='gray', alpha=0.5,
               label='25p shared μ_m=44.46°')
    ax.set_xlabel("year"); ax.set_ylabel("μ_m (deg)")
    ax.set_title(f"A. μ_m(y) — mlat sigmoid midpoint  (range {mus.min():.2f} − {mus.max():.2f}°)")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: k_m vs year
    ax = axes[0, 1]
    ax.plot(ys, kms, 'o-', lw=1.5, ms=6, color='C0')
    ax.axhline(6.33, ls='--', color='gray', alpha=0.5,
               label='25p shared k_m=6.33°')
    ax.set_xlabel("year"); ax.set_ylabel("k_m (deg)")
    ax.set_title(f"B. k_m(y) — mlat sigmoid width  (range {kms.min():.2f} − {kms.max():.2f}°)")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel C: b (mlat rise amp) vs year
    ax = axes[0, 2]
    ax.plot(ys, bs, 'o-', lw=1.5, ms=6, color='C2')
    ax.set_xlabel("year"); ax.set_ylabel("b (cnt/s)")
    ax.set_title(f"C. b(y) — high-mlat amplitude  (range {bs.min():.0f} − {bs.max():.0f} cnt/s)")
    ax.grid(alpha=0.3)

    # Panel D: a (low-mlat base) vs year
    ax = axes[1, 0]
    ax.plot(ys, as_, 'o-', lw=1.5, ms=6, color='C1')
    ax.set_xlabel("year"); ax.set_ylabel("a (cnt/s)")
    ax.set_title(f"D. a(y) — low-mlat baseline  (range {as_.min():.1f} − {as_.max():.1f} cnt/s)")
    ax.grid(alpha=0.3)

    # Panel E: marginal C(mlat) per year overlaid + fit line
    ax = axes[1, 1]
    cmap = plt.cm.viridis
    for k, y in enumerate(ys.astype(int)):
        d = results[y]
        color = cmap((y - ys.min()) / max(ys.max()-ys.min(), 1))
        ax.plot(mlat_centers[d["valid"]], d["C_y"][d["valid"]], 'o',
                ms=3, color=color, alpha=0.5)
        m_fine = np.linspace(0, 60, 400)
        ax.plot(m_fine, model_mlat_only([d["a"], d["b"], d["mu_m"], d["k_m"]], m_fine),
                '-', lw=1.5, color=color, label=str(y))
    ax.set_xlabel("|mlat| (deg)"); ax.set_ylabel("⟨C⟩_year (cnt/s)")
    ax.set_title("E. Per-year marginal C(|mlat|) + independent sigmoid fits")
    ax.legend(fontsize=8, ncol=2, loc='upper left'); ax.grid(alpha=0.3)

    # Panel F: midpoint trend with linear fit
    ax = axes[1, 2]
    ax.plot(ys, mus, 'o', ms=8, color='C3', label='μ_m(y)')
    # Fit linear
    coef = np.polyfit(ys - 2017, mus, 1)
    ax.plot(ys, np.polyval(coef, ys-2017), '--', color='black',
            label=f'linear: {coef[0]:+.3f}°/yr')
    coef_k = np.polyfit(ys - 2017, kms, 1)
    ax2 = ax.twinx()
    ax2.plot(ys, kms, 's', ms=8, color='C0', label='k_m(y)')
    ax2.plot(ys, np.polyval(coef_k, ys-2017), '--', color='gray',
             label=f'linear: {coef_k[0]:+.3f}°/yr')
    ax.set_xlabel("year")
    ax.set_ylabel("μ_m (deg)", color='C3')
    ax2.set_ylabel("k_m (deg)", color='C0')
    ax.set_title(f"F. Trend: μ_m slope {coef[0]:+.3f}°/yr,  k_m slope {coef_k[0]:+.3f}°/yr")
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    out = "plots/diag_mlat_shape_drift.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved {out}")

    # Numeric summary
    print(f"\n=== Slope summary (linear fit) ===")
    print(f"  μ_m drift: {coef[0]:+.4f}°/yr   (over 9 yr: {coef[0]*9:+.2f}°)")
    print(f"  k_m drift: {coef_k[0]:+.4f}°/yr  (over 9 yr: {coef_k[0]*9:+.2f}°)")


if __name__ == "__main__":
    main()
