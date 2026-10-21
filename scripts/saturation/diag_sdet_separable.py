#!/usr/bin/env python3
"""Test whether s_det(det, t) separates into s0_det * g(t) (shared outgassing curve).

Uses the per-day s_det already stored in v5_agg_full.npz. If, after dividing
each detector's curve by its own baseline, all 18 normalized curves collapse
onto one, then s_det is separable and we can replace per-day refit with a single
analytic g(t) calibrated once.

Output: plots/diag_sdet_separable.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-npz", default="n_below_study/v5_npz/v5_agg_full.npz")
    p.add_argument("--output", default="plots/diag_sdet_separable.png")
    args = p.parse_args()

    z = np.load(args.full_npz)
    dates = z["dates"]
    s = z["s_det_daily"]  # (n_days, 3, 6)
    dt = np.array([np.datetime64(d) for d in dates])
    t_years = (dt - dt[0]).astype("timedelta64[D]").astype(float) / 365.25

    # Baseline per det: median of first 180 valid days
    n_days = len(dates)
    s_flat = s.reshape(n_days, 18)  # det order A0..A5,B0..B5,C0..C5
    det_labels = [f"{b}-{d}" for b in "ABC" for d in range(6)]

    baseline = np.full(18, np.nan)
    for j in range(18):
        col = s_flat[:180, j]
        baseline[j] = np.nanmedian(col)

    norm = s_flat / baseline[None, :]  # each det divided by its own baseline

    # Monthly median of the collapsed normalized curve (the shared g(t))
    months = (dt.astype("datetime64[M]"))
    uniq_m = np.unique(months)
    g_t = []
    g_x = []
    for m in uniq_m:
        sel = months == m
        vals = norm[sel].ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 50:
            g_t.append(np.median(vals))
            g_x.append((m - dt[0].astype("datetime64[M]")).astype(int) / 12.0)
    g_x = np.array(g_x); g_t = np.array(g_t)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

    # Panel 1: raw s_det(t), all 18
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, 18))
    for j in range(18):
        ax.plot(t_years, s_flat[:, j], lw=0.4, alpha=0.5, color=colors[j])
    ax.set_xlabel("years since mission start"); ax.set_ylabel("s_det (cnt/s)")
    ax.set_title("raw s_det(t) — 18 detectors", fontsize=12)
    ax.grid(True, alpha=0.3); ax.set_ylim(50, 220)

    # Panel 2: normalized — do they collapse?
    ax = axes[1]
    for j in range(18):
        ax.plot(t_years, norm[:, j], lw=0.4, alpha=0.4, color=colors[j])
    ax.plot(g_x, g_t, "k-", lw=2.5, label="shared g(t) = monthly median")
    # exponential fit g(t)=A+(1-A)exp(-t/tau)
    from scipy.optimize import curve_fit
    def gfun(t, A, tau):
        return A + (1 - A) * np.exp(-t / tau)
    try:
        popt, _ = curve_fit(gfun, g_x, g_t, p0=[0.7, 2.0], maxfev=10000)
        tt = np.linspace(0, g_x.max(), 200)
        ax.plot(tt, gfun(tt, *popt), "r--", lw=2.0,
                label=f"exp fit: g={popt[0]:.2f}+{1-popt[0]:.2f}·e^(-t/{popt[1]:.1f}yr)")
    except Exception as e:
        print("exp fit failed:", e)
        popt = None
    ax.set_xlabel("years since mission start"); ax.set_ylabel("s_det / baseline")
    ax.set_title("normalized s_det/baseline — collapse = separable s0·g(t)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.3)

    # Panel 3: scatter of all (det,month) normalized vs shared g(t) — tightness = separability
    ax = axes[2]
    # residual of each det's normalized curve around shared monthly g
    g_interp = np.interp(t_years, g_x, g_t)
    spread = norm - g_interp[:, None]
    for j in range(18):
        ax.plot(t_years, spread[:, j], lw=0.3, alpha=0.4, color=colors[j])
    ax.axhline(0, color="k", lw=1)
    rms = np.nanstd(spread)
    ax.set_xlabel("years since mission start"); ax.set_ylabel("norm − g(t)")
    ax.set_title(f"deviation from shared g(t)  (RMS={rms:.3f})\nflat band = separable", fontsize=12)
    ax.grid(True, alpha=0.3); ax.set_ylim(-0.2, 0.2)

    fig.suptitle("Is s_det(det,t) separable into s0_det × g(t)?  (outgassing shared shape)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")

    # Numeric: per-det total drop ratio (late/early) — should be similar if separable
    print("\n=== late/early ratio per det (separable → all similar) ===")
    ratios = []
    for j in range(18):
        early = np.nanmedian(s_flat[:180, j])
        late = np.nanmedian(s_flat[-180:, j])
        r = late / early
        ratios.append(r)
        print(f"  {det_labels[j]}: {early:6.1f} → {late:6.1f}  ratio={r:.3f}")
    ratios = np.array(ratios)
    print(f"\n  ratio mean={ratios.mean():.3f}, std={ratios.std():.3f}, "
          f"spread={ratios.std()/ratios.mean()*100:.1f}%")
    if popt is not None:
        print(f"\n  shared g(t): asymptote A={popt[0]:.3f}, tau={popt[1]:.2f} yr")
        print(f"  => s_det drops to {popt[0]*100:.0f}% of initial as t→∞")
    print(f"  deviation from shared g(t): RMS={rms:.3f} ({rms*100:.1f}% of s_det)")


if __name__ == "__main__":
    main()
