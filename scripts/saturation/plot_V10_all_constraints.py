#!/usr/bin/env python3
"""V10: M7-merged-ACD + cross-detector (j-sum) terms — advisor's 2nd formula.

Model (9 free params + b = 10 per det, 180 total for 18 dets):
    PHO_i = c_pure  · Sci_pure_i  + c_ACD  · Sci_ACD_i  + β  · Wide_i  + γ  · Large_i
          + c_pure' · Sci_pure_js + c_ACD' · Sci_ACD_js + β' · Wide_js + γ' · Large_js
          + b

where _js = sum over the other 5 dets in the same box at the same second.

Bottom errorbar panel layout: 2 rows × 5 cols:
  row 1: b, c_pure, c_ACD, β, γ       (own-det, like V8 ghost compatible)
  row 2: -, c_pure', c_ACD', β', γ'   (cross-det, no V8 ghost since absent)

Compares against V8 baseline with RMS Δ% in per-panel annotations.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large
from plot_all_hypotheses import (
    load_260226A, _ols_with_err, fit_v8, fit_perdet, predict_pho,
    CACHE, BOX_OFFSET, SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP,
    N_MIN_PERDET, X_LO, N_SCATTER_PER_DET, OUT_DIR, DESKTOP, TRIGGER_260,
)


def add_crossdet(df):
    """Add per-(box, met_sec) j-sum columns: sum of OTHER 5 dets at same second."""
    for c in ["scipure_rate", "acd_rate", "wide_rate", "large_rate"]:
        bsum = df.groupby(["date", "box", "met_sec"])[c].transform("sum")
        df[c + "_js"] = bsum - df[c]
    return df


def fit_v10(sub):
    """V10 + b=0, c_pure=γ=1, dt k=1 — 6 free params (c_ACD, β, 4 cross-det).
       Target: PHO·(1−dt/L) − Sci_pure − Large = ...
       Packed as 9-vector [b, c_pure, c_ACD, β, γ, c_pure', c_ACD', β', γ']."""
    X = np.column_stack([sub["acd_rate"], sub["wide_rate"],
                          sub["scipure_rate_js"], sub["acd_rate_js"],
                          sub["wide_rate_js"], sub["large_rate_js"]])
    lf = 1.0 - sub["dt_frac"].values
    y = (sub["pho_rate"].values * lf
         - sub["scipure_rate"].values - sub["large_rate"].values)
    c6, e6 = _ols_with_err(X, y)
    coef = np.array([0.0, 1.0, c6[0], c6[1], 1.0, c6[2], c6[3], c6[4], c6[5]])
    err  = np.array([0.0, 0.0, e6[0], e6[1], 0.0, e6[2], e6[3], e6[4], e6[5]])
    return coef, err


def predict_pho_v10(df, coef_map):
    """Predict PHO (implied, since target was PHO·(1-dt/L)).
       PHO_pred = RHS / (1 - dt/L)"""
    n = len(df)
    coefs = np.zeros((n, 9))
    keys = list(zip(df["box"].astype(str).values, df["det"].astype(int).values))
    for i, k in enumerate(keys):
        coefs[i] = coef_map[k]
    b, c0, cA, bet, gam, c0j, cAj, betj, gamj = coefs.T
    rhs = (b + c0*df["scipure_rate"].values + cA*df["acd_rate"].values
              + bet*df["wide_rate"].values + gam*df["large_rate"].values
              + c0j*df["scipure_rate_js"].values + cAj*df["acd_rate_js"].values
              + betj*df["wide_rate_js"].values + gamj*df["large_rate_js"].values)
    return rhs / (1.0 - df["dt_frac"].values)


def invert_v10(df, coef_map):
    """Self-consistent inversion with own-det local r AND dt k=1.
       PHO·(1-dt/L) = (1-r)·Sci + c_ACD·r·Sci + β·Wide + γ·Large + cross + b
       Sci = (PHO·(1-dt/L) − cross − β·Wide − γ·Large − b) / [(1-r) + c_ACD·r]"""
    n = len(df)
    coefs = np.zeros((n, 9))
    keys = list(zip(df["box"].astype(str).values, df["det"].astype(int).values))
    for i, k in enumerate(keys):
        coefs[i] = coef_map[k]
    b, c0, cA, bet, gam, c0j, cAj, betj, gamj = coefs.T
    r = df["ratio_local"].values
    denom = c0 * (1.0 - r) + cA * r
    cross = (c0j*df["scipure_rate_js"].values + cAj*df["acd_rate_js"].values
              + betj*df["wide_rate_js"].values + gamj*df["large_rate_js"].values)
    lf = 1.0 - df["dt_frac"].values   # k=1 dt correction
    pred = (df["pho_rate"].values * lf - bet*df["wide_rate"].values
            - gam*df["large_rate"].values - cross - b) / denom
    return pred


def main():
    print(f"Loading {CACHE}...")
    train = pd.read_parquet(CACHE)
    print(f"  rows: {len(train):,}")
    print("Computing cross-det sums on training...")
    train = add_crossdet(train)

    print("Loading 260226A...")
    grb = load_260226A()
    # For burst: cross-det also needs to be computed.
    # date column missing → fake it for the groupby
    grb["date"] = "20260226"
    grb = add_crossdet(grb)

    clean = ((train["sci_rate"] >= SCI_LO_CLEAN) & (train["sci_rate"] < SCI_HI_CLEAN)
              & (train["group_rate"] < BOX_RATE_CAP))
    print(f"  CLEAN rows: {int(clean.sum()):,}")

    # ===== V8 baseline (5-param) for comparison =====
    print("\nFitting V8 baseline (per-det)...")
    v8_fits, v8_errs, _ = fit_perdet(train, clean, fit_v8)
    pho_pred_v8 = predict_pho(train[clean], v8_fits, dt_correct=False)
    actual_v8 = train.loc[clean, "pho_rate"].values
    sub_v8 = train[clean].reset_index()
    v8_rms_perdet = {}
    for box in "ABC":
        for det in range(6):
            m = ((sub_v8["box"]==box) & (sub_v8["det"]==det)).values
            v8_rms_perdet[(box, det)] = (
                float(np.sqrt(np.mean((pho_pred_v8[m] - actual_v8[m])**2)))
                if m.any() else float("nan"))

    # ===== V10 per-det fit =====
    print("\nFitting V10 + b=0 + c_pure=γ=1 (6 free params per det)...")
    v10_fits = {}
    v10_errs = {}
    fallback = []
    for box in "ABC":
        for det in range(6):
            m = ((train["box"]==box) & (train["det"]==det) & clean)
            n = int(m.sum())
            if n < N_MIN_PERDET:
                v10_fits[(box, det)] = np.zeros(9)
                v10_errs[(box, det)] = np.full(9, np.nan)
                fallback.append((box, det, n))
                continue
            c, e = fit_v10(train[m])
            v10_fits[(box, det)] = c
            v10_errs[(box, det)] = e
            print(f"  {box}-{det}  b={c[0]:+6.1f}  "
                  f"c0={c[1]:+.3f} cA={c[2]:+.3f} β={c[3]:+.3f} γ={c[4]:+.3f}  "
                  f"c0'={c[5]:+.4f} cA'={c[6]:+.4f} β'={c[7]:+.4f} γ'={c[8]:+.4f}")
    if fallback:
        print(f"  fallbacks: {fallback}")

    # ===== Per-det RMS for V10 =====
    pho_pred_v10 = predict_pho_v10(train[clean], v10_fits)
    v10_rms_perdet = {}
    for box in "ABC":
        for det in range(6):
            m = ((sub_v8["box"]==box) & (sub_v8["det"]==det)).values
            v10_rms_perdet[(box, det)] = (
                float(np.sqrt(np.mean((pho_pred_v10[m] - actual_v8[m])**2)))
                if m.any() else float("nan"))

    print("\nRMS comparison (V8 vs V10):")
    for box in "ABC":
        v8_rms_box = float(np.sqrt(np.mean(
            (pho_pred_v8[(sub_v8["box"]==box).values] - actual_v8[(sub_v8["box"]==box).values])**2)))
        v10_rms_box = float(np.sqrt(np.mean(
            (pho_pred_v10[(sub_v8["box"]==box).values] - actual_v8[(sub_v8["box"]==box).values])**2)))
        d = 100.0 * (v10_rms_box - v8_rms_box) / v8_rms_box
        print(f"  Box {box}:  V8 = {v8_rms_box:.2f}   V10 = {v10_rms_box:.2f}   "
              f"Δ = {d:+.2f}%")

    # ===== Compute Sci_pred for plotting =====
    train["sci_pred"] = invert_v10(train, v10_fits)
    grb["sci_pred"] = invert_v10(grb, v10_fits)
    grb_with_sci = grb[grb["Sci_obs"] > 0].copy()

    # ===== Plot =====
    fig = plt.figure(figsize=(24, 15))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1.4],
                               hspace=0.32,
                               top=0.93, bottom=0.04, left=0.05, right=0.93)

    # Top: 3×6 scatter
    gs_top = outer[0].subgridspec(3, 6, hspace=0.30, wspace=0.10)
    axes = np.empty((3, 6), dtype=object)
    for r in range(3):
        for c in range(6):
            sharex = axes[0, c] if r > 0 else None
            sharey = axes[r, 0] if c > 0 else None
            axes[r, c] = fig.add_subplot(gs_top[r, c], sharex=sharex, sharey=sharey)
            if r < 2: plt.setp(axes[r, c].get_xticklabels(), visible=False)
            if c > 0: plt.setp(axes[r, c].get_yticklabels(), visible=False)

    xb = np.logspace(np.log10(X_LO), np.log10(4500), 120)
    yb = np.logspace(np.log10(X_LO/2), np.log10(7000), 120)
    last_sc = None
    rng = np.random.RandomState(0)
    for row, box in enumerate("ABC"):
        for det in range(6):
            ax = axes[row, det]
            sub = train[(train["box"]==box) & (train["det"]==det)
                        & (train["sci_rate"] >= X_LO) & (train["sci_pred"] > 0)]
            if len(sub) > 0:
                H, xe, ye = np.histogram2d(sub["sci_rate"].values,
                                             sub["sci_pred"].values, bins=[xb, yb])
                ix = np.clip(np.searchsorted(xe, sub["sci_rate"].values) - 1, 0, len(xe)-2)
                iy = np.clip(np.searchsorted(ye, sub["sci_pred"].values) - 1, 0, len(ye)-2)
                density = H[ix, iy].astype(float); density[density < 1] = 1
                idx = (rng.choice(len(sub), N_SCATTER_PER_DET, replace=False)
                       if len(sub) > N_SCATTER_PER_DET else np.arange(len(sub)))
                order = np.argsort(density[idx])
                sc = ax.scatter(sub["sci_rate"].values[idx][order],
                                 sub["sci_pred"].values[idx][order],
                                 c=density[idx][order], cmap="viridis",
                                 norm=LogNorm(vmin=1, vmax=max(density.max(), 2)),
                                 s=1.5, alpha=0.6, rasterized=True, edgecolor="none")
                last_sc = sc

            # 260226A burst overlay
            g_own = grb_with_sci[(grb_with_sci["box"]==box)
                                  & (grb_with_sci["det"]==det)
                                  & (grb_with_sci["Sci_fill_box"] > 0)]
            for _, r in g_own.iterrows():
                ax.plot([r["sci_rate_obs"], r["sci_rate_recov"]],
                        [r["sci_pred"], r["sci_pred"]],
                        color="gray", lw=0.7, alpha=0.55, zorder=5)
            ax.scatter(g_own["sci_rate_obs"], g_own["sci_pred"],
                        s=18, color="blue", alpha=0.85, edgecolor="black", lw=0.4,
                        zorder=6, marker="o")
            ax.scatter(g_own["sci_rate_recov"], g_own["sci_pred"],
                        s=18, color="red", alpha=0.85, edgecolor="black", lw=0.4,
                        zorder=7, marker="^")

            line = np.array([X_LO, 4500])
            ax.plot(line, line, "--", color="red", lw=1.0)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(X_LO, 4500); ax.set_ylim(X_LO/2, 7000)
            c = v10_fits[(box, det)]
            ax.set_title(f"{box}-{det}  c0={c[1]:.2f} cA={c[2]:.2f} "
                          f"β={c[3]:.2f} γ={c[4]:.2f}", fontsize=8)
            rms_now = v10_rms_perdet[(box, det)]
            rms_v8 = v8_rms_perdet[(box, det)]
            dpct = 100.0 * (rms_now - rms_v8) / rms_v8 if rms_v8 > 0 else float("nan")
            sign = "−" if dpct < 0 else "+"
            ax.text(0.97, 0.05,
                     f"RMS={rms_now:.1f}\n(V8 {rms_v8:.1f}, {sign}{abs(dpct):.1f}%)",
                     transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=7, color="black",
                     bbox=dict(facecolor="white", alpha=0.78,
                               edgecolor="none", pad=1.5),
                     linespacing=1.1)
            ax.grid(alpha=0.3, which="both")
            if row == 2: ax.set_xlabel("Sci observed [cnt/s/det]")
            if det == 0: ax.set_ylabel(f"Box {box}\nSci predicted")

    legend_handles = [
        plt.Line2D([], [], color="red", ls="--", lw=1.5, label="y = x"),
        plt.Line2D([], [], marker="o", color="blue", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4,
                   markersize=7, label="260226A Sci_obs"),
        plt.Line2D([], [], marker="^", color="red", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4,
                   markersize=7, label="260226A Sci_recov"),
        plt.Line2D([], [], color="gray", lw=0.8, alpha=0.6,
                   label="same-second pair"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="lower left",
                       fontsize=7, frameon=True, framealpha=0.92)
    if last_sc is not None:
        cbar_ax = fig.add_axes([0.945, 0.45, 0.012, 0.40])
        fig.colorbar(last_sc, cax=cbar_ax, label="training density (log)")

    # Bottom: 2×5 errorbar panels (row 1: own-det b/c_p/c_A/β/γ, row 2: cross c_p'/c_A'/β'/γ' + RMS bar)
    box_color = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    det_order = [(b, d) for b in "ABC" for d in range(6)]
    det_labels = [f"{b}-{d}" for b, d in det_order]
    y_pos = np.arange(len(det_order))
    coef_arr = np.array([v10_fits[k] for k in det_order])   # (18, 9)
    err_arr  = np.array([v10_errs[k] for k in det_order])
    v8_coef = np.array([v8_fits[k]   for k in det_order])
    v8_err  = np.array([v8_errs[k]   for k in det_order])

    # Layout: 2 rows × 5 cols inside outer[1]
    gs_bot = outer[1].subgridspec(2, 5, hspace=0.45, wspace=0.10)
    own_idx = [0, 1, 2, 3, 4]          # b, c_pure, c_ACD, β, γ in coef[0..4]
    own_names = [r"$b$", r"$c_{\mathrm{pure}}$", r"$c_{\mathrm{ACD}}$",
                  r"$\beta$ (Wide)", r"$\gamma$ (Large)"]
    js_idx = [None, 5, 6, 7, 8]        # cross-det positions, None for column 0
    js_names = [None, r"$c_{\mathrm{pure}}'$ (js)", r"$c_{\mathrm{ACD}}'$ (js)",
                 r"$\beta'$ (js)", r"$\gamma'$ (js)"]

    axes_own = []
    for c in range(5):
        sharey = axes_own[0] if c > 0 else None
        ax_own = fig.add_subplot(gs_bot[0, c], sharey=sharey)
        if c > 0: plt.setp(ax_own.get_yticklabels(), visible=False)
        axes_own.append(ax_own)

    axes_js = []
    for c in range(5):
        if js_idx[c] is None:
            ax_js = None
            axes_js.append(None)
            continue
        sharey = axes_js[1] if c > 1 else (axes_own[0] if c == 1 else None)
        if c == 1:
            ax_js = fig.add_subplot(gs_bot[1, c], sharey=axes_own[0])
        else:
            ax_js = fig.add_subplot(gs_bot[1, c], sharey=axes_js[1])
            plt.setp(ax_js.get_yticklabels(), visible=False)
        axes_js.append(ax_js)

    # Row 1: own-det 5 panels with V8 ghost
    for col_i, (ax, idx, name) in enumerate(zip(axes_own, own_idx, own_names)):
        for i, (b, _) in enumerate(det_order):
            # Connecting line V8 → V10
            ax.plot([v8_coef[i, idx], coef_arr[i, idx]], [y_pos[i], y_pos[i]],
                    color=box_color[b], lw=0.7, alpha=0.35, zorder=2)
            ax.errorbar(v8_coef[i, idx], y_pos[i], xerr=v8_err[i, idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        alpha=0.30, elinewidth=0.6, capsize=6, capthick=1.0,
                        markersize=7, markeredgewidth=1.0, zorder=3)
            ax.errorbar(coef_arr[i, idx], y_pos[i], xerr=err_arr[i, idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        elinewidth=0.8, capsize=10, capthick=1.8,
                        markersize=10, markeredgewidth=1.8, zorder=5)
        ax.axhline(5.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axhline(11.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.set_title(name, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
    axes_own[0].set_yticks(y_pos)
    axes_own[0].set_yticklabels(det_labels, fontsize=8)
    axes_own[0].invert_yaxis()
    axes_own[0].set_ylabel("detector\n(own-det)")

    # Row 2: cross-det 4 panels (no V8 ghost since absent)
    axes_js[0] = None
    for col_i in range(1, 5):
        ax = axes_js[col_i]
        idx = js_idx[col_i]
        name = js_names[col_i]
        for i, (b, _) in enumerate(det_order):
            ax.errorbar(coef_arr[i, idx], y_pos[i], xerr=err_arr[i, idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        elinewidth=0.8, capsize=10, capthick=1.8,
                        markersize=10, markeredgewidth=1.8, zorder=5)
        ax.axhline(5.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axhline(11.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axvline(0, color='black', ls=':', lw=0.8, alpha=0.7)
        ax.set_title(name, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
    # axes_js[1] needs its own y labels (column 0 of row 2 is empty/missing)
    if axes_js[1] is not None:
        axes_js[1].set_yticks(y_pos)
        axes_js[1].set_yticklabels(det_labels, fontsize=8)
        axes_js[1].invert_yaxis()
        axes_js[1].set_ylabel("detector\n(cross j-sum)")

    legend_handles2 = [
        plt.Line2D([], [], marker='|', color="#d62728", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box A"),
        plt.Line2D([], [], marker='|', color="#2ca02c", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box B"),
        plt.Line2D([], [], marker='|', color="#1f77b4", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box C"),
        plt.Line2D([], [], marker='|', color='gray', lw=0, alpha=0.30,
                   markersize=7, markeredgewidth=1.0, label="V8 (own-det only)"),
    ]
    axes_own[0].legend(handles=legend_handles2, loc="upper right",
                        fontsize=7, ncol=1, frameon=True, framealpha=0.92)

    # ----- Title + header -----
    formula = (r"$\mathbf{PHO_i \cdot (1-dt/L) = Sci_{pure,i} + c_{ACD}\!\cdot\!Sci_{ACD,i} + \beta\!\cdot\!Wide_i + Large_i + "
               r"c_{pure}'\!\cdot\!Sci_{pure,js} + c_{ACD}'\!\cdot\!Sci_{ACD,js} + \beta'\!\cdot\!Wide_{js} + \gamma'\!\cdot\!Large_{js}}$"
               + "\n   (ALL: V10 cross-det + dt k=1 + b=0 + c_pure=γ=1)")
    fig.text(0.05, 0.945, formula, fontsize=11)
    fig.text(0.05, 0.27,
              "Per-det fit coefficients ± 1σ  "
              "(top row: own-det w/ V8 ghost;  bottom row: cross-det j-sum)",
              fontsize=12, weight='bold')
    fig.suptitle("V10 + dt(k=1) + b=0, c_pure=γ=1 — ALL constraints — training + 260226A burst overlay   "
                 "(6 free params × 18 dets = 108 total)",
                 fontsize=14, y=0.98)

    out = OUT_DIR / "sci_pred_M7merged_perdet_V10_all_constraints.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    desktop = DESKTOP / out.name
    fig.savefig(desktop, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out}\n       {desktop}")


if __name__ == "__main__":
    main()
