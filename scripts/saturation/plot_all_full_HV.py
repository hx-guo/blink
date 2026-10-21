#!/usr/bin/env python3
"""Unified plot generator: produces all hypothesis plots in one Python process.

Loads cache once → pre-extracts per-(box, det) numpy arrays → generates 14 plots.

Avoids the 8× redundant cache loading of running each script separately.
Expected total runtime: ~12-15 min for all plots on full-HV cache (133M rows).
"""
from pathlib import Path
import sys
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

CACHE = Path("n_below_study/train_cache.parquet")
PERDET_DIR = Path("n_below_study/perdet_npz")
BOX_TOTALS = Path("n_below_study/box_totals.parquet")
PLOT_DIR = Path("plots"); PLOT_DIR.mkdir(exist_ok=True)
DESKTOP_DIR = Path.home() / "Desktop" / "PHO_hypotheses_full_HV"
DESKTOP_DIR.mkdir(exist_ok=True, parents=True)

BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP = 400.0, 1000.0, 6000.0
N_MIN_PERDET = 100
X_LO = 300
N_SCATTER_PER_DET = 40_000
TRIGGER_260 = 446726273.0
MET_CORRECTION = 4.0


def _ols(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    n, p = X.shape
    resid = y - X @ coef
    sigma2 = float((resid @ resid)) / max(n - p, 1)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        err = np.full(p, np.nan)
    return coef, err


# ============= Fit functions (each takes data dict, returns (coef[5], err[5])) =============
def fit_v8(d, dt_correct=False):
    X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["wide"], d["large"]])
    y = d["pho_lf"] if dt_correct else d["pho"]
    return _ols(X, y)


def fit_cpure1_gamma1(d, dt_correct=False):
    X = np.column_stack([d["ones"], d["acd"], d["wide"]])
    rhs = d["pho_lf"] if dt_correct else d["pho"]
    y = rhs - d["scipure"] - d["large"]
    c, e = _ols(X, y)
    return (np.array([c[0], 1.0, c[1], c[2], 1.0]),
            np.array([e[0], 0.0, e[1], e[2], 0.0]))


def fit_cACD2(d):
    X = np.column_stack([d["ones"], d["scipure"], d["wide"], d["large"]])
    y = d["pho"] - 2.0 * d["acd"]
    c, e = _ols(X, y)
    return (np.array([c[0], c[1], 2.0, c[2], c[3]]),
            np.array([e[0], e[1], 0.0, e[2], e[3]]))


def fit_cpure1_gamma1_cACD2(d):
    X = np.column_stack([d["ones"], d["wide"]])
    y = d["pho"] - d["scipure"] - 2.0*d["acd"] - d["large"]
    c, e = _ols(X, y)
    return (np.array([c[0], 1.0, 2.0, c[1], 1.0]),
            np.array([e[0], 0.0, 0.0, e[1], 0.0]))


def fit_all_ones(d):
    X = np.column_stack([d["ones"], d["wide"]])
    y = d["pho"] - d["scipure"] - d["acd"] - d["large"]
    c, e = _ols(X, y)
    return (np.array([c[0], 1.0, 1.0, c[1], 1.0]),
            np.array([e[0], 0.0, 0.0, e[1], 0.0]))


def fit_b0_cpure1_gamma1(d):
    X = np.column_stack([d["acd"], d["wide"]])    # no intercept
    y = d["pho"] - d["scipure"] - d["large"]
    c, e = _ols(X, y)
    return (np.array([0.0, 1.0, c[0], c[1], 1.0]),
            np.array([0.0, 0.0, e[0], e[1], 0.0]))


def fit_no_wide(d):
    X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["large"]])
    c, e = _ols(X, d["pho"])
    return (np.array([c[0], c[1], c[2], 0.0, c[3]]),
            np.array([e[0], e[1], e[2], 0.0, e[3]]))


def fit_no_large(d):
    X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["wide"]])
    c, e = _ols(X, d["pho"])
    return (np.array([c[0], c[1], c[2], c[3], 0.0]),
            np.array([e[0], e[1], e[2], e[3], 0.0]))


def fit_v10(d, dt_correct=False, b0_cpg=False):
    """V10 cross-detector. Returns (coef[9], err[9])."""
    if b0_cpg:
        # b=0, c_pure=γ=1; fit (c_ACD, β, c_pure', c_ACD', β', γ')
        X = np.column_stack([d["acd"], d["wide"], d["scipure_js"], d["acd_js"],
                              d["wide_js"], d["large_js"]])
        rhs = d["pho_lf"] if dt_correct else d["pho"]
        y = rhs - d["scipure"] - d["large"]
        c, e = _ols(X, y)
        return (np.array([0.0, 1.0, c[0], c[1], 1.0, c[2], c[3], c[4], c[5]]),
                np.array([0.0, 0.0, e[0], e[1], 0.0, e[2], e[3], e[4], e[5]]))
    else:
        # All 9 free
        X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["wide"], d["large"],
                              d["scipure_js"], d["acd_js"], d["wide_js"], d["large_js"]])
        rhs = d["pho_lf"] if dt_correct else d["pho"]
        return _ols(X, rhs)


# ============= Predict / invert helpers =============
def predict_pho_v8(d, coef, dt_correct=False):
    b, c0, cA, bet, gam = coef
    rhs = b + c0*d["scipure"] + cA*d["acd"] + bet*d["wide"] + gam*d["large"]
    return rhs / (1.0 - d["dtfrac"]) if dt_correct else rhs


def predict_pho_v10(d, coef, dt_correct=False):
    b, c0, cA, bet, gam, c0j, cAj, betj, gamj = coef
    rhs = (b + c0*d["scipure"] + cA*d["acd"] + bet*d["wide"] + gam*d["large"]
            + c0j*d["scipure_js"] + cAj*d["acd_js"]
            + betj*d["wide_js"] + gamj*d["large_js"])
    return rhs / (1.0 - d["dtfrac"]) if dt_correct else rhs


def invert_v8(d, coef, dt_correct=False):
    """Self-consistent inversion. coef = [b, c_pure, c_ACD, β, γ]."""
    b, c0, cA, bet, gam = coef
    r = d["ratio"]
    denom = c0*(1.0 - r) + cA*r
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    lf = (1.0 - d["dtfrac"]) if dt_correct else 1.0
    return (d["pho"]*lf - bet*d["wide"] - gam*d["large"] - b) / denom


def invert_v10(d, coef, dt_correct=False):
    """V10 inversion. coef = [b, c_pure, c_ACD, β, γ, c_pure', c_ACD', β', γ']."""
    b, c0, cA, bet, gam, c0j, cAj, betj, gamj = coef
    r = d["ratio"]
    denom = c0*(1.0 - r) + cA*r
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    cross = (c0j*d["scipure_js"] + cAj*d["acd_js"]
             + betj*d["wide_js"] + gamj*d["large_js"])
    lf = (1.0 - d["dtfrac"]) if dt_correct else 1.0
    return (d["pho"]*lf - bet*d["wide"] - gam*d["large"] - cross - b) / denom


# ============= 260226A loader =============
def load_260226A():
    rows = []
    for box, code in [("A","0766"),("B","1009"),("C","1781")]:
        eng_file = f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits"
        fe = fits.open(eng_file, memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
        length_cyc = d["Length_Time_Cycle"].astype(float)
        length_s = length_cyc * 16e-6
        for det_local in range(6):
            det_g = BOX_OFFSET[box] + det_local
            pho = d[f"Cnt_PHODet_{det_g}"].astype(float)
            csi = d[f"Cnt_CsI_PHODet_{det_g}"].astype(float)
            large_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)
            dt = d[f"DeadTime_PHODet_{det_g}"].astype(float)
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                rows.append({"box": box, "det": det_local,
                              "met_sec": int(met_eng[i]),
                              "length_cyc": length_cyc[i], "length_s": length_s[i],
                              "dt_cyc": dt[i],
                              "PHO": pho[i], "Wide": csi[i], "Large": large[i]})
        fe.close()
    eng = pd.DataFrame(rows)
    eng["dt_frac"] = eng["dt_cyc"] / eng["length_cyc"]

    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    sci_obs = sci_obs[sci_obs["type"]=="EVT"].copy()
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    aminfo = sci_obs["aminfo"].values.astype(np.int64)
    popcount = np.zeros(len(aminfo), dtype=np.int32)
    for bit in range(18): popcount += ((aminfo >> bit) & 1).astype(np.int32)
    sci_obs["is_acd"] = (popcount > 0).astype("int32")
    sci_obs_pd = sci_obs.groupby(["box","det_id","met_sec"]).agg(
        Sci_obs=("box", "size"), Sci_ACD_obs=("is_acd", "sum"),
    ).reset_index().rename(columns={"det_id":"det"})

    sci_rec = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    sci_rec["box"] = sci_rec["box"].astype(str)
    sci_rec["met_sec"] = sci_rec["met"].astype("int64")
    sci_fill_box = (sci_rec[sci_rec["type"]=="FILL_GAP"]
                    .groupby(["box","met_sec"]).size()
                    .rename("Sci_fill_box").reset_index())

    df = eng.merge(sci_obs_pd, on=["box","det","met_sec"], how="left")
    df["Sci_obs"] = df["Sci_obs"].fillna(0).astype("int64")
    df["Sci_ACD_obs"] = df["Sci_ACD_obs"].fillna(0).astype("int64")
    df = df.merge(sci_fill_box, on=["box","met_sec"], how="left")
    df["Sci_fill_box"] = df["Sci_fill_box"].fillna(0)
    box_obs_sum = df.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    df["Sci_recov_box"] = box_obs_sum + df["Sci_fill_box"]
    df["Sci_recov"] = np.where(box_obs_sum > 0,
                                df["Sci_recov_box"] * df["Sci_obs"]/box_obs_sum.clip(lower=1),
                                df["Sci_recov_box"]/6)
    df["ratio_local"] = np.where(df["Sci_obs"] > 0,
                                  df["Sci_ACD_obs"] / df["Sci_obs"].clip(lower=1),
                                  np.nan)
    box_sec_ratio = df.groupby(["box","met_sec"])["ratio_local"].transform("mean")
    df["ratio_local"] = df["ratio_local"].fillna(box_sec_ratio).fillna(0.22)
    df["ratio_local"] = df["ratio_local"].astype("float32").clip(0, 1)

    df["sci_rate_obs"] = df["Sci_obs"] / df["length_s"]
    df["sci_rate_recov"] = df["Sci_recov"] / df["length_s"]
    df["wide_rate"]  = df["Wide"]  / df["length_s"]
    df["large_rate"] = df["Large"] / df["length_s"]
    df["pho_rate"]   = df["PHO"]   / df["length_s"]
    df["scipure_rate"] = (df["sci_rate_obs"] * (1 - df["ratio_local"])).astype("float32")
    df["acd_rate"]     = (df["sci_rate_obs"] * df["ratio_local"]).astype("float32")

    # Cross-det sums for V10
    df["date"] = "20260226"
    for c in ["scipure_rate", "acd_rate", "wide_rate", "large_rate"]:
        bsum = df.groupby(["date", "box", "met_sec"])[c].transform("sum")
        df[c + "_js"] = bsum - df[c]
    return df


# ============= Plot helpers =============
def plot_v8_style(name, suffix, formula, fits_dict, errs_dict, rms_perdet,
                   v8_fits, v8_errs, v8_rms_perdet, data, grb, fixed_values,
                   subtitle_params):
    """Generic plot: 3×6 scatter top + 1×5 errorbar bottom. coef = 5-vector."""
    fig = plt.figure(figsize=(24, 13.5))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.30,
                               top=0.92, bottom=0.05, left=0.05, right=0.93)
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
    grb_with_sci = grb[grb["Sci_obs"] > 0].copy()

    for row, box in enumerate("ABC"):
        for det in range(6):
            ax = axes[row, det]
            d = data[(box, det)]
            sci_pred = d["sci_pred"]
            sci_rate = d["sci_rate"]
            m = (sci_rate >= X_LO) & (sci_pred > 0)
            sub_sci = sci_rate[m]
            sub_pred = sci_pred[m]
            if len(sub_sci) > 0:
                H, xe, ye = np.histogram2d(sub_sci, sub_pred, bins=[xb, yb])
                ix = np.clip(np.searchsorted(xe, sub_sci) - 1, 0, len(xe)-2)
                iy = np.clip(np.searchsorted(ye, sub_pred) - 1, 0, len(ye)-2)
                dens = H[ix, iy].astype(float); dens[dens < 1] = 1
                idx = (rng.choice(len(sub_sci), N_SCATTER_PER_DET, replace=False)
                       if len(sub_sci) > N_SCATTER_PER_DET else np.arange(len(sub_sci)))
                order = np.argsort(dens[idx])
                sc = ax.scatter(sub_sci[idx][order], sub_pred[idx][order],
                                 c=dens[idx][order], cmap="viridis",
                                 norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
                                 s=1.5, alpha=0.6, rasterized=True, edgecolor="none")
                last_sc = sc

            # Burst overlay
            g_own = grb_with_sci[(grb_with_sci["box"]==box)
                                  & (grb_with_sci["det"]==det)
                                  & (grb_with_sci["Sci_fill_box"] > 0)]
            for _, rr in g_own.iterrows():
                ax.plot([rr["sci_rate_obs"], rr["sci_rate_recov"]],
                        [rr["sci_pred"], rr["sci_pred"]],
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
            c = fits_dict[(box, det)][:5]
            ax.set_title(f"{box}-{det}  c0={c[1]:.2f} cA={c[2]:.2f} "
                          f"β={c[3]:.2f} γ={c[4]:.2f}", fontsize=8)
            rms_now = rms_perdet[(box, det)]
            rms_v8 = v8_rms_perdet[(box, det)]
            dpct = 100.0 * (rms_now - rms_v8) / rms_v8 if rms_v8 > 0 else float("nan")
            sign = "−" if dpct < 0 else "+"
            ax.text(0.97, 0.05,
                     f"RMS={rms_now:.1f}\n(V8 {rms_v8:.1f}, {sign}{abs(dpct):.1f}%)",
                     transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=7, color="black",
                     bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5),
                     linespacing=1.1)
            ax.grid(alpha=0.3, which="both")
            if row == 2: ax.set_xlabel("Sci observed [cnt/s/det]")
            if det == 0: ax.set_ylabel(f"Box {box}\nSci predicted")

    legend_handles = [
        plt.Line2D([], [], color="red", ls="--", lw=1.5, label="y = x"),
        plt.Line2D([], [], marker="o", color="blue", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4,
                   markersize=7, label="Sci_obs"),
        plt.Line2D([], [], marker="^", color="red", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4,
                   markersize=7, label="Sci_recov"),
        plt.Line2D([], [], color="gray", lw=0.8, alpha=0.6, label="pair"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="lower left",
                       fontsize=7, frameon=True, framealpha=0.92)
    if last_sc is not None:
        cbar_ax = fig.add_axes([0.945, 0.45, 0.012, 0.40])
        fig.colorbar(last_sc, cax=cbar_ax, label="training density (log)")

    # Bottom errorbar 1×5
    box_color = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    det_order = [(b, dd) for b in "ABC" for dd in range(6)]
    det_labels = [f"{b}-{dd}" for b, dd in det_order]
    y_pos = np.arange(18)
    coef_arr = np.array([fits_dict[k][:5] for k in det_order])
    err_arr  = np.array([errs_dict[k][:5] for k in det_order])
    coef_v8a = np.array([v8_fits[k][:5] for k in det_order])
    err_v8a  = np.array([v8_errs[k][:5] for k in det_order])

    gs_bot = outer[1].subgridspec(1, 5, wspace=0.08)
    axes3 = np.empty(5, dtype=object)
    for c in range(5):
        sharey = axes3[0] if c > 0 else None
        axes3[c] = fig.add_subplot(gs_bot[0, c], sharey=sharey)
        if c > 0: plt.setp(axes3[c].get_yticklabels(), visible=False)
    param_names = [r"$b$ (cnt/s)", r"$c_{\mathrm{pure}}$", r"$c_{\mathrm{ACD}}$",
                    r"$\beta$ (Wide)", r"$\gamma$ (Large)"]
    for p_idx, (ax, pname) in enumerate(zip(axes3, param_names)):
        for i, (b, _) in enumerate(det_order):
            ax.plot([coef_v8a[i, p_idx], coef_arr[i, p_idx]], [y_pos[i], y_pos[i]],
                    color=box_color[b], lw=0.7, alpha=0.35, zorder=2)
            ax.errorbar(coef_v8a[i, p_idx], y_pos[i], xerr=err_v8a[i, p_idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b], alpha=0.30,
                        elinewidth=0.6, capsize=6, capthick=1.0,
                        markersize=7, markeredgewidth=1.0, zorder=3)
            ax.errorbar(coef_arr[i, p_idx], y_pos[i], xerr=err_arr[i, p_idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        elinewidth=0.8, capsize=10, capthick=1.8,
                        markersize=10, markeredgewidth=1.8, zorder=5)
        if p_idx in fixed_values:
            ax.axvline(fixed_values[p_idx], color='black', ls=':', lw=1.0,
                        alpha=0.7, zorder=1)
        ax.axhline(5.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axhline(11.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.set_title(pname, fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlabel("coefficient value")
    axes3[0].set_yticks(y_pos)
    axes3[0].set_yticklabels(det_labels, fontsize=9)
    axes3[0].invert_yaxis()
    axes3[0].set_ylabel("detector")
    legend2 = [
        plt.Line2D([], [], marker='|', color=box_color[b], lw=0,
                   markersize=10, markeredgewidth=1.8, label=f"Box {b}")
        for b in "ABC"
    ] + [
        plt.Line2D([], [], marker='|', color='gray', lw=0, alpha=0.30,
                   markersize=7, markeredgewidth=1.0, label="V8 ghost"),
        plt.Line2D([], [], color='black', ls=':', lw=1.0, label="fixed"),
    ]
    axes3[0].legend(handles=legend2, loc="upper right", fontsize=7, ncol=1,
                     frameon=True, framealpha=0.92)

    fig.text(0.05, 0.94, formula, fontsize=12)
    fig.text(0.05, 0.255,
              r"Per-detector fit coefficients $\pm$ 1$\sigma$  (V8 ghost: 5 free)",
              fontsize=12, weight='bold')
    fig.suptitle(f"{name} — full HV data (133M training rows)   ({subtitle_params})",
                 fontsize=14, y=0.975)

    out = PLOT_DIR / f"sci_pred_full_HV_{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(DESKTOP_DIR / f"{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_v10_style(name, suffix, formula, fits_dict, errs_dict, rms_perdet,
                    v8_fits, v8_errs, v8_rms_perdet, data, grb, subtitle_params,
                    fixed_values=None):
    """V10 plot: 3×6 scatter top + 2×5 errorbar (own row + cross-det row)."""
    fig = plt.figure(figsize=(24, 15))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1.4], hspace=0.32,
                               top=0.93, bottom=0.04, left=0.05, right=0.93)
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
    grb_with_sci = grb[grb["Sci_obs"] > 0].copy()

    for row, box in enumerate("ABC"):
        for det in range(6):
            ax = axes[row, det]
            d = data[(box, det)]
            sci_pred = d["sci_pred"]
            sci_rate = d["sci_rate"]
            m = (sci_rate >= X_LO) & (sci_pred > 0)
            sub_sci = sci_rate[m]
            sub_pred = sci_pred[m]
            if len(sub_sci) > 0:
                H, xe, ye = np.histogram2d(sub_sci, sub_pred, bins=[xb, yb])
                ix = np.clip(np.searchsorted(xe, sub_sci) - 1, 0, len(xe)-2)
                iy = np.clip(np.searchsorted(ye, sub_pred) - 1, 0, len(ye)-2)
                dens = H[ix, iy].astype(float); dens[dens < 1] = 1
                idx = (rng.choice(len(sub_sci), N_SCATTER_PER_DET, replace=False)
                       if len(sub_sci) > N_SCATTER_PER_DET else np.arange(len(sub_sci)))
                order = np.argsort(dens[idx])
                sc = ax.scatter(sub_sci[idx][order], sub_pred[idx][order],
                                 c=dens[idx][order], cmap="viridis",
                                 norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
                                 s=1.5, alpha=0.6, rasterized=True, edgecolor="none")
                last_sc = sc

            g_own = grb_with_sci[(grb_with_sci["box"]==box)
                                  & (grb_with_sci["det"]==det)
                                  & (grb_with_sci["Sci_fill_box"] > 0)]
            for _, rr in g_own.iterrows():
                ax.plot([rr["sci_rate_obs"], rr["sci_rate_recov"]],
                        [rr["sci_pred"], rr["sci_pred"]],
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
            c = fits_dict[(box, det)]
            ax.set_title(f"{box}-{det}  c0={c[1]:.2f} cA={c[2]:.2f} "
                          f"β={c[3]:.2f} γ={c[4]:.2f}", fontsize=8)
            rms_now = rms_perdet[(box, det)]
            rms_v8 = v8_rms_perdet[(box, det)]
            dpct = 100.0 * (rms_now - rms_v8) / rms_v8 if rms_v8 > 0 else float("nan")
            sign = "−" if dpct < 0 else "+"
            ax.text(0.97, 0.05,
                     f"RMS={rms_now:.1f}\n(V8 {rms_v8:.1f}, {sign}{abs(dpct):.1f}%)",
                     transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=7, color="black",
                     bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5),
                     linespacing=1.1)
            ax.grid(alpha=0.3, which="both")
            if row == 2: ax.set_xlabel("Sci observed [cnt/s/det]")
            if det == 0: ax.set_ylabel(f"Box {box}\nSci predicted")

    legend_handles = [
        plt.Line2D([], [], color="red", ls="--", lw=1.5, label="y = x"),
        plt.Line2D([], [], marker="o", color="blue", lw=0, markeredgecolor="black",
                   markeredgewidth=0.4, markersize=7, label="Sci_obs"),
        plt.Line2D([], [], marker="^", color="red", lw=0, markeredgecolor="black",
                   markeredgewidth=0.4, markersize=7, label="Sci_recov"),
        plt.Line2D([], [], color="gray", lw=0.8, alpha=0.6, label="pair"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="lower left",
                       fontsize=7, frameon=True, framealpha=0.92)
    if last_sc is not None:
        cbar_ax = fig.add_axes([0.945, 0.45, 0.012, 0.40])
        fig.colorbar(last_sc, cax=cbar_ax, label="training density (log)")

    # Bottom: 2 rows × 5 cols
    box_color = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    det_order = [(b, dd) for b in "ABC" for dd in range(6)]
    det_labels = [f"{b}-{dd}" for b, dd in det_order]
    y_pos = np.arange(18)
    coef_arr = np.array([fits_dict[k] for k in det_order])
    err_arr  = np.array([errs_dict[k] for k in det_order])
    v8_coef = np.array([v8_fits[k][:5] for k in det_order])
    v8_err  = np.array([v8_errs[k][:5] for k in det_order])

    gs_bot = outer[1].subgridspec(2, 5, hspace=0.45, wspace=0.10)
    own_names = [r"$b$", r"$c_{pure}$", r"$c_{ACD}$", r"$\beta$ Wide", r"$\gamma$ Large"]
    js_names = [None, r"$c_{pure}'$", r"$c_{ACD}'$", r"$\beta'$", r"$\gamma'$"]
    fixed_values = fixed_values or {}

    axes_own = []
    for c in range(5):
        sharey = axes_own[0] if c > 0 else None
        ax_own = fig.add_subplot(gs_bot[0, c], sharey=sharey)
        if c > 0: plt.setp(ax_own.get_yticklabels(), visible=False)
        axes_own.append(ax_own)
    axes_js = [None]
    for c in range(1, 5):
        if c == 1:
            ax_js = fig.add_subplot(gs_bot[1, c], sharey=axes_own[0])
        else:
            ax_js = fig.add_subplot(gs_bot[1, c], sharey=axes_js[1])
            plt.setp(ax_js.get_yticklabels(), visible=False)
        axes_js.append(ax_js)

    # Row 1 (own) with V8 ghost
    for col_i, (ax, pname) in enumerate(zip(axes_own, own_names)):
        for i, (b, _) in enumerate(det_order):
            ax.plot([v8_coef[i, col_i], coef_arr[i, col_i]], [y_pos[i], y_pos[i]],
                    color=box_color[b], lw=0.7, alpha=0.35, zorder=2)
            ax.errorbar(v8_coef[i, col_i], y_pos[i], xerr=v8_err[i, col_i],
                        fmt='|', color=box_color[b], ecolor=box_color[b], alpha=0.30,
                        elinewidth=0.6, capsize=6, capthick=1.0,
                        markersize=7, markeredgewidth=1.0, zorder=3)
            ax.errorbar(coef_arr[i, col_i], y_pos[i], xerr=err_arr[i, col_i],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        elinewidth=0.8, capsize=10, capthick=1.8,
                        markersize=10, markeredgewidth=1.8, zorder=5)
        if col_i in fixed_values:
            ax.axvline(fixed_values[col_i], color='black', ls=':', lw=1.0,
                        alpha=0.7, zorder=1)
        ax.axhline(5.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axhline(11.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.set_title(pname, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
    axes_own[0].set_yticks(y_pos)
    axes_own[0].set_yticklabels(det_labels, fontsize=8)
    axes_own[0].invert_yaxis()
    axes_own[0].set_ylabel("detector\n(own-det)")

    # Row 2 (cross-det) — no V8 ghost
    for col_i in range(1, 5):
        ax = axes_js[col_i]
        idx = col_i + 4   # coef[5..8] are c_pure', c_ACD', β', γ'
        for i, (b, _) in enumerate(det_order):
            ax.errorbar(coef_arr[i, idx], y_pos[i], xerr=err_arr[i, idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        elinewidth=0.8, capsize=10, capthick=1.8,
                        markersize=10, markeredgewidth=1.8, zorder=5)
        ax.axhline(5.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axhline(11.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axvline(0, color='black', ls=':', lw=0.8, alpha=0.7)
        ax.set_title(js_names[col_i], fontsize=11)
        ax.grid(axis='x', alpha=0.3)
    axes_js[1].set_yticks(y_pos)
    axes_js[1].set_yticklabels(det_labels, fontsize=8)
    axes_js[1].invert_yaxis()
    axes_js[1].set_ylabel("detector\n(cross j-sum)")
    legend2 = [plt.Line2D([], [], marker='|', color=box_color[b], lw=0,
                          markersize=10, markeredgewidth=1.8, label=f"Box {b}")
                for b in "ABC"]
    legend2.append(plt.Line2D([], [], marker='|', color='gray', lw=0, alpha=0.30,
                              markersize=7, markeredgewidth=1.0, label="V8 ghost"))
    axes_own[0].legend(handles=legend2, loc="upper right", fontsize=7, ncol=1,
                        frameon=True, framealpha=0.92)

    fig.text(0.05, 0.945, formula, fontsize=11)
    fig.text(0.05, 0.27,
              "Per-det fit coefficients ± 1σ  (top: own-det; bottom: cross-det j-sum)",
              fontsize=12, weight='bold')
    fig.suptitle(f"{name} — full HV data (133M training rows)   ({subtitle_params})",
                 fontsize=14, y=0.98)

    out = PLOT_DIR / f"sci_pred_full_HV_{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(DESKTOP_DIR / f"{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============= Main =============
def main():
    t_global = time.time()
    # ===== Load 18 per-(box, det) npz files + box_totals =====
    print(f"Loading 18 per-(box, det) npz files from {PERDET_DIR}/...", flush=True)
    t0 = time.time()
    data_full = {}
    for box in "ABC":
        for det in range(6):
            npz = np.load(PERDET_DIR / f"{box}_{det}.npz")
            d = {
                "scipure":   npz["scipure_rate"].astype(np.float32),
                "acd":       npz["acd_rate"].astype(np.float32),
                "wide":      npz["wide_rate"].astype(np.float32),
                "large":     npz["large_rate"].astype(np.float32),
                "pho":       npz["pho_rate"].astype(np.float32),
                "dtfrac":    npz["dt_frac"].astype(np.float32),
                "ratio":     npz["ratio_local"].astype(np.float32),
                "sci_rate":  npz["sci_rate"].astype(np.float32),
                "group_rate": npz["group_rate"].astype(np.float32),
                "date":      npz["date"],
                "met_sec":   npz["met_sec"],
            }
            d["clean_mask"] = ((d["sci_rate"] >= SCI_LO_CLEAN)
                                & (d["sci_rate"] < SCI_HI_CLEAN)
                                & (d["group_rate"] < BOX_RATE_CAP))
            d["ones"] = np.ones(len(d["pho"]), dtype=np.float32)
            d["pho_lf"] = d["pho"] * (1.0 - d["dtfrac"])
            data_full[(box, det)] = d
    print(f"  loaded 18 npz files in {time.time()-t0:.0f}s", flush=True)

    print(f"Loading box_totals for cross-det...", flush=True)
    t0 = time.time()
    box_totals = pd.read_parquet(BOX_TOTALS)
    # Convert date to int32 YYYYMMDD to match per-det npz date format
    box_totals["date"] = box_totals["date"].str.replace("-", "").astype(np.int32)
    print(f"  {len(box_totals):,} (date,box,sec) totals loaded in {time.time()-t0:.0f}s",
          flush=True)
    print(f"Computing cross-det j-sums via merge on (date, met_sec)...", flush=True)
    t0 = time.time()
    for box in "ABC":
        bt = box_totals[box_totals["box"] == box].set_index(["date", "met_sec"]).sort_index()
        for det in range(6):
            d = data_full[(box, det)]
            idx = pd.MultiIndex.from_arrays([d["date"], d["met_sec"]])
            aligned = bt.reindex(idx)
            d["scipure_js"] = (aligned["scipure_sum"].values - d["scipure"]).astype(np.float32)
            d["acd_js"]     = (aligned["acd_sum"].values     - d["acd"]).astype(np.float32)
            d["wide_js"]    = (aligned["wide_sum"].values    - d["wide"]).astype(np.float32)
            d["large_js"]   = (aligned["large_sum"].values   - d["large"]).astype(np.float32)
            # date/met_sec no longer needed
            del d["date"], d["met_sec"]
    del box_totals
    gc.collect()
    print(f"  cross-det merged in {time.time()-t0:.0f}s", flush=True)

    # Build CLEAN-only views (no copy, just numpy array slicing)
    print("Building CLEAN-band views for fitting...", flush=True)
    data = {}
    for k, d in data_full.items():
        cm = d["clean_mask"]
        data[k] = {kk: vv[cm] for kk, vv in d.items() if kk != "clean_mask"}

    # 260226A
    print("Loading 260226A...", flush=True)
    grb = load_260226A()

    # ===== V8 baseline fit (used as ghost for all other plots) =====
    print("\nFitting V8 baseline (per-det)...", flush=True)
    v8_fits, v8_errs = {}, {}
    for box in "ABC":
        for det in range(6):
            d = data[(box, det)]
            if len(d["pho"]) < N_MIN_PERDET:
                v8_fits[(box, det)] = np.zeros(5)
                v8_errs[(box, det)] = np.full(5, np.nan)
                continue
            c, e = fit_v8(d)
            v8_fits[(box, det)] = c
            v8_errs[(box, det)] = e

    # V8 per-(box, det) RMS (using FULL data — sci_rate, pho, etc on full not CLEAN)
    # But RMS is computed on CLEAN band data for fair comparison.
    v8_rms = {}
    for box in "ABC":
        for det in range(6):
            d = data[(box, det)]
            if len(d["pho"]) < N_MIN_PERDET:
                v8_rms[(box, det)] = float("nan"); continue
            pred = predict_pho_v8(d, v8_fits[(box, det)])
            v8_rms[(box, det)] = float(np.sqrt(np.mean((pred - d["pho"])**2)))
    print(f"  V8 RMS sample: A-0={v8_rms[('A',0)]:.2f} B-2={v8_rms[('B',2)]:.2f} C-5={v8_rms[('C',5)]:.2f}",
          flush=True)

    # Compute sci_pred (V8) for plotting — use FULL data (with rate > X_LO mask in plot)
    print("Computing V8 Sci_pred for all 18 dets...", flush=True)
    for box in "ABC":
        for det in range(6):
            data_full[(box, det)]["sci_pred"] = invert_v8(data_full[(box, det)], v8_fits[(box, det)])

    # Add 260226A burst Sci_pred (V8 baseline)
    n = len(grb)
    coefs_grb = np.zeros((n, 5))
    keys = list(zip(grb["box"].astype(str).values, grb["det"].astype(int).values))
    for i, k in enumerate(keys): coefs_grb[i] = v8_fits[k]
    b, c0, cA, bet, gam = coefs_grb.T
    r = grb["ratio_local"].values
    denom = c0*(1-r) + cA*r
    grb["sci_pred"] = (grb["pho_rate"].values - bet*grb["wide_rate"].values
                       - gam*grb["large_rate"].values - b) / denom

    # ===== Generate V8 baseline plot =====
    print("\n[1/14] V8 baseline plot...", flush=True)
    t0 = time.time()
    plot_v8_style("V8 baseline (full HV)", "01_V8_baseline",
                   r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} + \beta\cdot Wide + \gamma\cdot Large + b}$",
                   v8_fits, v8_errs, v8_rms, v8_fits, v8_errs, v8_rms,
                   data_full, grb, {}, "5 params × 18 dets = 90")
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # ===== List of V8-style hypotheses =====
    v8_hypotheses = [
        ("dt k=1",                   "02_dt_k1",                  fit_v8, True, {},
         r"$\mathbf{PHO\cdot(1-dt/L) = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} + \beta\cdot Wide + \gamma\cdot Large + b}$",
         "5 params × 18 dets = 90 + dt k=1"),
        ("c_pure=γ=1",               "03_cpure_gamma_eq_1",       lambda d: fit_cpure1_gamma1(d, dt_correct=False), False, {1: 1.0, 4: 1.0},
         r"$\mathbf{PHO = Sci_{pure} + c_{ACD}\cdot Sci_{ACD} + \beta\cdot Wide + Large + b}$  ($c_{pure}=\gamma=1$)",
         "3 params × 18 dets = 54"),
        ("dt + c_pure=γ=1",          "04_dt_plus_cpure_gamma_eq_1", lambda d: fit_cpure1_gamma1(d, dt_correct=True), True, {1: 1.0, 4: 1.0},
         r"$\mathbf{PHO\cdot(1-dt/L) = Sci_{pure} + c_{ACD}\cdot Sci_{ACD} + \beta\cdot Wide + Large + b}$",
         "3 params × 18 dets = 54 + dt k=1"),
        ("c_ACD=2",                  "05_cACD_eq_2",              fit_cACD2, False, {2: 2.0},
         r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + 2\cdot Sci_{ACD} + \beta\cdot Wide + \gamma\cdot Large + b}$",
         "4 params × 18 dets = 72"),
        ("c_pure=γ=1, c_ACD=2",      "06_three_coefs_fixed",      fit_cpure1_gamma1_cACD2, False, {1: 1.0, 2: 2.0, 4: 1.0},
         r"$\mathbf{PHO = Sci_{pure} + 2\cdot Sci_{ACD} + \beta\cdot Wide + Large + b}$",
         "2 params × 18 dets = 36"),
        ("all 4 = 1",                "07_all_event_coefs_eq_1",   fit_all_ones, False, {1: 1.0, 2: 1.0, 4: 1.0},
         r"$\mathbf{PHO = Sci_{pure} + Sci_{ACD} + \beta\cdot Wide + Large + b}$",
         "2 params × 18 dets = 36"),
        ("b=0, c_pure=γ=1 (真香)",   "08_b0_cpure_gamma_eq_1",    fit_b0_cpure1_gamma1, False, {0: 0.0, 1: 1.0, 4: 1.0},
         r"$\mathbf{PHO = Sci_{pure} + c_{ACD}\cdot Sci_{ACD} + \beta\cdot Wide + Large}$  ($b=0$)",
         "2 params × 18 dets = 36"),
        ("no Wide (β=0)",            "09_no_Wide",                fit_no_wide, False, {3: 0.0},
         r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} + \gamma\cdot Large + b}$",
         "4 params × 18 dets = 72"),
        ("no Large (γ=0)",           "10_no_Large",               fit_no_large, False, {4: 0.0},
         r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} + \beta\cdot Wide + b}$",
         "4 params × 18 dets = 72"),
    ]

    # ===== β global (special: 2-stage) =====
    print(f"\n[β global] pooling 18 dets for global β...", flush=True)
    t0 = time.time()
    all_ones    = np.concatenate([data[(b,d)]["ones"]    for b in "ABC" for d in range(6)])
    all_scipure = np.concatenate([data[(b,d)]["scipure"] for b in "ABC" for d in range(6)])
    all_acd     = np.concatenate([data[(b,d)]["acd"]     for b in "ABC" for d in range(6)])
    all_wide    = np.concatenate([data[(b,d)]["wide"]    for b in "ABC" for d in range(6)])
    all_large   = np.concatenate([data[(b,d)]["large"]   for b in "ABC" for d in range(6)])
    all_pho     = np.concatenate([data[(b,d)]["pho"]     for b in "ABC" for d in range(6)])
    X_pool = np.column_stack([all_ones, all_scipure, all_acd, all_wide, all_large])
    c_pool, *_ = np.linalg.lstsq(X_pool, all_pho, rcond=None)
    beta_g = float(c_pool[3])
    del all_ones, all_scipure, all_acd, all_wide, all_large, all_pho, X_pool
    gc.collect()
    print(f"  β_global = {beta_g:.4f}  in {time.time()-t0:.0f}s", flush=True)

    # ===== Loop through V8-style hypotheses =====
    for idx, (name, suffix, fit_fn, dt_correct, fixed_values, formula, subtitle) in enumerate(v8_hypotheses):
        print(f"\n[{idx+2}/14] {name}...", flush=True)
        t0 = time.time()
        fits_d, errs_d = {}, {}
        for box in "ABC":
            for det in range(6):
                d = data[(box, det)]
                if len(d["pho"]) < N_MIN_PERDET:
                    fits_d[(box, det)] = np.zeros(5); errs_d[(box, det)] = np.full(5, np.nan)
                    continue
                fits_d[(box, det)], errs_d[(box, det)] = fit_fn(d)

        # Compute RMS on CLEAN data
        rms_d = {}
        for box in "ABC":
            for det in range(6):
                d = data[(box, det)]
                if len(d["pho"]) < N_MIN_PERDET:
                    rms_d[(box, det)] = float("nan"); continue
                pred = predict_pho_v8(d, fits_d[(box, det)], dt_correct=dt_correct)
                rms_d[(box, det)] = float(np.sqrt(np.mean((pred - d["pho"])**2)))

        # Update full-data sci_pred for plotting (use FULL data, not CLEAN)
        for box in "ABC":
            for det in range(6):
                df = data_full[(box, det)]
                df["sci_pred"] = invert_v8(df, fits_d[(box, det)], dt_correct=dt_correct)

        # Update grb sci_pred for THIS hypothesis
        for i, k in enumerate(keys): coefs_grb[i] = fits_d[k]
        b, c0, cA, bet, gam = coefs_grb.T
        r = grb["ratio_local"].values
        denom = c0*(1-r) + cA*r
        denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        lf = (1.0 - grb["dt_frac"].values) if dt_correct else 1.0
        grb["sci_pred"] = (grb["pho_rate"].values*lf - bet*grb["wide_rate"].values
                           - gam*grb["large_rate"].values - b) / denom

        plot_v8_style(name, suffix, formula, fits_d, errs_d, rms_d,
                       v8_fits, v8_errs, v8_rms, data_full, grb, fixed_values, subtitle)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # ===== β global hypothesis =====
    print(f"\n[11/14] β global shared...", flush=True)
    t0 = time.time()
    fits_d, errs_d = {}, {}
    for box in "ABC":
        for det in range(6):
            d = data[(box, det)]
            if len(d["pho"]) < N_MIN_PERDET:
                fits_d[(box, det)] = np.zeros(5); errs_d[(box, det)] = np.full(5, np.nan)
                continue
            X = np.column_stack([d["ones"], d["scipure"], d["acd"], d["large"]])
            y = d["pho"] - beta_g * d["wide"]
            c, e = _ols(X, y)
            fits_d[(box, det)] = np.array([c[0], c[1], c[2], beta_g, c[3]])
            errs_d[(box, det)] = np.array([e[0], e[1], e[2], 0.0, e[3]])
    rms_d = {}
    for box in "ABC":
        for det in range(6):
            d = data[(box, det)]
            if len(d["pho"]) < N_MIN_PERDET:
                rms_d[(box, det)] = float("nan"); continue
            pred = predict_pho_v8(d, fits_d[(box, det)])
            rms_d[(box, det)] = float(np.sqrt(np.mean((pred - d["pho"])**2)))
    for box in "ABC":
        for det in range(6):
            df = data_full[(box, det)]
            df["sci_pred"] = invert_v8(df, fits_d[(box, det)])
    for i, k in enumerate(keys): coefs_grb[i] = fits_d[k]
    b, c0, cA, bet, gam = coefs_grb.T
    r = grb["ratio_local"].values
    denom = c0*(1-r) + cA*r
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    grb["sci_pred"] = (grb["pho_rate"].values - bet*grb["wide_rate"].values
                       - gam*grb["large_rate"].values - b) / denom
    plot_v8_style("β global shared", "11_beta_global",
                   r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} + \beta_{global}\cdot Wide + \gamma\cdot Large + b}$  ($\beta$ shared)",
                   fits_d, errs_d, rms_d, v8_fits, v8_errs, v8_rms,
                   data_full, grb, {3: beta_g}, f"4 params/det + 1 global β = 73 total  (β={beta_g:.3f})")
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # ===== V10 hypotheses =====
    v10_hypotheses = [
        ("V10 free (cross-det)",      "12_V10_crossdet",        False, False, {},
         r"$\mathbf{PHO_i = c_{pure}\cdot Sci_{pure,i} + c_{ACD}\cdot Sci_{ACD,i} + \beta\cdot Wide_i + \gamma\cdot Large_i + c_{pure}'\cdot Sci_{pure,js} + c_{ACD}'\cdot Sci_{ACD,js} + \beta'\cdot Wide_{js} + \gamma'\cdot Large_{js} + b}$",
         "9 params + b = 10 per det = 180 total"),
        ("V10 + b=0, c_pure=γ=1 (真香)", "13_V10_plus_b0_cpure_gamma_eq_1", False, True, {0: 0.0, 1: 1.0, 4: 1.0},
         r"$\mathbf{PHO_i = Sci_{pure,i} + c_{ACD}\cdot Sci_{ACD,i} + \beta\cdot Wide_i + Large_i + cross\;terms}$  ($b=0$, $c_{pure}=\gamma=1$)",
         "6 free params × 18 dets = 108"),
        ("V10 + dt + b=0, c_pure=γ=1", "14_V10_all_constraints", True,  True, {0: 0.0, 1: 1.0, 4: 1.0},
         r"$\mathbf{PHO_i\cdot(1-dt/L) = Sci_{pure,i} + c_{ACD}\cdot Sci_{ACD,i} + \beta\cdot Wide_i + Large_i + cross\;terms}$",
         "6 free params × 18 dets = 108 + dt k=1"),
    ]
    for idx, (name, suffix, dt_correct, b0_cpg, fixed_values, formula, subtitle) in enumerate(v10_hypotheses):
        print(f"\n[{idx+12}/14] {name}...", flush=True)
        t0 = time.time()
        fits_d, errs_d = {}, {}
        for box in "ABC":
            for det in range(6):
                d = data[(box, det)]
                if len(d["pho"]) < N_MIN_PERDET:
                    fits_d[(box, det)] = np.zeros(9); errs_d[(box, det)] = np.full(9, np.nan)
                    continue
                fits_d[(box, det)], errs_d[(box, det)] = fit_v10(d, dt_correct=dt_correct, b0_cpg=b0_cpg)
        rms_d = {}
        for box in "ABC":
            for det in range(6):
                d = data[(box, det)]
                if len(d["pho"]) < N_MIN_PERDET:
                    rms_d[(box, det)] = float("nan"); continue
                pred = predict_pho_v10(d, fits_d[(box, det)], dt_correct=dt_correct)
                rms_d[(box, det)] = float(np.sqrt(np.mean((pred - d["pho"])**2)))
        for box in "ABC":
            for det in range(6):
                df = data_full[(box, det)]
                df["sci_pred"] = invert_v10(df, fits_d[(box, det)], dt_correct=dt_correct)
        coefs_grb9 = np.zeros((n, 9))
        for i, k in enumerate(keys): coefs_grb9[i] = fits_d[k]
        b, c0, cA, bet, gam, c0j, cAj, betj, gamj = coefs_grb9.T
        r = grb["ratio_local"].values
        denom = c0*(1-r) + cA*r
        denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        cross = (c0j*grb["scipure_rate_js"].values + cAj*grb["acd_rate_js"].values
                  + betj*grb["wide_rate_js"].values + gamj*grb["large_rate_js"].values)
        lf = (1.0 - grb["dt_frac"].values) if dt_correct else 1.0
        grb["sci_pred"] = (grb["pho_rate"].values*lf - bet*grb["wide_rate"].values
                           - gam*grb["large_rate"].values - cross - b) / denom
        plot_v10_style(name, suffix, formula, fits_d, errs_d, rms_d,
                        v8_fits, v8_errs, v8_rms, data_full, grb, subtitle, fixed_values)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    print(f"\n[ALL DONE] total {(time.time()-t_global)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
