#!/usr/bin/env python3
"""Generate combined plots (3×6 scatter + 1×5 errorbar) for ALL bake-off
hypotheses, in the same style as sci_pred_M7merged_perdet_combined_dt_constraints.png.

Reads from parquet cache. Outputs one PNG per hypothesis to plots/ and ~/Desktop/.
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

CACHE = Path("n_below_study/train_cache.parquet")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
DESKTOP = Path.home() / "Desktop"

BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP = 400.0, 1000.0, 6000.0
N_MIN_PERDET = 100
X_LO = 300
N_SCATTER_PER_DET = 40_000
TRIGGER_260 = 446726273.0
MET_CORRECTION = 4.0


# ============= Hypothesis configurations =============
# Each hypothesis defines:
#   fit_fn(sub)        → returns (coef[5], std_err[5]) where coef=[b,c_pure,c_ACD,β,γ]
#   invert_fn(df, coef_map) → returns sci_pred array (self-consistent w/ ratio_local)
#   formula (TeX),  suffix (filename),  desc (title)

def _ols_with_err(X, y):
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


def fit_v8(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"], sub["large_rate"]])
    c, e = _ols_with_err(X, sub["pho_rate"].values)
    return c, e


def fit_cpure1_gamma1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["acd_rate"], sub["wide_rate"]])
    y = sub["pho_rate"].values - sub["scipure_rate"].values - sub["large_rate"].values
    c3, e3 = _ols_with_err(X, y)
    return (np.array([c3[0], 1.0, c3[1], c3[2], 1.0]),
            np.array([e3[0], 0.0, e3[1], e3[2], 0.0]))


def fit_dt_k1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"], sub["large_rate"]])
    y = sub["pho_rate"].values * (1.0 - sub["dt_frac"].values)
    return _ols_with_err(X, y)


def fit_cACD2(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"],
                          sub["wide_rate"], sub["large_rate"]])
    y = sub["pho_rate"].values - 2.0 * sub["acd_rate"].values
    c4, e4 = _ols_with_err(X, y)
    return (np.array([c4[0], c4[1], 2.0, c4[2], c4[3]]),
            np.array([e4[0], e4[1], 0.0, e4[2], e4[3]]))


def fit_cpure1_gamma1_cACD2(sub):
    X = np.column_stack([np.ones(len(sub)), sub["wide_rate"]])
    y = (sub["pho_rate"].values - sub["scipure_rate"].values
         - 2.0*sub["acd_rate"].values - sub["large_rate"].values)
    c2, e2 = _ols_with_err(X, y)
    return (np.array([c2[0], 1.0, 2.0, c2[1], 1.0]),
            np.array([e2[0], 0.0, 0.0, e2[1], 0.0]))


def fit_all_1_or_2(sub):
    """c_pure = γ = c_ACD = 1, fit (b, β)."""
    X = np.column_stack([np.ones(len(sub)), sub["wide_rate"]])
    y = (sub["pho_rate"].values - sub["scipure_rate"].values
         - sub["acd_rate"].values - sub["large_rate"].values)
    c2, e2 = _ols_with_err(X, y)
    return (np.array([c2[0], 1.0, 1.0, c2[1], 1.0]),
            np.array([e2[0], 0.0, 0.0, e2[1], 0.0]))


def fit_no_wide(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["large_rate"]])
    c4, e4 = _ols_with_err(X, sub["pho_rate"].values)
    return (np.array([c4[0], c4[1], c4[2], 0.0, c4[3]]),
            np.array([e4[0], e4[1], e4[2], 0.0, e4[3]]))


def fit_no_large(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"]])
    c4, e4 = _ols_with_err(X, sub["pho_rate"].values)
    return (np.array([c4[0], c4[1], c4[2], c4[3], 0.0]),
            np.array([e4[0], e4[1], e4[2], e4[3], 0.0]))


def fit_b0_cpure1_gamma1(sub):
    """Most parsimonious physical model: b=0, c_pure=γ=1.
       Regression through origin: PHO - Sci_pure - Large = c_ACD·Sci_ACD + β·Wide
       Only 2 free params: c_ACD and β."""
    X = np.column_stack([sub["acd_rate"], sub["wide_rate"]])    # no intercept column
    y = sub["pho_rate"].values - sub["scipure_rate"].values - sub["large_rate"].values
    c2, e2 = _ols_with_err(X, y)
    return (np.array([0.0, 1.0, c2[0], c2[1], 1.0]),
            np.array([0.0, 0.0, e2[0], e2[1], 0.0]))


# β global is handled separately (2-stage pool then per-det)


# ============= Inversion =============
def invert(df, coef_map, dt_correct=False):
    """Self-consistent inversion: Sci_pred = (PHO·(1-k·dt/L) - β·Wide - γ·Large - b)
       / [c_pure·(1-r) + c_ACD·r]"""
    n = len(df)
    coefs = np.zeros((n, 5))
    keys = list(zip(df["box"].astype(str).values, df["det"].astype(int).values))
    for i, k in enumerate(keys):
        coefs[i] = coef_map[k]
    b, c0, cA, bet, gam = coefs.T
    r = df["ratio_local"].values
    denom = c0 * (1.0 - r) + cA * r
    lf = (1.0 - df["dt_frac"].values) if dt_correct else 1.0
    pred = (df["pho_rate"].values * lf
            - bet*df["wide_rate"].values
            - gam*df["large_rate"].values
            - b) / np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return pred


def predict_pho(df, coef_map, dt_correct=False):
    n = len(df)
    coefs = np.zeros((n, 5))
    keys = list(zip(df["box"].astype(str).values, df["det"].astype(int).values))
    for i, k in enumerate(keys):
        coefs[i] = coef_map[k]
    b, c0, cA, bet, gam = coefs.T
    rhs = (b + c0*df["scipure_rate"].values + cA*df["acd_rate"].values
           + bet*df["wide_rate"].values + gam*df["large_rate"].values)
    if dt_correct:
        return rhs / (1.0 - df["dt_frac"].values)
    return rhs


# ============= Burst loader =============
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
        Sci_obs=("box", "size"),
        Sci_ACD_obs=("is_acd", "sum"),
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
    df["t_rel"] = df["met_sec"] - TRIGGER_260
    return df


# ============= Per-hypothesis fit-and-plot =============
def fit_perdet(train, sub_clean, fit_fn, beta_global=None):
    """Fit each (box, det). For β global hypothesis, beta_global is set externally."""
    fits, errs, fallbacks = {}, {}, []
    box_fits = {}
    for box in "ABC":
        m = (train["box"]==box) & (sub_clean)
        c, _ = fit_fn(train[m])
        box_fits[box] = c
    for box in "ABC":
        for det in range(6):
            m = (train["box"]==box) & (train["det"]==det) & sub_clean
            n = int(m.sum())
            if n < N_MIN_PERDET:
                fits[(box, det)] = box_fits[box]
                errs[(box, det)] = np.full(5, np.nan)
                fallbacks.append((box, det, n))
            else:
                if beta_global is None:
                    c, e = fit_fn(train[m])
                else:
                    # β-global: fit 4-param with β fixed
                    s = train[m]
                    X = np.column_stack([np.ones(n), s["scipure_rate"], s["acd_rate"],
                                          s["large_rate"]])
                    y = s["pho_rate"].values - beta_global * s["wide_rate"].values
                    c4, e4 = _ols_with_err(X, y)
                    c = np.array([c4[0], c4[1], c4[2], beta_global, c4[3]])
                    e = np.array([e4[0], e4[1], e4[2], 0.0, e4[3]])
                fits[(box, det)] = c
                errs[(box, det)] = e
    return fits, errs, fallbacks


def run_hypothesis(train, grb, hypo, v8_fits, v8_errs, v8_rms_perdet):
    print(f"\n{'='*80}\n  Hypothesis: {hypo['name']}\n{'='*80}")
    clean = ((train["sci_rate"] >= SCI_LO_CLEAN) & (train["sci_rate"] < SCI_HI_CLEAN)
              & (train["group_rate"] < BOX_RATE_CAP))

    if hypo.get("beta_global", False):
        # Pool first to get β_global
        sub_pool = train[clean]
        c_pool, _ = fit_v8(sub_pool)
        beta_g = c_pool[3]
        fits, errs, fbk = fit_perdet(train, clean, hypo["fit_fn"], beta_global=beta_g)
        print(f"  β_global = {beta_g:.4f}")
    else:
        fits, errs, fbk = fit_perdet(train, clean, hypo["fit_fn"])

    dt = hypo["dt_correct"]
    # RMS per-(box, det)
    rms_perdet = {}
    sub_clean = train[clean]
    pred_pho_clean = predict_pho(sub_clean, fits, dt_correct=dt)
    actual = sub_clean["pho_rate"].values
    for box in "ABC":
        for det in range(6):
            mask = ((sub_clean["box"]==box) & (sub_clean["det"]==det)).values
            if mask.any():
                rms_perdet[(box, det)] = float(np.sqrt(np.mean(
                    (pred_pho_clean[mask] - actual[mask])**2)))
            else:
                rms_perdet[(box, det)] = float("nan")

    # Print per-box mean RMS
    for box in "ABC":
        rms_vals = [rms_perdet[(box, d)] for d in range(6)]
        print(f"  Box {box} mean per-det RMS = {np.mean(rms_vals):.2f}")

    # Compute Sci_pred for plotting — apply to ALL train rows (not just CLEAN band)
    # so the scatter shows model behavior beyond the fitting region (sci > 1000)
    train_plot = train.copy()
    train_plot["sci_pred"] = invert(train_plot, fits, dt_correct=dt)

    # Burst
    grb_local = grb.copy()
    grb_local["sci_pred"] = invert(grb_local, fits, dt_correct=dt)
    grb_with_sci = grb_local[grb_local["Sci_obs"] > 0].copy()

    # ====== Plot ======
    fig = plt.figure(figsize=(24, 13.5))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1],
                               hspace=0.30,
                               top=0.92, bottom=0.05, left=0.05, right=0.93)

    # Top: scatter
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
            sub = train_plot[(train_plot["box"]==box) & (train_plot["det"]==det)
                              & (train_plot["sci_rate"] >= X_LO)
                              & (train_plot["sci_pred"] > 0)]
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

            # Burst overlay
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
            is_fb = any(k[0]==box and k[1]==det for k in fbk)
            star = " *" if is_fb else ""
            c = fits[(box, det)]
            ax.set_title(f"{box}-{det}{star}  c0={c[1]:.2f} cA={c[2]:.2f} "
                          f"β={c[3]:.2f} γ={c[4]:.2f}", fontsize=8)
            rms_now = rms_perdet[(box, det)]
            rms_v8  = v8_rms_perdet[(box, det)]
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
                   markersize=7, label="260226A Sci_obs (per-det)"),
        plt.Line2D([], [], marker="^", color="red", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4,
                   markersize=7, label="260226A Sci_recov (per-det)"),
        plt.Line2D([], [], color="gray", lw=0.8, alpha=0.6,
                   label="same-second pair"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="lower left",
                       fontsize=7, frameon=True, framealpha=0.92)
    if last_sc is not None:
        cbar_ax = fig.add_axes([0.945, 0.45, 0.012, 0.40])
        fig.colorbar(last_sc, cax=cbar_ax, label="training density (log)")

    # Bottom: errorbar (V8 ghost + current bold)
    box_color = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    det_order = [(b, d) for b in "ABC" for d in range(6)]
    det_labels = [f"{b}-{d}" for b, d in det_order]
    y_pos = np.arange(len(det_order))
    coef_arr = np.array([fits[k] for k in det_order])
    err_arr  = np.array([errs[k] for k in det_order])
    coef_v8a = np.array([v8_fits[k] for k in det_order])
    err_v8a  = np.array([v8_errs[k] for k in det_order])
    box_idx  = np.array([k[0] for k in det_order])

    param_names = [r"$b$ (cnt/s)", r"$c_{\mathrm{pure}}$",
                    r"$c_{\mathrm{ACD}}$", r"$\beta$ (Wide)", r"$\gamma$ (Large)"]
    gs_bot = outer[1].subgridspec(1, 5, wspace=0.08)
    axes3 = np.empty(5, dtype=object)
    for c in range(5):
        sharey = axes3[0] if c > 0 else None
        axes3[c] = fig.add_subplot(gs_bot[0, c], sharey=sharey)
        if c > 0: plt.setp(axes3[c].get_yticklabels(), visible=False)

    fixed_vals = hypo.get("fixed_values", {})
    for p_idx, (ax, name) in enumerate(zip(axes3, param_names)):
        for i, (b, _) in enumerate(det_order):
            # Connecting line V8 → constrained
            ax.plot([coef_v8a[i, p_idx], coef_arr[i, p_idx]],
                    [y_pos[i], y_pos[i]],
                    color=box_color[b], lw=0.7, alpha=0.35, zorder=2)
            ax.errorbar(coef_v8a[i, p_idx], y_pos[i], xerr=err_v8a[i, p_idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b], alpha=0.30,
                        elinewidth=0.6, capsize=6, capthick=1.0,
                        markersize=7, markeredgewidth=1.0, zorder=3)
            ax.errorbar(coef_arr[i, p_idx], y_pos[i], xerr=err_arr[i, p_idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        elinewidth=0.8, capsize=10, capthick=1.8,
                        markersize=10, markeredgewidth=1.8, zorder=5)
        # x=fixed reference line if applicable
        if p_idx in fixed_vals:
            ax.axvline(fixed_vals[p_idx], color='black', ls=':', lw=1.0,
                        alpha=0.7, zorder=1)
        ax.axhline(5.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axhline(11.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.set_title(name, fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlabel("coefficient value")

    axes3[0].set_yticks(y_pos)
    axes3[0].set_yticklabels(det_labels, fontsize=9)
    axes3[0].invert_yaxis()
    axes3[0].set_ylabel("detector")
    legend_handles2 = [
        plt.Line2D([], [], marker='|', color="#d62728", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box A"),
        plt.Line2D([], [], marker='|', color="#2ca02c", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box B"),
        plt.Line2D([], [], marker='|', color="#1f77b4", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box C"),
        plt.Line2D([], [], marker='|', color='gray', lw=0, alpha=0.30,
                   markersize=7, markeredgewidth=1.0, label="V8 (ghost)"),
        plt.Line2D([], [], color='black', ls=':', lw=1.0, alpha=0.7,
                   label="fixed value"),
    ]
    axes3[0].legend(handles=legend_handles2, loc="upper right",
                     fontsize=7, ncol=1, frameon=True, framealpha=0.92)

    fig.text(0.05, 0.94, hypo["formula"], fontsize=12)
    fig.text(0.05, 0.255,
              r"Per-detector fit coefficients $\pm$ 1$\sigma$  (V8 ghost: 5 free)",
              fontsize=12, weight='bold')
    fig.suptitle(f"{hypo['name']} — training + 260226A burst overlay   "
                 f"({hypo['params']})",
                 fontsize=14, y=0.975)

    out = OUT_DIR / f"sci_pred_M7merged_perdet_{hypo['suffix']}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    desktop = DESKTOP / out.name
    fig.savefig(desktop, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}\n         {desktop}")


def main():
    print(f"Loading cache {CACHE}...")
    train = pd.read_parquet(CACHE)
    print(f"  rows: {len(train):,}")
    print("Loading 260226A...")
    grb = load_260226A()
    grb_with_sci = grb[grb["Sci_obs"] > 0]
    print(f"  GRB cached Sci rows: {len(grb_with_sci):,}")

    # First, fit V8 baseline (for ghost comparison)
    clean = ((train["sci_rate"] >= SCI_LO_CLEAN) & (train["sci_rate"] < SCI_HI_CLEAN)
              & (train["group_rate"] < BOX_RATE_CAP))
    v8_fits, v8_errs, _ = fit_perdet(train, clean, fit_v8)
    pho_pred_v8 = predict_pho(train[clean], v8_fits, dt_correct=False)
    actual_v8 = train.loc[clean, "pho_rate"].values
    sub_v8 = train[clean].reset_index()
    v8_rms_perdet = {}
    for box in "ABC":
        for det in range(6):
            m = ((sub_v8["box"]==box) & (sub_v8["det"]==det)).values
            v8_rms_perdet[(box, det)] = (float(np.sqrt(np.mean((pho_pred_v8[m] - actual_v8[m])**2)))
                                          if m.any() else float("nan"))

    # ===== Hypotheses to plot =====
    hypotheses = [
        {"name": "c_ACD = 2 fixed",
         "suffix": "cACD2",
         "fit_fn": fit_cACD2, "dt_correct": False,
         "params": "4 params × 18 dets = 72",
         "fixed_values": {2: 2.0},
         "formula": (r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + 2\cdot Sci_{ACD} "
                     r"+ \beta\cdot Wide + \gamma\cdot Large + b}$"
                     r"   ($c_{ACD}=2$ fixed)")},
        {"name": "c_pure = γ = 1, c_ACD = 2",
         "suffix": "cpure1_gamma1_cACD2",
         "fit_fn": fit_cpure1_gamma1_cACD2, "dt_correct": False,
         "params": "2 params × 18 dets = 36",
         "fixed_values": {1: 1.0, 2: 2.0, 4: 1.0},
         "formula": (r"$\mathbf{PHO = Sci_{pure} + 2\cdot Sci_{ACD} "
                     r"+ \beta\cdot Wide + Large + b}$"
                     r"   ($c_{pure}=\gamma=1$, $c_{ACD}=2$ all fixed)")},
        {"name": "All event coefs = 1",
         "suffix": "all_ones",
         "fit_fn": fit_all_1_or_2, "dt_correct": False,
         "params": "2 params × 18 dets = 36",
         "fixed_values": {1: 1.0, 2: 1.0, 4: 1.0},
         "formula": (r"$\mathbf{PHO = Sci_{pure} + Sci_{ACD} "
                     r"+ \beta\cdot Wide + Large + b}$"
                     r"   ($c_{pure}=c_{ACD}=\gamma=1$ all fixed)")},
        {"name": "β global shared",
         "suffix": "beta_global",
         "fit_fn": fit_v8, "dt_correct": False,
         "beta_global": True,
         "params": "4 params/det + 1 global β = 73 total",
         "fixed_values": {},
         "formula": (r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} "
                     r"+ \beta_{global}\cdot Wide + \gamma\cdot Large + b}$"
                     r"   ($\beta$ shared across all 18 dets)")},
        {"name": "no Wide (β = 0)",
         "suffix": "no_wide",
         "fit_fn": fit_no_wide, "dt_correct": False,
         "params": "4 params × 18 dets = 72",
         "fixed_values": {3: 0.0},
         "formula": (r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} "
                     r"+ \gamma\cdot Large + b}$"
                     r"   (Wide term dropped, $\beta=0$)")},
        {"name": "no Large (γ = 0)",
         "suffix": "no_large",
         "fit_fn": fit_no_large, "dt_correct": False,
         "params": "4 params × 18 dets = 72",
         "fixed_values": {4: 0.0},
         "formula": (r"$\mathbf{PHO = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} "
                     r"+ \beta\cdot Wide + b}$"
                     r"   (Large term dropped, $\gamma=0$)")},
        {"name": "b = 0, c_pure = γ = 1  (3 fixed, no offset)",
         "suffix": "b0_cpure1_gamma1",
         "fit_fn": fit_b0_cpure1_gamma1, "dt_correct": False,
         "params": "2 params × 18 dets = 36",
         "fixed_values": {0: 0.0, 1: 1.0, 4: 1.0},
         "formula": (r"$\mathbf{PHO = Sci_{pure} + c_{ACD}\cdot Sci_{ACD} "
                     r"+ \beta\cdot Wide + Large}$"
                     r"   ($b=0$, $c_{pure}=\gamma=1$ all fixed)")},
    ]

    for hypo in hypotheses:
        run_hypothesis(train, grb, hypo, v8_fits, v8_errs, v8_rms_perdet)

    print("\n[all hypotheses plotted]")


if __name__ == "__main__":
    main()
