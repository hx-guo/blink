#!/usr/bin/env python3
"""M7 merged-ACD per-detector with deadtime correction:
    PHO · (1 − k·dt/L) = c_pure·Sci_pure + c_ACD·Sci_ACD + β·Wide + γ·Large + b

Each detector gets its own (k, c_pure, c_ACD, β, γ, b) = 6 parameters × 18 dets.
k is found by grid-scan over [-0.5, 6] minimising RMS in PHO units.

Plot layout (combined figure):
  top: 3×6 scatter (training viridis + 260226A overlay)
  bottom: 1×6 H-bar errorbars (5 coefficients + k)
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

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP = 400.0, 1000.0, 6000.0
N_MIN_PERDET = 100
X_LO = 300
N_SCATTER_PER_DET = 40_000

K_GRID = np.linspace(-0.5, 6.0, 131)   # for per-det k_opt grid scan

TRIGGER_260 = 446726273.0
MET_CORRECTION = 4.0


def load_training():
    dtype = {"date":"string","box":"category","met_sec":"int64","det":"int8",
             "L_cycles":"int32","PHO":"int32","Wide":"int32","Large":"int32",
             "Dt":"int32","Sci":"int32","Sci_ACD1":"int32","Sci_ACDN":"int32"}
    parts = []
    for f in sorted(CSV_DIR.glob("*.csv")):
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
    df["Sci_ACD"] = df["Sci_ACD1"] + df["Sci_ACDN"]
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("acd_rate","Sci_ACD"),("wide_rate","Wide"),
                    ("large_rate","Large"),("pho_rate","PHO")]:
        df[c] = df[src] / df["length"]
    df["group_rate"] = df["sci_sec_total"] / df["length"]
    df["dt_frac"] = df["Dt"].astype("float32") / df["L_cycles"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    hv = pd.read_csv(HV_TABLE, dtype={"date":"string","met_sec":"int64",
        **{f"hv{i}":"float32" for i in range(18)}})
    hv = hv.set_index(["date","met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    return df


def fit_dt_corrected(sub, k):
    """OLS on PHO·(1-k·dt/L) ~ RHS. Returns (coef, std_err)."""
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"], sub["large_rate"]]).astype(np.float64)
    lf = 1.0 - k * sub["dt_frac"].values.astype(np.float64)
    y = sub["pho_rate"].values.astype(np.float64) * lf
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    n, p = X.shape
    resid = y - X @ coef
    sigma2 = float((resid @ resid)) / max(n - p, 1)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        std_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        std_err = np.full(p, np.nan)
    return coef, std_err


def find_kopt(sub):
    """Grid scan k to minimise RMS in PHO units. Return (k_opt, coef, err, rms)."""
    Xmat = np.column_stack([np.ones(len(sub)), sub["scipure_rate"],
                             sub["acd_rate"], sub["wide_rate"],
                             sub["large_rate"]]).astype(np.float64)
    pho = sub["pho_rate"].values.astype(np.float64)
    dtf = sub["dt_frac"].values.astype(np.float64)
    best = (None, None, None, float('inf'))
    for k in K_GRID:
        lf = 1.0 - k * dtf
        if np.any(lf <= 0):
            continue
        target = pho * lf
        coef, *_ = np.linalg.lstsq(Xmat, target, rcond=None)
        pred_rhs = Xmat @ coef
        pred_pho = pred_rhs / lf
        rms = float(np.sqrt(np.mean((pho - pred_pho)**2)))
        if rms < best[3]:
            # Compute std_err for this fit
            n, p = Xmat.shape
            resid = target - pred_rhs
            sigma2 = float((resid @ resid)) / max(n - p, 1)
            try:
                cov = sigma2 * np.linalg.inv(Xmat.T @ Xmat)
                err = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                err = np.full(p, np.nan)
            best = (float(k), coef, err, rms)
    return best


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
            dt   = d[f"DeadTime_PHODet_{det_g}"].astype(float)
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                rows.append({
                    "box": box, "det": det_local,
                    "met_sec": int(met_eng[i]),
                    "length_cyc": length_cyc[i], "length_s": length_s[i],
                    "dt_cyc": dt[i],
                    "PHO": pho[i], "Wide": csi[i], "Large": large[i],
                })
        fe.close()
    eng = pd.DataFrame(rows)
    eng["dt_frac"] = eng["dt_cyc"] / eng["length_cyc"]

    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    sci_obs = sci_obs[sci_obs["type"]=="EVT"]
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    sci_obs_pd = sci_obs.groupby(["box","det_id","met_sec"]).size().rename("Sci_obs").reset_index()
    sci_obs_pd = sci_obs_pd.rename(columns={"det_id":"det"})

    sci_rec = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    sci_rec["box"] = sci_rec["box"].astype(str)
    sci_rec["met_sec"] = sci_rec["met"].astype("int64")
    sci_fill_box = sci_rec[sci_rec["type"]=="FILL_GAP"].groupby(["box","met_sec"]).size().rename("Sci_fill_box").reset_index()

    df = eng.merge(sci_obs_pd, on=["box","det","met_sec"], how="left")
    df["Sci_obs"] = df["Sci_obs"].fillna(0)
    df = df.merge(sci_fill_box, on=["box","met_sec"], how="left")
    df["Sci_fill_box"] = df["Sci_fill_box"].fillna(0)
    box_obs_sum = df.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    df["Sci_recov_box"] = box_obs_sum + df["Sci_fill_box"]
    df["Sci_recov"] = np.where(box_obs_sum > 0,
                                df["Sci_recov_box"] * df["Sci_obs"]/box_obs_sum.clip(lower=1),
                                df["Sci_recov_box"]/6)

    df["Sci_ACD_obs"] = 0.22 * df["Sci_obs"]
    df["Sci_pure_obs"] = df["Sci_obs"] - df["Sci_ACD_obs"]
    df["Sci_ACD_recov"] = 0.22 * df["Sci_recov"]
    df["Sci_pure_recov"] = df["Sci_recov"] - df["Sci_ACD_recov"]

    df["sci_rate_obs"] = df["Sci_obs"] / df["length_s"]
    df["scipure_rate_obs"] = df["Sci_pure_obs"] / df["length_s"]
    df["acd_rate_obs"] = df["Sci_ACD_obs"] / df["length_s"]
    df["sci_rate_recov"] = df["Sci_recov"] / df["length_s"]
    df["scipure_rate_recov"] = df["Sci_pure_recov"] / df["length_s"]
    df["acd_rate_recov"] = df["Sci_ACD_recov"] / df["length_s"]
    df["wide_rate"]  = df["Wide"]  / df["length_s"]
    df["large_rate"] = df["Large"] / df["length_s"]
    df["pho_rate"]   = df["PHO"]   / df["length_s"]
    df["t_rel"] = df["met_sec"] - TRIGGER_260
    return df


def apply_coefs_dt(df, kk_map, coef_map, acd_col):
    """Apply per-(box,det) dt-corrected M7-merged-ACD to derive Sci_pred.
       Inversion: Sci = [PHO·(1−k·dt/L) − (c_ACD−c_pure)·Sci_ACD − β·Wide − γ·Large − b] / c_pure
    """
    n = len(df)
    coefs = np.zeros((n, 5))
    kk    = np.zeros(n)
    keys = list(zip(df["box"].astype(str).values, df["det"].astype(int).values))
    for i, k in enumerate(keys):
        coefs[i] = coef_map[k]
        kk[i]    = kk_map[k]
    b, c0, cA, bet, gam = coefs.T
    lf = 1.0 - kk * df["dt_frac"].values
    pred = ((df["pho_rate"].values * lf
             - (cA-c0)*df[acd_col].values
             - bet*df["wide_rate"].values
             - gam*df["large_rate"].values - b) / c0)
    return pred


def main():
    print("Loading 2017-2019 training data...")
    train = load_training()
    print(f"  rows: {len(train):,}")
    print(f"  Dt/L: mean={train['dt_frac'].mean()*100:.2f}%  "
          f"max={train['dt_frac'].max()*100:.2f}%")

    # Box-level fits (fallback)
    print("\nBox-level fallback fits (uncorrected, for sparse per-det cases)...")
    box_fits = {}
    for box in "ABC":
        mask = ((train["box"] == box)
                & (train["sci_rate"] >= SCI_LO_CLEAN)
                & (train["sci_rate"] < SCI_HI_CLEAN)
                & (train["group_rate"] < BOX_RATE_CAP))
        coef, _ = fit_dt_corrected(train[mask], k=0.0)
        box_fits[box] = coef

    # Per-det fits with k-scan
    print("\nPer-(box,det) fits with k-scan over [-0.5, 6.0]...")
    print(f"  {'box-det':>8s}  {'N':>7s}  {'k_opt':>6s}  {'b':>8s}  "
          f"{'c_pure':>10s}  {'c_ACD':>10s}  {'beta':>10s}  {'gamma':>10s}  {'RMS':>7s}")
    fits_dict, errs_dict, k_dict, fallback_keys = {}, {}, {}, []
    for box in "ABC":
        for det in range(6):
            mask = ((train["box"] == box) & (train["det"] == det)
                    & (train["sci_rate"] >= SCI_LO_CLEAN)
                    & (train["sci_rate"] < SCI_HI_CLEAN)
                    & (train["group_rate"] < BOX_RATE_CAP))
            N = int(mask.sum())
            if N < N_MIN_PERDET:
                fits_dict[(box, det)] = box_fits[box]
                errs_dict[(box, det)] = np.full(5, np.nan)
                k_dict[(box, det)] = 0.0
                fallback_keys.append((box, det, N))
                c = fits_dict[(box, det)]
                print(f"  {box}-{det}      {N:>7d}  {0.0:>+6.2f}  {c[0]:>+8.2f}  "
                      f"{c[1]:>10.4f}  {c[2]:>10.4f}  {c[3]:>10.4f}  {c[4]:>10.4f}  ← fb")
            else:
                k_opt, coef, err, rms = find_kopt(train[mask])
                fits_dict[(box, det)] = coef
                errs_dict[(box, det)] = err
                k_dict[(box, det)] = k_opt
                print(f"  {box}-{det}      {N:>7d}  {k_opt:>+6.2f}  "
                      f"{coef[0]:>+5.2f}±{err[0]:>4.2f}  "
                      f"{coef[1]:>5.3f}±{err[1]:.4f}  "
                      f"{coef[2]:>5.3f}±{err[2]:.4f}  "
                      f"{coef[3]:>5.3f}±{err[3]:.4f}  "
                      f"{coef[4]:>5.3f}±{err[4]:.4f}  {rms:>6.2f}")

    # Apply for Sci_pred (training)
    train["sci_pred"] = apply_coefs_dt(train, k_dict, fits_dict, "acd_rate")

    # RMS comparison: uncorrected baseline vs dt-corrected per-det
    print("\nRMS comparison (CLEAN training band, PHO units):")
    print(f"  {'box':>3s}  {'N':>9s}  {'per-box no-dt':>14s}  "
          f"{'per-det+dt':>11s}  {'Δ%':>7s}")
    clean = ((train["sci_rate"] >= SCI_LO_CLEAN)
             & (train["sci_rate"] < SCI_HI_CLEAN)
             & (train["group_rate"] < BOX_RATE_CAP))
    # per-box no-dt baseline
    b_box, c0_box, cA_box, bt_box, gm_box = (
        train["box"].astype(str).map(lambda x: box_fits[x][i]).values for i in range(5))
    pho_pred_box = (b_box + c0_box*train["scipure_rate"].values
                     + cA_box*train["acd_rate"].values
                     + bt_box*train["wide_rate"].values
                     + gm_box*train["large_rate"].values)
    # per-det dt-corrected
    n = len(train)
    coefs = np.zeros((n, 5)); kk = np.zeros(n)
    keys = list(zip(train["box"].astype(str).values, train["det"].astype(int).values))
    for i, k in enumerate(keys):
        coefs[i] = fits_dict[k]; kk[i] = k_dict[k]
    bb, c0d, cAd, btd, gmd = coefs.T
    lf = 1.0 - kk*train["dt_frac"].values
    pho_pred_dt = (bb + c0d*train["scipure_rate"].values
                    + cAd*train["acd_rate"].values
                    + btd*train["wide_rate"].values
                    + gmd*train["large_rate"].values) / lf
    for box in "ABC":
        sel = clean & (train["box"]==box)
        actual = train.loc[sel, "pho_rate"].values
        rms_b = float(np.sqrt(np.mean((pho_pred_box[sel] - actual)**2)))
        rms_d = float(np.sqrt(np.mean((pho_pred_dt[sel]  - actual)**2)))
        print(f"  {box:>3s}  {int(sel.sum()):>9d}  {rms_b:>14.2f}  "
              f"{rms_d:>11.2f}  {100*(rms_d-rms_b)/rms_b:>+7.2f}")

    # 260226A
    print("\nLoading 260226A...")
    grb = load_260226A()
    print(f"  GRB dt/L: mean={grb['dt_frac'].mean()*100:.2f}%, "
          f"max={grb['dt_frac'].max()*100:.2f}%")
    grb["sci_pred_obs"]   = apply_coefs_dt(grb, k_dict, fits_dict, "acd_rate_obs")
    grb["sci_pred_recov"] = apply_coefs_dt(grb, k_dict, fits_dict, "acd_rate_recov")
    grb_with_sci = grb[grb["Sci_obs"] > 0].copy()
    g_burst = grb_with_sci[grb_with_sci["Sci_fill_box"] > 0]
    box_recov = (g_burst.groupby(["box", "met_sec"])
                  .agg(sci_recov_box=("sci_rate_recov", "mean"),
                       sci_pred_recov_box=("sci_pred_recov", "mean"))
                  .reset_index())
    print(f"  Box-level recovery seconds: {len(box_recov):,} "
          f"({box_recov['box'].value_counts().to_dict()})")

    # ============= Combined figure: 3×6 scatter + 1×6 errorbars (b, c_pure, c_ACD, β, γ, k) =============
    fig = plt.figure(figsize=(24, 20))
    outer = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[12, 7], hspace=0.30,
                               top=0.95, bottom=0.04, left=0.05, right=0.93)

    # ----- Top: 3×6 scatter -----
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
                dens = H[ix, iy].astype(float); dens[dens < 1] = 1
                idx = rng.choice(len(sub), N_SCATTER_PER_DET, replace=False) \
                       if len(sub) > N_SCATTER_PER_DET else np.arange(len(sub))
                order = np.argsort(dens[idx])
                sc = ax.scatter(sub["sci_rate"].values[idx][order],
                                 sub["sci_pred"].values[idx][order],
                                 c=dens[idx][order], cmap="viridis",
                                 norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
                                 s=1.5, alpha=0.6, rasterized=True, edgecolor="none")
                last_sc = sc

            g_own = grb_with_sci[(grb_with_sci["box"]==box)
                                  & (grb_with_sci["det"]==det)
                                  & (grb_with_sci["Sci_fill_box"] > 0)]
            ax.scatter(g_own["sci_rate_obs"], g_own["sci_pred_obs"],
                        s=18, color="blue", alpha=0.85, edgecolor="black", lw=0.4,
                        zorder=6, marker="o")
            g_brc = box_recov[box_recov["box"]==box]
            ax.scatter(g_brc["sci_recov_box"], g_brc["sci_pred_recov_box"],
                        s=18, color="red", alpha=0.85, edgecolor="black", lw=0.4,
                        zorder=7, marker="^")

            line = np.array([X_LO, 4500])
            ax.plot(line, line, "--", color="red", lw=1.0)

            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(X_LO, 4500); ax.set_ylim(X_LO/2, 7000)
            c = fits_dict[(box, det)]
            kk_disp = k_dict[(box, det)]
            is_fb = any(kk[0]==box and kk[1]==det for kk in fallback_keys)
            star = " *" if is_fb else ""
            ax.set_title(f"{box}-{det}{star}  k={kk_disp:+.2f}  "
                          f"c0={c[1]:.2f} cA={c[2]:.2f} β={c[3]:.2f} γ={c[4]:.2f}",
                          fontsize=8)
            ax.grid(alpha=0.3, which="both")
            if row == 2: ax.set_xlabel("Sci observed [cnt/s/det]")
            if det == 0: ax.set_ylabel(f"Box {box}\nSci predicted")

    legend_handles = [
        plt.Line2D([], [], color="red", ls="--", lw=1.0, label="y = x"),
        plt.Line2D([], [], marker="o", color="blue", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4, markersize=7,
                   label="260226A Sci_obs (per-det)"),
        plt.Line2D([], [], marker="^", color="red", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4, markersize=7,
                   label="260226A Sci_recov (box-level, same on 6 panels)"),
    ]
    fig.legend(handles=legend_handles, loc="upper left",
                bbox_to_anchor=(0.05, 0.945), fontsize=9, ncol=3,
                frameon=True, framealpha=0.92)
    if last_sc is not None:
        cbar_ax = fig.add_axes([0.945, 0.50, 0.012, 0.36])
        fig.colorbar(last_sc, cax=cbar_ax, label="2017-2019 local density (log)")

    # ----- Bottom: 1×6 H-bar errorbars (k, b, c_pure, c_ACD, β, γ) -----
    box_color = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    det_order = [(b, d) for b in "ABC" for d in range(6)]
    det_labels = [f"{b}-{d}" for b, d in det_order]
    n_dets = len(det_order)
    y_pos = np.arange(n_dets)

    # k has no formal OLS error (it's a profile-likelihood scan), so we'll
    # show k as a point only; the other 5 have ±1σ from inner OLS.
    param_names = [r"$k$ (dt-strength)", r"$b$ (cnt/s)",
                    r"$c_{\mathrm{pure}}$", r"$c_{\mathrm{ACD}}$",
                    r"$\beta$ (Wide)", r"$\gamma$ (Large)"]
    # 6 panels: index 0 is k, 1..5 are b, c_pure, c_ACD, β, γ from coef[0..4]
    coef_arr = np.array([fits_dict[k_] for k_ in det_order])
    err_arr  = np.array([errs_dict[k_] for k_ in det_order])
    k_arr    = np.array([k_dict[k_]   for k_ in det_order])
    box_idx  = np.array([k_[0] for k_ in det_order])

    gs_bot = outer[1].subgridspec(1, 6, wspace=0.08)
    axes3 = np.empty(6, dtype=object)
    for c in range(6):
        sharey = axes3[0] if c > 0 else None
        axes3[c] = fig.add_subplot(gs_bot[0, c], sharey=sharey)
        if c > 0: plt.setp(axes3[c].get_yticklabels(), visible=False)

    for p_idx, (ax, name) in enumerate(zip(axes3, param_names)):
        if p_idx == 0:
            # k panel: no formal 1σ — draw as filled circle only
            for i, (b, _) in enumerate(det_order):
                ax.scatter(k_arr[i], y_pos[i],
                           color=box_color[b], s=60, edgecolor='black', lw=0.6,
                           zorder=5)
            ax.axvline(0, color='gray', ls=':', lw=1, label="k=0 (no dt)")
            ax.axvline(1, color='orange', ls=':', lw=1, label="k=1 (user hypothesis)")
            for box in "ABC":
                sel = box_idx == box
                ax.axvline(float(np.mean(k_arr[sel])), color=box_color[box],
                            ls='--', lw=1.0, alpha=0.55, zorder=1)
            ax.set_xlim(K_GRID.min(), min(K_GRID.max(), float(np.nanmax(k_arr))+0.5))
        else:
            coef_idx = p_idx - 1   # map panel index → coef index
            for i, (b, _) in enumerate(det_order):
                ax.errorbar(coef_arr[i, coef_idx], y_pos[i],
                            xerr=err_arr[i, coef_idx],
                            fmt='|', color=box_color[b], ecolor=box_color[b],
                            elinewidth=0.8, capsize=10, capthick=1.8,
                            markersize=10, markeredgewidth=1.8, zorder=5)
            for box in "ABC":
                sel = box_idx == box
                ax.axvline(float(np.mean(coef_arr[sel, coef_idx])),
                            color=box_color[box], ls='--', lw=1.0,
                            alpha=0.55, zorder=1)

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
        plt.Line2D([], [], color='gray', ls='--', lw=1.0, label="per-box mean"),
        plt.Line2D([], [], color='gray', ls=':',  lw=1.0, label="k=0 / k=1 reference"),
    ]
    fig.legend(handles=legend_handles2, loc="upper right",
                bbox_to_anchor=(0.93, 0.35), fontsize=9, ncol=5,
                frameon=True, framealpha=0.92)

    fig.text(0.05, 0.965,
              "M7 merged-ACD per detector  +  deadtime correction  PHO·(1−k·dt/L) = RHS",
              fontsize=13, weight='bold')
    fig.text(0.05, 0.345,
              "Per-detector fit coefficients (left: k, no formal σ; right 5: OLS coefficients ± 1σ)",
              fontsize=13, weight='bold')

    formula = (r"$\bf{PHO \cdot (1 - k\,dt/L) = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} "
               r"+ \beta\cdot Wide + \gamma\cdot Large + b}$"
               + "    (6 params × 18 dets = 108 params)")
    fig.suptitle("M7 merged-ACD per-detector with deadtime correction\n" + formula,
                 fontsize=14, y=0.998)

    out = OUT_DIR / "sci_pred_M7merged_perdet_dt_corrected.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out}")
    desktop = Path.home() / "Desktop" / "sci_pred_M7merged_perdet_dt_corrected.png"
    fig.savefig(desktop, dpi=180, bbox_inches="tight")
    print(f"Saved: {desktop}")


if __name__ == "__main__":
    main()
