#!/usr/bin/env python3
"""M7 merged-ACD per-detector (5 params × 18 dets = 90 params), 260226A overlaid.
3 rows (boxes A/B/C) × 6 cols (det 0-5) = 18 panels.

For each (box, det) we fit an independent 5-parameter linear model:
  PHO_rate = c_pure·Sci_pure + c_ACD·Sci_ACD + β·Wide + γ·Large + b
This lets each detector carry its own gain / dark / ACD efficiency.
Box C's blind detector (insufficient CLEAN-band data) falls back to
the box-level fit so the panel still has a sensible prediction line."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from unwrap_large import unwrap_large

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
CACHE = Path("n_below_study/train_cache.parquet")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP = 400.0, 1000.0, 6000.0
N_MIN_PERDET = 100         # below this, fall back to box-level fit
X_LO = 300
N_SCATTER_PER_DET = 40_000
MAX_FILES = 2000           # cap CSV files loaded (memory safety); None = all 9726

TRIGGER_260 = 446726273.0
MET_CORRECTION = 4.0


def load_training():
    # Fast path: read parquet cache if available
    if CACHE.exists():
        print(f"  loading cache {CACHE} ...")
        df = pd.read_parquet(CACHE)
        print(f"  cache loaded: {len(df):,} rows, "
              f"date range {df['date'].min()} – {df['date'].max()}")
        return df
    # Fallback: parse CSVs (slow)
    print(f"  WARN: cache not found, falling back to CSV parsing")
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Sci": "int32", "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
    parts = []
    files = sorted(CSV_DIR.glob("*.csv"))
    if MAX_FILES is not None and len(files) > MAX_FILES:
        stride = len(files) // MAX_FILES
        files = files[::stride][:MAX_FILES]
    print(f"  loading {len(files)} CSV files...")
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
    df["Sci_ACD"] = df["Sci_ACD1"] + df["Sci_ACDN"]
    # per-row local ACD ratio (exact from CSV, used for self-consistent inversion)
    df["ratio_local"] = (df["Sci_ACD"].astype("float32")
                         / df["Sci"].astype("float32").clip(lower=1)).clip(0, 1)
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("acd_rate","Sci_ACD"),
                    ("wide_rate","Wide"),("large_rate","Large"),
                    ("pho_rate","PHO")]:
        df[c] = df[src] / df["length"]
    df["group_rate"] = df["sci_sec_total"] / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
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
    return df


def fit_m7_merged(sub, constrained=True):
    """constrained=True: fix c_pure=1, γ=1, dt k=1; fit (b, c_ACD, β).
         Target: PHO·(1−dt/L) − Sci_pure − Large = c_ACD·Sci_ACD + β·Wide + b
       constrained=False: V8 baseline 5-param fit (no dt, no constraints)."""
    if constrained:
        X = np.column_stack([np.ones(len(sub)), sub["acd_rate"], sub["wide_rate"]])
        lf = 1.0 - sub["dt_frac"].values
        y = (sub["pho_rate"].values * lf
             - sub["scipure_rate"].values
             - sub["large_rate"].values)
        c3, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.array([c3[0], 1.0, c3[1], c3[2], 1.0])
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"], sub["large_rate"]])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def fit_m7_merged_with_err(sub, constrained=True):
    """Return (coef, std_err) — 5-vectors. constrained: c_pure=γ=1, dt k=1."""
    if constrained:
        X = np.column_stack([np.ones(len(sub)), sub["acd_rate"], sub["wide_rate"]])
        lf = 1.0 - sub["dt_frac"].values
        y = (sub["pho_rate"].values * lf
             - sub["scipure_rate"].values
             - sub["large_rate"].values)
        c3, *_ = np.linalg.lstsq(X, y, rcond=None)
        n, p = X.shape
        resid = y - X @ c3
        sigma2 = float((resid @ resid)) / max(n - p, 1)
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            e3 = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            e3 = np.full(p, np.nan)
        coef = np.array([c3[0], 1.0, c3[1], c3[2], 1.0])
        err  = np.array([e3[0], 0.0, e3[1], e3[2], 0.0])
        return coef, err
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"], sub["large_rate"]])
    y = sub["pho_rate"].values
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
            dt = d[f"DeadTime_PHODet_{det_g}"].astype(float)   # for k=1 dt
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
    eng["dt_frac"] = eng["dt_cyc"] / eng["length_cyc"]   # k=1 dt correction

    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    sci_obs = sci_obs[sci_obs["type"]=="EVT"].copy()
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    # Decode aminfo (18-bit ACD shield mask) → ACD classification per event
    aminfo = sci_obs["aminfo"].values.astype(np.int64)
    popcount = np.zeros(len(aminfo), dtype=np.int32)
    for bit in range(18):
        popcount += ((aminfo >> bit) & 1).astype(np.int32)
    sci_obs["is_acd"] = (popcount > 0).astype("int32")     # ACD1 or ACDN
    # Aggregate per (box, det, sec)
    sci_obs_pd = sci_obs.groupby(["box","det_id","met_sec"]).agg(
        Sci_obs=("box", "size"),
        Sci_ACD_obs=("is_acd", "sum"),
    ).reset_index().rename(columns={"det_id":"det"})

    sci_rec = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    sci_rec["box"] = sci_rec["box"].astype(str)
    sci_rec["met_sec"] = sci_rec["met"].astype("int64")
    sci_fill_box = sci_rec[sci_rec["type"]=="FILL_GAP"].groupby(["box","met_sec"]).size().rename("Sci_fill_box").reset_index()

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

    # Per-(det, sec) local ACD ratio from observed events (fall back to box-sec
    # mean, then global, when this det observed 0 events at this second)
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
    df["t_rel"] = df["met_sec"] - TRIGGER_260
    return df


def apply_coefs(df, coef_map):
    """Self-consistent Sci_pred under k=1 dt + c_pure=γ=1 constraints:

       PHO·(1-dt/L) = Sci + (c_ACD-1)·Sci_ACD + β·Wide + b   (since c_pure=γ=1, Sci_pure = Sci-Sci_ACD)

    with Sci_ACD = r·Sci (self-consistent):

       Sci_pred = (PHO·(1-dt/L) - β·Wide - b) / [1 + (c_ACD-1)·r]

    coef_map keys: (box_str, det_int) → [b, c_pure=1, c_ACD, β, γ=1]"""
    n = len(df)
    coefs = np.zeros((n, 5))
    keys = list(zip(df["box"].astype(str).values, df["det"].astype(int).values))
    for i, k in enumerate(keys):
        coefs[i] = coef_map[k]
    b, c0, cA, bet, gam = coefs.T   # c0=γ should be 1, gam should be 1
    r = df["ratio_local"].values
    denom = c0 * (1.0 - r) + cA * r        # = 1 - r + c_ACD·r = 1 + (c_ACD-1)·r
    lf = 1.0 - df["dt_frac"].values        # k=1 dt correction
    pred = (df["pho_rate"].values * lf
            - bet*df["wide_rate"].values
            - gam*df["large_rate"].values     # = Large × 1, see note below
            - b) / denom
    # Note: gam*Large appears in the inversion identical to V8 formula —
    # algebraically, "Sci_pred = (PHO·lf - β·Wide - γ·Large - b) / denom" is
    # the correct inversion of "PHO·lf = c_pure·Sci_pure + c_ACD·Sci_ACD + β·Wide + γ·Large + b"
    # with c_pure=γ=1. The (1-r), c_ACD·r terms appear because Sci_pure = (1-r)·Sci.
    return pred


def main():
    print("Loading 2017-2019 training data...")
    train = load_training()
    print(f"  rows: {len(train):,}")

    # ---------- 1. Per-detector fit (with box-level fallback for sparse dets) ----------
    print("\nFitting M7-merged-ACD per (box, det)...")
    # First: box-level fits to use as fallback
    box_fits = {}
    for box in "ABC":
        mask = ((train["box"] == box)
                & (train["sci_rate"] >= SCI_LO_CLEAN) & (train["sci_rate"] < SCI_HI_CLEAN)
                & (train["group_rate"] < BOX_RATE_CAP))
        box_fits[box] = fit_m7_merged(train[mask])

    # Then per-(box, det) fits: constrained (c_pure=γ=1) + V8 baseline for compare
    fits_dict, errs_dict = {}, {}
    fits_v8,   errs_v8   = {}, {}
    fallback_keys = []
    print(f"  {'box-det':>8s}  {'N':>7s}  {'b':>10s}  {'c_pure':>12s}  "
          f"{'c_ACD':>12s}  {'beta':>12s}  {'gamma':>12s}")
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
                sub_box = train[((train["box"]==box)
                                  & (train["sci_rate"] >= SCI_LO_CLEAN)
                                  & (train["sci_rate"] < SCI_HI_CLEAN)
                                  & (train["group_rate"] < BOX_RATE_CAP))]
                fits_v8[(box, det)] = fit_m7_merged(sub_box, constrained=False)
                errs_v8[(box, det)] = np.full(5, np.nan)
                fallback_keys.append((box, det, N))
                c = fits_dict[(box, det)]
                print(f"  {box}-{det}      {N:>7d}  {c[0]:+10.2f}  "
                      f"{c[1]:>12.4f}  {c[2]:>12.4f}  {c[3]:>12.4f}  "
                      f"{c[4]:>12.4f}  ← fallback")
            else:
                c, e = fit_m7_merged_with_err(train[mask], constrained=True)
                fits_dict[(box, det)] = c
                errs_dict[(box, det)] = e
                c8, e8 = fit_m7_merged_with_err(train[mask], constrained=False)
                fits_v8[(box, det)] = c8
                errs_v8[(box, det)] = e8
                print(f"  {box}-{det}      {N:>7d}  "
                      f"{c[0]:+7.2f}±{e[0]:.2f}  {c[1]:>7.4f}±{e[1]:.4f}  "
                      f"{c[2]:>7.4f}±{e[2]:.4f}  {c[3]:>7.4f}±{e[3]:.4f}  "
                      f"{c[4]:>7.4f}±{e[4]:.4f}")
    if fallback_keys:
        print(f"\n  ⚠️  fallback used for {len(fallback_keys)} detector(s) "
              f"(N<{N_MIN_PERDET} CLEAN rows): {fallback_keys}")

    # ---------- 2. RMS comparison: per-box vs per-det ----------
    # NOTE: Target was PHO·(1-dt/L) → implied PHO = RHS_fit / (1-dt/L)
    train["box_str"] = train["box"].astype(str)
    lf_inv = 1.0 / (1.0 - train["dt_frac"].values)
    b_box  = train["box_str"].map(lambda x: box_fits[x][0]).values
    c0_box = train["box_str"].map(lambda x: box_fits[x][1]).values
    cA_box = train["box_str"].map(lambda x: box_fits[x][2]).values
    bt_box = train["box_str"].map(lambda x: box_fits[x][3]).values
    gm_box = train["box_str"].map(lambda x: box_fits[x][4]).values
    pho_pred_box = (b_box + c0_box*train["scipure_rate"].values
                     + cA_box*train["acd_rate"].values
                     + bt_box*train["wide_rate"].values
                     + gm_box*train["large_rate"].values) * lf_inv

    # Per-det prediction on training data
    n = len(train)
    coefs = np.zeros((n, 5))
    keys = list(zip(train["box"].astype(str).values, train["det"].astype(int).values))
    for i, k in enumerate(keys):
        coefs[i] = fits_dict[k]
    b_pd, c0_pd, cA_pd, bt_pd, gm_pd = coefs.T
    pho_pred_pd = (b_pd + c0_pd*train["scipure_rate"].values
                    + cA_pd*train["acd_rate"].values
                    + bt_pd*train["wide_rate"].values
                    + gm_pd*train["large_rate"].values) * lf_inv

    print("\nRMS comparison on CLEAN training band, per box:")
    print(f"  {'box':>3s}  {'N_clean':>9s}  {'per-box RMS':>12s}  "
          f"{'per-det RMS':>12s}  {'Δ%':>7s}")
    clean_mask = ((train["sci_rate"] >= SCI_LO_CLEAN)
                  & (train["sci_rate"] < SCI_HI_CLEAN)
                  & (train["group_rate"] < BOX_RATE_CAP))
    # V8 PHO prediction (5-param fit, no constraint)
    coefs_v8 = np.zeros((n, 5))
    for i, k in enumerate(keys):
        coefs_v8[i] = fits_v8[k]
    b8, c08, cA8, bt8, gm8 = coefs_v8.T
    pho_pred_v8 = (b8 + c08*train["scipure_rate"].values
                    + cA8*train["acd_rate"].values
                    + bt8*train["wide_rate"].values
                    + gm8*train["large_rate"].values)
    # Per-(box, det) RMS for panel annotation
    rms_perdet = {}
    rms_perdet_v8 = {}
    for box in "ABC":
        for det in range(6):
            sel = (clean_mask & (train["box"] == box) & (train["det"] == det))
            if sel.sum() > 0:
                a = train.loc[sel, "pho_rate"].values
                rms_perdet[(box, det)] = float(np.sqrt(np.mean((pho_pred_pd[sel] - a)**2)))
                rms_perdet_v8[(box, det)] = float(np.sqrt(np.mean((pho_pred_v8[sel] - a)**2)))
            else:
                rms_perdet[(box, det)] = float("nan")
                rms_perdet_v8[(box, det)] = float("nan")
    for box in "ABC":
        sel = clean_mask & (train["box"] == box)
        actual = train.loc[sel, "pho_rate"].values
        rms_b = float(np.sqrt(np.mean((pho_pred_box[sel] - actual)**2)))
        rms_d = float(np.sqrt(np.mean((pho_pred_pd[sel] - actual)**2)))
        delta = 100.0 * (rms_d - rms_b) / rms_b
        print(f"  {box:>3s}  {int(sel.sum()):>9d}  {rms_b:>12.2f}  "
              f"{rms_d:>12.2f}  {delta:>+7.2f}")

    # Sci_pred (self-consistent, per-row local ratio) for training scatter
    train["sci_pred"] = apply_coefs(train, fits_dict)

    # ---------- 3. 260226A ----------
    print("\nLoading 260226A...")
    grb = load_260226A()
    # Single Sci_pred per (det, sec) — function of engineering quantities
    # and local ACD ratio, INDEPENDENT of whether we look at Sci_obs or Sci_recov
    grb["sci_pred"] = apply_coefs(grb, fits_dict)
    grb_with_sci = grb[grb["Sci_obs"] > 0].copy()
    print(f"  GRB rows with cached Sci: {len(grb_with_sci):,}")

    # Per-(det, sec) recovery: keep this det's observed part exactly, distribute
    # the box-level FILL_GAP back to per-det by Sci_obs share. The result is
    # already in grb_with_sci["sci_rate_recov"] from load_260226A. No box-level
    # averaging — each det gets its own RED X reflecting "if all 6 dets were
    # truncated by FIFO at the same per-event probability, my recovered share
    # is in proportion to my observed share."
    g_burst = grb_with_sci[grb_with_sci["Sci_fill_box"] > 0]
    print(f"  Burst (FILL_GAP>0) rows: {len(g_burst):,} "
          f"({g_burst.groupby('box').size().to_dict()} per box; "
          f"× 6 dets each)")

    # ---------- 4. Combined figure: 3×6 scatter (top) + 1×5 coefficients (bottom) ----------
    # Use nested GridSpec because the two sections have different column counts
    # (top: 6 panels per row of scatter; bottom: 5 panels for the 5 parameters).
    import matplotlib.gridspec as gridspec

    # 16:9 aspect, top section gets ~75% of height
    # top=0.92 leaves a wide header strip for suptitle + formula
    fig = plt.figure(figsize=(24, 13.5))
    outer = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[3, 1],
                               hspace=0.30,
                               top=0.92, bottom=0.05, left=0.05, right=0.93)

    # ----- Top: 3 rows × 6 cols scatter -----
    gs_top = outer[0].subgridspec(3, 6, hspace=0.30, wspace=0.10)
    axes = np.empty((3, 6), dtype=object)
    for r in range(3):
        for c in range(6):
            sharex = axes[0, c] if r > 0 else None
            sharey = axes[r, 0] if c > 0 else None
            axes[r, c] = fig.add_subplot(gs_top[r, c], sharex=sharex, sharey=sharey)
            if r < 2:
                plt.setp(axes[r, c].get_xticklabels(), visible=False)
            if c > 0:
                plt.setp(axes[r, c].get_yticklabels(), visible=False)

    xb = np.logspace(np.log10(X_LO), np.log10(4500), 120)
    yb = np.logspace(np.log10(X_LO/2), np.log10(7000), 120)
    last_sc = None
    rng = np.random.RandomState(0)

    for row, box in enumerate("ABC"):
        for col in range(6):
            det = col
            ax = axes[row, col]
            sub = train[(train["box"]==box) & (train["det"]==det)
                        & (train["sci_rate"] >= X_LO) & (train["sci_pred"] > 0)]
            if len(sub) > 0:
                H, xedges, yedges = np.histogram2d(sub["sci_rate"].values,
                                                     sub["sci_pred"].values,
                                                     bins=[xb, yb])
                ix = np.clip(np.searchsorted(xedges, sub["sci_rate"].values) - 1,
                             0, len(xedges)-2)
                iy = np.clip(np.searchsorted(yedges, sub["sci_pred"].values) - 1,
                             0, len(yedges)-2)
                density = H[ix, iy].astype(float)
                density[density < 1] = 1
                if len(sub) > N_SCATTER_PER_DET:
                    idx = rng.choice(len(sub), N_SCATTER_PER_DET, replace=False)
                else:
                    idx = np.arange(len(sub))
                order = np.argsort(density[idx])
                sc = ax.scatter(sub["sci_rate"].values[idx][order],
                                 sub["sci_pred"].values[idx][order],
                                 c=density[idx][order],
                                 cmap="viridis",
                                 norm=LogNorm(vmin=1, vmax=max(density.max(), 2)),
                                 s=1.5, alpha=0.6, rasterized=True, edgecolor="none")
                last_sc = sc

            # 260226A burst overlay (FILL_GAP > 0)
            # Both blue and red use the same per-(det, sec) Sci_pred (self-
            # consistent, engineering-data only).
            # Blue X = this det's observed Sci_obs rate.
            # Red X = this det's recovered rate = Sci_obs + (box_fill share by Sci_obs).
            # → connecting line is HORIZONTAL, pointing right (red always ≥ blue
            #   since recovery only ADDs events). Line length = FIFO loss for this det.
            g_own = grb_with_sci[(grb_with_sci["box"] == box)
                                  & (grb_with_sci["det"] == det)
                                  & (grb_with_sci["Sci_fill_box"] > 0)]
            for _, r in g_own.iterrows():
                ax.plot([r["sci_rate_obs"], r["sci_rate_recov"]],
                        [r["sci_pred"], r["sci_pred"]],     # horizontal
                        color="gray", lw=0.7, alpha=0.55, zorder=5)
            ax.scatter(g_own["sci_rate_obs"], g_own["sci_pred"],
                        s=18, color="blue", alpha=0.85, edgecolor="black", lw=0.4,
                        zorder=6, marker="o")
            ax.scatter(g_own["sci_rate_recov"], g_own["sci_pred"],
                        s=18, color="red", alpha=0.85, edgecolor="black", lw=0.4,
                        zorder=7, marker="^")

            line = np.array([X_LO, 4500])
            c = fits_dict[(box, det)]
            ax.plot(line, line, "--", color="red", lw=1.0)

            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(X_LO, 4500); ax.set_ylim(X_LO/2, 7000)
            is_fb = any(k[0]==box and k[1]==det for k in fallback_keys)
            star = " *" if is_fb else ""
            ax.set_title(f"{box}-{det}{star}  c0={c[1]:.2f} cA={c[2]:.2f} "
                          f"β={c[3]:.2f} γ={c[4]:.2f}", fontsize=8)
            # RMS annotation (per-det, PHO units)
            rms_now = rms_perdet[(box, det)]
            rms_v8  = rms_perdet_v8[(box, det)]
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
            if row == 2:
                ax.set_xlabel("Sci observed [cnt/s/det]")
            if col == 0:
                ax.set_ylabel(f"Box {box}\nSci predicted")

    # Top section legend — placed INSIDE the first scatter panel (A-0) at
    # upper-left, where there is empty space above the y=x diagonal. This
    # avoids overlap with both the suptitle and the bottom x-axis labels.
    legend_handles = [
        plt.Line2D([], [], color="red", ls="--", lw=1.5, label="y = x"),
        plt.Line2D([], [], marker="o", color="blue", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4,
                   markersize=7, label="260226A Sci_obs (per-det)"),
        plt.Line2D([], [], marker="^", color="red", lw=0,
                   markeredgecolor="black", markeredgewidth=0.4,
                   markersize=7,
                   label="260226A Sci_recov (per-det: obs + FILL_GAP × obs-share)"),
        plt.Line2D([], [], color="gray", lw=0.8, alpha=0.6,
                   label="same-second pair (Sci_obs → Sci_recov)"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="lower left",
                       fontsize=7, frameon=True, framealpha=0.92)

    # Shared colorbar for the scatter section
    if last_sc is not None:
        cbar_ax = fig.add_axes([0.945, 0.50, 0.012, 0.36])
        fig.colorbar(last_sc, cax=cbar_ax, label="2017-2019 local density (log)")

    # ----- Bottom: 1 row × 5 cols coefficient + error -----
    param_names = [r"$b$ (cnt/s)",
                   r"$c_{\mathrm{pure}}$",
                   r"$c_{\mathrm{ACD}}$",
                   r"$\beta$ (Wide)",
                   r"$\gamma$ (Large)"]
    box_color = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    det_order = [(b, d) for b in "ABC" for d in range(6)]
    det_labels = [f"{b}-{d}" for b, d in det_order]
    n_dets = len(det_order)
    y_pos = np.arange(n_dets)

    coef_arr = np.array([fits_dict[k] for k in det_order])
    err_arr  = np.array([errs_dict[k] for k in det_order])
    coef_v8a = np.array([fits_v8[k]   for k in det_order])
    err_v8a  = np.array([errs_v8[k]   for k in det_order])
    box_idx  = np.array([k[0] for k in det_order])

    gs_bot = outer[1].subgridspec(1, 5, wspace=0.08)
    axes3 = np.empty(5, dtype=object)
    for c in range(5):
        sharey = axes3[0] if c > 0 else None
        axes3[c] = fig.add_subplot(gs_bot[0, c], sharey=sharey)
        if c > 0:
            plt.setp(axes3[c].get_yticklabels(), visible=False)

    for p_idx, (ax, name) in enumerate(zip(axes3, param_names)):
        for i, (b, _) in enumerate(det_order):
            # Connecting line V8 → constrained (per det)
            ax.plot([coef_v8a[i, p_idx], coef_arr[i, p_idx]],
                    [y_pos[i], y_pos[i]],
                    color=box_color[b], lw=0.7, alpha=0.35, zorder=2)
            # V8 baseline (5 free params) — light ghost
            ax.errorbar(coef_v8a[i, p_idx], y_pos[i],
                        xerr=err_v8a[i, p_idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        alpha=0.30,
                        elinewidth=0.6, capsize=6, capthick=1.0,
                        markersize=7, markeredgewidth=1.0, zorder=3)
            # Constrained (c_pure=γ=1) — bold foreground
            ax.errorbar(coef_arr[i, p_idx], y_pos[i],
                        xerr=err_arr[i, p_idx],
                        fmt='|', color=box_color[b], ecolor=box_color[b],
                        elinewidth=0.8, capsize=10, capthick=1.8,
                        markersize=10, markeredgewidth=1.8, zorder=5)

        for box in "ABC":
            sel = box_idx == box
            box_mean = float(np.mean(coef_arr[sel, p_idx]))
            ax.axvline(box_mean, color=box_color[box], ls='--', lw=1.0,
                       alpha=0.55, zorder=1)
        # Highlight x=1 reference for c_pure and γ panels
        if p_idx in (1, 4):
            ax.axvline(1.0, color='black', ls=':', lw=1.0, alpha=0.7, zorder=1)

        ax.axhline(5.5, color='black', ls=':', lw=0.6, alpha=0.5)
        ax.axhline(11.5, color='black', ls=':', lw=0.6, alpha=0.5)

        ax.set_title(name, fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlabel("coefficient value")

    axes3[0].set_yticks(y_pos)
    axes3[0].set_yticklabels(det_labels, fontsize=9)
    axes3[0].invert_yaxis()
    axes3[0].set_ylabel("detector")

    # Bottom section legend — placed INSIDE the first coefficient panel (b)
    # at upper-right corner where x is large (no data points there).
    legend_handles2 = [
        plt.Line2D([], [], marker='|', color="#d62728", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box A (constr.)"),
        plt.Line2D([], [], marker='|', color="#2ca02c", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box B (constr.)"),
        plt.Line2D([], [], marker='|', color="#1f77b4", lw=0,
                   markersize=10, markeredgewidth=1.8, label="Box C (constr.)"),
        plt.Line2D([], [], marker='|', color='gray', lw=0, alpha=0.30,
                   markersize=7, markeredgewidth=1.0, label="V8 (5 free, ghost)"),
        plt.Line2D([], [], color='black', ls=':', lw=1.0, alpha=0.7,
                   label="x = 1 (fixed at)"),
    ]
    axes3[0].legend(handles=legend_handles2, loc="upper right",
                     fontsize=7, ncol=1, frameon=True, framealpha=0.92)

    # ----- Section titles -----
    formula = (r"$\mathbf{PHO \cdot (1 - dt/L) = Sci_{pure} + c_{ACD}\cdot Sci_{ACD} "
               r"+ \beta\cdot Wide + Large + b}$"
               r"   ($c_{pure} = \gamma = 1$, $k=1$ dt-correction)")
    fig.text(0.05, 0.94, formula, fontsize=12)
    fig.text(0.05, 0.255,
              r"Per-detector fit coefficients $\pm$ 1$\sigma$  "
              r"(V8 ghost: 5 free; constrained: $c_{pure}=\gamma=1$ + dt $k=1$)",
              fontsize=12, weight='bold')

    fig.suptitle("M7 merged-ACD per-detector + dt(k=1) + $c_{pure}=\\gamma=1$ — "
                 "training + 260226A burst overlay   (3 params × 18 dets = 54 params)",
                 fontsize=14, y=0.975)

    out = OUT_DIR / "sci_pred_M7merged_perdet_combined_dt_constraints.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"\nSaved: {out}")
    desktop = Path.home() / "Desktop" / "sci_pred_M7merged_perdet_combined_dt_constraints.png"
    fig.savefig(desktop, dpi=240, bbox_inches="tight")
    print(f"Saved: {desktop}")

    # Print per-detector coefficient spread summary
    print("\nPer-detector coefficient spread vs typical 1σ error:")
    print(f"  {'param':>8s}  {'inter-det std':>14s}  {'median 1σ err':>14s}  "
          f"{'ratio (spread/err)':>18s}")
    for p_idx, name in enumerate(["b", "c_pure", "c_ACD", "beta", "gamma"]):
        spread = float(np.std(coef_arr[:, p_idx], ddof=1))
        med_err = float(np.median(err_arr[:, p_idx]))
        ratio = spread / med_err if med_err > 0 else float("nan")
        print(f"  {name:>8s}  {spread:>14.4f}  {med_err:>14.4f}  {ratio:>18.1f}")


if __name__ == "__main__":
    main()
