#!/usr/bin/env python3
"""Diagnostic: directly compare V8 (no dt) vs V9-linear vs V9-exp Sci_pred for 260226A burst.
Plot all three predictions on the same Sci_recov vs Sci_pred panel per box,
to see how red triangles shift between models."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
TRIGGER_260 = 446726273.0
MET_CORRECTION = 4.0


def load_training():
    dtype = {"date":"string","box":"category","met_sec":"int64","det":"int8",
             "L_cycles":"int32","PHO":"int32","Wide":"int32","Large":"int32",
             "Dt":"int32","Sci":"int32","Sci_ACD1":"int32","Sci_ACDN":"int32"}
    parts = []
    for f in sorted(CSV_DIR.glob("*.csv")):
        try: parts.append(pd.read_csv(f, usecols=list(dtype), dtype=dtype))
        except: pass
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


def fit_v8(sub):
    """V8: OLS on PHO ~ RHS (no dt correction)."""
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"], sub["large_rate"]]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values.astype(np.float64), rcond=None)
    return coef


def find_kopt(sub, form):
    """Grid scan k for `form` ∈ {'lin', 'exp'}."""
    Xmat = np.column_stack([np.ones(len(sub)), sub["scipure_rate"],
                             sub["acd_rate"], sub["wide_rate"],
                             sub["large_rate"]]).astype(np.float64)
    pho = sub["pho_rate"].values.astype(np.float64)
    dtf = sub["dt_frac"].values.astype(np.float64)
    K = np.linspace(-0.5, 10.0, 211) if form == "exp" else np.linspace(-0.5, 6.0, 131)
    best = (None, None, float('inf'))
    for k in K:
        lf = np.exp(-k * dtf) if form == "exp" else (1.0 - k * dtf)
        if (form == "lin") and np.any(lf <= 0): continue
        target = pho * lf
        coef, *_ = np.linalg.lstsq(Xmat, target, rcond=None)
        pred = Xmat @ coef
        pho_pred = pred / lf
        rms = float(np.sqrt(np.mean((pho - pho_pred)**2)))
        if rms < best[2]: best = (float(k), coef, rms)
    return best


def load_260226A_engineering():
    rows = []
    for box, code in [("A","0766"),("B","1009"),("C","1781")]:
        fe = fits.open(f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits",
                        memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
        L = d["Length_Time_Cycle"].astype(float)
        length_s = L * 16e-6
        for det in range(6):
            det_g = BOX_OFFSET[box] + det
            pho = d[f"Cnt_PHODet_{det_g}"].astype(float)
            wide = d[f"Cnt_CsI_PHODet_{det_g}"].astype(float)
            large_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)
            dt = d[f"DeadTime_PHODet_{det_g}"].astype(float)
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                rows.append({"box":box,"det":det,"met_sec":int(met_eng[i]),
                              "L_cyc":L[i],"length_s":length_s[i],"dt_cyc":dt[i],
                              "PHO":pho[i],"Wide":wide[i],"Large":large[i]})
        fe.close()
    df = pd.DataFrame(rows)
    df["dt_frac"] = df["dt_cyc"] / df["L_cyc"]
    return df


def main():
    print("Loading training (this may take ~30s)...")
    train = load_training()

    # Per-det fits for all three models
    print("Fitting V8 / V9-linear / V9-exp per (box,det)...")
    fits_v8, fits_lin, fits_exp = {}, {}, {}
    k_lin, k_exp = {}, {}
    for box in "ABC":
        for det in range(6):
            mask = ((train["box"]==box) & (train["det"]==det)
                    & (train["sci_rate"] >= SCI_LO_CLEAN)
                    & (train["sci_rate"] < SCI_HI_CLEAN)
                    & (train["group_rate"] < BOX_RATE_CAP))
            sub = train[mask]
            fits_v8[(box, det)] = fit_v8(sub)
            kl, cl, _ = find_kopt(sub, "lin");  fits_lin[(box, det)] = cl;  k_lin[(box, det)] = kl
            ke, ce, _ = find_kopt(sub, "exp");  fits_exp[(box, det)] = ce;  k_exp[(box, det)] = ke

    # Load 260226A and identify burst seconds (FILL_GAP > 0)
    print("Loading 260226A engineering + Sci_obs + FILL_GAP...")
    grb = load_260226A_engineering()
    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"])
    sci_obs = sci_obs[sci_obs["type"]=="EVT"]
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    sci_obs_pd = (sci_obs.groupby(["box","det_id","met_sec"]).size()
                  .rename("Sci_obs").reset_index().rename(columns={"det_id":"det"}))
    sci_rec = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    sci_rec["box"] = sci_rec["box"].astype(str); sci_rec["met_sec"] = sci_rec["met"].astype("int64")
    sci_fill_box = (sci_rec[sci_rec["type"]=="FILL_GAP"].groupby(["box","met_sec"]).size()
                    .rename("Sci_fill_box").reset_index())

    grb = grb.merge(sci_obs_pd, on=["box","det","met_sec"], how="left")
    grb["Sci_obs"] = grb["Sci_obs"].fillna(0)
    grb = grb.merge(sci_fill_box, on=["box","met_sec"], how="left")
    grb["Sci_fill_box"] = grb["Sci_fill_box"].fillna(0)
    box_obs_sum = grb.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    grb["Sci_recov_box_total"] = box_obs_sum + grb["Sci_fill_box"]
    # Rates
    for c, s in [("pho_rate","PHO"),("wide_rate","Wide"),("large_rate","Large"),
                 ("sci_rate_obs","Sci_obs")]:
        grb[c] = grb[s] / grb["length_s"]
    grb["scipure_rate_obs"] = 0.78 * grb["sci_rate_obs"]
    grb["acd_rate_obs"]     = 0.22 * grb["sci_rate_obs"]
    grb["sci_rate_recov_box_avg"] = grb["Sci_recov_box_total"] / grb["length_s"] / 6

    # Apply each model
    def pred_v8(row):
        c = fits_v8[(row["box"], row["det"])]
        return (row["pho_rate"] - (c[2]-c[1])*row["acd_rate_obs"]
                - c[3]*row["wide_rate"] - c[4]*row["large_rate"] - c[0]) / c[1]
    def pred_lin(row):
        c = fits_lin[(row["box"], row["det"])]
        k = k_lin[(row["box"], row["det"])]
        lf = 1.0 - k * row["dt_frac"]
        return (row["pho_rate"]*lf - (c[2]-c[1])*row["acd_rate_obs"]
                - c[3]*row["wide_rate"] - c[4]*row["large_rate"] - c[0]) / c[1]
    def pred_exp(row):
        c = fits_exp[(row["box"], row["det"])]
        k = k_exp[(row["box"], row["det"])]
        lf = float(np.exp(-k * row["dt_frac"]))
        return (row["pho_rate"]*lf - (c[2]-c[1])*row["acd_rate_obs"]
                - c[3]*row["wide_rate"] - c[4]*row["large_rate"] - c[0]) / c[1]

    print("Computing per-row Sci_pred for V8 / linear / exp...")
    grb["sci_pred_v8"]  = grb.apply(pred_v8,  axis=1)
    grb["sci_pred_lin"] = grb.apply(pred_lin, axis=1)
    grb["sci_pred_exp"] = grb.apply(pred_exp, axis=1)

    # Per-box avg over 6 dets at each burst second
    burst = grb[(grb["Sci_fill_box"] > 0) & (grb["Sci_obs"] > 0)]
    agg = burst.groupby(["box","met_sec"]).agg(
        sci_recov_avg=("sci_rate_recov_box_avg","first"),
        v8 =("sci_pred_v8", "mean"),
        lin=("sci_pred_lin","mean"),
        exp=("sci_pred_exp","mean"),
        dt_frac=("dt_frac","mean"),
    ).reset_index()
    print(f"\nBurst seconds: {len(agg)} ({agg['box'].value_counts().to_dict()})")

    # Print mean offset per box (Sci_pred − Sci_recov), in cnt/s/det
    print(f"\n{'box':>3s}  {'<Sci_recov>':>11s}  {'<V8 pred>':>10s}  {'<V9-lin pred>':>13s}  {'<V9-exp pred>':>13s}")
    for box in "ABC":
        a = agg[agg["box"]==box]
        print(f"  {box}  {a['sci_recov_avg'].mean():>11.1f}  {a['v8'].mean():>10.1f}  "
              f"{a['lin'].mean():>13.1f}  {a['exp'].mean():>13.1f}")

    # Plot 1×3: per box scatter, 3 colors for 3 models
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    colors = {"v8":"#1f77b4", "lin":"#ff7f0e", "exp":"#2ca02c"}
    labels = {"v8":"V8 (no dt)", "lin":"V9 linear (1−k·dt/L)", "exp":"V9 exp (exp(−k·dt/L))"}
    for ax, box in zip(axes, "ABC"):
        a = agg[agg["box"]==box]
        x = a["sci_recov_avg"].values
        for key in ["v8","lin","exp"]:
            ax.scatter(x, a[key].values, s=40, color=colors[key], edgecolor="black",
                        lw=0.4, alpha=0.85, label=labels[key])
            # draw connecting line per second showing v8→lin→exp
        line = np.array([200, 5000])
        ax.plot(line, line, "--", color="red", lw=1.2, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(200, 5000); ax.set_ylim(200, 8000)
        ax.set_xlabel("Sci_recov_box_avg [cnt/s/det]")
        if box == "A":
            ax.set_ylabel("Sci_pred [cnt/s/det]")
        ax.set_title(f"Box {box}  (burst seconds, mean over 6 dets)\n"
                      f"<dt_frac> ≈ {a['dt_frac'].mean()*100:.1f}%, "
                      f"k_lin̅={np.mean([k_lin[(box,d)] for d in range(6)]):.1f}  "
                      f"k_exp̅={np.mean([k_exp[(box,d)] for d in range(6)]):.1f}",
                      fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("260226A burst: how Sci_pred differs between V8, V9-linear, V9-exp",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    out = OUT_DIR / "diag_v8_vs_v9_burst.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {out}")
    desktop = Path.home() / "Desktop" / "diag_v8_vs_v9_burst.png"
    fig.savefig(desktop, dpi=160, bbox_inches="tight")
    print(f"Saved: {desktop}")


if __name__ == "__main__":
    main()
