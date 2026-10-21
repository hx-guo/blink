#!/usr/bin/env python3
"""M7 with merged ACD channel, 260226A overlaid.
V2: actually decode aminfo to get per-bin ACD ratio, scale to Sci_recov."""
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
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}

SCI_LO_CLEAN, SCI_HI_CLEAN, BOX_RATE_CAP = 400.0, 1000.0, 6000.0
X_LO = 300
N_SCATTER = 200_000

TRIGGER_260 = 446726273.0
MET_CORRECTION = 4.0


def load_training():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Sci": "int32", "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
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


def fit_m7_merged(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd_rate"],
                          sub["wide_rate"], sub["large_rate"]])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def load_260226A_decoded():
    """Load 260226A with REAL ACD decoded from aminfo bits."""
    # Engineering FITS
    rows = []
    for box, code in [("A","0766"),("B","1009"),("C","1781")]:
        eng_file = f"data/1B/2026/20260226/{code}/HXMT_1B_{code}_20260226T100000_G076262_000_004.fits"
        fe = fits.open(eng_file, memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met_eng = d["Time"].astype(float) + offset + MET_CORRECTION
        length_s = d["Length_Time_Cycle"].astype(float) * 16e-6
        for det_local in range(6):
            det_g = BOX_OFFSET[box] + det_local
            pho = d[f"Cnt_PHODet_{det_g}"].astype(float)
            csi = d[f"Cnt_CsI_PHODet_{det_g}"].astype(float)
            large_raw = d[f"Cnt_LargeEvt_{det_g}"].astype(float)
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                rows.append({
                    "box": box, "det": det_local,
                    "met_sec": int(met_eng[i]),
                    "length_s": length_s[i],
                    "PHO": pho[i], "Wide": csi[i], "Large": large[i],
                })
        fe.close()
    eng = pd.DataFrame(rows)

    # Decode 1B events: aminfo -> ACD class
    print("  Reading 1B solved events + decoding aminfo bits...")
    sci_obs = pd.read_csv("/tmp/260226A_validate/solved.csv",
        names=["box","type","met","channel","det_id","pkt_idx","evt_idx","aminfo","pulinfo"],
        dtype={"aminfo":"int64"})
    sci_obs = sci_obs[sci_obs["type"]=="EVT"]
    sci_obs["box"] = sci_obs["box"].astype(str)
    sci_obs["met_sec"] = sci_obs["met"].astype("int64")
    # ACD class:
    aminfo = sci_obs["aminfo"].values
    # popcount for 18-bit
    popcount = np.zeros_like(aminfo)
    for bit in range(18):
        popcount += (aminfo >> bit) & 1
    sci_obs["acd_class"] = np.where(popcount == 0, "pure",
                            np.where(popcount == 1, "ACD1", "ACDN"))
    print(f"  ACD class distribution:")
    print(sci_obs["acd_class"].value_counts())
    print(f"  Overall ACD/Sci ratio: "
          f"{(sci_obs['acd_class']!='pure').sum()/len(sci_obs):.3f}")

    # Aggregate per (box, det, met_sec)
    sci_obs["sci_count"] = 1
    sci_obs["pure_count"] = (sci_obs["acd_class"]=="pure").astype(int)
    sci_obs["acd1_count"] = (sci_obs["acd_class"]=="ACD1").astype(int)
    sci_obs["acdn_count"] = (sci_obs["acd_class"]=="ACDN").astype(int)
    agg = sci_obs.groupby(["box","det_id","met_sec"]).agg(
        Sci_obs=("sci_count","sum"),
        Sci_pure_obs=("pure_count","sum"),
        Sci_ACD1_obs=("acd1_count","sum"),
        Sci_ACDN_obs=("acdn_count","sum"),
    ).reset_index()
    agg = agg.rename(columns={"det_id":"det"})
    agg["Sci_ACD_obs"] = agg["Sci_ACD1_obs"] + agg["Sci_ACDN_obs"]
    print(f"  per-(box,det,met_sec) bins with EVT: {len(agg):,}")

    # FILL_GAP counts per (box, met_sec)
    sci_rec = pd.read_csv("/tmp/260226A_validate/reconstructed.csv",
        names=["box","type","met","channel","pkt_idx","evt_idx"])
    sci_rec["box"] = sci_rec["box"].astype(str)
    sci_rec["met_sec"] = sci_rec["met"].astype("int64")
    sci_fill_box = sci_rec[sci_rec["type"]=="FILL_GAP"].groupby(["box","met_sec"]).size().rename("Sci_fill_box").reset_index()

    # Merge
    df = eng.merge(agg, on=["box","det","met_sec"], how="left")
    for c in ["Sci_obs","Sci_pure_obs","Sci_ACD1_obs","Sci_ACDN_obs","Sci_ACD_obs"]:
        df[c] = df[c].fillna(0)
    df = df.merge(sci_fill_box, on=["box","met_sec"], how="left")
    df["Sci_fill_box"] = df["Sci_fill_box"].fillna(0)

    # Sci_recov per det: scale fill events to dets proportional to Sci_obs
    box_obs_sum = df.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    df["Sci_recov_box"] = box_obs_sum + df["Sci_fill_box"]
    df["Sci_recov"] = np.where(box_obs_sum > 0,
                                df["Sci_recov_box"] * df["Sci_obs"]/box_obs_sum.clip(lower=1),
                                df["Sci_recov_box"]/6)
    # For recovered: ACD ratio = real per-bin ratio from observed events
    # Assumption: FIFO drop is random by event-arrival, doesn't discriminate by ACD
    ratio_acd = np.where(df["Sci_obs"]>0, df["Sci_ACD_obs"]/df["Sci_obs"], 0.20)  # 0.20 fallback
    df["Sci_ACD_recov"]  = ratio_acd * df["Sci_recov"]
    df["Sci_pure_recov"] = df["Sci_recov"] - df["Sci_ACD_recov"]

    # Convert to rates
    df["sci_rate_obs"]      = df["Sci_obs"]      / df["length_s"]
    df["scipure_rate_obs"]  = df["Sci_pure_obs"] / df["length_s"]
    df["acd_rate_obs"]      = df["Sci_ACD_obs"]  / df["length_s"]
    df["sci_rate_recov"]    = df["Sci_recov"]    / df["length_s"]
    df["scipure_rate_recov"]= df["Sci_pure_recov"]/ df["length_s"]
    df["acd_rate_recov"]    = df["Sci_ACD_recov"]/ df["length_s"]
    df["wide_rate"]         = df["Wide"]  / df["length_s"]
    df["large_rate"]        = df["Large"] / df["length_s"]
    df["pho_rate"]          = df["PHO"]   / df["length_s"]
    df["t_rel"] = df["met_sec"] - TRIGGER_260

    return df


def main():
    print("Loading 2017-2019 training data...")
    train = load_training()
    print(f"  rows: {len(train):,}")

    print("\nFitting M7-merged-ACD per box (CLEAN band)...")
    fits_dict = {}
    for box in "ABC":
        mask = ((train["box"] == box)
                & (train["sci_rate"] >= SCI_LO_CLEAN) & (train["sci_rate"] < SCI_HI_CLEAN)
                & (train["group_rate"] < BOX_RATE_CAP))
        fits_dict[box] = fit_m7_merged(train[mask])
        c = fits_dict[box]
        print(f"  Box {box}: b={c[0]:+.1f}, c_pure={c[1]:.3f}, c_ACD={c[2]:.3f}, "
              f"β={c[3]:.3f}, γ={c[4]:.3f}")

    # Sci_pred for training data
    train["box_str"] = train["box"].astype(str)
    b   = train["box_str"].map(lambda x: fits_dict[x][0]).values
    c0  = train["box_str"].map(lambda x: fits_dict[x][1]).values
    cA  = train["box_str"].map(lambda x: fits_dict[x][2]).values
    bet = train["box_str"].map(lambda x: fits_dict[x][3]).values
    gam = train["box_str"].map(lambda x: fits_dict[x][4]).values
    train["sci_pred"] = ((train["pho_rate"].values
                          - (cA-c0)*train["acd_rate"].values
                          - bet*train["wide_rate"].values
                          - gam*train["large_rate"].values - b) / c0)

    print("\nLoading 260226A with decoded ACD bits...")
    grb = load_260226A_decoded()
    print(f"  GRB rows: {len(grb):,}")

    # Sci_pred for GRB
    grb["box_str"] = grb["box"].astype(str)
    gb   = grb["box_str"].map(lambda x: fits_dict[x][0]).values
    gc0  = grb["box_str"].map(lambda x: fits_dict[x][1]).values
    gcA  = grb["box_str"].map(lambda x: fits_dict[x][2]).values
    gbet = grb["box_str"].map(lambda x: fits_dict[x][3]).values
    ggam = grb["box_str"].map(lambda x: fits_dict[x][4]).values
    grb["sci_pred_obs"] = ((grb["pho_rate"].values
                             - (gcA-gc0)*grb["acd_rate_obs"].values
                             - gbet*grb["wide_rate"].values
                             - ggam*grb["large_rate"].values - gb) / gc0)
    grb["sci_pred_recov"] = ((grb["pho_rate"].values
                               - (gcA-gc0)*grb["acd_rate_recov"].values
                               - gbet*grb["wide_rate"].values
                               - ggam*grb["large_rate"].values - gb) / gc0)
    grb_with_sci = grb[grb["Sci_obs"] > 0].copy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    xb = np.logspace(np.log10(X_LO), np.log10(4500), 150)
    yb = np.logspace(np.log10(X_LO/2), np.log10(7000), 150)

    for ax, box in zip(axes, "ABC"):
        sub = train[(train["box"]==box) & (train["sci_rate"] >= X_LO)
                    & (train["sci_pred"] > 0)]
        H, xedges, yedges = np.histogram2d(sub["sci_rate"].values,
                                             sub["sci_pred"].values,
                                             bins=[xb, yb])
        ix = np.clip(np.searchsorted(xedges, sub["sci_rate"].values) - 1,
                     0, len(xedges)-2)
        iy = np.clip(np.searchsorted(yedges, sub["sci_pred"].values) - 1,
                     0, len(yedges)-2)
        density = H[ix, iy].astype(float)
        density[density < 1] = 1
        if len(sub) > N_SCATTER:
            idx = np.random.RandomState(0).choice(len(sub), N_SCATTER, replace=False)
        else:
            idx = np.arange(len(sub))
        order = np.argsort(density[idx])
        sc = ax.scatter(sub["sci_rate"].values[idx][order],
                         sub["sci_pred"].values[idx][order],
                         c=density[idx][order],
                         cmap="viridis", norm=LogNorm(vmin=1, vmax=density.max()),
                         s=2, alpha=0.6, rasterized=True, edgecolor="none")

        g_box = grb_with_sci[grb_with_sci["box"]==box]
        burst_mask = g_box["Sci_fill_box"] > 0

        ax.scatter(g_box.loc[burst_mask, "sci_rate_obs"],
                    g_box.loc[burst_mask, "sci_pred_obs"],
                    s=12, color="blue", alpha=0.7, edgecolor="black", lw=0.3,
                    label="260226A burst (Sci_obs)", zorder=5, marker="o")
        ax.scatter(g_box.loc[burst_mask, "sci_rate_recov"],
                    g_box.loc[burst_mask, "sci_pred_recov"],
                    s=12, color="red", alpha=0.7, edgecolor="black", lw=0.3,
                    label="260226A burst (Sci_recov)", zorder=6, marker="^")

        line = np.array([X_LO, 4500])
        c = fits_dict[box]
        ax.plot(line, line, "--", color="red", lw=1.5, label="y=x")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(X_LO, 4500); ax.set_ylim(X_LO/2, 7000)
        ax.set_xlabel("Sci [cnt/s/det]")
        if box == "A":
            ax.set_ylabel("Sci predicted (M7-merged-ACD)")
        ax.set_title(f"Box {box}", fontsize=12, fontweight="bold")
        coef_text = (f"c_pure = {c[1]:.2f}\n"
                     f"c_ACD = {c[2]:.2f}\n"
                     f"β = {c[3]:.2f}\n"
                     f"γ = {c[4]:.2f}\n"
                     f"b = {c[0]:+.0f}")
        ax.text(0.97, 0.03, coef_text, transform=ax.transAxes,
                fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.85))
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
        ax.grid(alpha=0.3, which="both")

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85,
                          label="2017-2019 local density (log)")
    formula = (r"$\bf{PHO = c_{pure}\cdot Sci_{pure} + c_{ACD}\cdot Sci_{ACD} "
               r"+ \beta\cdot Wide + \gamma\cdot Large + b}$")
    fig.suptitle("M7 with merged ACD channel (5 params per box) — 260226A overlaid\n"
                 + formula,
                 fontsize=14, y=1.005)
    out = OUT_DIR / "sci_pred_M7merged_with_260226A_v2.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
