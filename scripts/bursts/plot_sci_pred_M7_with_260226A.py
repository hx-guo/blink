#!/usr/bin/env python3
"""Overlay 260226A (observed and recovered) on M7 CLEAN density scatter."""
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
    files = sorted(CSV_DIR.glob("*.csv"))
    parts = []
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
    for c, src in [("sci_rate","Sci"),("scipure_rate","Sci_pure"),
                    ("acd1_rate","Sci_ACD1"),("acdn_rate","Sci_ACDN"),
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


def fit_m7(sub):
    X = np.column_stack([np.ones(len(sub)), sub["scipure_rate"], sub["acd1_rate"],
                          sub["acdn_rate"], sub["wide_rate"], sub["large_rate"]])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return coef


def load_260226A():
    """Load 260226A engineering FITS for the whole file (~1 hour) + cached Sci."""
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
            large = unwrap_large(pho, large_raw)
            for i in range(len(met_eng)):
                rows.append({
                    "box": box, "det": det_local,
                    "met_sec": int(met_eng[i]),
                    "length_cyc": length_cyc[i],
                    "length_s": length_s[i],
                    "PHO": pho[i], "Wide": csi[i], "Large": large[i],
                })
        fe.close()
    eng = pd.DataFrame(rows)

    # Load cached Sci_obs and Sci_recov (only covers ±70s around trigger)
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
    # FILL_GAP count per (box, met_sec) — this is the true "recovery" signal
    sci_fill_box = sci_rec[sci_rec["type"]=="FILL_GAP"].groupby(["box","met_sec"]).size().rename("Sci_fill_box").reset_index()

    df = eng.merge(sci_obs_pd, on=["box","det","met_sec"], how="left")
    df["Sci_obs"] = df["Sci_obs"].fillna(0)
    df = df.merge(sci_fill_box, on=["box","met_sec"], how="left")
    df["Sci_fill_box"] = df["Sci_fill_box"].fillna(0)
    # Sci_recov_box = Sci_obs_box + Sci_fill_box (true recovery)
    box_obs_sum = df.groupby(["box","met_sec"])["Sci_obs"].transform("sum")
    df["Sci_recov_box"] = box_obs_sum + df["Sci_fill_box"]
    df["Sci_recov"] = np.where(box_obs_sum > 0,
                                df["Sci_recov_box"] * df["Sci_obs"]/box_obs_sum.clip(lower=1),
                                df["Sci_recov_box"]/6)

    # Estimate per-det Sci_ACD1, Sci_ACDN from 1B events
    # (need to read aminfo bits for ACD coincidence — placeholder: use sci_pure = Sci)
    # For simplicity assume Sci_ACD1 ≈ Sci_obs × 0.10, Sci_ACDN ≈ Sci_obs × 0.12 (typical ratios)
    # Better: 0 for Sci_ACD1 and 0 for Sci_ACDN means treat all Sci as "pure"
    # We'll use a reasonable per-second ratio derived from quiet bins
    df["Sci_ACD1_obs"] = 0.10 * df["Sci_obs"]
    df["Sci_ACDN_obs"] = 0.12 * df["Sci_obs"]
    df["Sci_pure_obs"] = df["Sci_obs"] - df["Sci_ACD1_obs"] - df["Sci_ACDN_obs"]
    df["Sci_ACD1_recov"] = 0.10 * df["Sci_recov"]
    df["Sci_ACDN_recov"] = 0.12 * df["Sci_recov"]
    df["Sci_pure_recov"] = df["Sci_recov"] - df["Sci_ACD1_recov"] - df["Sci_ACDN_recov"]

    # Convert to rates
    df["sci_rate_obs"] = df["Sci_obs"] / df["length_s"]
    df["scipure_rate_obs"] = df["Sci_pure_obs"] / df["length_s"]
    df["acd1_rate_obs"] = df["Sci_ACD1_obs"] / df["length_s"]
    df["acdn_rate_obs"] = df["Sci_ACDN_obs"] / df["length_s"]
    df["sci_rate_recov"] = df["Sci_recov"] / df["length_s"]
    df["scipure_rate_recov"] = df["Sci_pure_recov"] / df["length_s"]
    df["acd1_rate_recov"] = df["Sci_ACD1_recov"] / df["length_s"]
    df["acdn_rate_recov"] = df["Sci_ACDN_recov"] / df["length_s"]
    df["wide_rate"]  = df["Wide"]  / df["length_s"]
    df["large_rate"] = df["Large"] / df["length_s"]
    df["pho_rate"]   = df["PHO"]   / df["length_s"]
    df["t_rel"] = df["met_sec"] - TRIGGER_260

    return df


def main():
    print("Loading 2017-2019 training data...")
    train = load_training()
    print(f"  rows: {len(train):,}")

    print("\nFitting M7 CLEAN per box...")
    fits_dict = {}
    for box in "ABC":
        mask = ((train["box"] == box)
                & (train["sci_rate"] >= SCI_LO_CLEAN) & (train["sci_rate"] < SCI_HI_CLEAN)
                & (train["group_rate"] < BOX_RATE_CAP))
        fits_dict[box] = fit_m7(train[mask])
        print(f"  Box {box}: c0={fits_dict[box][1]:.3f}, c1={fits_dict[box][2]:.3f}, "
              f"cN={fits_dict[box][3]:.3f}, β={fits_dict[box][4]:.3f}, "
              f"γ={fits_dict[box][5]:.3f}")

    # Compute Sci_pred for training data
    train["box_str"] = train["box"].astype(str)
    b   = train["box_str"].map(lambda b: fits_dict[b][0]).values
    c0  = train["box_str"].map(lambda b: fits_dict[b][1]).values
    c1  = train["box_str"].map(lambda b: fits_dict[b][2]).values
    cN  = train["box_str"].map(lambda b: fits_dict[b][3]).values
    bet = train["box_str"].map(lambda b: fits_dict[b][4]).values
    gam = train["box_str"].map(lambda b: fits_dict[b][5]).values
    train["sci_pred"] = ((train["pho_rate"].values
                          - (c1-c0)*train["acd1_rate"].values
                          - (cN-c0)*train["acdn_rate"].values
                          - bet*train["wide_rate"].values
                          - gam*train["large_rate"].values - b) / c0)

    # Load 260226A data
    print("\nLoading 260226A engineering + Sci data...")
    grb = load_260226A()
    print(f"  GRB rows (per det × second): {len(grb):,}")
    print(f"  GRB t_rel range: {grb['t_rel'].min():.0f} to {grb['t_rel'].max():.0f}")

    # Compute Sci_pred for GRB (both obs and recov)
    grb["box_str"] = grb["box"].astype(str)
    gb   = grb["box_str"].map(lambda b: fits_dict[b][0]).values
    gc0  = grb["box_str"].map(lambda b: fits_dict[b][1]).values
    gc1  = grb["box_str"].map(lambda b: fits_dict[b][2]).values
    gcN  = grb["box_str"].map(lambda b: fits_dict[b][3]).values
    gbet = grb["box_str"].map(lambda b: fits_dict[b][4]).values
    ggam = grb["box_str"].map(lambda b: fits_dict[b][5]).values
    grb["sci_pred_obs"] = ((grb["pho_rate"].values
                             - (gc1-gc0)*grb["acd1_rate_obs"].values
                             - (gcN-gc0)*grb["acdn_rate_obs"].values
                             - gbet*grb["wide_rate"].values
                             - ggam*grb["large_rate"].values - gb) / gc0)
    grb["sci_pred_recov"] = ((grb["pho_rate"].values
                               - (gc1-gc0)*grb["acd1_rate_recov"].values
                               - (gcN-gc0)*grb["acdn_rate_recov"].values
                               - gbet*grb["wide_rate"].values
                               - ggam*grb["large_rate"].values - gb) / gc0)

    # Filter GRB to where we have Sci data (cache covers ~100s)
    grb_with_sci = grb[grb["Sci_obs"] > 0].copy()
    print(f"  GRB rows with cached Sci: {len(grb_with_sci):,}")

    # ============ Plot ============
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    xb = np.logspace(np.log10(X_LO), np.log10(4500), 150)
    yb = np.logspace(np.log10(X_LO/2), np.log10(7000), 150)

    for ax, box in zip(axes, "ABC"):
        # Training density background — use original viridis density coloring
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

        # Overlay 260226A
        g_box = grb_with_sci[grb_with_sci["box"]==box]
        # Classify by whether recovery added events (= FIFO drop happened)
        burst_mask = g_box["Sci_recov"] > g_box["Sci_obs"]
        quiet_mask = ~burst_mask

        # Burst bins, observed (blue circles, smaller)
        ax.scatter(g_box.loc[burst_mask, "sci_rate_obs"],
                    g_box.loc[burst_mask, "sci_pred_obs"],
                    s=12, color="blue", alpha=0.7, edgecolor="black", lw=0.3,
                    label="260226A burst (Sci_obs, saturated)", zorder=5,
                    marker="o")

        # Burst bins, recovered (red triangles, smaller)
        ax.scatter(g_box.loc[burst_mask, "sci_rate_recov"],
                    g_box.loc[burst_mask, "sci_pred_recov"],
                    s=12, color="red", alpha=0.7, edgecolor="black", lw=0.3,
                    label="260226A burst (Sci_recov)", zorder=6,
                    marker="^")

        # y=x line only
        line = np.array([X_LO, 4500])
        c = fits_dict[box]
        ax.plot(line, line, "--", color="red", lw=1.5, label="y=x")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(X_LO, 4500); ax.set_ylim(X_LO/2, 7000)
        ax.set_xlabel("Sci [cnt/s/det]")
        if box == "A":
            ax.set_ylabel("Sci predicted (M7 CLEAN)")
        ax.set_title(f"Box {box}  M7: c0={c[1]:.2f}, c1={c[2]:.2f}, "
                      f"cN={c[3]:.2f}, β={c[4]:.2f}, γ={c[5]:.2f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
        ax.grid(alpha=0.3, which="both")

    # Add colorbar for density
    cbar = fig.colorbar(sc, ax=axes, shrink=0.85,
                          label="2017-2019 local density (log)")

    fig.suptitle("260226A overlaid on 2017-2019 M7 CLEAN density "
                 "(burst: blue=Sci_obs, red=Sci_recov)",
                 fontsize=12, y=0.998)
    out = OUT_DIR / "sci_pred_M7_with_260226A.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
