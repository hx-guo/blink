#!/usr/bin/env python3
"""M1 + post-hoc cross-detector dead-time correction.

Pipeline:
  1. Fit M1 on main band (Sci > 300) per Box → (β, γ, α, b)
  2. Compute Sci_pred_M1 for all rows
  3. residual_M1 = Sci_pred_M1 - Sci_obs
  4. Fit δ on the Sci > 1500 subset only: residual_M1 = δ · Sci_others
     (no intercept — must vanish at Sci_others = 0)
  5. Apply correction: Sci_pred_final = Sci_pred_M1 - δ · Sci_others
     (subtract because residual = pred - obs, we want to push pred down where
      δ·Sci_others is positive at high rates: δ should be negative)

Final plots:
  - Sci_pred_final vs Sci_obs (density-colored scatter, 3 Boxes)
  - Residual binned median before vs after correction
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
MAIN_BAND_LO = 300.0
XDET_FIT_LO = 1500.0


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs...")
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
    df["sci_rate"]   = df["Sci"]   / df["length"]
    df["wide_rate"]  = df["Wide"]  / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"]   = df["PHO"]   / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

    print("Computing Sci_others...")
    tot = df.groupby(["date","box","met_sec"], observed=True)["sci_rate"].sum()
    tot.name = "sci_tot_rate"
    df = df.merge(tot, on=["date","box","met_sec"])
    df["sci_others_rate"] = df["sci_tot_rate"] - df["sci_rate"]
    df = df.drop(columns="sci_tot_rate")

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
    print(f"normal-mode rows: {len(df):,}")
    return df


def main():
    df = load()

    print(f"\n=== Per-Box M1 fit on Sci > {MAIN_BAND_LO} ===")
    box_fits = {}
    for box in "ABC":
        sub = df[(df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)]
        X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                             sub["wide_rate"].values, sub["large_rate"].values])
        coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
        b, one_plus_a, beta, gamma = coef
        box_fits[box] = dict(b=b, alpha=one_plus_a - 1, beta=beta, gamma=gamma)
        print(f"  Box {box}: b={b:.1f}, α={one_plus_a-1:.3f}, β={beta:.3f}, γ={gamma:.4f}")

    # Compute Sci_pred_M1 for all rows
    df["sci_pred_m1"] = np.nan
    for box in "ABC":
        p = box_fits[box]
        mask = df["box"] == box
        pho_corr = df.loc[mask, "pho_rate"].values \
                    - p["beta"]*df.loc[mask, "wide_rate"].values \
                    - p["gamma"]*df.loc[mask, "large_rate"].values
        df.loc[mask, "sci_pred_m1"] = (pho_corr - p["b"]) / (1 + p["alpha"])
    df["resid_m1"] = df["sci_pred_m1"] - df["sci_rate"]

    print(f"\n=== Cross-detector δ fit on Sci > {XDET_FIT_LO} (no intercept) ===")
    box_delta = {}
    for box in "ABC":
        sub = df[(df["box"] == box) & (df["sci_rate"] > XDET_FIT_LO)]
        # OLS: resid_m1 = δ · Sci_others, no intercept
        x = sub["sci_others_rate"].values
        y = sub["resid_m1"].values
        delta = float((x * y).sum() / (x * x).sum())
        box_delta[box] = delta
        print(f"  Box {box}: δ = {delta:+.6f} cnt/s/det per (cnt/s/det × 5 dets sum)  "
              f"(N={len(sub):,})")

    # Apply correction: Sci_pred_final = Sci_pred_M1 - δ · Sci_others
    df["sci_pred_final"] = np.nan
    for box in "ABC":
        mask = df["box"] == box
        df.loc[mask, "sci_pred_final"] = (df.loc[mask, "sci_pred_m1"]
                                          - box_delta[box]
                                          * df.loc[mask, "sci_others_rate"])
    df["resid_final"] = df["sci_pred_final"] - df["sci_rate"]

    print(f"\n=== RMS comparison ===")
    print(f"{'Box':>4s} {'RMS_M1':>9s} {'RMS_M1_main':>13s} {'RMS_final':>10s} {'RMS_final_main':>16s}")
    for box in "ABC":
        sub = df[df["box"] == box]
        rms_m1 = float(np.sqrt(np.mean(sub["resid_m1"].values ** 2)))
        sub_main = sub[sub["sci_rate"] > MAIN_BAND_LO]
        rms_m1_m = float(np.sqrt(np.mean(sub_main["resid_m1"].values ** 2)))
        rms_fin = float(np.sqrt(np.mean(sub["resid_final"].values ** 2)))
        rms_fin_m = float(np.sqrt(np.mean(sub_main["resid_final"].values ** 2)))
        print(f"  {box}    {rms_m1:>9.1f} {rms_m1_m:>13.1f} {rms_fin:>10.1f} {rms_fin_m:>16.1f}")

    # === Plot 1: Sci_pred_final vs Sci_obs, density-colored scatter ===
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 17), sharex=True, sharey=True)
    SCI_MIN, SCI_MAX = 40.0, 5000.0
    Y_MIN, Y_MAX = 1.0, 5000.0
    last_hb = None
    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"] == box]
        sci = sub["sci_rate"].values
        sp = sub["sci_pred_final"].values
        sp_pos = np.maximum(sp, Y_MIN * 0.5)
        keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
        x = sci[keep]; y = sp_pos[keep]
        nbins = 200
        x_edges = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), nbins+1)
        y_edges = np.logspace(np.log10(Y_MIN), np.log10(Y_MAX), nbins+1)
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        ix = np.clip(np.searchsorted(x_edges, x) - 1, 0, nbins-1)
        iy = np.clip(np.searchsorted(y_edges, y) - 1, 0, nbins-1)
        density = H[ix, iy]
        order = np.argsort(density)
        hb = ax.scatter(x[order], y[order], c=density[order], s=1.5,
                        cmap="viridis", norm=LogNorm(vmin=1, vmax=density.max()),
                        rasterized=True, linewidths=0)
        last_hb = hb
        ax.set_xscale("log"); ax.set_yscale("log")

        # Binned median (≥500 pts) for main band
        bins_e = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 50)
        bc = 0.5 * (bins_e[:-1] + bins_e[1:])
        med = []
        for i in range(len(bins_e) - 1):
            m = (sci >= bins_e[i]) & (sci < bins_e[i+1])
            med.append(np.median(sp[m]) if m.sum() > 500 else np.nan)
        ax.plot(bc, np.array(med), "-", color="orange", lw=2.2, zorder=5,
                label="binned median (≥500 pts)")
        ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "r--", lw=1.8, zorder=6,
                label=f"y = x  (δ={box_delta[box]:+.5f})")

        ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlabel("Sci observed [cnt/s/det]")
        ax.set_ylabel("Sci predicted [cnt/s/det]")
        rms_box = float(np.sqrt(np.mean((sub["resid_final"].values) ** 2)))
        ax.set_title(f"Box {box}  (N={len(sub):,})  RMS = {rms_box:.0f}")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
        ax.grid(alpha=0.3, which="both")

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.08, 0.02, 0.84])
    cb = fig.colorbar(last_hb, cax=cax)
    cb.set_label("per-det-sec bin count (log scale)")

    fig.suptitle("M1 + cross-detector dead-time correction: "
                 r"Sci$_{\rm pred}$ = Sci$_{\rm M1}$ − δ·Sci$_{\rm others}$",
                 fontsize=12, y=0.995)
    out1 = OUT_DIR / "n_below_m1_xdet.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out1}")

    # === Plot 2: Residual binned median before vs after correction ===
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    SCI_MIN_R, SCI_MAX_R = MAIN_BAND_LO, 4000.0
    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"] == box]
        sci = sub["sci_rate"].values
        for col, label, color in [("resid_m1", "M1 (no correction)", "C0"),
                                   ("resid_final", "M1 + δ·Sci_others", "C3")]:
            resid = sub[col].values
            bins = np.logspace(np.log10(SCI_MIN_R), np.log10(SCI_MAX_R), 40)
            bc = 0.5 * (bins[:-1] + bins[1:])
            med = []
            for i in range(len(bins) - 1):
                m = (sci >= bins[i]) & (sci < bins[i+1])
                med.append(np.median(resid[m]) if m.sum() > 500 else np.nan)
            ax.plot(bc, np.array(med), "o-", color=color, lw=2, ms=4, label=label)
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.axvline(XDET_FIT_LO, color="gray", ls=":", lw=1,
                   label=f"δ fit cut (Sci > {XDET_FIT_LO:.0f})")
        ax.set_xscale("log")
        ax.set_xlim(SCI_MIN_R, SCI_MAX_R)
        ax.set_xlabel("Sci obs [cnt/s/det]")
        ax.set_ylabel("residual = Sci_pred - Sci [cnt/s/det]")
        ax.set_title(f"Box {box}  δ = {box_delta[box]:+.5f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Residual binned-median: M1 vs M1 + cross-detector correction",
                 fontsize=12)
    fig.tight_layout()
    out2 = OUT_DIR / "n_below_m1_xdet_residual.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
