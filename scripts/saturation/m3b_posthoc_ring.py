#!/usr/bin/env python3
"""M3b: Post-hoc ring correction (M3 with multicollinearity defused).

Step 1: Fit M1 per Box (4 coefs: b, α, β, γ) — same as baseline.
Step 2: On the M1 residual, fit a *single* coefficient δ_ring per (Box × Ring)
        on Sci_others.  Because we fit on the residual (not PHO directly),
        Sci has already been absorbed; Sci_others can be a clean regressor.

  resid_m1 ≈ δ_ring × (Sci_others − ⟨Sci_others⟩)        for each (box, ring)

  sci_pred_m3b = sci_pred_m1 + δ_ring × Sci_others_residual

If δ_inner ≈ 0 and δ_outer < 0 (each box), and the inner-outer gap at
high Sci collapses → cross-detector geometric coupling is confirmed
as the source of the 150 cnt/s/det residual split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE = Path("n_below_study/hv_table_partial.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
MAIN_BAND_LO = 300.0
HIGH_SCI = 1500.0


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
    df["ring"] = np.where(df["det"] < 2, "in", "out")

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


def fit_m1(sub):
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    b, one_plus_a, beta, gamma = coef
    return b, one_plus_a - 1, beta, gamma


def main():
    df = load()

    # ============ M1 baseline ============
    print(f"\n=== M1 per-Box fit ===")
    print(f"{'Box':>4s}  {'b':>8s} {'α':>8s} {'β':>8s} {'γ':>8s}")
    df["sci_pred_m1"] = np.nan
    df["resid_m1"] = np.nan
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        b, a, beta, gamma = fit_m1(df[mask_fit])
        print(f"  {box}    {b:>8.1f} {a:>+8.4f} {beta:>+8.4f} {gamma:>+8.4f}")
        mask_apply = (df["box"] == box)
        sub = df[mask_apply]
        pho_corr = sub["pho_rate"].values - beta*sub["wide_rate"].values - gamma*sub["large_rate"].values
        sci_pred = (pho_corr - b) / (1 + a)
        df.loc[mask_apply, "sci_pred_m1"] = sci_pred
        df.loc[mask_apply, "resid_m1"] = sci_pred - sub["sci_rate"].values

    # ============ M3b: post-hoc fit δ_ring on M1 residual vs Sci_others ============
    # fit only on Sci > MAIN_BAND_LO data (same domain as M1 fit)
    print(f"\n=== M3b: post-hoc δ_ring fit on resid_m1 vs Sci_others ===")
    print(f"{'Box':>4s} {'Ring':>5s}  {'N_fit':>10s}  {'⟨resid⟩':>10s}  {'⟨S_oth⟩':>10s}  {'δ':>10s}  {'δ²·var':>10s}")
    deltas = {}
    df["resid_m3b"] = df["resid_m1"]
    for box in "ABC":
        for ring in ("in", "out"):
            mask_fit = ((df["box"] == box) & (df["ring"] == ring)
                        & (df["sci_rate"] > MAIN_BAND_LO))
            sub = df[mask_fit]
            # Fit resid = δ·(Sci_others - mean) + const ⇒ slope = cov/var
            x = sub["sci_others_rate"].values
            y = sub["resid_m1"].values
            x_mean, y_mean = x.mean(), y.mean()
            dx, dy = x - x_mean, y - y_mean
            delta = float((dx * dy).sum() / (dx * dx).sum())
            deltas[(box, ring)] = (delta, x_mean)
            var_explained = delta**2 * dx.var()
            print(f"  {box}    {ring:>5s}  {mask_fit.sum():>10,d}  "
                  f"{y_mean:>+10.2f}  {x_mean:>+10.1f}  {delta:>+10.5f}  {var_explained:>10.1f}")
            # Apply globally to that (box, ring): subtract δ·(Sci_others - x_mean)
            mask_apply = (df["box"] == box) & (df["ring"] == ring)
            x_all = df.loc[mask_apply, "sci_others_rate"].values
            df.loc[mask_apply, "resid_m3b"] = (
                df.loc[mask_apply, "resid_m1"].values - delta * (x_all - x_mean)
            )

    # ============ Per-det median residual: M1 vs M3b ============
    print(f"\n=== Per-det median residual at Sci > {HIGH_SCI} ===")
    print(f"{'det':>4s} {'ring':>5s}  {'N':>8s}  {'M1_med':>8s}  {'M3b_med':>8s}  Δ")
    hi = df[df["sci_rate"] > HIGH_SCI]
    for d in range(18):
        sub = hi[hi["det_global"] == d]
        if len(sub) == 0:
            continue
        ring = "in" if (d % 6) < 2 else "out"
        m1 = sub["resid_m1"].median()
        m3 = sub["resid_m3b"].median()
        print(f"  {d:>2d}  {ring:>5s}  {len(sub):>8,d}  {m1:>+8.1f}  {m3:>+8.1f}  {m1-m3:>+8.1f}")

    # ============ RMS comparison ============
    print(f"\n=== RMS comparison ===")
    print(f"{'Box':>4s}  {'Sci range':>15s}  {'M1 RMS':>10s}  {'M3b RMS':>10s}  Δ%")
    for box in "ABC":
        for label, mask_extra in (("all Sci>300", df["sci_rate"] > 300),
                                  ("Sci>1500",    df["sci_rate"] > HIGH_SCI)):
            sub = df[(df["box"] == box) & mask_extra]
            r1 = np.sqrt(np.mean(sub["resid_m1"]**2))
            r3 = np.sqrt(np.mean(sub["resid_m3b"]**2))
            print(f"  {box}    {label:>15s}  {r1:>10.1f}  {r3:>10.1f}  {(r3-r1)/r1*100:>+6.1f}%")

    print(f"\n=== Inner-Outer gap at Sci > {HIGH_SCI} ===")
    print(f"{'Box':>4s}  {'M1 in':>10s}  {'M1 out':>10s}  {'M1 gap':>8s} | "
          f"{'M3b in':>10s}  {'M3b out':>10s}  {'M3b gap':>8s}")
    for box in "ABC":
        sub_hi = df[(df["box"] == box) & (df["sci_rate"] > HIGH_SCI)]
        m1_in = sub_hi.loc[sub_hi["ring"]=="in", "resid_m1"].median()
        m1_out = sub_hi.loc[sub_hi["ring"]=="out", "resid_m1"].median()
        m3_in = sub_hi.loc[sub_hi["ring"]=="in", "resid_m3b"].median()
        m3_out = sub_hi.loc[sub_hi["ring"]=="out", "resid_m3b"].median()
        print(f"  {box}    {m1_in:>+10.1f}  {m1_out:>+10.1f}  {m1_in-m1_out:>+8.1f} | "
              f"{m3_in:>+10.1f}  {m3_out:>+10.1f}  {m3_in-m3_out:>+8.1f}")

    # ============ Plot: residual vs Sci, M1 vs M3b, by ring ============
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, sharey=True)
    SCI_MIN, SCI_MAX = MAIN_BAND_LO, 4000.0
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])
    for row_i, box in enumerate("ABC"):
        for col_i, model_name in enumerate(("M1", "M3b")):
            ax = axes[row_i, col_i]
            col_resid = "resid_m1" if model_name == "M1" else "resid_m3b"
            sub_box = df[df["box"] == box]
            for ring, color in (("in", "C2"), ("out", "C3")):
                sub = sub_box[sub_box["ring"] == ring]
                sci = sub["sci_rate"].values
                resid = sub[col_resid].values
                med = []
                for i in range(len(bins) - 1):
                    m = (sci >= bins[i]) & (sci < bins[i+1])
                    med.append(np.median(resid[m]) if m.sum() > 200 else np.nan)
                ax.plot(bc, np.array(med), "o-", color=color, lw=2, ms=4,
                        label=f"{ring}er (n={len(sub):,})")
            ax.axhline(0, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlim(SCI_MIN, SCI_MAX)
            ax.set_ylim(-400, 200)
            if row_i == 0:
                ax.set_title(f"{model_name}", fontsize=12)
            if col_i == 0:
                ax.set_ylabel(f"Box {box}\nresid [cnt/s/det]")
            if row_i == 2:
                ax.set_xlabel("Sci [cnt/s/det]")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, which="both")
    fig.suptitle("M1 (Box only) vs M3b (Box + post-hoc δ_ring·Sci_others)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "m3b_resid_vs_sci.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Per-det histogram
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=True, sharey=True)
    boxes = ["A"]*6 + ["B"]*6 + ["C"]*6
    for d in range(18):
        ax = axes[d // 6, d % 6]
        sub = hi[hi["det_global"] == d]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes, ha="center")
            continue
        ring = "in" if (d % 6) < 2 else "out"
        ax.hist(sub["resid_m1"], bins=80, alpha=0.45, color="C0",
                range=(-1500, 500), label=f"M1 med={sub['resid_m1'].median():+.0f}")
        ax.hist(sub["resid_m3b"], bins=80, alpha=0.45, color="C3",
                range=(-1500, 500), label=f"M3b med={sub['resid_m3b'].median():+.0f}")
        ax.axvline(0, color="k", ls="--", lw=1)
        ring_label = "INNER" if ring == "in" else "outer"
        ax.set_title(f"Box {boxes[d]} det {d%6} [{ring_label}]\nN={len(sub):,}",
                     fontsize=9, fontweight="bold" if ring == "in" else "normal")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-1500, 500)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Per-det M1 (blue) vs M3b (red) at Sci > {HIGH_SCI}", fontsize=12)
    fig.tight_layout()
    out2 = OUT_DIR / "m3b_per_det_residual_hi.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
