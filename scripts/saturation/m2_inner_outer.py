#!/usr/bin/env python3
"""M2 model: split per-Box fit into Inner vs Outer ring sub-groups.

Hypothesis from astro_sift detector.py layout:
  inner ring (6 dets):  global = {0,1, 6,7, 12,13}   (each box's det 0,1)
  outer ring (12 dets): global = {2-5, 8-11, 14-17}  (each box's det 2-5)

M1: fit one (b, α, β, γ) per Box (6 dets pooled)
    → residual splits: inner ≈ 0, outer ≈ −150 at high Sci

M2: fit one (b, α, β, γ) per (Box × Ring) — 6 groups total
    Test: does the inner/outer residual split vanish?

If yes → 100% confirmed cross-detector geometry / lateral HVT background
        is the real cause of "two parallel bands" at high Sci.
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
    # Ring: inner = each box's det 0,1   outer = each box's det 2-5
    df["ring"] = np.where(df["det"] < 2, "in", "out")

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
    print(f"  inner: {(df['ring']=='in').sum():,}   outer: {(df['ring']=='out').sum():,}")
    return df


def fit_group(sub):
    """Fit PHO_rate = b + (1+α)Sci + β Wide + γ Large; return b,α,β,γ."""
    X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                         sub["wide_rate"].values, sub["large_rate"].values])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    b, one_plus_a, beta, gamma = coef
    return b, one_plus_a - 1, beta, gamma


def apply_fit(sub, b, alpha, beta, gamma):
    pho_corr = sub["pho_rate"].values - beta*sub["wide_rate"].values - gamma*sub["large_rate"].values
    sci_pred = (pho_corr - b) / (1 + alpha)
    return sci_pred - sub["sci_rate"].values  # residual


def main():
    df = load()

    # ============ M1: fit per Box (pool all 6 dets) ============
    print(f"\n=== M1: per-Box fit (pool 6 dets) on Sci>{MAIN_BAND_LO} ===")
    print(f"{'Box':>4s}  {'b':>8s} {'α':>8s} {'β':>8s} {'γ':>8s}")
    resid_m1 = np.full(len(df), np.nan)
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        b, a, beta, gamma = fit_group(df[mask_fit])
        print(f"  {box}    {b:>8.1f} {a:>+8.4f} {beta:>+8.4f} {gamma:>+8.4f}")
        mask_apply = df["box"] == box
        resid_m1[mask_apply.values] = apply_fit(df[mask_apply], b, a, beta, gamma)
    df["resid_m1"] = resid_m1

    # ============ M2: fit per (Box × Ring) ============
    print(f"\n=== M2: per-(Box × Ring) fit on Sci>{MAIN_BAND_LO} ===")
    print(f"{'Box':>4s} {'Ring':>5s}  {'N_fit':>10s}  {'b':>8s} {'α':>8s} {'β':>8s} {'γ':>8s}")
    resid_m2 = np.full(len(df), np.nan)
    m2_coefs = {}
    for box in "ABC":
        for ring in ("in", "out"):
            mask_fit = (df["box"] == box) & (df["ring"] == ring) & (df["sci_rate"] > MAIN_BAND_LO)
            n_fit = mask_fit.sum()
            b, a, beta, gamma = fit_group(df[mask_fit])
            m2_coefs[(box, ring)] = (b, a, beta, gamma)
            print(f"  {box}    {ring:>5s}  {n_fit:>10,d}  {b:>8.1f} {a:>+8.4f} {beta:>+8.4f} {gamma:>+8.4f}")
            mask_apply = (df["box"] == box) & (df["ring"] == ring)
            resid_m2[mask_apply.values] = apply_fit(df[mask_apply], b, a, beta, gamma)
    df["resid_m2"] = resid_m2

    # ============ Per-det residual table at high Sci ============
    print(f"\n=== Per-det median residual at Sci > {HIGH_SCI} ===")
    print(f"{'det':>4s} {'ring':>5s}  {'N':>8s}  {'M1_med':>8s}  {'M2_med':>8s}  Δ")
    hi = df[df["sci_rate"] > HIGH_SCI]
    for d in range(18):
        sub = hi[hi["det_global"] == d]
        if len(sub) == 0:
            continue
        ring = "in" if (d % 6) < 2 else "out"
        m1 = sub["resid_m1"].median()
        m2 = sub["resid_m2"].median()
        print(f"  {d:>2d}  {ring:>5s}  {len(sub):>8,d}  {m1:>+8.1f}  {m2:>+8.1f}  {m1-m2:>+8.1f}")

    # ============ RMS comparison ============
    print(f"\n=== RMS comparison on Sci>{HIGH_SCI} (per-Box) ===")
    print(f"{'Box':>4s}  {'M1 RMS':>10s}  {'M2 RMS':>10s}  Δ%")
    for box in "ABC":
        sub_hi = df[(df["box"] == box) & (df["sci_rate"] > HIGH_SCI)]
        r1 = np.sqrt(np.mean(sub_hi["resid_m1"]**2))
        r2 = np.sqrt(np.mean(sub_hi["resid_m2"]**2))
        print(f"  {box}    {r1:>10.1f}  {r2:>10.1f}  {(r2-r1)/r1*100:>+6.1f}%")

    # ============ Plot 1: per-det residual histograms M1 vs M2 ============
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
        ax.hist(sub["resid_m2"], bins=80, alpha=0.45, color="C3",
                range=(-1500, 500), label=f"M2 med={sub['resid_m2'].median():+.0f}")
        ax.axvline(0, color="k", ls="--", lw=1)
        ring_label = "INNER" if ring == "in" else "outer"
        ax.set_title(f"Box {boxes[d]} det {d%6} (global {d}) [{ring_label}]\nN={len(sub):,}",
                     fontsize=9, fontweight="bold" if ring == "in" else "normal")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-1500, 500)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Per-detector M1 (blue) vs M2 (red) residual at Sci > {HIGH_SCI}",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m2_per_det_residual_hi.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # ============ Plot 2: residual vs Sci, side-by-side M1/M2 ============
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, sharey=True)
    SCI_MIN, SCI_MAX = MAIN_BAND_LO, 4000.0
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])
    for row_i, box in enumerate("ABC"):
        for col_i, model_name in enumerate(("M1", "M2")):
            ax = axes[row_i, col_i]
            col_resid = "resid_m1" if model_name == "M1" else "resid_m2"
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
    fig.suptitle("Residual vs Sci, by inner/outer ring", fontsize=13, y=0.995)
    fig.tight_layout()
    out2 = OUT_DIR / "m2_resid_vs_sci.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
