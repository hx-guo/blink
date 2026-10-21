#!/usr/bin/env python3
"""M5: Test if dead-time correction removes the S-curve residual.

Background:
  Dt is reported per-second deadtime (units unclear: dead cycles or dead events).
  Currently rates are computed as count / (L_cycles × 16μs). This treats
  L_cycles as the active time, ignoring dead time. If active time is actually
  L_cycles × (1 − dt_frac), all rates are underestimated by factor (1 − dt_frac)⁻¹.

Test scenarios:
  M1   : baseline (no dt correction)
  M5a  : all rates corrected by 1/(1 − dt_frac), with dt_frac = Dt/L_cycles
  M5b  : just rescale, check residual SHAPE
  diag : print dt_frac vs Sci medians; do they explain the 25% normalized
         residual at Sci=3000?
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


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32"}
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

    df["dt_frac"] = df["Dt"] / df["L_cycles"].clip(lower=1)

    # Raw rates (M1 baseline) — no dt correction
    df["sci_rate"]   = df["Sci"]   / df["length"]
    df["wide_rate"]  = df["Wide"]  / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"]   = df["PHO"]   / df["length"]

    # Dead-time-corrected rates (active length = length × (1 - dt_frac))
    live_frac = 1.0 - df["dt_frac"]
    live_frac = live_frac.clip(lower=0.05)  # protect against pathological
    df["sci_rate_dt"]   = df["sci_rate"]   / live_frac
    df["wide_rate_dt"]  = df["wide_rate"]  / live_frac
    df["large_rate_dt"] = df["large_rate"] / live_frac
    df["pho_rate_dt"]   = df["pho_rate"]   / live_frac

    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    df["ring"] = np.where(df["det"] < 2, "in", "out")

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


def fit_m1(sub, suffix=""):
    sci  = sub[f"sci_rate{suffix}"].values
    wide = sub[f"wide_rate{suffix}"].values
    large= sub[f"large_rate{suffix}"].values
    pho  = sub[f"pho_rate{suffix}"].values
    X = np.column_stack([np.ones(len(sub)), sci, wide, large])
    coef, *_ = np.linalg.lstsq(X, pho, rcond=None)
    b, ap1, beta, gamma = coef
    return b, ap1 - 1, beta, gamma


def apply_m1(sub, b, alpha, beta, gamma, suffix=""):
    sci  = sub[f"sci_rate{suffix}"].values
    wide = sub[f"wide_rate{suffix}"].values
    large= sub[f"large_rate{suffix}"].values
    pho  = sub[f"pho_rate{suffix}"].values
    pho_corr = pho - beta*wide - gamma*large
    sci_pred = (pho_corr - b) / (1 + alpha)
    return sci_pred - sci


def median_per_bin(sci, y, bins, min_count=200):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (sci >= bins[i]) & (sci < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    # ============ Step 0: dt_frac vs Sci profile ============
    print(f"\n=== dt_frac vs Sci profile ===")
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'median dt_frac':>15s}  "
          f"{'expected loss':>15s}")
    sci_bin_edges = np.array([50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000, 4000, 6000])
    for i in range(len(sci_bin_edges) - 1):
        lo, hi = sci_bin_edges[i], sci_bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        dt_med = df.loc[mask, "dt_frac"].median()
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  {dt_med:>15.4f}  "
              f"{dt_med*100:>14.1f}%")

    # ============ M1 baseline ============
    print(f"\n=== M1 baseline (raw rates, no dt correction) ===")
    df["resid_m1"] = np.nan
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        b, a, beta, gamma = fit_m1(df[mask_fit])
        print(f"  Box {box}: b={b:+8.1f}, α={a:+.4f}, β={beta:+.4f}, γ={gamma:+.4f}")
        mask_apply = df["box"] == box
        df.loc[mask_apply, "resid_m1"] = apply_m1(df[mask_apply], b, a, beta, gamma)

    # ============ M5: dead-time corrected ============
    print(f"\n=== M5: dt-corrected rates ===")
    df["resid_m5"] = np.nan
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate_dt"] > MAIN_BAND_LO)
        b, a, beta, gamma = fit_m1(df[mask_fit], suffix="_dt")
        print(f"  Box {box}: b={b:+8.1f}, α={a:+.4f}, β={beta:+.4f}, γ={gamma:+.4f}")
        mask_apply = df["box"] == box
        df.loc[mask_apply, "resid_m5"] = apply_m1(df[mask_apply], b, a, beta, gamma, suffix="_dt")

    # ============ RMS comparison by Sci bin ============
    print(f"\n=== RMS by Sci bin ===")
    print(f"{'Sci bin':>15s}  {'N':>10s}  {'M1 RMS':>10s}  {'M5 RMS':>10s}  Δ%")
    sci_bin_edges2 = np.array([300, 600, 1000, 1500, 2000, 2500, 3000, 4500])
    for i in range(len(sci_bin_edges2) - 1):
        lo, hi = sci_bin_edges2[i], sci_bin_edges2[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        r1 = np.sqrt(np.mean(df.loc[mask, "resid_m1"]**2))
        r5 = np.sqrt(np.mean(df.loc[mask, "resid_m5"]**2))
        delta_pct = (r5 - r1) / r1 * 100 if r1 > 0 else 0.0
        print(f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  {r1:>10.1f}  {r5:>10.1f}  {delta_pct:>+6.1f}%")

    # ============ Plot: residual vs Sci (M1 raw vs M5 dt-corrected) ============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    SCI_MIN, SCI_MAX = MAIN_BAND_LO, 4500.0
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])
    bins2 = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc2 = 0.5 * (bins2[:-1] + bins2[1:])

    # (0,0) M1 residual vs raw Sci
    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = df[df["box"] == box]
        med = median_per_bin(sub["sci_rate"].values, sub["resid_m1"].values, bins)
        axes[0,0].plot(bc, med, "-", color=color, lw=2, label=f"Box {box}")
    axes[0,0].axhline(0, color="k", ls=":", lw=1)
    axes[0,0].set_xscale("log")
    axes[0,0].set_xlim(SCI_MIN, SCI_MAX)
    axes[0,0].set_title("M1 residual vs raw Sci")
    axes[0,0].set_ylabel("resid [cnt/s/det]")
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3, which="both")

    # (0,1) M5 residual vs dt-corrected Sci
    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = df[df["box"] == box]
        med = median_per_bin(sub["sci_rate_dt"].values, sub["resid_m5"].values, bins2)
        axes[0,1].plot(bc2, med, "-", color=color, lw=2, label=f"Box {box}")
    axes[0,1].axhline(0, color="k", ls=":", lw=1)
    axes[0,1].set_xscale("log")
    axes[0,1].set_xlim(SCI_MIN, SCI_MAX)
    axes[0,1].set_title("M5 residual vs dt-corrected Sci")
    axes[0,1].set_ylabel("resid [cnt/s/det]")
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3, which="both")

    # (1,0) dt_frac vs raw Sci
    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = df[df["box"] == box]
        med = median_per_bin(sub["sci_rate"].values, sub["dt_frac"].values, bins)
        axes[1,0].plot(bc, med, "-", color=color, lw=2, label=f"Box {box}")
    axes[1,0].set_xscale("log")
    axes[1,0].set_xlim(SCI_MIN, SCI_MAX)
    axes[1,0].set_title("Median dt_frac vs Sci (raw)")
    axes[1,0].set_ylabel("Dt/L_cycles")
    axes[1,0].set_xlabel("Sci [cnt/s/det]")
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3, which="both")

    # (1,1) ratio sci_rate_dt / sci_rate vs raw Sci — i.e. 1/(1-dt_frac)
    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = df[df["box"] == box]
        med = median_per_bin(sub["sci_rate"].values,
                             (sub["sci_rate_dt"] / sub["sci_rate"]).values, bins)
        axes[1,1].plot(bc, med, "-", color=color, lw=2, label=f"Box {box}")
    axes[1,1].set_xscale("log")
    axes[1,1].set_xlim(SCI_MIN, SCI_MAX)
    axes[1,1].set_title("sci_rate_dt / sci_rate = 1/(1−dt_frac)")
    axes[1,1].set_ylabel("ratio")
    axes[1,1].set_xlabel("Sci [cnt/s/det]")
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3, which="both")

    fig.suptitle("Dead-time correction test: M1 vs M5 residual + dt_frac profile",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m5_deadtime_test.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # ============ Verify: M5 residual ~ M1 residual / (1-dt_frac) ? ============
    # If M5 just rescales rates, residual should also rescale.
    # If S-curve is REAL deformation, M5 will show same shape after rescaling.
    print(f"\n=== Sanity check: at Sci_raw=3000, are M5 / M1 residual consistent with scaling? ===")
    mask_check = (df["sci_rate"] > 2800) & (df["sci_rate"] < 3200)
    sub = df[mask_check]
    if len(sub) > 100:
        print(f"  N={len(sub):,}")
        print(f"  median M1 resid:           {sub['resid_m1'].median():+.1f}")
        print(f"  median M5 resid:           {sub['resid_m5'].median():+.1f}")
        print(f"  median dt_frac:            {sub['dt_frac'].median():.4f}")
        print(f"  median 1/(1-dt_frac):      {(1/(1-sub['dt_frac'])).median():.4f}")
        # If correction is just rescaling, M5_resid ≈ M1_resid / (1 - dt_frac)
        expected = (sub["resid_m1"] / (1 - sub["dt_frac"])).median()
        print(f"  expected M5 = M1/(1-dtf):  {expected:+.1f}")


if __name__ == "__main__":
    main()
