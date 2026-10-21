#!/usr/bin/env python3
"""Physics diagnostic for inner vs outer ring difference.

Question: at high Sci, outer ring has resid −150 cnt/s/det while inner ≈ 0.
The fact that this only appears at high Sci means it's source-induced.

Candidate physical mechanisms:
  C: lateral-HVT tantalum K-edge (67.4 keV) scattering. Source X-rays hit
     the 1 mm Ta shell around lateral HVTs, produce characteristic 67 keV
     X-rays, deposit in adjacent outer NaI. inner has no nearby Ta.
  D: ME/LE Box / satellite structure scattering. outer closer to these.
  B: large-event deadtime, if outer sees more Large events.

Diagnostic plots (no model fit, just data):
  - PHO/Sci  vs Sci, by ring   → constant if no Sci-prop bg, rises if there is
  - (PHO − β·Wide − γ·Large)/Sci vs Sci  → "effective 1+α"
  - Wide/Sci, Large/Sci          vs Sci, by ring  → does ring change PSA mix?
  - Sci_ACDN/Sci, Sci_ACD1/Sci   vs Sci, by ring  → veto rate diff (lateral HVT)
  - Dt/L_cycles                  vs Sci, by ring  → reported deadtime fraction
  - PHO − Sci_pred_M1            vs Sci, by ring  → residual shape
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
    # NOW also load Dt, Sci_ACD1, Sci_ACDN
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32",
             "Dt": "int32", "Sci": "int32",
             "Sci_ACD1": "int32", "Sci_ACDN": "int32"}
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
    df["dt_rate"]    = df["Dt"]    / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    df["ring"] = np.where(df["det"] < 2, "in", "out")

    # Diagnostic ratios — protect against tiny Sci
    sci_safe = df["Sci"].clip(lower=1)
    df["pho_over_sci"]    = df["PHO"]      / sci_safe
    df["wide_over_sci"]   = df["Wide"]     / sci_safe
    df["large_over_sci"]  = df["Large"]    / sci_safe
    df["acd1_over_sci"]   = df["Sci_ACD1"] / sci_safe
    df["acdn_over_sci"]   = df["Sci_ACDN"] / sci_safe
    df["dt_frac"]         = df["Dt"]       / df["L_cycles"].clip(lower=1)

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


def median_per_bin(sci, y, bins, min_count=200):
    """Median of y in Sci bins. NaN if too few points."""
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (sci >= bins[i]) & (sci < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    # Also compute M1 residual for reference
    print(f"\n=== M1 per-Box fit ===")
    df["resid_m1"] = np.nan
    for box in "ABC":
        mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
        sub = df[mask_fit]
        X = np.column_stack([np.ones(len(sub)), sub["sci_rate"].values,
                             sub["wide_rate"].values, sub["large_rate"].values])
        coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
        b, ap1, beta, gamma = coef
        mask_apply = df["box"] == box
        sub_a = df[mask_apply]
        pho_corr = sub_a["pho_rate"].values - beta*sub_a["wide_rate"].values - gamma*sub_a["large_rate"].values
        sci_pred = (pho_corr - b) / ap1
        df.loc[mask_apply, "resid_m1"] = sci_pred - sub_a["sci_rate"].values
        print(f"  Box {box}: b={b:.1f}, α={ap1-1:+.4f}, β={beta:+.4f}, γ={gamma:+.4f}")

    # ============ Diagnostic ratios at high Sci ============
    print(f"\n=== Median diagnostic ratios at high Sci (1500-3000) ===")
    print(f"{'Box':>4s} {'Ring':>5s}  "
          f"{'PHO/Sci':>10s} {'Wide/Sci':>10s} {'Large/Sci':>10s} "
          f"{'ACD1/Sci':>10s} {'ACDN/Sci':>10s} {'Dt/L':>10s}")
    for box in "ABC":
        for ring in ("in", "out"):
            sub = df[(df["box"] == box) & (df["ring"] == ring)
                     & (df["sci_rate"] > 1500) & (df["sci_rate"] < 3000)]
            if len(sub) == 0:
                continue
            print(f"  {box}    {ring:>5s}  "
                  f"{sub['pho_over_sci'].median():>10.4f} "
                  f"{sub['wide_over_sci'].median():>10.4f} "
                  f"{sub['large_over_sci'].median():>10.4f} "
                  f"{sub['acd1_over_sci'].median():>10.4f} "
                  f"{sub['acdn_over_sci'].median():>10.4f} "
                  f"{sub['dt_frac'].median():>10.4f}")

    # And at low-mid Sci (300-800) for comparison
    print(f"\n=== Median diagnostic ratios at low-mid Sci (300-800) ===")
    print(f"{'Box':>4s} {'Ring':>5s}  "
          f"{'PHO/Sci':>10s} {'Wide/Sci':>10s} {'Large/Sci':>10s} "
          f"{'ACD1/Sci':>10s} {'ACDN/Sci':>10s} {'Dt/L':>10s}")
    for box in "ABC":
        for ring in ("in", "out"):
            sub = df[(df["box"] == box) & (df["ring"] == ring)
                     & (df["sci_rate"] > 300) & (df["sci_rate"] < 800)]
            if len(sub) == 0:
                continue
            print(f"  {box}    {ring:>5s}  "
                  f"{sub['pho_over_sci'].median():>10.4f} "
                  f"{sub['wide_over_sci'].median():>10.4f} "
                  f"{sub['large_over_sci'].median():>10.4f} "
                  f"{sub['acd1_over_sci'].median():>10.4f} "
                  f"{sub['acdn_over_sci'].median():>10.4f} "
                  f"{sub['dt_frac'].median():>10.4f}")

    # ============ Plot: 6 diagnostic ratios vs Sci, by ring ============
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
    SCI_MIN, SCI_MAX = MAIN_BAND_LO, 4000.0
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])
    panels = [
        ("pho_over_sci",   "PHO / Sci",           "ratio"),
        ("wide_over_sci",  "Wide / Sci",          "ratio"),
        ("large_over_sci", "Large / Sci",         "ratio"),
        ("acdn_over_sci",  "Sci_ACDN / Sci",      "ratio"),
        ("acd1_over_sci",  "Sci_ACD1 / Sci",      "ratio"),
        ("dt_frac",        "Dt / L_cycles",       "fraction"),
    ]
    for ax, (col, title, ylab) in zip(axes.flat, panels):
        for ring, color in (("in", "C2"), ("out", "C3")):
            for box, ls in (("A", "-"), ("B", "--"), ("C", ":")):
                sub = df[(df["box"] == box) & (df["ring"] == ring)]
                med = median_per_bin(sub["sci_rate"].values, sub[col].values, bins)
                ax.plot(bc, med, ls, color=color, lw=1.5,
                        label=f"{box} {ring}")
        ax.set_xscale("log")
        ax.set_xlim(SCI_MIN, SCI_MAX)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Sci [cnt/s/det]")
        ax.grid(alpha=0.3, which="both")
        if col == "pho_over_sci":
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Diagnostic ratios vs Sci, inner (green) vs outer (red)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "diag_ratios_vs_sci.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # ============ Plot: residual_M1 vs Sci, by ring, normalized & raw ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for box in "ABC":
        for ring, color in (("in", "C2"), ("out", "C3")):
            sub = df[(df["box"] == box) & (df["ring"] == ring)]
            sci = sub["sci_rate"].values
            resid = sub["resid_m1"].values

            med = median_per_bin(sci, resid, bins)
            axes[0].plot(bc, med, "-" if ring == "in" else "--", color=f"C{ord(box)-ord('A')}",
                         lw=1.5, label=f"{box} {ring}")

            med_norm = median_per_bin(sci, resid / np.maximum(sci, 1), bins)
            axes[1].plot(bc, med_norm, "-" if ring == "in" else "--", color=f"C{ord(box)-ord('A')}",
                         lw=1.5, label=f"{box} {ring}")
    for ax in axes:
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(SCI_MIN, SCI_MAX)
        ax.set_xlabel("Sci [cnt/s/det]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("resid_M1 [cnt/s/det]")
    axes[0].set_title("Raw residual")
    axes[1].set_ylabel("resid_M1 / Sci")
    axes[1].set_title("Normalized residual (if ∝ Sci, this is flat)")
    fig.suptitle("Residual shape vs Sci, by ring", fontsize=12)
    fig.tight_layout()
    out2 = OUT_DIR / "diag_resid_shape.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
