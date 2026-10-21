#!/usr/bin/env python3
"""M6: Test if adding Sci_ACDN as an explicit term removes the +150 residual
bump at Sci 1500-2000.

Hypothesis:
  Sci 1000-2000 region shows a peak in Sci_ACDN/Sci ratio (from diag plot).
  Same region shows residual_M1 +150 bump (from diag_resid_shape).
  If ACD-coincident NaI events go into PHO but NOT into Sci, M1 mistakes
  these as extra source signal → sci_pred over-estimates → resid positive.

Test models:
  M1   : PHO = (1+α)Sci + β·Wide + γ·Large + b
  M6a  : PHO = (1+α)Sci + β·Wide + γ·Large + δ_n·Sci_ACDN + b
  M6b  : PHO = (1+α)Sci + β·Wide + γ·Large + δ_n·Sci_ACDN + δ_1·Sci_ACD1 + b

If M6 flattens the +150 bump, ACD coincidence is the mechanism.
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

    df["sci_rate"]    = df["Sci"]      / df["length"]
    df["wide_rate"]   = df["Wide"]     / df["length"]
    df["large_rate"]  = df["Large"]    / df["length"]
    df["pho_rate"]    = df["PHO"]      / df["length"]
    df["acd1_rate"]   = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]   = df["Sci_ACDN"] / df["length"]
    df["det_global"]  = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

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


def fit_lstsq(sub, terms):
    var = {
        "1":     np.ones(len(sub)),
        "Sci":   sub["sci_rate"].values,
        "Wide":  sub["wide_rate"].values,
        "Large": sub["large_rate"].values,
        "ACD1":  sub["acd1_rate"].values,
        "ACDN":  sub["acdn_rate"].values,
    }
    X = np.column_stack([var[t] for t in terms])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    return dict(zip(terms, coef))


def apply_fit(sub, coefs):
    """sci_pred = (PHO - sum(c·X for X != Sci, 1)) / coef_Sci  — minus const."""
    pho = sub["pho_rate"].values
    sci_term_coef = coefs.get("Sci", 1.0)
    const = coefs.get("1", 0.0)
    other = np.zeros(len(sub))
    for t, c in coefs.items():
        if t in ("Sci", "1"):
            continue
        if t == "Wide":
            other += c * sub["wide_rate"].values
        elif t == "Large":
            other += c * sub["large_rate"].values
        elif t == "ACD1":
            other += c * sub["acd1_rate"].values
        elif t == "ACDN":
            other += c * sub["acdn_rate"].values
    sci_pred = (pho - other - const) / sci_term_coef
    return sci_pred - sub["sci_rate"].values


def median_per_bin(sci, y, bins, min_count=200):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (sci >= bins[i]) & (sci < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    models = {
        "M1":  ["1", "Sci", "Wide", "Large"],
        "M6a": ["1", "Sci", "Wide", "Large", "ACDN"],
        "M6b": ["1", "Sci", "Wide", "Large", "ACDN", "ACD1"],
    }

    print(f"\n=== Coefficients (per Box) ===")
    print(f"{'Box':>4s} {'Model':>5s}  "
          f"{'b':>10s} {'α':>9s} {'β':>9s} {'γ':>9s} {'δ_N':>9s} {'δ_1':>9s}")
    residuals = {}
    for box in "ABC":
        for name, terms in models.items():
            mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
            coefs = fit_lstsq(df[mask_fit], terms)
            alpha = coefs.get("Sci", 1.0) - 1
            b = coefs.get("1", 0)
            beta = coefs.get("Wide", 0)
            gamma = coefs.get("Large", 0)
            delta_N = coefs.get("ACDN", 0)
            delta_1 = coefs.get("ACD1", 0)
            print(f"  {box}    {name:>5s}  "
                  f"{b:>+10.1f} {alpha:>+9.4f} {beta:>+9.4f} {gamma:>+9.4f} "
                  f"{delta_N:>+9.4f} {delta_1:>+9.4f}")
            mask_apply = df["box"] == box
            res = apply_fit(df[mask_apply], coefs)
            residuals[(box, name)] = (mask_apply, res)

    # Attach residuals to df
    for name in models:
        col = f"resid_{name}"
        df[col] = np.nan
        for box in "ABC":
            mask, res = residuals[(box, name)]
            df.loc[mask, col] = res

    # ============ RMS by Sci bin ============
    print(f"\n=== RMS by Sci bin ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    header = f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>10s}" for n in models)
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>10.1f}" for r in rmss)
        print(row)

    # ============ Median residual by Sci bin ============
    print(f"\n=== Median residual by Sci bin (the +150 bump test) ===")
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+10.1f}" for m in meds)
        print(row)

    # ============ Plot: residual vs Sci, one panel per model ============
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    SCI_MIN, SCI_MAX = MAIN_BAND_LO, 4500.0
    bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])
    for ax, name in zip(axes, models):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub = df[df["box"] == box]
            med = median_per_bin(sub["sci_rate"].values, sub[f"resid_{name}"].values, bins)
            ax.plot(bc, med, "-", color=color, lw=2, label=f"Box {box}")
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(SCI_MIN, SCI_MAX)
        ax.set_ylim(-700, 250)
        ax.set_title(f"{name}: {' + '.join(t for t in models[name] if t != '1')}",
                     fontsize=10)
        ax.set_xlabel("Sci [cnt/s/det]")
        if name == "M1":
            ax.set_ylabel("median residual [cnt/s/det]")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Effect of adding ACD coincidence terms to PHO model", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m6_acd_term_test.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # ============ Also: plot Sci_ACDN/Sci side-by-side residual to make link visible ============
    fig, ax_resid = plt.subplots(1, 1, figsize=(9, 5))
    ax_acdn = ax_resid.twinx()
    for box, color in zip("ABC", ["C0","C1","C2"]):
        sub = df[df["box"] == box]
        med_resid = median_per_bin(sub["sci_rate"].values, sub["resid_M1"].values, bins)
        sci_safe = sub["Sci"].clip(lower=1)
        med_acdn = median_per_bin(sub["sci_rate"].values,
                                  (sub["Sci_ACDN"] / sci_safe).values, bins)
        ax_resid.plot(bc, med_resid, "-", color=color, lw=2, label=f"M1 resid {box}")
        ax_acdn.plot(bc, med_acdn, "--", color=color, lw=1, alpha=0.6, label=f"ACDN/Sci {box}")
    ax_resid.axhline(0, color="k", ls=":", lw=1)
    ax_resid.set_xscale("log")
    ax_resid.set_xlim(SCI_MIN, SCI_MAX)
    ax_resid.set_xlabel("Sci [cnt/s/det]")
    ax_resid.set_ylabel("M1 residual [cnt/s/det]", color="black")
    ax_acdn.set_ylabel("Sci_ACDN / Sci", color="gray")
    ax_resid.grid(alpha=0.3, which="both")
    ax_resid.set_title("Visual link: M1 residual bump @1500-2000 vs ACDN/Sci peak @1000-2000")
    ax_resid.legend(loc="lower left", fontsize=9)
    ax_acdn.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out2 = OUT_DIR / "m6_resid_vs_acdn_link.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
