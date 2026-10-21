#!/usr/bin/env python3
"""M8: Add group_rate (PDAU-shared ADC total rate) terms to M1.

Physical motivation (verified in diag_resid_vs_group_rate + diag_pileup_signature):
  The S-curve in residual is determined by group_rate (6-det sum sharing one
  ADC), NOT by per-det Sci. The mechanism is pile-up + dead time at the
  PDAU-shared ADC level:

  - Mid group_rate (6-10k cnt/s): pile-up gain — sub-threshold events combine
    to cross PHO threshold → residual POSITIVE (+160 at grp 8-10k)
  - High group_rate (>10k cnt/s): pile-up loss — PHO events shift up to Wide,
    plus dead-time effects → residual NEGATIVE (-600 at grp >14k)

Model variants:
  M1:  PHO = (1+α)·Sci + β·Wide + γ·Large + b                          [baseline]
  M8a: PHO = (1+α)·Sci + β·Wide + γ·Large + δ·group_rate + b           [linear rate]
  M8b: PHO = (1+α)·Sci + β·Wide + γ·Large + δ·G + ε·G²/G_norm + b      [+pile-up²]
  M8c: PHO = (1+α)·Sci + β·Wide + γ·Large + δ·G + ε·G²/G_norm
                + ζ·Sci·G/G_norm + b                                    [+dead-time]

  where G = group_rate, G_norm = 10000 (numerical conditioning).

Hypothesis: M8b/c should flatten the bump (positive at mid grp) AND the drop
(negative at high grp) simultaneously.
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
G_NORM = 10000.0


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

    df["sci_rate"]    = df["Sci"]      / df["length"]
    df["wide_rate"]   = df["Wide"]     / df["length"]
    df["large_rate"]  = df["Large"]    / df["length"]
    df["pho_rate"]    = df["PHO"]      / df["length"]
    df["group_rate"]  = df["sci_sec_total"] / df["length"]
    df["G"]           = df["group_rate"] / G_NORM
    df["G2"]          = df["G"] ** 2
    df["SciG"]        = df["sci_rate"] * df["G"]
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
        "G":     sub["G"].values,
        "G2":    sub["G2"].values,
        "SciG":  sub["SciG"].values,
    }
    X = np.column_stack([var[t] for t in terms])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    cond = np.linalg.cond(X)
    return dict(zip(terms, coef)), cond


def apply_fit(sub, coefs):
    pho_pred = np.zeros(len(sub))
    contrib_no_sci = np.zeros(len(sub))
    for t, c in coefs.items():
        v = {
            "1":     1.0,
            "Sci":   sub["sci_rate"].values,
            "Wide":  sub["wide_rate"].values,
            "Large": sub["large_rate"].values,
            "G":     sub["G"].values,
            "G2":    sub["G2"].values,
            "SciG":  sub["SciG"].values,
        }[t]
        pho_pred += c * v
        if t != "Sci":
            contrib_no_sci += c * v
    sci_coef = coefs.get("Sci", 1.0)
    sci_pred = (sub["pho_rate"].values - contrib_no_sci) / sci_coef
    return sci_pred - sub["sci_rate"].values


def median_per_bin(x, y, bins, min_count=200):
    med = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        if m.sum() > min_count:
            med[i] = np.median(y[m])
    return med


def main():
    df = load()

    models = {
        "M1":   ["1", "Sci", "Wide", "Large"],
        "M8a":  ["1", "Sci", "Wide", "Large", "G"],
        "M8b":  ["1", "Sci", "Wide", "Large", "G", "G2"],
        "M8c":  ["1", "Sci", "Wide", "Large", "G", "G2", "SciG"],
    }

    print(f"\n=== Coefficients (per Box) ===")
    print(f"{'Box':>4s} {'Model':>5s}  {'cond':>10s}  "
          f"{'b':>10s} {'1+α':>9s} {'β':>9s} {'γ':>9s} "
          f"{'δ(G)':>9s} {'ε(G²)':>9s} {'ζ(SciG)':>10s}")
    residuals = {}
    for box in "ABC":
        for name, terms in models.items():
            mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
            coefs, cond = fit_lstsq(df[mask_fit], terms)
            b      = coefs.get("1", 0)
            c_sci  = coefs.get("Sci", 1.0)
            beta   = coefs.get("Wide", 0)
            gamma  = coefs.get("Large", 0)
            d_g    = coefs.get("G", np.nan)
            e_g2   = coefs.get("G2", np.nan)
            z_scig = coefs.get("SciG", np.nan)
            print(f"  {box}    {name:>5s}  {cond:>10.1f}  "
                  f"{b:>+10.1f} {c_sci:>+9.4f} {beta:>+9.4f} {gamma:>+9.4f} "
                  f"{d_g:>+9.2f} {e_g2:>+9.2f} {z_scig:>+10.4f}")
            mask_apply = df["box"] == box
            res = apply_fit(df[mask_apply], coefs)
            residuals[(box, name)] = (mask_apply, res)

    for name in models:
        col = f"resid_{name}"
        df[col] = np.nan
        for box in "ABC":
            mask, res = residuals[(box, name)]
            df.loc[mask, col] = res

    # ============ RMS by Sci bin ============
    print(f"\n=== RMS by Sci bin (improvement vs M1) ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    header = f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>9s}" for n in models)
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>9.1f}" for r in rmss)
        print(row)

    print(f"\n=== Median residual by Sci bin ===")
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+9.1f}" for m in meds)
        print(row)

    # ============ RMS by GROUP rate bin (the real variable) ============
    print(f"\n=== RMS by GROUP rate bin ===")
    g_edges = [1800, 4000, 6000, 8000, 10000, 14000, 25000]
    header2 = f"{'group_rate':>15s}  {'N':>10s}  " + "  ".join(f"{n:>9s}" for n in models)
    print(header2)
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>9.1f}" for r in rmss)
        print(row)

    print(f"\n=== Median residual by GROUP rate bin ===")
    print(header2)
    for i in range(len(g_edges) - 1):
        lo, hi = g_edges[i], g_edges[i+1]
        mask = (df["group_rate"] >= lo) & (df["group_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+9.1f}" for m in meds)
        print(row)

    # ============ Plot: residual vs group_rate, one panel per model ============
    fig, axes = plt.subplots(1, len(models), figsize=(20, 5), sharey=True)
    G_MIN, G_MAX = 1800, 25000
    bins = np.logspace(np.log10(G_MIN), np.log10(G_MAX), 40)
    bc = 0.5 * (bins[:-1] + bins[1:])
    for ax, name in zip(axes, models):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub = df[df["box"] == box]
            med = median_per_bin(sub["group_rate"].values,
                                 sub[f"resid_{name}"].values, bins)
            ax.plot(bc, med, "-", color=color, lw=2, label=f"Box {box}")
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(G_MIN, G_MAX)
        ax.set_ylim(-700, 250)
        terms_str = " + ".join(t for t in models[name] if t != "1")
        ax.set_title(f"{name}: {terms_str}", fontsize=9)
        ax.set_xlabel("group_rate [cnt/s/box]")
        if name == "M1":
            ax.set_ylabel("median residual [cnt/s/det]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("M8: add group_rate (PDAU-shared ADC rate) terms — pile-up & dead-time",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m8_grouprate.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Also vs per-det Sci, for comparison
    fig2, axes2 = plt.subplots(1, len(models), figsize=(20, 5), sharey=True)
    SCI_MIN, SCI_MAX = 300, 4500
    bins_s = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bc_s = 0.5 * (bins_s[:-1] + bins_s[1:])
    for ax, name in zip(axes2, models):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub = df[df["box"] == box]
            med = median_per_bin(sub["sci_rate"].values,
                                 sub[f"resid_{name}"].values, bins_s)
            ax.plot(bc_s, med, "-", color=color, lw=2, label=f"Box {box}")
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(SCI_MIN, SCI_MAX)
        ax.set_ylim(-700, 250)
        terms_str = " + ".join(t for t in models[name] if t != "1")
        ax.set_title(f"{name}: {terms_str}", fontsize=9)
        ax.set_xlabel("Sci [cnt/s/det]")
        if name == "M1":
            ax.set_ylabel("median residual [cnt/s/det]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig2.suptitle("M8: residual vs per-det Sci (alternative view)", fontsize=12)
    fig2.tight_layout()
    out2 = OUT_DIR / "m8_grouprate_vs_sci.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
