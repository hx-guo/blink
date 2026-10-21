#!/usr/bin/env python3
"""M10: combine M7 (ACD channel decomposition) with M9 (group_rate non-linear).

Are they orthogonal physical effects or the same thing seen two ways?

Models compared:
  M1:  PHO = (1+α)·Sci + β·Wide + γ·Large + b                                     [baseline]
  M7:  PHO = c0·Sci_pure + c1·Sci_ACD1 + cN·Sci_ACDN + β·Wide + γ·Large + b      [ACD channels]
  M9b: PHO = (1+α)·Sci + β·Wide + γ·Large + δ·Other + ε·Other² + b               [group_rate²]
  M10: PHO = c0·Sci_pure + c1·Sci_ACD1 + cN·Sci_ACDN + β·Wide + γ·Large
              + δ·Other + ε·Other² + b                                            [combined]

  Sci_pure = Sci - Sci_ACD1 - Sci_ACDN (disjoint).
  Other    = (group_rate - sci_rate) / R0,  R0 = 10000.

If M10 RMS ≈ M7 RMS: M9's pile-up term is REDUNDANT with ACD term → same physics
If M10 RMS < both M7 and M9b RMS: independent physical effects, both real
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
R0 = 10000.0
SCI_REF = 2.24


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

    df["Sci_pure"] = df["Sci"] - df["Sci_ACD1"] - df["Sci_ACDN"]

    df["sci_rate"]      = df["Sci"]      / df["length"]
    df["scipure_rate"]  = df["Sci_pure"] / df["length"]
    df["acd1_rate"]     = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"]     = df["Sci_ACDN"] / df["length"]
    df["wide_rate"]     = df["Wide"]     / df["length"]
    df["large_rate"]    = df["Large"]    / df["length"]
    df["pho_rate"]      = df["PHO"]      / df["length"]
    df["group_rate"]    = df["sci_sec_total"] / df["length"]
    df["other_rate"]    = (df["group_rate"] - df["sci_rate"]) / R0
    df["other2"]        = df["other_rate"] ** 2
    df["det_global"]    = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

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


VAR_MAP_KEYS = [
    "1", "Sci", "SciPure", "ACD1", "ACDN", "Wide", "Large", "Other", "Other2"
]


def var_map(sub):
    return {
        "1":       np.ones(len(sub)),
        "Sci":     sub["sci_rate"].values,
        "SciPure": sub["scipure_rate"].values,
        "ACD1":    sub["acd1_rate"].values,
        "ACDN":    sub["acdn_rate"].values,
        "Wide":    sub["wide_rate"].values,
        "Large":   sub["large_rate"].values,
        "Other":   sub["other_rate"].values,
        "Other2":  sub["other2"].values,
    }


def fit_lstsq(sub, terms):
    vm = var_map(sub)
    X = np.column_stack([vm[t] for t in terms])
    coef, *_ = np.linalg.lstsq(X, sub["pho_rate"].values, rcond=None)
    cond = np.linalg.cond(X)
    return dict(zip(terms, coef)), cond


def predict_pho(sub, coefs):
    vm = var_map(sub)
    pho_pred = np.zeros(len(sub))
    for t, c in coefs.items():
        pho_pred += c * vm[t]
    return pho_pred


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
        "M7":   ["1", "SciPure", "ACD1", "ACDN", "Wide", "Large"],
        "M9b":  ["1", "Sci", "Wide", "Large", "Other", "Other2"],
        "M10":  ["1", "SciPure", "ACD1", "ACDN", "Wide", "Large", "Other", "Other2"],
    }

    print(f"\n=== Coefficients (per Box) ===")
    print(f"{'Box':>4s} {'Model':>5s}  {'cond':>10s}  "
          f"{'b':>10s} {'cPure':>9s} {'c1':>9s} {'cN':>9s} {'β':>9s} {'γ':>9s} "
          f"{'δ':>9s} {'ε':>9s}")
    residuals_pho = {}
    for box in "ABC":
        for name, terms in models.items():
            mask_fit = (df["box"] == box) & (df["sci_rate"] > MAIN_BAND_LO)
            coefs, cond = fit_lstsq(df[mask_fit], terms)
            b      = coefs.get("1", 0)
            c_p    = coefs.get("Sci", coefs.get("SciPure", np.nan))
            c_1    = coefs.get("ACD1", np.nan)
            c_n    = coefs.get("ACDN", np.nan)
            beta   = coefs.get("Wide", 0)
            gamma  = coefs.get("Large", 0)
            d_o    = coefs.get("Other", np.nan)
            e_o2   = coefs.get("Other2", np.nan)
            print(f"  {box}    {name:>5s}  {cond:>10.1f}  "
                  f"{b:>+10.1f} {c_p:>+9.4f} {c_1:>+9.4f} {c_n:>+9.4f} "
                  f"{beta:>+9.4f} {gamma:>+9.4f} {d_o:>+9.2f} {e_o2:>+9.2f}")
            mask_apply = df["box"] == box
            pho_pred = predict_pho(df[mask_apply], coefs)
            resid_pho = df.loc[mask_apply, "pho_rate"].values - pho_pred
            residuals_pho[(box, name)] = (mask_apply, resid_pho)

    for name in models:
        col = f"resid_{name}"
        df[col] = np.nan
        for box in "ABC":
            mask, res_pho = residuals_pho[(box, name)]
            df.loc[mask, col] = res_pho / SCI_REF  # CORRECT SIGN: positive PHO_meas - PHO_pred → positive sci_pred - sci

    # ============ RMS by Sci bin ============
    print(f"\n=== RMS by Sci bin (Sci-equiv units, common SCI_REF={SCI_REF}) ===")
    bin_edges = [300, 600, 1000, 1500, 2000, 2500, 4500]
    header = f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>9s}" for n in models) \
             + "    M10-M7%   M10-M9b%"
    print(header)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        rmss = [np.sqrt(np.mean(df.loc[mask, f"resid_{n}"]**2)) for n in models]
        d_m7  = 100 * (rmss[3] - rmss[1]) / rmss[1]
        d_m9b = 100 * (rmss[3] - rmss[2]) / rmss[2]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{r:>9.1f}" for r in rmss)
        row += f"   {d_m7:>+8.1f}%   {d_m9b:>+8.1f}%"
        print(row)

    print(f"\n=== Median residual by Sci bin ===")
    print(f"{'Sci bin':>15s}  {'N':>10s}  " + "  ".join(f"{n:>9s}" for n in models))
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (df["sci_rate"] >= lo) & (df["sci_rate"] < hi)
        if mask.sum() < 100:
            continue
        meds = [df.loc[mask, f"resid_{n}"].median() for n in models]
        row = f"  {lo:>5d}-{hi:>5d}  {mask.sum():>10,d}  " + "  ".join(f"{m:>+9.1f}" for m in meds)
        print(row)

    # ============ Plot: residual vs Sci and vs group_rate, side by side ============
    fig, axes = plt.subplots(2, len(models), figsize=(20, 9), sharey="row")
    SCI_MIN, SCI_MAX = 300, 4500
    G_MIN, G_MAX = 1800, 25000
    bins_s = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
    bins_g = np.logspace(np.log10(G_MIN), np.log10(G_MAX), 40)
    bc_s = 0.5 * (bins_s[:-1] + bins_s[1:])
    bc_g = 0.5 * (bins_g[:-1] + bins_g[1:])

    for col_idx, name in enumerate(models):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub = df[df["box"] == box]
            med_s = median_per_bin(sub["sci_rate"].values,
                                   sub[f"resid_{name}"].values, bins_s)
            med_g = median_per_bin(sub["group_rate"].values,
                                   sub[f"resid_{name}"].values, bins_g)
            axes[0, col_idx].plot(bc_s, med_s, "-", color=color, lw=2, label=f"Box {box}")
            axes[1, col_idx].plot(bc_g, med_g, "-", color=color, lw=2, label=f"Box {box}")

        for ax, xlim, xlab in zip(axes[:, col_idx],
                                   [(SCI_MIN, SCI_MAX), (G_MIN, G_MAX)],
                                   ["Sci [cnt/s/det]", "group_rate [cnt/s/box]"]):
            ax.axhline(0, color="k", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(-250, 700)
            ax.set_xlabel(xlab)
            ax.grid(alpha=0.3, which="both")
        axes[0, col_idx].set_title(f"{name}", fontsize=11)

    axes[0, 0].set_ylabel("Sci-equiv residual\n(vs per-det Sci)")
    axes[1, 0].set_ylabel("Sci-equiv residual\n(vs group_rate)")
    axes[0, 0].legend(fontsize=9)

    fig.suptitle("M10 = M7 + M9b. Are ACD and pile-up the same physics or independent?",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "m10_combine.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
