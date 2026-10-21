#!/usr/bin/env python3
"""Add second-order terms to flatten the high-Sci residual S-curve.

Conservation extended: PHO = (1+α)·N_n + β·Wide + γ·Large + b
                          + ε_pp·N_n² + ε_pw·N_n·Wide + ε_pl·N_n·Large
                          + ε_ww·Wide² + ε_wl·Wide·Large + ε_ll·Large²

Test progressively richer models:
  N0: M1 baseline (β, γ, α, b only) — linear
  N1: + N_n² only
  N2: N1 + N_n·Wide
  N3: N2 + N_n·Large
  N4: full 10-coeff 2nd-order

Fit each on Sci > 300 (excluding clump), report RMS and check whether residual
S-curve at Sci > 1500 is flattened.
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
CLUMP_HI = 300.0


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
    df["sci_rate"]  = df["Sci"]  / df["length"]
    df["wide_rate"] = df["Wide"] / df["length"]
    df["large_rate"]= df["Large"]/ df["length"]
    df["pho_rate"]  = df["PHO"]  / df["length"]
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
    df = df[df["sci_rate"] > CLUMP_HI].copy()
    print(f"main-band rows: {len(df):,}")
    return df


def fit_model(sub, terms):
    """Fit PHO_rate = sum_i coef_i * term_i.
    terms is a list of strings naming columns (or composite products).
    Returns coefficients, predicted Sci, and Sci-vs-Sci_pred RMS.
    """
    sci  = sub["sci_rate"].values
    wide = sub["wide_rate"].values
    large= sub["large_rate"].values
    pho  = sub["pho_rate"].values

    var = {
        "1": np.ones(len(sub)),
        "Sci": sci, "Wide": wide, "Large": large,
        "Sci²": sci*sci, "Wide²": wide*wide, "Large²": large*large,
        "Sci·Wide": sci*wide, "Sci·Large": sci*large, "Wide·Large": wide*large,
    }
    X = np.column_stack([var[t] for t in terms])
    coef, *_ = np.linalg.lstsq(X, pho, rcond=None)
    # For Sci_pred, invert PHO = ... only when terms are linear in Sci.
    # General approach: numerical inversion is hard. Use the residual on PHO
    # as a proxy, and an approximation:
    #   At fixed (Wide, Large), PHO is a quadratic in Sci. Solve quadratic
    #   for each row.
    # Coefficients: collect Sci-dependence
    a0 = 0.0  # constant + Wide + Large + Wide² + Large² + WideLarge
    a1 = 0.0  # linear in Sci
    a2 = 0.0  # Sci²
    for c, t in zip(coef, terms):
        if t == "1":
            a0_term = c
            a0 += a0_term
        elif t == "Sci":
            a1 += c
        elif t == "Wide":
            a0 += c * wide
        elif t == "Large":
            a0 += c * large
        elif t == "Sci²":
            a2 += c
        elif t == "Sci·Wide":
            a1 += c * wide
        elif t == "Sci·Large":
            a1 += c * large
        elif t == "Wide²":
            a0 += c * wide * wide
        elif t == "Large²":
            a0 += c * large * large
        elif t == "Wide·Large":
            a0 += c * wide * large
    # Solve a2·s² + a1·s + (a0 - pho) = 0  for sci_pred
    # Quadratic formula, take positive root closest to actual sci_rate
    if isinstance(a2, float) and a2 == 0.0:
        sci_pred = (pho - a0) / a1
    else:
        # numerical
        disc = a1*a1 - 4*a2*(a0 - pho)
        disc = np.maximum(disc, 0)
        sqrt_d = np.sqrt(disc)
        # Two roots:
        r1 = (-a1 + sqrt_d) / (2*a2)
        r2 = (-a1 - sqrt_d) / (2*a2)
        # Pick the positive root
        sci_pred = np.where(r1 > 0, r1, r2)
    rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))
    return coef, sci_pred, rms


def main():
    df = load()

    models = {
        "N0": ["1", "Sci", "Wide", "Large"],
        "N1": ["1", "Sci", "Wide", "Large", "Sci²"],
        "N2": ["1", "Sci", "Wide", "Large", "Sci²", "Sci·Wide"],
        "N3": ["1", "Sci", "Wide", "Large", "Sci²", "Sci·Wide", "Sci·Large"],
        "N4": ["1", "Sci", "Wide", "Large", "Sci²", "Sci·Wide", "Sci·Large",
               "Wide²", "Large²", "Wide·Large"],
    }

    print(f"\n=== Nonlinear model comparison (main band Sci > {CLUMP_HI}) ===")
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharey="row")
    SCI_MIN, SCI_MAX = CLUMP_HI, 4000.0
    coefs_table = {}
    rms_table = {}

    for row_i, box in enumerate("ABC"):
        sub = df[df["box"] == box]
        for col_i, (name, terms) in enumerate(models.items()):
            coef, sci_pred, rms = fit_model(sub, terms)
            rms_table[(box, name)] = rms
            coefs_table[(box, name)] = (terms, coef)

            ax = axes[row_i, col_i]
            sci = sub["sci_rate"].values
            resid = sci_pred - sci
            bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
            bc = 0.5*(bins[:-1] + bins[1:])
            med = []
            for i in range(len(bins)-1):
                m = (sci >= bins[i]) & (sci < bins[i+1])
                med.append(np.median(resid[m]) if m.sum() > 300 else np.nan)
            ax.plot(bc, np.array(med), "o-", color="orange", lw=2, ms=3)
            ax.axhline(0, color="r", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlim(SCI_MIN, SCI_MAX)
            ax.set_ylim(-800, 200)
            if row_i == 0:
                ax.set_title(f"{name}\nN_terms={len(terms)}  RMS={rms:.0f}", fontsize=10)
            else:
                ax.set_title(f"RMS={rms:.0f}", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"Box {box}\nresid [cnt/s/det]")
            if row_i == 2:
                ax.set_xlabel("Sci obs [cnt/s/det]")
            ax.grid(alpha=0.3, which="both")

    fig.suptitle("Residual binned-median vs Sci, for increasingly rich nonlinear models",
                 fontsize=12, y=0.998)
    fig.tight_layout()
    out = OUT_DIR / "nonlinear_residuals.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # RMS summary
    print(f"\n=== RMS summary (main band) ===")
    print(f"{'Box':>4s}  " + "  ".join(f"{m:>8s}" for m in models))
    for box in "ABC":
        print(f"  {box:>2s}   " + "  ".join(f"{rms_table[(box,m)]:>8.1f}" for m in models))

    # Print N1 coefficients (most informative additional term)
    print(f"\n=== N1 (with Sci² term) coefficients per Box ===")
    for box in "ABC":
        terms, coef = coefs_table[(box, "N1")]
        for t, c in zip(terms, coef):
            print(f"  {box} | {t:>8s}: {c:>14.6g}")
        print()


if __name__ == "__main__":
    main()
