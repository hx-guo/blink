#!/usr/bin/env python3
"""Test Xiao 2020 cross-detector dead-time coupling hypothesis.

For each (date, box, met_sec), compute Sci_others = sum of Sci over the OTHER
5 detectors in the same Box. If 6 detectors share one ADC, this should appear
as a dead-time penalty on PHO that scales with Sci_others.

Models tested:
  C0:  PHO = (1+α)·Sci + β·Wide + γ·Large + b
       — M1 baseline
  C1:  C0 + δ·Sci_others
       — linear cross-detector term
  C2:  C0 + δ·Sci_others + ε·Sci·Sci_others
       — cross-detector dead-time (true Xiao 2020 form)

For each model: residual vs Sci, RMS.
Also: residual at high Sci vs Sci_others to check the correlation directly.
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
    df["sci_rate"]   = df["Sci"]   / df["length"]
    df["wide_rate"]  = df["Wide"]  / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"]   = df["PHO"]   / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")

    # Compute Sci_others = total Sci rate in same (date, box, met_sec) MINUS own
    print("Computing Sci_others (sum of other 5 detectors)...")
    tot_sci = df.groupby(["date","box","met_sec"], observed=True)["sci_rate"].sum()
    tot_sci.name = "sci_tot_rate"
    df = df.merge(tot_sci, on=["date","box","met_sec"])
    df["sci_others_rate"] = df["sci_tot_rate"] - df["sci_rate"]
    df = df.drop(columns="sci_tot_rate")

    # HV filter
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


def fit_and_predict(sub, terms_for_pho):
    """Build design matrix from terms (column names), fit PHO_rate = X·coef.
    Return coefs and Sci_pred via numerical solve where needed."""
    sci  = sub["sci_rate"].values
    wide = sub["wide_rate"].values
    large= sub["large_rate"].values
    pho  = sub["pho_rate"].values
    sci_oth = sub["sci_others_rate"].values

    var = {
        "1": np.ones(len(sub)),
        "Sci": sci, "Wide": wide, "Large": large,
        "Sci_oth": sci_oth, "Sci·Sci_oth": sci*sci_oth,
    }
    X = np.column_stack([var[t] for t in terms_for_pho])
    coef, *_ = np.linalg.lstsq(X, pho, rcond=None)

    # For inversion: collect Sci-dependent parts
    # PHO = a0 + a1·Sci  (no Sci² in any of our test models, only Sci·Sci_oth
    #     which is also linear in Sci at fixed Sci_oth)
    a0 = np.zeros(len(sub))
    a1 = np.zeros(len(sub))
    for c, t in zip(coef, terms_for_pho):
        if t == "1":
            a0 += c
        elif t == "Sci":
            a1 += c
        elif t == "Wide":
            a0 += c * wide
        elif t == "Large":
            a0 += c * large
        elif t == "Sci_oth":
            a0 += c * sci_oth
        elif t == "Sci·Sci_oth":
            a1 += c * sci_oth
    sci_pred = (pho - a0) / a1
    rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))
    return coef, sci_pred, rms


def main():
    df = load()

    # Quick correlation: residual from M1 vs Sci_others
    print(f"\n=== Spearman corr of M1 residual with Sci_others ===")
    for box in "ABC":
        sub = df[df["box"] == box]
        coef, sp, rms = fit_and_predict(sub, ["1","Sci","Wide","Large"])
        sub_local = sub.assign(resid=sp - sub["sci_rate"].values)
        c = sub_local[["resid","sci_others_rate"]].corr(method="spearman").iloc[0,1]
        # Just in high-Sci region
        c_hi = sub_local.loc[sub_local["sci_rate"] > 1500,
                             ["resid","sci_others_rate"]].corr(method="spearman").iloc[0,1]
        print(f"  Box {box}: all Sci ρ={c:+.4f}  Sci>1500 ρ={c_hi:+.4f}  "
              f"(n_high={(sub_local['sci_rate']>1500).sum():,})")

    models = {
        "C0": ["1", "Sci", "Wide", "Large"],
        "C1": ["1", "Sci", "Wide", "Large", "Sci_oth"],
        "C2": ["1", "Sci", "Wide", "Large", "Sci_oth", "Sci·Sci_oth"],
    }

    print(f"\n=== Model fits with Sci_others ===")
    print(f"{'Box':>4s} {'Model':>5s}  " + " ".join(f"{t:>10s}" for t in
          ["b","α+1","β","γ","δ_oth","ε_cross"]) + "    RMS")
    fits = {}
    for box in "ABC":
        sub = df[df["box"] == box]
        for name, terms in models.items():
            coef, sp, rms = fit_and_predict(sub, terms)
            fits[(box, name)] = (coef, sp, rms, terms)
            # Pad coef to 6-wide for printing
            vals = {t: c for t, c in zip(terms, coef)}
            row = [vals.get("1", 0), vals.get("Sci", 0), vals.get("Wide", 0),
                   vals.get("Large", 0), vals.get("Sci_oth", 0),
                   vals.get("Sci·Sci_oth", 0)]
            print(f"  {box}    {name}   " + " ".join(f"{v:>10.5g}" for v in row) +
                  f"  {rms:>6.1f}")

    # Plot: residual vs Sci for each model
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    SCI_MIN, SCI_MAX = CLUMP_HI, 4000.0
    for row_i, box in enumerate("ABC"):
        for col_i, name in enumerate(models):
            ax = axes[row_i, col_i]
            coef, sp, rms, terms = fits[(box, name)]
            sci = df[df["box"] == box]["sci_rate"].values
            resid = sp - sci
            bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 40)
            bc = 0.5 * (bins[:-1] + bins[1:])
            med = []
            for i in range(len(bins) - 1):
                m = (sci >= bins[i]) & (sci < bins[i+1])
                med.append(np.median(resid[m]) if m.sum() > 300 else np.nan)
            ax.plot(bc, np.array(med), "o-", color="orange", lw=2, ms=3)
            ax.axhline(0, color="r", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlim(SCI_MIN, SCI_MAX)
            ax.set_ylim(-800, 200)
            if row_i == 0:
                title = {"C0": "C0: baseline M1",
                         "C1": "C1: + δ·Sci_others",
                         "C2": "C2: + δ·Sci_oth + ε·Sci·Sci_oth"}[name]
                ax.set_title(f"{title}\nRMS={rms:.0f}", fontsize=10)
            else:
                ax.set_title(f"RMS={rms:.0f}", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"Box {box}\nresid [cnt/s/det]")
            if row_i == 2:
                ax.set_xlabel("Sci obs [cnt/s/det]")
            ax.grid(alpha=0.3, which="both")

    fig.suptitle("Cross-detector dead-time test (Xiao 2020 form)",
                 fontsize=12, y=0.998)
    fig.tight_layout()
    out = OUT_DIR / "cross_detector_test.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Also: 2D plot of residual (M1) vs Sci_others, for high-Sci subset only
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, box in zip(axes, "ABC"):
        sub = df[df["box"] == box]
        coef, sp, rms = fit_and_predict(sub, ["1","Sci","Wide","Large"])
        resid = sp - sub["sci_rate"].values
        hi = sub["sci_rate"].values > 1500
        x = sub["sci_others_rate"].values[hi]
        y = resid[hi]
        if len(x) > 100:
            ax.hexbin(x, y, gridsize=60, cmap="viridis",
                      norm=LogNorm(vmin=1), mincnt=1, rasterized=True)
        ax.axhline(0, color="r", ls="--", lw=1)
        ax.set_xlabel("Sci_others [cnt/s/det × 5 dets sum]")
        ax.set_ylabel("M1 residual = Sci_pred − Sci [cnt/s/det]")
        ax.set_title(f"Box {box}: high Sci (>1500) only, N={hi.sum():,}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = OUT_DIR / "resid_vs_sci_others.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
