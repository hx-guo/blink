#!/usr/bin/env python3
"""Try multiple models to fix the dual-band residual structure:

M0: PHO = (1+α)·Sci + 2·Wide + 1.2·Large + b
    — baseline (β=2, γ=1.2 fixed, fit α, b per Box)

M1: PHO = (1+α)·Sci + β·Wide + γ·Large + b
    — free (α, β, γ, b)

M2: PHO = (1+α)·Sci + 2·Wide + 1.2·Large + δ·Sci_ACDN + b
    — baseline + ACDN as 5th term (particle background tracker)

M3: PHO = (1+α)·Sci + 2·Wide + 1.2·Large + δ·Sci_ACD1 + b
    — same with ACD1

M4: PHO = (1+α)·Sci + β·Wide + γ·Large + δ·Sci_ACDN + b
    — fully free with ACDN

For each model, fit per Box on pooled normal-mode data, then plot
Sci_pred vs Sci as 3x5 grid (Box × Model).
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
    df["sci_rate"] = df["Sci"] / df["length"]
    df["wide_rate"] = df["Wide"] / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"] = df["PHO"] / df["length"]
    df["acd1_rate"] = df["Sci_ACD1"] / df["length"]
    df["acdn_rate"] = df["Sci_ACDN"] / df["length"]
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
    print(f"normal-mode rows: {len(df):,}")
    return df


def fit_model(sub, model):
    """Fit a given model. Returns (coef_dict, sci_pred, rms).
       sub has columns: sci_rate, wide_rate, large_rate, pho_rate, acd1_rate, acdn_rate
    """
    y = sub["pho_rate"].values
    sci = sub["sci_rate"].values
    wide = sub["wide_rate"].values
    large = sub["large_rate"].values
    acd1 = sub["acd1_rate"].values
    acdn = sub["acdn_rate"].values

    if model == "M0":
        # Fixed β=2, γ=1.2. Subtract fixed terms then fit (1+α)·Sci + b.
        target = y - 2.0*wide - 1.2*large
        X = np.column_stack([np.ones(len(sub)), sci])
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        b, one_plus_a = coef
        alpha, beta, gamma, delta_acd = one_plus_a - 1, 2.0, 1.2, 0.0
    elif model == "M1":
        # Free (α, β, γ, b)
        X = np.column_stack([np.ones(len(sub)), sci, wide, large])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b, one_plus_a, beta, gamma = coef
        alpha, delta_acd = one_plus_a - 1, 0.0
    elif model == "M2":
        # β=2, γ=1.2 fixed + ACDN
        target = y - 2.0*wide - 1.2*large
        X = np.column_stack([np.ones(len(sub)), sci, acdn])
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        b, one_plus_a, delta_acd = coef
        alpha, beta, gamma = one_plus_a - 1, 2.0, 1.2
    elif model == "M3":
        target = y - 2.0*wide - 1.2*large
        X = np.column_stack([np.ones(len(sub)), sci, acd1])
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        b, one_plus_a, delta_acd = coef
        alpha, beta, gamma = one_plus_a - 1, 2.0, 1.2
    elif model == "M4":
        # Fully free
        X = np.column_stack([np.ones(len(sub)), sci, wide, large, acdn])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b, one_plus_a, beta, gamma, delta_acd = coef
        alpha = one_plus_a - 1
    else:
        raise ValueError(f"unknown model {model}")

    # Predicted Sci using inverted conservation
    pho_corr = y - beta*wide - gamma*large - delta_acd*acdn
    sci_pred = (pho_corr - b) / (1 + alpha)
    rms = float(np.sqrt(np.mean((sci - sci_pred) ** 2)))
    return dict(b=b, alpha=alpha, beta=beta, gamma=gamma, delta_acd=delta_acd,
                rms=rms, n=len(sub)), sci_pred


def main():
    df = load()
    models = ["M0", "M1", "M2", "M3", "M4"]
    titles = {
        "M0": "M0: β=2, γ=1.2 (baseline)",
        "M1": "M1: free β, γ",
        "M2": "M2: M0 + δ·ACDN",
        "M3": "M3: M0 + δ·ACD1",
        "M4": "M4: free β, γ + δ·ACDN",
    }

    print(f"\n{'Box':>4s} {'Model':>5s} {'b':>10s} {'α':>8s} {'β':>7s} {'γ':>7s} "
          f"{'δ_ACD':>8s} {'RMS':>7s} {'N':>10s}")
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharex=True, sharey=True)
    SCI_MIN, SCI_MAX = 40.0, 5000.0
    Y_MIN, Y_MAX = 1.0, 5000.0

    results = {}
    for row_i, box in enumerate("ABC"):
        sub = df[df["box"] == box].copy()
        for col_i, model in enumerate(models):
            ax = axes[row_i, col_i]
            info, sci_pred = fit_model(sub, model)
            results[(box, model)] = info
            sub[f"sci_pred_{model}"] = sci_pred
            print(f"  {box:>2s}  {model:>5s} {info['b']:>10.1f} {info['alpha']:>8.3f} "
                  f"{info['beta']:>7.3f} {info['gamma']:>7.3f} {info['delta_acd']:>8.3f} "
                  f"{info['rms']:>7.1f} {info['n']:>10,d}")
            # Plot hexbin
            sci = sub["sci_rate"].values
            sp_pos = np.maximum(sci_pred, Y_MIN * 0.5)
            keep = (sci >= SCI_MIN) & (sci <= SCI_MAX) & (sp_pos <= Y_MAX)
            ax.hexbin(sci[keep], sp_pos[keep], gridsize=80,
                      xscale="log", yscale="log",
                      extent=(np.log10(SCI_MIN), np.log10(SCI_MAX),
                              np.log10(Y_MIN), np.log10(Y_MAX)),
                      cmap="viridis", norm=LogNorm(vmin=1), mincnt=1, rasterized=True)
            ax.plot([SCI_MIN, SCI_MAX], [SCI_MIN, SCI_MAX], "r--", lw=1.2)
            # Binned median
            bins = np.logspace(np.log10(SCI_MIN), np.log10(SCI_MAX), 30)
            bc = 0.5 * (bins[:-1] + bins[1:])
            med = []
            for i in range(len(bins) - 1):
                m = (sci >= bins[i]) & (sci < bins[i+1])
                med.append(np.median(sci_pred[m]) if m.sum() > 500 else np.nan)
            ax.plot(bc, np.array(med), "-", color="orange", lw=1.8, zorder=5)

            ax.set_xlim(SCI_MIN, SCI_MAX); ax.set_ylim(Y_MIN, Y_MAX)
            if row_i == 0:
                ax.set_title(f"{titles[model]}\nRMS={info['rms']:.0f}", fontsize=9)
            else:
                ax.set_title(f"RMS={info['rms']:.0f}", fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f"Box {box}\nSci_pred")
            if row_i == 2:
                ax.set_xlabel("Sci_obs [cnt/s/det]")
            ax.grid(alpha=0.3, which="both")

    fig.suptitle("Multi-model comparison: which collapses the dual band?",
                 fontsize=12, y=0.998)
    fig.tight_layout()
    out = OUT_DIR / "multi_model_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # RMS summary
    print(f"\n=== RMS summary (Sci_pred vs Sci_obs) ===")
    print(f"{'Box':>4s} {'M0':>7s} {'M1':>7s} {'M2':>7s} {'M3':>7s} {'M4':>7s}")
    for box in "ABC":
        print(f"  {box:>2s}  " + "  ".join(
            f"{results[(box, m)]['rms']:>7.0f}" for m in models))


if __name__ == "__main__":
    main()
