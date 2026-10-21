#!/usr/bin/env python3
"""Free-γ fit: instead of PHO = N_n + 2W + 1.2L + N_below with fixed γ=1.2,
fit γ as a free parameter (along with β-free for completeness).

Conservation:   PHO = N_n + β·Wide + γ·Large + N_below
With empirical: N_below = b + α·N_n

Rearranged for regression:
    PHO - N_n = β·Wide + γ·Large + (b + α·N_n)
    => Sci_obs and counters are known; fit (β, γ, b, α) by OLS:

    PHO = N_n + α·N_n + β·Wide + γ·Large + b
        = (1+α)·N_n + β·Wide + γ·Large + b
    Or:  PHO/length - Sci - β·Wide/length - γ·Large/length = b + α·Sci

This script does:
  1. Per-observation (date-level) free-γ fit
  2. Histogram γ across observations
  3. If γ is bimodal → split observations by γ, refit pooled
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("n_below_study/per_sec_csvs")
HV_TABLE_PARTIAL = Path("n_below_study/hv_table_partial.csv.gz")
HV_TABLE_FULL = Path("n_below_study/hv_table.csv.gz")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)
L_THRESH = 50_000
SCI_SEC_TOTAL_MIN = 100
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


def load():
    dtype = {"date": "string", "box": "category", "met_sec": "int64",
             "det": "int8", "L_cycles": "int32",
             "PHO": "int32", "Wide": "int32", "Large": "int32", "Sci": "int32"}
    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Loading {len(files)} CSVs...")
    parts = []
    for i, f in enumerate(files):
        try:
            parts.append(pd.read_csv(f, usecols=list(dtype), dtype=dtype))
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True)
    df["length"] = df["L_cycles"].astype("float32") * 16e-6
    df = df[df["L_cycles"] > L_THRESH].copy()
    g = df.groupby(["date","box","met_sec"], observed=True)["Sci"].sum()
    g.name = "sci_sec_total"
    df = df.merge(g, on=["date","box","met_sec"])
    df = df[df["sci_sec_total"] > SCI_SEC_TOTAL_MIN]
    df["sci_rate"] = df["Sci"] / df["length"]
    df["wide_rate"] = df["Wide"] / df["length"]
    df["large_rate"] = df["Large"] / df["length"]
    df["pho_rate"] = df["PHO"] / df["length"]
    df["det_global"] = (df["box"].map(BOX_OFFSET) + df["det"]).astype("int8")
    df["year"] = df["date"].str.slice(0, 4).astype("int16")

    hv_path = HV_TABLE_PARTIAL if HV_TABLE_PARTIAL.exists() else HV_TABLE_FULL
    print(f"Loading HV table {hv_path}...")
    hv = pd.read_csv(hv_path, dtype={"date": "string", "met_sec": "int64",
                                     **{f"hv{i}": "float32" for i in range(18)}})
    hv = hv.set_index(["date","met_sec"]).sort_index()
    keys = pd.MultiIndex.from_arrays(
        [df["date"].astype(str).str.replace("-","",regex=False).values,
         df["met_sec"].values], names=["date","met_sec"])
    hv_arr = hv.reindex(keys).values
    rows = np.arange(len(df))
    df["hv"] = hv_arr[rows, df["det_global"].values.astype(int)]
    # keep only normal-mode (HV around -1000V)
    df = df[(df["hv"] < -900) & (df["hv"] > -1100)].copy()
    print(f"  normal-mode rows: {len(df):,}")
    return df


def free_fit(sub):
    """Fit (b, α, β, γ) jointly:
       PHO/length = (1+α)·Sci_rate + β·Wide_rate + γ·Large_rate + b
       i.e. fit pho_rate vs [1, sci_rate, wide_rate, large_rate]
    """
    if len(sub) < 100:
        return None
    X = np.column_stack([
        np.ones(len(sub)),
        sub["sci_rate"].values,
        sub["wide_rate"].values,
        sub["large_rate"].values,
    ])
    y = sub["pho_rate"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, one_plus_alpha, beta, gamma = coef
    alpha = one_plus_alpha - 1
    # residual after fit
    yhat = X @ coef
    rms = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return dict(b=b, alpha=alpha, beta=beta, gamma=gamma, rms=rms, n=len(sub))


def main():
    df = load()

    print(f"\n=== Per-Box global free-γ fit ===")
    for box in "ABC":
        f = free_fit(df[df["box"] == box])
        if f:
            print(f"  Box {box} (N={f['n']:,}):  b={f['b']:.1f}  α={f['alpha']:.3f}  "
                  f"β={f['beta']:.3f}  γ={f['gamma']:.3f}  RMS={f['rms']:.1f}")

    print(f"\n=== Per-(date, box) free-γ fit ===")
    results = []
    for (date, box), sub in df.groupby(["date","box"], observed=True):
        f = free_fit(sub)
        if f is None or f["n"] < 500:
            continue
        results.append({"date": date, "box": box, **f})
    res = pd.DataFrame(results)
    print(f"  N per-obs fits: {len(res):,}")
    print(f"  γ statistics: mean={res['gamma'].mean():.3f}  "
          f"median={res['gamma'].median():.3f}  std={res['gamma'].std():.3f}")
    print(f"  β statistics: mean={res['beta'].mean():.3f}  "
          f"median={res['beta'].median():.3f}  std={res['beta'].std():.3f}")
    print(f"  α statistics: mean={res['alpha'].mean():.3f}  std={res['alpha'].std():.3f}")
    print(f"  b statistics: mean={res['b'].mean():.1f}  std={res['b'].std():.1f}")

    # Histogram γ — check bimodality
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (var, ax) in enumerate(zip(["gamma","beta","alpha","b"], axes.flat)):
        for box, color in zip("ABC", ["C0","C1","C2"]):
            sub = res[res["box"] == box]
            ax.hist(sub[var], bins=80, alpha=0.5, label=f"Box {box} (N={len(sub)})",
                    color=color)
        ax.set_xlabel(var)
        ax.set_ylabel("# observations")
        ax.legend()
        ax.grid(alpha=0.3)
        if var == "gamma":
            ax.axvline(1.2, color="r", ls="--", lw=1.5, label="fixed γ=1.2")
            ax.legend()
    fig.suptitle("Per-observation free-γ fit: parameter distributions")
    fig.tight_layout()
    out = OUT_DIR / "free_gamma_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Also save per-obs fits for further use
    res.to_csv("plots/per_obs_free_gamma_fits.csv", index=False)
    print(f"Saved: plots/per_obs_free_gamma_fits.csv")


if __name__ == "__main__":
    main()
