#!/usr/bin/env python3
"""Compare predictors of (a0, a1, a2, a3): time, state, time+state.
Cross-validated to avoid overfitting.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import UnivariateSpline

df = pd.read_csv("coef_table_big.csv")
def date_to_year(s):
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.year + (dt - datetime(dt.year, 1, 1)).total_seconds() / (365.25 * 86400)
df["year"] = df["date"].apply(date_to_year)

# === For each parameter and each box, test 4 models ===
print(f"\n{'Box':>3s} {'Param':>5s}  "
      f"{'const σ':>8s} {'time σ':>8s} {'state σ':>8s} {'both σ':>8s}  "
      f"{'state R²':>8s} {'both R²':>8s}")
results = []
for box in "ABC":
    for param in ["a0", "a1", "a2", "a3"]:
        sub = df[df["box"] == box].copy()
        # Aggregate per-date (mean across 6 detectors)
        agg = sub.groupby("year").agg({param: "mean", "PHO_med": "mean",
                                        "Wide_med": "mean", "Large_med": "mean",
                                        "Sci_med": "mean"}).reset_index()
        if len(agg) < 30: continue
        y = agg[param].values
        t = agg["year"].values
        pho = agg["PHO_med"].values
        large = agg["Large_med"].values
        wide = agg["Wide_med"].values
        # leave-one-out: predict each point from the others
        n = len(y)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        kfold = 5
        sigmas = {"const": [], "time": [], "state": [], "both": []}
        for k in range(kfold):
            test = idx[k::kfold]
            train = np.array([i for i in range(n) if i not in test])
            ytrain, ytest = y[train], y[test]
            # const: just mean
            pred_const = np.full_like(ytest, ytrain.mean())
            sigmas["const"].append(ytest - pred_const)
            # time: linear in t
            X = np.column_stack([np.ones(len(train)), t[train]])
            c, *_ = np.linalg.lstsq(X, ytrain, rcond=None)
            pred_time = c[0] + c[1] * t[test]
            sigmas["time"].append(ytest - pred_time)
            # state: linear in (PHO_med, Large_med, Wide_med)
            Xs = np.column_stack([np.ones(len(train)), pho[train], large[train], wide[train]])
            cs, *_ = np.linalg.lstsq(Xs, ytrain, rcond=None)
            pred_state = (cs[0] + cs[1] * pho[test] + cs[2] * large[test] + cs[3] * wide[test])
            sigmas["state"].append(ytest - pred_state)
            # both
            Xb = np.column_stack([np.ones(len(train)), t[train], pho[train], large[train], wide[train]])
            cb, *_ = np.linalg.lstsq(Xb, ytrain, rcond=None)
            pred_both = (cb[0] + cb[1] * t[test] + cb[2] * pho[test] + cb[3] * large[test] + cb[4] * wide[test])
            sigmas["both"].append(ytest - pred_both)

        std_const = np.std(np.concatenate(sigmas["const"]))
        std_time = np.std(np.concatenate(sigmas["time"]))
        std_state = np.std(np.concatenate(sigmas["state"]))
        std_both = np.std(np.concatenate(sigmas["both"]))
        # R² explained by state vs by both
        var_y = np.var(y)
        r2_state = 1 - std_state ** 2 / var_y
        r2_both = 1 - std_both ** 2 / var_y
        print(f"  {box} {param:>5s}  {std_const:>8.3f} {std_time:>8.3f} {std_state:>8.3f} {std_both:>8.3f}  "
              f"{r2_state:>8.3f} {r2_both:>8.3f}")
        results.append({"box": box, "param": param,
                        "std_const": std_const, "std_time": std_time,
                        "std_state": std_state, "std_both": std_both,
                        "r2_state": r2_state, "r2_both": r2_both})

# === Visualize: parameter vs PHO_med (the key state variable) ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}
for ax, param, title in zip(axes.flat,
                             ["a0", "a1", "a2", "a3"],
                             ["a₀ vs PHO_med", "a₁ vs PHO_med",
                              "a₂ vs PHO_med", "a₃ vs PHO_med"]):
    for box in "ABC":
        sub = df[df["box"] == box].groupby("year").agg({param: "mean", "PHO_med": "mean"}).reset_index()
        ax.scatter(sub["PHO_med"], sub[param], s=10, alpha=0.4, color=box_colors[box],
                    label=f"Box {box}")
        # fit
        x = sub["PHO_med"].values; y = sub[param].values
        A = np.column_stack([np.ones_like(x), x])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        rho = np.corrcoef(x, y)[0, 1]
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, c[0] + c[1] * xs, "-", color=box_colors[box], lw=1.5,
                 alpha=0.85)
    ax.set_xlabel("Daily median PHO [cnt/s/det]")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

fig.tight_layout()
out = "plots/time_state_param.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")

# === Final composite predictor: how does it work in practice? ===
# Use ALL 4 box × param fits, build a per-(box, param) predictor: a = c0 + c1·t + c2·PHO + c3·Large + c4·Wide
print(f"\n=== Composite predictor formulas (per box × param) ===")
formulas = {}
for box in "ABC":
    for param in ["a0", "a1", "a2", "a3"]:
        sub = df[df["box"] == box]
        agg = sub.groupby("year").agg({param: "mean", "PHO_med": "mean",
                                        "Large_med": "mean", "Wide_med": "mean"}).reset_index()
        if len(agg) < 30: continue
        y = agg[param].values
        Xb = np.column_stack([np.ones(len(agg)), agg["year"].values, agg["PHO_med"].values,
                              agg["Large_med"].values, agg["Wide_med"].values])
        cb, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        formulas[(box, param)] = cb
        print(f"  {box}.{param} = {cb[0]:+.4g} {cb[1]:+.4g}·yr {cb[2]:+.4g}·PHO {cb[3]:+.4g}·L {cb[4]:+.4g}·W")
