#!/usr/bin/env python3
"""Apply composite (time + state) predictor and measure end-to-end Sci RMS.
Compare to: global linear, daily recalibration.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("coef_table_big.csv")
def date_to_year(s):
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.year + (dt - datetime(dt.year, 1, 1)).total_seconds() / (365.25 * 86400)
df["year"] = df["date"].apply(date_to_year)

# === Build composite predictor: param = c0 + c1·yr + c2·PHO + c3·Large + c4·Wide ===
formulas = {}
for box in "ABC":
    for param in ["a0", "a1", "a2", "a3"]:
        sub = df[df["box"] == box]
        agg = sub.groupby("year").agg({param: "mean", "PHO_med": "mean",
                                        "Large_med": "mean", "Wide_med": "mean"}).reset_index()
        if len(agg) < 30: continue
        y = agg[param].values
        X = np.column_stack([np.ones(len(agg)), agg["year"].values, agg["PHO_med"].values,
                              agg["Large_med"].values, agg["Wide_med"].values])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        formulas[(box, param)] = c

# === For each (box, det, date) compute predicted Sci using:
#   Method 1: global mean a (4 box-level constants)
#   Method 2: composite predictor a(t, state)
#   Method 3: actual fitted a (oracle, assumes daily recalib) ===

# We don't have raw data here, but we have median PHO/Wide/Large/Sci per (box, det, date)
# So we use the predicted vs actual Sci mean as proxy

# First — global box means
global_means = df.groupby("box")[["a0", "a1", "a2", "a3"]].mean().to_dict("index")

methods = {"global": [], "composite": [], "oracle": []}
for _, row in df.iterrows():
    box = row["box"]
    PHO = row["PHO_med"]; W = row["Wide_med"]; L = row["Large_med"]
    Sci_actual = row["Sci_med"]
    yr = row["year"]
    # Method 1: global box mean
    g = global_means[box]
    Sci_g = g["a0"] + g["a1"] * PHO + g["a2"] * W + g["a3"] * L
    # Method 2: composite predictor
    c0 = formulas[(box, "a0")]; c1 = formulas[(box, "a1")]
    c2 = formulas[(box, "a2")]; c3 = formulas[(box, "a3")]
    a0 = c0[0] + c0[1]*yr + c0[2]*PHO + c0[3]*L + c0[4]*W
    a1 = c1[0] + c1[1]*yr + c1[2]*PHO + c1[3]*L + c1[4]*W
    a2 = c2[0] + c2[1]*yr + c2[2]*PHO + c2[3]*L + c2[4]*W
    a3 = c3[0] + c3[1]*yr + c3[2]*PHO + c3[3]*L + c3[4]*W
    Sci_c = a0 + a1 * PHO + a2 * W + a3 * L
    # Method 3: oracle (per-(date, box, det) fit)
    Sci_o = row["a0"] + row["a1"] * PHO + row["a2"] * W + row["a3"] * L
    methods["global"].append((Sci_actual, Sci_g))
    methods["composite"].append((Sci_actual, Sci_c))
    methods["oracle"].append((Sci_actual, Sci_o))

print(f"\n=== Per-(box, det, date) median Sci prediction RMS (N={len(df)}) ===")
for name, pairs in methods.items():
    arr = np.array(pairs)
    resid = arr[:, 0] - arr[:, 1]
    rms = np.sqrt(np.mean(resid ** 2))
    print(f"  {name:>10s}: RMS = {rms:>6.1f} cnt/s")

# === Also test composite predictor in cross-validation
# Split dates into train/test, fit composite on train, predict test
print(f"\n=== 5-fold CV on dates (predict held-out dates) ===")
dates = sorted(df["year"].unique())
rng = np.random.default_rng(0)
shuffled = rng.permutation(len(dates))
fold_size = len(dates) // 5
cv_results = {"composite": [], "global": []}
for fold in range(5):
    test_idx = shuffled[fold*fold_size:(fold+1)*fold_size]
    test_dates = set([dates[i] for i in test_idx])
    train_df = df[~df["year"].isin(test_dates)]
    test_df = df[df["year"].isin(test_dates)]
    # Fit on train
    train_formulas = {}
    train_means = {}
    for box in "ABC":
        for param in ["a0", "a1", "a2", "a3"]:
            sub = train_df[train_df["box"] == box]
            agg = sub.groupby("year").agg({param: "mean", "PHO_med": "mean",
                                            "Large_med": "mean", "Wide_med": "mean"}).reset_index()
            y = agg[param].values
            X = np.column_stack([np.ones(len(agg)), agg["year"].values, agg["PHO_med"].values,
                                  agg["Large_med"].values, agg["Wide_med"].values])
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            train_formulas[(box, param)] = c
        gm = train_df[train_df["box"] == box][["a0","a1","a2","a3"]].mean()
        train_means[box] = gm.to_dict()
    # Predict test
    for _, row in test_df.iterrows():
        box = row["box"]
        PHO = row["PHO_med"]; W = row["Wide_med"]; L = row["Large_med"]
        Sci_actual = row["Sci_med"]
        yr = row["year"]
        g = train_means[box]
        Sci_g = g["a0"] + g["a1"]*PHO + g["a2"]*W + g["a3"]*L
        c0 = train_formulas[(box,"a0")]; c1 = train_formulas[(box,"a1")]
        c2 = train_formulas[(box,"a2")]; c3 = train_formulas[(box,"a3")]
        a0 = c0[0]+c0[1]*yr+c0[2]*PHO+c0[3]*L+c0[4]*W
        a1 = c1[0]+c1[1]*yr+c1[2]*PHO+c1[3]*L+c1[4]*W
        a2 = c2[0]+c2[1]*yr+c2[2]*PHO+c2[3]*L+c2[4]*W
        a3 = c3[0]+c3[1]*yr+c3[2]*PHO+c3[3]*L+c3[4]*W
        Sci_c = a0 + a1*PHO + a2*W + a3*L
        cv_results["composite"].append(Sci_actual - Sci_c)
        cv_results["global"].append(Sci_actual - Sci_g)

print(f"{'Method':>10s}  CV RMS")
for name, residuals in cv_results.items():
    rms = np.sqrt(np.mean(np.array(residuals) ** 2))
    print(f"  {name:>10s}: {rms:>6.1f} cnt/s")

# === Plot ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for k, (name, pairs) in enumerate(methods.items()):
    arr = np.array(pairs)
    actual = arr[:, 0]; pred = arr[:, 1]
    rms = np.sqrt(np.mean((actual - pred) ** 2))
    ax = axes[k]
    ax.scatter(actual, pred, s=2, alpha=0.3, color="C0", rasterized=True)
    lo, hi = max(actual.min(), 0), min(actual.max(), 2500)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_xlabel("Sci_actual (median per day) [cnt/s]")
    ax.set_ylabel("Sci_predicted [cnt/s]")
    ax.set_title(f"{name} predictor\nRMS = {rms:.1f} cnt/s")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 2500); ax.set_ylim(0, 2500)
fig.suptitle(f"Predicted vs actual median Sci ({len(df)} (box, det, date) measurements)", fontsize=11)
fig.tight_layout()
out = "plots/composite_predictor.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")
