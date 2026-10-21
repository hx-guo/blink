#!/usr/bin/env python3
"""Fit a₀ vs time as exponential decay (PMT outgassing model)
   a₀(t) = a∞ + (a₀ - a∞) · exp(-(t - t₀)/τ)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit

df = pd.read_csv("coef_table_big.csv")
def date_to_year(s):
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.year + (dt - datetime(dt.year, 1, 1)).total_seconds() / (365.25 * 86400)
df["year"] = df["date"].apply(date_to_year)
df["t"] = df["year"] - 2017.5  # years since launch

# === Fit a₀ exponential decay per box (and per detector) ===
def expmodel(t, a_inf, a0_minus_inf, tau):
    return a_inf + a0_minus_inf * np.exp(-t / tau)

def linmodel(t, a, b):
    return a + b * t

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
box_colors = {"A": "C0", "B": "C1", "C": "C2"}

# Top-left: per-box mean a₀ vs t with exp + linear fits
ax = axes[0, 0]
print(f"\n=== a₀ exponential decay fits ===")
print(f"{'Box':>3s}  {'a∞':>8s}  {'a₀-a∞':>8s}  {'τ [yr]':>8s}  {'RMS_exp':>8s}  {'RMS_lin':>8s}  {'RMS_const':>10s}")
for box in "ABC":
    sub = df[df["box"] == box].groupby("t")["a0"].mean().reset_index()
    t = sub["t"].values; y = sub["a0"].values
    ax.scatter(t, y, s=8, alpha=0.4, color=box_colors[box], label=f"Box {box}")
    # Fit exponential
    try:
        popt, _ = curve_fit(expmodel, t, y, p0=[50, 100, 5], maxfev=5000)
        a_inf, a0i, tau = popt
        pred = expmodel(t, *popt)
        rms_exp = np.sqrt(np.mean((y - pred) ** 2))
        # Linear
        c, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), y, rcond=None)
        rms_lin = np.sqrt(np.mean((y - (c[0] + c[1] * t)) ** 2))
        # Constant (mean)
        rms_const = y.std()
        ts = np.linspace(t.min(), t.max(), 200)
        ax.plot(ts, expmodel(ts, *popt), "-", color=box_colors[box], lw=2.5,
                label=f"Box {box}: a∞={a_inf:.0f}, τ={tau:.1f}yr")
        print(f"  {box}  {a_inf:>+8.1f}  {a0i:>+8.1f}  {tau:>8.2f}  {rms_exp:>8.2f}  {rms_lin:>8.2f}  {rms_const:>10.2f}")
    except Exception as e:
        print(f"  {box}: fit failed: {e}")

ax.set_xlabel("Years since 2017.5")
ax.set_ylabel("a₀ (intercept) [cnt/s]")
ax.set_title("a₀ vs time: exponential decay fit (per-box)\n"
             "Hypothesis: PMT outgassing → dark count decreases")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)


# Top-right: a₀ residual after exp fit (should be ~Poisson-noise level)
ax = axes[0, 1]
for box in "ABC":
    sub = df[df["box"] == box].groupby("t")["a0"].mean().reset_index()
    t = sub["t"].values; y = sub["a0"].values
    try:
        popt, _ = curve_fit(expmodel, t, y, p0=[50, 100, 5], maxfev=5000)
        resid = y - expmodel(t, *popt)
        ax.scatter(t, resid, s=10, alpha=0.5, color=box_colors[box],
                    label=f"Box {box} (resid σ={resid.std():.1f})")
    except Exception:
        pass
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Years since 2017.5")
ax.set_ylabel("a₀ - exp_fit residual [cnt/s]")
ax.set_title("Residual after exponential subtraction\nshould be small if PMT outgassing dominates")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)


# Bottom-left: same for a₁ (PHO coef)
ax = axes[1, 0]
for box in "ABC":
    sub = df[df["box"] == box].groupby("t")["a1"].mean().reset_index()
    t = sub["t"].values; y = sub["a1"].values
    ax.scatter(t, y, s=8, alpha=0.4, color=box_colors[box], label=f"Box {box}")
    try:
        # Try exp from above
        popt, _ = curve_fit(expmodel, t, y, p0=[0.7, -0.1, 5], maxfev=5000)
        ts = np.linspace(t.min(), t.max(), 200)
        ax.plot(ts, expmodel(ts, *popt), "-", color=box_colors[box], lw=2,
                label=f"Box {box}: a∞={popt[0]:.3f}, τ={popt[2]:.1f}yr")
    except Exception:
        pass
ax.set_xlabel("Years since 2017.5")
ax.set_ylabel("a₁ (PHO coef)")
ax.set_title("a₁ vs time")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)


# Bottom-right: comparison of residual std (constant vs linear vs exp)
ax = axes[1, 1]
labels = []
const_rms = []; lin_rms = []; exp_rms = []
for box in "ABC":
    for param in ["a0", "a1", "a2", "a3"]:
        sub = df[df["box"] == box].groupby("t")[param].mean().reset_index()
        t = sub["t"].values; y = sub[param].values
        labels.append(f"{box}.{param}")
        const_rms.append(y.std())
        c, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), y, rcond=None)
        lin_rms.append(np.sqrt(np.mean((y - (c[0] + c[1] * t)) ** 2)))
        try:
            popt, _ = curve_fit(expmodel, t, y, p0=[y.mean(), y[0] - y.mean(), 5], maxfev=5000)
            exp_rms.append(np.sqrt(np.mean((y - expmodel(t, *popt)) ** 2)))
        except Exception:
            exp_rms.append(np.nan)

xs = np.arange(len(labels))
ax.bar(xs - 0.27, const_rms, 0.27, color="C3", label="Constant")
ax.bar(xs, lin_rms, 0.27, color="C1", label="Linear in t")
ax.bar(xs + 0.27, exp_rms, 0.27, color="C0", label="Exponential")
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, fontsize=8)
ax.set_ylabel("Residual std")
ax.set_title("Per-box × param: const vs linear vs exp model")
ax.legend()
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = "plots/a0_exponential.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out}")
