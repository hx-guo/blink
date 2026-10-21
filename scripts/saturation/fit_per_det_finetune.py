#!/usr/bin/env python3
"""Two fine-tuned 25p variants:
  A: 24p-linT  — replace sigmoid σ_t(t) by linear (1−β·(t−t_anchor))
  B: 28p-dualM — replace single sigmoid mlat by dual sigmoid (low+high)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_det_mlat_t_heatmap.npz"
T_ANCHOR = 4.0   # t anchor where decay reaches roughly half (year 2021 mid)


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# Baseline 25p model for comparison
def model_25(p, mlat, t):
    a_det = p[:18]
    alpha, mu_m, k_m, amp0, mu_t, k_t, C0 = p[18:25]
    sm = sigm((mlat - mu_m) / k_m); st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return (a_det[:, None, None] * g[None, :, None]
            * (1.0 - amp0 * g[None, :, None] * st[None, None, :]) + C0)


# Variant A: 24p with linear time (use a normalised t centered at t_anchor)
#   C = a_det · g · (1 − amp₀·g·(t − t_anchor)) + C₀
def model_linT(p, mlat, t):
    a_det = p[:18]
    alpha, mu_m, k_m, amp0, C0 = p[18:23]
    sm = sigm((mlat - mu_m) / k_m)
    g = 1.0 + alpha * sm
    t_norm = t - T_ANCHOR
    return (a_det[:, None, None] * g[None, :, None]
            * (1.0 - amp0 * g[None, :, None] * t_norm[None, None, :]) + C0)


# Variant B: 28p with dual mlat sigmoid + sigmoid time
#   g(mlat) = 1 + α₁·σ_m1 + α₂·σ_m2
def model_dualM(p, mlat, t):
    a_det = p[:18]
    alpha1, mu1, k1, alpha2, mu2, k2, amp0, mu_t, k_t, C0 = p[18:28]
    s1 = sigm((mlat - mu1) / k1)
    s2 = sigm((mlat - mu2) / k2)
    st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha1 * s1 + alpha2 * s2
    return (a_det[:, None, None] * g[None, :, None]
            * (1.0 - amp0 * g[None, :, None] * st[None, None, :]) + C0)


def fit(model_fn, p0, lo, hi, C_clean, mask, mlat, t, name):
    def res_fn(p):
        return ((model_fn(p, mlat, t) - C_clean) * mask).ravel()
    res = least_squares(res_fn, p0, bounds=(lo, hi), method='trf', max_nfev=10000)
    Cp = model_fn(res.x, mlat, t)
    r = (C_clean - Cp) * mask
    valid = r[mask]
    std = np.std(valid)
    print(f"\n=== {name} ===")
    print(f"  fit: cost={res.cost:.0f}, nfev={res.nfev}, success={res.success}")
    print(f"  heatmap std: {std:.3f} cnt/s")
    return res.x, std


def main():
    z = np.load(NPZ)
    C_data = z["C_mean"]; n_data = z["n"]
    months = z["months"]; edges = z["mlat_edges"]
    mlat = 0.5 * (edges[:-1] + edges[1:])
    month_dt = np.array([np.datetime64(m + "-15") for m in months])
    t0 = np.datetime64("2017-06-22")
    t_yr = ((month_dt - t0).astype("timedelta64[D]").astype(float)) / 365.25
    mask = n_data > 50
    C_clean = np.where(mask, C_data, 0.0)
    print(f"valid bins: {mask.sum()}/{mask.size}")

    # ─── Variant A: 23p linear time (replace σ_t) ───
    # params: a_det[18], α, μ_m, k_m, amp₀, C₀  → 23
    p0 = np.concatenate([np.full(18, 200.0),
                         [1.7, 44.0, 6.0, 0.10, -80.0]])
    lo = np.concatenate([np.full(18, 10.0),
                         [0.1, 30.0, 0.5, -1.0, -300.0]])
    hi = np.concatenate([np.full(18, 500.0),
                         [10.0, 60.0, 15.0, 1.0, 300.0]])
    xL, stdL = fit(model_linT, p0, lo, hi, C_clean, mask, mlat, t_yr,
                   "Variant A: 23p linear time (replace σ_t)")
    print(f"  a_det range: {xL[:18].min():.1f} - {xL[:18].max():.1f}")
    print(f"  α={xL[18]:.3f}, μ_m={xL[19]:.2f}°, k_m={xL[20]:.2f}°, amp₀={xL[21]:.4f}/yr, C₀={xL[22]:+.2f}")
    Path("/tmp/per_det_23p_linT.json").write_text(json.dumps({
        "a_det": xL[:18].tolist(),
        "alpha": float(xL[18]), "mu_m": float(xL[19]), "k_m": float(xL[20]),
        "amp0": float(xL[21]), "C0": float(xL[22]),
        "t_anchor": T_ANCHOR,
    }))

    # ─── Variant B: 28p dual mlat sigmoid + sigmoid time ───
    # params: a_det[18], α₁, μ_1, k_1, α₂, μ_2, k_2, amp₀, μ_t, k_t, C₀  → 28
    p0 = np.concatenate([np.full(18, 200.0),
                         [1.84, 44.9, 7.3, -0.07, 14.5, 1.3, 0.155, 5.25, 1.0, -80.0]])
    lo = np.concatenate([np.full(18, 10.0),
                         [0.1, 30.0, 0.5, -2.0, 5.0, 0.1, 0.0, 0.5, 0.2, -300.0]])
    hi = np.concatenate([np.full(18, 500.0),
                         [10.0, 60.0, 15.0, 2.0, 30.0, 10.0, 1.0, 8.0, 8.0, 300.0]])
    xD, stdD = fit(model_dualM, p0, lo, hi, C_clean, mask, mlat, t_yr,
                   "Variant B: 28p dual mlat sigmoid + sigmoid time")
    print(f"  a_det range: {xD[:18].min():.1f} - {xD[:18].max():.1f}")
    print(f"  α₁={xD[18]:.3f}, μ_1={xD[19]:.2f}°, k_1={xD[20]:.2f}°")
    print(f"  α₂={xD[21]:.4f}, μ_2={xD[22]:.2f}°, k_2={xD[23]:.3f}°")
    print(f"  amp₀={xD[24]:.3f}, μ_t={xD[25]:.2f}yr, k_t={xD[26]:.2f}yr, C₀={xD[27]:+.2f}")
    Path("/tmp/per_det_28p_dualM.json").write_text(json.dumps({
        "a_det": xD[:18].tolist(),
        "alpha1": float(xD[18]), "mu1": float(xD[19]), "k1": float(xD[20]),
        "alpha2": float(xD[21]), "mu2": float(xD[22]), "k2": float(xD[23]),
        "amp0": float(xD[24]), "mu_t": float(xD[25]), "k_t": float(xD[26]),
        "C0": float(xD[27]),
    }))

    print("\n=== Heatmap std summary ===")
    print(f"  25p (sigmoid time, single mlat sigm):   12.17 cnt/s")
    print(f"  23p-linT (linear time):                 {stdL:.2f} cnt/s")
    print(f"  28p-dualM (dual mlat sigmoid):          {stdD:.2f} cnt/s")
    print("  Saved /tmp/per_det_23p_linT.json, /tmp/per_det_28p_dualM.json")


if __name__ == "__main__":
    main()
