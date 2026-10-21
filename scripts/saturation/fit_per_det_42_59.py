#!/usr/bin/env python3
"""Fit 42-param (per-det a + C₀) and 59-param (+ per-det amp₀) models.

Common shared shape:
  g(mlat) = 1 + α · σ_m(mlat)
  σ_t(t) = sigmoid((t - μ_t)/k_t)

42p model:
  C(det, mlat, t) = a_det · g · [1 − amp₀ · g · σ_t] + C₀_det
  params: a_det[18], C₀_det[18], α, μ_m, k_m, amp₀, μ_t, k_t  → 42

59p model:
  C(det, mlat, t) = a_det · g · [1 − amp₀_det · g · σ_t] + C₀_det
  params: a_det[18], C₀_det[18], amp₀_det[18], α, μ_m, k_m, μ_t, k_t  → 59
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_det_mlat_t_heatmap.npz"


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def model_42(p, mlat, t):
    # p layout: a_det[18], C0_det[18], α, μ_m, k_m, amp₀, μ_t, k_t
    a_det  = p[:18]
    C0_det = p[18:36]
    alpha, mu_m, k_m, amp0, mu_t, k_t = p[36:42]
    sm = sigm((mlat - mu_m) / k_m); st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return (a_det[:, None, None]
            * g[None, :, None] * (1.0 - amp0 * g[None, :, None] * st[None, None, :])
            + C0_det[:, None, None])


def model_59(p, mlat, t):
    # p layout: a_det[18], C0_det[18], amp0_det[18], α, μ_m, k_m, μ_t, k_t
    a_det    = p[:18]
    C0_det   = p[18:36]
    amp0_det = p[36:54]
    alpha, mu_m, k_m, mu_t, k_t = p[54:59]
    sm = sigm((mlat - mu_m) / k_m); st = sigm((t - mu_t) / k_t)
    g = 1.0 + alpha * sm
    return (a_det[:, None, None]
            * g[None, :, None]
            * (1.0 - amp0_det[:, None, None] * g[None, :, None] * st[None, None, :])
            + C0_det[:, None, None])


def fit(model_fn, p0, lo, hi, C_clean, mask, mlat, t, name):
    def residual(p):
        return ((model_fn(p, mlat, t) - C_clean) * mask).ravel()
    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=8000)
    Cp = model_fn(res.x, mlat, t)
    r = (C_clean - Cp) * mask
    valid = r[mask]
    std = np.std(valid); rms_pct = np.sqrt(np.mean(valid**2))/np.mean(C_clean[mask])*100
    print(f"\n=== {name} ===")
    print(f"  fit: cost={res.cost:.0f}, nfev={res.nfev}, success={res.success}")
    print(f"  heatmap std: {std:.3f} cnt/s,  RMS/mean: {rms_pct:.2f}%")
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

    # ─── 42p: per-det a + per-det C₀ ───
    p0_42 = np.concatenate([
        np.full(18, 200.0),                  # a_det
        np.full(18, -80.0),                  # C0_det
        [1.7, 44.0, 6.0, 0.15, 5.2, 1.0],    # α, μ_m, k_m, amp₀, μ_t, k_t
    ])
    lo_42 = np.concatenate([
        np.full(18, 10.0), np.full(18, -300.0),
        [0.1, 30.0, 0.5, 0.0, 0.5, 0.2],
    ])
    hi_42 = np.concatenate([
        np.full(18, 500.0), np.full(18, 300.0),
        [10.0, 60.0, 15.0, 1.0, 8.0, 8.0],
    ])
    x42, std42 = fit(model_42, p0_42, lo_42, hi_42, C_clean, mask, mlat, t_yr, "42p (a_det + C₀_det)")
    a_det42 = x42[:18]; C0_det42 = x42[18:36]
    alpha42, mu_m42, k_m42, amp0_42, mu_t42, k_t42 = x42[36:42]
    print(f"  a_det range: {a_det42.min():.1f} - {a_det42.max():.1f}")
    print(f"  C₀_det range: {C0_det42.min():.1f} - {C0_det42.max():.1f}")
    print(f"  shared: α={alpha42:.3f}, μ_m={mu_m42:.2f}°, k_m={k_m42:.2f}°, "
          f"amp₀={amp0_42:.3f}, μ_t={mu_t42:.2f}yr, k_t={k_t42:.2f}yr")
    Path("/tmp/per_det_42p.json").write_text(json.dumps({
        "a_det": a_det42.tolist(), "C0_det": C0_det42.tolist(),
        "alpha": float(alpha42), "mu_m": float(mu_m42), "k_m": float(k_m42),
        "amp0": float(amp0_42), "mu_t": float(mu_t42), "k_t": float(k_t42),
    }))
    print("  Saved /tmp/per_det_42p.json")

    # ─── 59p: per-det a + per-det C₀ + per-det amp₀ ───
    p0_59 = np.concatenate([
        a_det42, C0_det42,                   # use 42p result as init
        np.full(18, amp0_42),                # amp₀_det all shared init
        [alpha42, mu_m42, k_m42, mu_t42, k_t42],
    ])
    lo_59 = np.concatenate([
        np.full(18, 10.0), np.full(18, -300.0), np.full(18, 0.0),
        [0.1, 30.0, 0.5, 0.5, 0.2],
    ])
    hi_59 = np.concatenate([
        np.full(18, 500.0), np.full(18, 300.0), np.full(18, 1.0),
        [10.0, 60.0, 15.0, 8.0, 8.0],
    ])
    x59, std59 = fit(model_59, p0_59, lo_59, hi_59, C_clean, mask, mlat, t_yr, "59p (a_det + C₀_det + amp₀_det)")
    a_det59 = x59[:18]; C0_det59 = x59[18:36]; amp0_det59 = x59[36:54]
    alpha59, mu_m59, k_m59, mu_t59, k_t59 = x59[54:59]
    print(f"  a_det range: {a_det59.min():.1f} - {a_det59.max():.1f}")
    print(f"  C₀_det range: {C0_det59.min():.1f} - {C0_det59.max():.1f}")
    print(f"  amp₀_det range: {amp0_det59.min():.3f} - {amp0_det59.max():.3f}")
    print(f"  shared: α={alpha59:.3f}, μ_m={mu_m59:.2f}°, k_m={k_m59:.2f}°, "
          f"μ_t={mu_t59:.2f}yr, k_t={k_t59:.2f}yr")
    Path("/tmp/per_det_59p.json").write_text(json.dumps({
        "a_det": a_det59.tolist(), "C0_det": C0_det59.tolist(),
        "amp0_det": amp0_det59.tolist(),
        "alpha": float(alpha59), "mu_m": float(mu_m59), "k_m": float(k_m59),
        "mu_t": float(mu_t59), "k_t": float(k_t59),
    }))
    print("  Saved /tmp/per_det_59p.json")

    print("\n=== Heatmap std summary ===")
    print(f"  8p (single a):           6.49 cnt/s  (mlat-only heatmap, 60×108)")
    print(f"  25p (per-det a):        12.17 cnt/s  (per-det heatmap, 18×60×108)")
    print(f"  42p (per-det a + C₀):   {std42:6.2f} cnt/s  (per-det heatmap)")
    print(f"  59p (+ amp₀_det):       {std59:6.2f} cnt/s  (per-det heatmap)")


if __name__ == "__main__":
    main()
