#!/usr/bin/env python3
"""27-param model: shared time-sigmoid σ_t(t) controls BOTH amplitude decay
AND mlat shape evolution (μ_m, k_m drift).

  σ_t(t) = sigmoid((t − μ_t)/k_t)
  μ_m(t) = μ_m_0 + Δμ · σ_t(t)
  k_m(t) = k_m_0 + Δk · σ_t(t)
  g(mlat, t) = 1 + α · sigmoid((mlat − μ_m(t)) / k_m(t))
  C(det, mlat, t) = a_det · g(mlat,t) · [1 − amp₀ · g(mlat,t) · σ_t(t)] + C₀

27 params: a_det[18], α, μ_m_0, k_m_0, Δμ, Δk, amp₀, μ_t, k_t, C₀
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

NPZ = "n_below_study/v5_npz/C_det_mlat_t_heatmap.npz"


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def model(p, mlat, t):
    a_det = p[:18]
    alpha, mu_m0, k_m0, dmu, dk, amp0, mu_t, k_t, C0 = p[18:27]
    st = sigm((t - mu_t) / k_t)                       # (n_t,)
    mu_m_t = mu_m0 + dmu * st                          # (n_t,)
    k_m_t = k_m0 + dk * st                             # (n_t,)
    sm = sigm((mlat[None, :, None] - mu_m_t[None, None, :]) / k_m_t[None, None, :])  # (1, n_mlat, n_t)
    g = 1.0 + alpha * sm                               # (1, n_mlat, n_t)
    return (a_det[:, None, None] * g
            * (1.0 - amp0 * g * st[None, None, :]) + C0)


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

    def residual(p):
        return ((model(p, mlat, t_yr) - C_clean) * mask).ravel()

    p0 = np.concatenate([
        np.full(18, 200.0),               # a_det
        [1.7, 44.5, 6.4, -3.5, -0.5, 0.15, 5.25, 1.0, -80.0],
    ])
    lo = np.concatenate([
        np.full(18, 10.0),
        [0.1, 30.0, 0.5, -15.0, -10.0, 0.0, 0.5, 0.2, -300.0],
    ])
    hi = np.concatenate([
        np.full(18, 500.0),
        [10.0, 60.0, 15.0, +15.0, +10.0, 1.0, 8.0, 8.0, +300.0],
    ])
    print(f"Fitting {len(p0)} params...")
    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=10000)
    print(f"  cost={res.cost:.0f}, nfev={res.nfev}, success={res.success}")

    a_det = res.x[:18]
    alpha, mu_m0, k_m0, dmu, dk, amp0, mu_t, k_t, C0 = res.x[18:27]
    print("\n=== Fitted parameters (27-param shape-drift) ===")
    print(f"  a_det:   range {a_det.min():.1f} - {a_det.max():.1f}, median {np.median(a_det):.1f}")
    print(f"  α        = {alpha:.3f}")
    print(f"  μ_m_0    = {mu_m0:.2f}°       (early plateau)")
    print(f"  k_m_0    = {k_m0:.2f}°        (early plateau)")
    print(f"  Δμ       = {dmu:+.3f}°       (μ_m drift)")
    print(f"  Δk       = {dk:+.3f}°        (k_m drift)")
    print(f"  μ_m_late = {mu_m0+dmu:.2f}°   (late plateau)")
    print(f"  k_m_late = {k_m0+dk:.2f}°    (late plateau)")
    print(f"  amp₀     = {amp0:.3f}")
    print(f"  μ_t      = {mu_t:.2f} yr = "
          f"{(t0 + np.timedelta64(int(mu_t*365.25), 'D')).astype(str)}")
    print(f"  k_t      = {k_t:.2f} yr")
    print(f"  C_0      = {C0:+.2f}")

    Cp = model(res.x, mlat, t_yr)
    r = (C_data - Cp) * mask
    valid = r[mask]
    print(f"\n=== Heatmap residual stats ===")
    print(f"  std: {np.std(valid):.3f} cnt/s")
    print(f"  cf. 25p (no shape drift): 12.17 cnt/s")

    Path("/tmp/per_det_27p.json").write_text(json.dumps({
        "a_det": a_det.tolist(), "alpha": float(alpha),
        "mu_m0": float(mu_m0), "k_m0": float(k_m0),
        "dmu": float(dmu), "dk": float(dk),
        "amp0": float(amp0), "mu_t": float(mu_t), "k_t": float(k_t),
        "C0": float(C0),
    }))
    print("Saved /tmp/per_det_27p.json")


if __name__ == "__main__":
    main()
