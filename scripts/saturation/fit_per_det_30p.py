#!/usr/bin/env python3
"""30-param model: α and β decoupled, plus independent σ_drift(t) for mlat shape drift.

  σ_t(t)      = sigmoid((t - μ_t1)/k_t1)        [amplitude decay]
  σ_drift(t)  = sigmoid((t - μ_t2)/k_t2)        [mlat shape drift, independent]
  μ_m(t)      = μ_m_0 + Δμ · σ_drift(t)
  k_m(t)      = k_m_0 + Δk · σ_drift(t)
  σ_m(m, t)   = sigmoid((m - μ_m(t)) / k_m(t))
  g_b(m, t)   = 1 + α · σ_m(m, t)               [mlat baseline]
  g_d(m, t)   = 1 + β · σ_m(m, t)               [mlat decay weight, decoupled]
  C(det, m, t) = a_det · g_b · [1 − amp₀ · g_d · σ_t(t)] + C₀

30 params: a_det[18], α, β, μ_m_0, k_m_0, Δμ, Δk, amp₀, μ_t1, k_t1, μ_t2, k_t2, C₀
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
    alpha, beta, mu_m0, k_m0, dmu, dk, amp0, mu_t1, k_t1, mu_t2, k_t2, C0 = p[18:30]
    st = sigm((t - mu_t1) / k_t1)                              # (n_t,) amp decay
    sd = sigm((t - mu_t2) / k_t2)                              # (n_t,) shape drift
    mu_m_t = mu_m0 + dmu * sd                                   # (n_t,)
    k_m_t = k_m0 + dk * sd                                      # (n_t,)
    # broadcast: mlat (n_m,), mu_m_t/k_m_t (n_t,)
    sm = sigm((mlat[None, :, None] - mu_m_t[None, None, :])     # (1, n_m, n_t)
              / k_m_t[None, None, :])
    g_b = 1.0 + alpha * sm                                       # (1, n_m, n_t)
    g_d = 1.0 + beta * sm                                        # (1, n_m, n_t)
    return (a_det[:, None, None] * g_b
            * (1.0 - amp0 * g_d * st[None, None, :]) + C0)


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

    # Init: α=β=1.7 (from 25p), μ_t1=5.25 (amp center), μ_t2=5.8 (drift center),
    # k_t1=1.0, k_t2=0.5 (drift sharper), Δμ=-3.5, Δk=-0.6 from per-year fit
    p0 = np.concatenate([
        np.full(18, 200.0),
        [1.7, 1.7, 44.5, 6.4, -3.5, -0.6, 0.15, 5.25, 1.0, 5.8, 0.5, -80.0],
    ])
    lo = np.concatenate([
        np.full(18, 10.0),
        [0.1, 0.1, 30.0, 0.5, -15.0, -10.0, 0.0, 0.5, 0.1, 0.5, 0.1, -300.0],
    ])
    hi = np.concatenate([
        np.full(18, 500.0),
        [10.0, 10.0, 60.0, 15.0, +15.0, +10.0, 1.0, 8.0, 8.0, 8.0, 8.0, +300.0],
    ])
    print(f"Fitting {len(p0)} params...")
    res = least_squares(residual, p0, bounds=(lo, hi), method='trf', max_nfev=20000)
    print(f"  cost={res.cost:.0f}, nfev={res.nfev}, success={res.success}")

    a_det = res.x[:18]
    alpha, beta, mu_m0, k_m0, dmu, dk, amp0, mu_t1, k_t1, mu_t2, k_t2, C0 = res.x[18:30]
    print("\n=== Fitted parameters (30-param decoupled shape-drift) ===")
    print(f"  a_det:    range {a_det.min():.1f} − {a_det.max():.1f}, median {np.median(a_det):.1f}")
    print(f"  α (baseline mlat amp)  = {alpha:.3f}")
    print(f"  β (decay-weight mlat)  = {beta:.3f}    ← α ≈ β? diff={alpha-beta:+.3f}")
    print(f"  μ_m_0    = {mu_m0:.2f}°  (early plateau center)")
    print(f"  k_m_0    = {k_m0:.2f}°   (early plateau width)")
    print(f"  Δμ       = {dmu:+.3f}°  → μ_m_late = {mu_m0+dmu:.2f}°")
    print(f"  Δk       = {dk:+.3f}°   → k_m_late = {k_m0+dk:.2f}°")
    print(f"  amp₀     = {amp0:.3f}")
    print(f"  μ_t1     = {mu_t1:.2f} yr = "
          f"{(t0 + np.timedelta64(int(mu_t1*365.25), 'D')).astype(str)}  [amp decay center]")
    print(f"  k_t1     = {k_t1:.2f} yr  [amp decay width]")
    print(f"  μ_t2     = {mu_t2:.2f} yr = "
          f"{(t0 + np.timedelta64(int(mu_t2*365.25), 'D')).astype(str)}  [shape drift center]")
    print(f"  k_t2     = {k_t2:.2f} yr  [shape drift width]")
    print(f"  C_0      = {C0:+.2f}")

    Cp = model(res.x, mlat, t_yr)
    r = (C_data - Cp) * mask
    valid = r[mask]
    print(f"\n=== Heatmap residual stats ===")
    print(f"  std: {np.std(valid):.3f} cnt/s")
    print(f"  cf. 25p (α=β, no drift):     12.17 cnt/s")
    print(f"  cf. 27p (α=β, locked drift): 12.15 cnt/s")

    Path("/tmp/per_det_30p.json").write_text(json.dumps({
        "a_det": a_det.tolist(),
        "alpha": float(alpha), "beta": float(beta),
        "mu_m0": float(mu_m0), "k_m0": float(k_m0),
        "dmu": float(dmu), "dk": float(dk),
        "amp0": float(amp0),
        "mu_t1": float(mu_t1), "k_t1": float(k_t1),
        "mu_t2": float(mu_t2), "k_t2": float(k_t2),
        "C0": float(C0),
    }))
    print("Saved /tmp/per_det_30p.json")


if __name__ == "__main__":
    main()
