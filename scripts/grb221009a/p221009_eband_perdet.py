#!/usr/bin/env python3
"""GRB 221009A: per-detector, rate-dependent loss not covered by the box live fraction.

The box live fraction f_live (DeadTime counter = busy of the box's single shared
processor) compensates photons arriving while the box is busy, the same fraction for
all six detectors of the box. A loss that depends on each detector's OWN rate
(pulse occupancy of its discriminator / per-channel end window / baseline shift) is
not compensated. Its signature: detectors that see more of the burst lose a larger
fraction, so the spread of per-detector shares compresses (or inverts) at high rate.

Model per detector i in a box: obs_i ∝ rho_i exp(-rho_i x), x = n_bar*tau_occ,
rho_i = per-detector share at low rate (reference phase), n_bar = box-mean true rate.
Survival (uncompensated) S(x) = mean(rho e^{-rho x}) / mean(rho).

Inputs : data/p221009/eband/events_obs.npz, hr_1s_const.csv
Output : data/p221009/eband/perdet_loss.csv (per second), stdout summary
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

OUT = Path("data/p221009/eband")
REFS = {"main rise": (180, 216), "P2 rise": (250, 256), "post-P2": (277, 300),
        "flare": (440, 508), "flare decay": (512, 600)}
TEST = {"P1 rise": (219, 226), "P1 peak": (226, 229), "P2 peak": (257, 264),
        "P2 decay": (264, 268), "flare peak": (508, 512)}


def load():
    d = np.load(OUT / "events_obs.npz")
    t = d["met"] - 339945423.0
    det = d["det"].astype(int)
    ed = np.arange(-60, 761, 1.0)
    H = np.vstack([np.histogram(t[det == i], ed)[0] for i in range(18)]).astype(float)
    x = ed[:-1]
    hr = pd.read_csv(OUT / "hr_1s_const.csv").set_index("t").reindex(x)
    bk = (x >= -55) & (x < -5)
    thin = (hr.o_all / np.maximum(hr.r_all, 1)).to_numpy()
    fl = hr.flive.to_numpy()
    ex = H - np.outer(H[:, bk].mean(1), thin * fl / np.nanmean(fl[bk]))
    return x, ex, hr


def fit_x(obs, rho):
    """obs, rho: (18,) -> x minimising chi2 of box-normalised shares (Poisson weights)."""
    def chi2(xv):
        c = 0.0
        for b in range(3):
            o, r = obs[6*b:6*b+6], rho[6*b:6*b+6]
            m = r * np.exp(-r * xv)
            m = m / m.sum() * o.sum()
            c += np.sum((o - m) ** 2 / np.maximum(m, 1))
        return c
    res = minimize_scalar(chi2, bounds=(0, 5), method="bounded")
    return res.x, res.fun


def survival(rho, xv):
    s = []
    for b in range(3):
        r = rho[6*b:6*b+6]
        s.append(np.mean(r * np.exp(-r * xv)) / np.mean(r))
    return float(np.mean(s))


def norm(e):
    return np.concatenate([e[6*b:6*b+6] / e[6*b:6*b+6].mean() for b in range(3)])


def main():
    x, ex, hr = load()
    rhos = {k: norm(ex[:, (x >= lo) & (x < hi)].sum(1)) for k, (lo, hi) in REFS.items()}
    # leave-one-out consistency of the references themselves (should give x ~ 0)
    print("reference phases fitted against each other (x should be ~0 if shares are stable):")
    for k, (lo, hi) in REFS.items():
        o = ex[:, (x >= lo) & (x < hi)].sum(1)
        xs = [fit_x(o, r)[0] for kk, r in rhos.items() if kk != k]
        print(f"  {k:12s} x = {np.min(xs):.3f}..{np.max(xs):.3f}")
    print("\nphase        n_corr/det   x range (5 refs)     tau_occ (us)      uncompensated survival S   1/S")
    rows = []
    for k, (lo, hi) in TEST.items():
        s = (x >= lo) & (x < hi)
        o = ex[:, s].sum(1)
        ncorr = np.nanmean((hr.r_all / hr.flive).to_numpy()[s]) / 18
        xs, ss = [], []
        for kk, r in rhos.items():
            xv, _ = fit_x(o, r)
            xs.append(xv); ss.append(survival(r, xv))
        # n_bar(true) = n_corr / S  -> tau = x * S / n_corr
        taus = [xv * sv / ncorr * 1e6 for xv, sv in zip(xs, ss)]
        print(f"{k:12s} {ncorr:10.3g}   {min(xs):.2f}–{max(xs):.2f}   "
              f"{min(taus):.2f}–{max(taus):.2f}   {min(ss):.2f}–{max(ss):.2f}   "
              f"{1/max(ss):.2f}–{1/min(ss):.2f}")
        rows.append(dict(phase=k, n_corr=ncorr, x_min=min(xs), x_max=max(xs),
                         tau_min_us=min(taus), tau_max_us=max(taus),
                         S_min=min(ss), S_max=max(ss)))
    # per-second fit with the pooled reference
    rho = norm(sum(ex[:, (x >= lo) & (x < hi)].sum(1) for lo, hi in REFS.values()))
    per = []
    for j, tt in enumerate(x):
        o = ex[:, j]
        if o.sum() < 3000 or not np.isfinite(hr.flive.iloc[j]):
            continue
        xv, c2 = fit_x(o, rho)
        per.append(dict(t=tt, n_corr=hr.r_all.iloc[j] / hr.flive.iloc[j] / 18, x=xv,
                        S=survival(rho, xv), chi2=c2, flive=hr.flive.iloc[j]))
    pd.DataFrame(per).to_csv(OUT / "perdet_loss.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "perdet_loss_phases.csv", index=False)


if __name__ == "__main__":
    main()
