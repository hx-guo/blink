#!/usr/bin/env python3
"""GRB 221009A: split of pile-up into f_live-compensated and uncompensated count loss.

Per 1-s bin:
  n_corr      dead-time-corrected input rate per detector (r_all / f_live / 18)
  P_pile_pw   fraction of recorded events carrying a second pulse, from the pulse-width
              indicator (wide-pulse excess at fixed channel / its asymptote A = 0.338)
  P_pile_mod  1 - exp(-n_corr * w), w = 2.1 us (CsI pulse width): second photon arrives
              AFTER the trigger, i.e. inside the box busy -> compensated by 1/f_live
  B_rise      1 - exp(-n_corr * tau_rise), tau_rise = 0.1-0.3 us: second photon in the
              leading edge before the threshold crossing -> merged, NOT compensated
  C_lo, C_hi  uncompensated loss from the per-detector share compression:
              C_lo = loss if only detectors above the box's least-affected level lose
                     (scale fixed by the 3 detectors with the lowest reference share)
              C_hi = paralyzable-occupancy model of p221009_eband_perdet.py (1 - S)
Reference shares: 200-216 s for P1, 250-256 s for P2, 500-508 s for the flare,
180-216 s otherwise.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from p221009_eband_perdet import load, fit_x, survival, norm  # noqa: E402

OUT = Path("data/p221009/eband")
A_PU, W_US = 0.338, 2.1
BINS = [185, 195, 216, 218, 220, 222, 224, 225, 226, 227, 228,
        255, 256, 257, 258, 260, 262, 263, 264, 510]


def ref_for(t):
    if 216 <= t < 230:
        return (200, 216)
    if 250 <= t < 268:
        return (250, 256)
    if 500 <= t < 515:
        return (500, 508)
    return (180, 216)


def c_lo(obs, rho):
    out = []
    for b in range(3):
        o, r = obs[6*b:6*b+6], rho[6*b:6*b+6]
        low = np.argsort(r)[:3]
        k = np.median(o[low] / r[low])
        out.append(1 - o.sum() / (k * r.sum()))
    return max(0.0, float(np.mean(out)))


def main():
    x, ex, hr = load()
    h = pd.read_csv(OUT / "hr_1s_const.csv").set_index("t")
    rows = []
    for tb in BINS:
        j = int(np.where(x == tb)[0][0])
        lo, hi = ref_for(tb)
        rho = norm(ex[:, (x >= lo) & (x < hi)].sum(1))
        o = ex[:, j]
        n = h.r_all[tb] / h.flive[tb] / 18
        xv, _ = fit_x(o, rho)
        rows.append(dict(
            t=tb, flive=h.flive[tb], n_corr=n,
            P_pile_pw=h.pu_excess[tb] / A_PU,
            P_pile_mod=1 - np.exp(-n * W_US * 1e-6),
            B_rise=f"{1-np.exp(-n*0.1e-6):.3f}-{1-np.exp(-n*0.3e-6):.3f}",
            C_lo=c_lo(o, rho), C_hi=1 - survival(rho, xv)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "countloss.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
