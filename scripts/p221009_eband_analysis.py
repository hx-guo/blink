#!/usr/bin/env python3
"""GRB 221009A: pile-up diagnostics and hardness ratios from HE 1B (observed events).

Reads data/p221009/eband/ts_1s.csv (p221009_eband_timeseries.py) and
data/p221009/eband/events_obs.npz (p221009_eband_decode.py).

Pile-up indicator (instrumental, spectrum-independent): among events at a FIXED
pulse-height channel (raw ch 60-140, CsI ~170-430 keV), the fraction whose
pulse width is >= 110 (2.3 us). The intrinsic pulse width at fixed amplitude is
a crystal/electronics property; a source that is merely harder does not widen
it. Widening at fixed channel means a second pulse overlapped the digitised one.

Hardness ratios (CsI bands, net of the pre-burst background):
  HR1 = C(150-300) / C(70-150),  HR2 = C(300-700) / C(150-300)   [keV deposited]
from observed events only (FIFO loss is energy-blind; fillers never used).

Writes: data/p221009/eband/hr_1s.csv, fit summary to stdout.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/p221009/eband")
BKG = (-55, -5)
HE_OFF = [(230, 249), (268, 277)]
PU_CH = (60, 140)


def pileup_indicator(edges):
    ev = np.load(OUT / "events_obs.npz")
    t = ev["met"] - 339945423.0
    ch = ev["raw_ch"].astype(int)
    pw = ev["pw"].astype(int)
    m = (ch >= PU_CH[0]) & (ch < PU_CH[1])
    n = np.histogram(t[m], edges)[0]
    w = np.histogram(t[m & (pw >= 110)], edges)[0]
    return n, w


def bkg_linear_1k(bands, win=(-400, -5)):
    """Per-band linear background fit on the pre-burst 1K events (no saturation there)."""
    from astropy.io import fits
    d = fits.open("data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS",
                  memmap=True)[1].data
    t = d["Time"] - 339945423.0
    s = (t >= win[0]) & (t < win[1])
    t, ch, pw = t[s], d["Channel"][s].astype(int), d["Pulse_Width"][s].astype(int)
    lut = np.load("data/hxmt_aux/csi_ch2e_211211.npy")
    e = lut[np.clip(ch, 0, 255)]
    csi = (pw >= 90) & (pw <= 255) & (ch >= 30)
    ed = np.arange(win[0], win[1] + 1, 1.0)
    x = ed[:-1] + 0.5
    out = {}
    for b in bands:
        lo, hi = map(int, b[1:].split("_"))
        h = np.histogram(t[csi & (e >= lo) & (e < hi)], ed)[0]
        out[b] = np.polyfit(x, h, 1)
    return out


def main(bkg_mode="const"):
    ts = pd.read_csv(OUT / "ts_1s.csv")
    a = ts.groupby("t").sum(numeric_only=True)
    a["flive"] = ts.groupby("t").flive.mean()
    t = a.index.to_numpy()
    edges = np.r_[t, t[-1] + 1]
    n_pu, w_pu = pileup_indicator(edges)
    a["pu_n"], a["pu_w"] = n_pu, w_pu
    bk = (t >= BKG[0]) & (t < BKG[1])
    f_bkg = w_pu[bk].sum() / n_pu[bk].sum()
    a["pu_frac"] = np.where(n_pu > 0, w_pu / np.maximum(n_pu, 1), np.nan)
    a["pu_excess"] = a.pu_frac - f_bkg

    # input rate per detector: dead-time-corrected gross rate / 18 (paper eq. 1)
    a["R_in"] = np.where(a.flive > 0.05, a.r_all / a.flive, np.nan)
    a["n_det"] = a.R_in / 18.0

    # net CsI band counts from observed events; background per observed event:
    # the observed stream is a thinned sample of (src + bkg), so scale the
    # pre-burst band background by o_all/(gross recorded) -> use band fractions.
    bands = ["c70_150", "c150_300", "c300_700"]
    B_tot = a.r_all[bk].mean()
    lin = bkg_linear_1k(bands) if bkg_mode == "linear" else None
    for b in bands:
        B_b = a["o_" + b][bk].mean()                       # bkg counts/s in band
        if lin is not None:
            B_b = np.polyval(lin[b], t + 0.5) * a["o_" + b][bk].mean() / np.polyval(lin[b], t[bk] + 0.5).mean()
        thin = a.o_all / np.maximum(a.r_all, 1)            # FIFO thinning of the second
        bkg_obs = B_b * thin * a.flive / a.flive[bk].mean()  # bkg recorded, thinned
        a["net_" + b] = a["o_" + b] - bkg_obs
        a["var_" + b] = a["o_" + b]                        # Poisson on observed counts
    a["src_frac"] = 1 - B_tot * a.flive / a.flive[bk].mean() / np.maximum(a.r_all, 1)
    for name, num, den in (("HR1", "c150_300", "c70_150"), ("HR2", "c300_700", "c150_300")):
        N, D = a["net_" + num], a["net_" + den]
        a[name] = N / D
        a[name + "_err"] = np.abs(a[name]) * np.sqrt(a["var_" + num] / N**2 + a["var_" + den] / D**2)
    off = np.zeros(len(t), bool)
    for lo, hi in HE_OFF:
        off |= (t >= lo) & (t < hi)
    a["he_off"] = off | (a.r_all < 10)
    a.reset_index().to_csv(OUT / f"hr_1s_{bkg_mode}.csv", index=False)

    # ---- empirical pile-up time: pu_excess = A (1 - exp(-n tau)) ----
    from scipy.optimize import curve_fit
    g = (~a.he_off) & a.n_det.notna() & (a.pu_n > 2000)
    x, y = a.n_det[g].to_numpy(), a.pu_excess[g].to_numpy()
    e = np.sqrt(a.pu_frac[g] * (1 - a.pu_frac[g]) / a.pu_n[g]).to_numpy() + 1e-3
    p, cov = curve_fit(lambda n, A, tau: A * (1 - np.exp(-n * tau)), x, y,
                       p0=(0.5, 2e-6), sigma=e, bounds=([0, 0], [1, 1e-4]))
    print(f"pile-up indicator: bkg frac {f_bkg:.4f}; fit A={p[0]:.3f}, "
          f"tau_eff={p[1]*1e6:.2f} us  (n_det range {x.min():.0f}-{x.max():.0f}/s)")

    # ---- per-phase summary (pooled counts) ----
    phases = {
        "pre-burst bkg": (-55, -5), "precursor": (0, 6),
        "main rise (clean)": (180, 216), "P1 rise": (219, 226), "P1 peak": (226, 229),
        "P2 rise": (252, 257), "P2 peak": (257, 264), "P2 decay": (264, 268),
        "post-P2 (clean)": (277, 300), "flare (clean, excl. 508-512)": (440, 508),
        "flare peak": (508, 512), "flare decay (clean)": (512, 600), "tail": (600, 700),
    }
    rows = []
    for k, (lo, hi) in phases.items():
        s = (t >= lo) & (t < hi) & ~a.he_off.to_numpy()
        r = {"phase": k, "t": f"{lo}-{hi}", "flive_min": a.flive[s].min(),
             "n_det_max": a.n_det[s].max(), "pu_excess": (w_pu[s].sum() / n_pu[s].sum() - f_bkg),
             "src_frac": a.src_frac[s].median()}
        for name, num, den in (("HR1", "c150_300", "c70_150"), ("HR2", "c300_700", "c150_300")):
            N, D = a["net_" + num][s].sum(), a["net_" + den][s].sum()
            vn, vd = a["var_" + num][s].sum(), a["var_" + den][s].sum()
            r[name] = N / D
            r[name + "_err"] = abs(N / D) * np.sqrt(vn / N**2 + vd / D**2)
        rows.append(r)
    summ = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))
    summ.to_csv(OUT / f"phase_summary_{bkg_mode}.csv", index=False)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "const")
