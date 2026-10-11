#!/usr/bin/env python3
"""X-ray sub-peak times of the HXMT/HE recovered light curve relative to the
two CHIME radio pulses of FRB 200428 (independent of the HXMT-IBIS lag).

Radio reference (CHIME/FRB Collaboration 2020, Nature 587, 54, Table 1,
arXiv:2005.10324v2): arrival times at INFINITE frequency (DM 332.7206),
GEOCENTRIC: 14:34:24.42650(2) and 14:34:24.45547(2) UTC.  Our T0 is
14:34:24.011 UTC geocentric, so the pulses sit at 0.41550 and 0.44447 s.

Sign convention: delay > 0  <=>  X-ray later than radio.

Peak-time estimators on the 1-ms net light curve (all deterministic):
  * local maximum of the 3-bin boxcar within [radio, radio + 15 ms]
  * centroid of the net counts in +/-15 ms around that maximum (Poisson err)
  * weighted parabola through the 5 unsmoothed bins around the maximum
  * weighted centre of the contiguous region above half maximum (baseline =
    minimum of the +/-15 ms window)
Errors are propagated from Poisson counts (fill events treated as Poisson);
they are statistical only and do not include reconstruction systematics.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import talk20_lag_200428 as L  # noqa: E402

OUT = L.OUT
BIN = L.BIN
T0_OFF = 0.011                                   # our T0 - 14:34:24.000 UTC
RADIO = {"P1": 0.42650 - T0_OFF, "P2": 0.45547 - T0_OFF}   # CHIME Table 1, geocentric
RADIO_ERR = 0.00002
HALF = 0.015
SEARCH = 0.015                                   # local-max search: [radio, radio+SEARCH]

# Literature X-ray peak delays (ms, X-ray minus radio, same pulse)
LIT = {
    "Ge+2023 HE multi-Gaussian (15-250 keV)": (2.77, 0.45, 34.30 - 28.97, np.hypot(0.56, 0.02)),
    "Ge+2023 all-instrument weighted": (2.92, 0.43, 6.20, 0.43),
    "Mereghetti+2020 IBIS t1/t2 - CHIME (0.42648/0.45545)":
        ((0.434 - 0.42648) * 1e3, 3.0, (0.462 - 0.45545) * 1e3, 1.0),
    "Li+2021 HXMT peaks quoted by CHIME (14:34:24.4289/.4589)":
        ((0.4289 - 0.42650) * 1e3, np.nan, (0.4589 - 0.45547) * 1e3, np.nan),
}


def curve(t, lo, hi, bkg):
    edges = np.arange(lo, hi + BIN / 2, BIN)
    x = 0.5 * (edges[1:] + edges[:-1])
    raw = L.hist(t, edges)
    return x, raw, raw - bkg


def peak_estimators(t, radio, bkg):
    lo, hi = radio - 0.040, radio + 0.055
    x, raw, net = curve(t, lo, hi, bkg)
    sm = np.convolve(net, np.ones(3) / 3.0, mode="same")
    sel = (x >= radio) & (x <= radio + SEARCH)
    k = int(np.flatnonzero(sel)[np.argmax(sm[sel])])
    t_max = x[k]
    R = dict(x=x, raw=raw, net=net, t_max=t_max)

    # centroid +/-15 ms around the local maximum
    w = (x >= t_max - HALF - 1e-9) & (x <= t_max + HALF + 1e-9)
    N = net[w].sum()
    c = np.sum(net[w] * x[w]) / N
    R["cen"] = c
    R["cen_err"] = np.sqrt(np.sum(raw[w] * (x[w] - c) ** 2)) / N
    R["cen_win"] = (t_max - HALF, t_max + HALF)

    # weighted parabola through 5 bins around the maximum (unsmoothed)
    idx = np.arange(k - 2, k + 3)
    xs, ys, vs = x[idx] - t_max, net[idx], np.clip(raw[idx], 1, None)
    A = np.vstack([xs ** 2, xs, np.ones_like(xs)]).T
    Wm = np.diag(1.0 / vs)
    cov = np.linalg.inv(A.T @ Wm @ A)
    p = cov @ A.T @ Wm @ ys
    a, b = p[0], p[1]
    v = -b / (2 * a)
    g = np.array([b / (2 * a * a), -1.0 / (2 * a), 0.0])
    R["par"] = t_max + v
    R["par_err"] = np.sqrt(g @ cov @ g) if a < 0 else np.nan
    R["par_ok"] = a < 0

    # half-maximum weighted centre
    base = net[w].min()
    thr = base + 0.5 * (net[k] - base)
    i0 = k
    while i0 > 0 and net[i0 - 1] >= thr:
        i0 -= 1
    i1 = k
    while i1 < len(net) - 1 and net[i1 + 1] >= thr:
        i1 += 1
    hm = slice(i0, i1 + 1)
    wt = net[hm] - base
    Nh = wt.sum()
    ch = np.sum(wt * x[hm]) / Nh
    R["hm"] = ch
    R["hm_err"] = np.sqrt(np.sum(raw[hm] * (x[hm] - ch) ** 2)) / Nh
    R["hm_win"] = (x[i0], x[i1])
    return R


def main():
    D = L.load_all()
    bkg_h = L.bkg_rate_per_bin(D["hxmt_obs"])
    bkg_i = L.bkg_rate_per_bin(D["ibis"])
    lines = []

    def say(s=""):
        print(s); lines.append(s)

    say("")
    say("# X-ray sub-peaks vs CHIME radio pulses (independent of the HXMT-IBIS lag)")
    say()
    say("- Radio reference: CHIME/FRB 2020 Table 1, infinite-frequency (DM 332.7206) GEOCENTRIC arrival "
        "times 14:34:24.42650(2) and 14:34:24.45547(2) UTC -> on our T0 (14:34:24.011 geocentric): "
        f"P1 = {RADIO['P1']:.5f} s, P2 = {RADIO['P2']:.5f} s (separation 28.97 ms).")
    say("- Provenance of RADIO_PEAKS in plot_hxmt_vs_ibis_bands.py (0.4155, 0.4155+0.0289): P1 matches the "
        "CHIME geocentric time (14:34:24.4265 = Ge+2023 T0); P2 used 28.9 ms instead of 28.97 ms "
        "(0.07 ms off, negligible).  Both already geocentric + infinite frequency; no further correction.")
    say("- HXMT events carry the +11.37 ms HXMT->geocentre projection; sign: delay > 0 = X-ray later than radio.")
    say(f"- Estimators on the 1-ms net curve: local max of 3-bin boxcar in [radio, radio+{SEARCH*1e3:.0f} ms]; "
        f"centroid +/-{HALF*1e3:.0f} ms around it; weighted parabola (5 bins); half-max weighted centre. "
        "Errors: Poisson statistics only (fill treated as Poisson), no reconstruction systematics.")
    say()
    say("| curve | pulse | t_max (s) | centroid (s) | delay_cen (ms) | parabola (s) | delay_par (ms) | half-max centre (s) | delay_hm (ms) | HM region (ms) |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    curves = [("HXMT recovered (EVT+fill)", D["hxmt_all"], bkg_h),
              ("HXMT observed only (no fill)", D["hxmt_obs"], bkg_h),
              ("IBIS-ISGRI 20-200 keV", D["ibis"], bkg_i)]
    res = {}
    for cname, t, bkg in curves:
        for pk, tr in RADIO.items():
            R = peak_estimators(t, tr, bkg)
            res[(cname, pk)] = R
            dc, dp, dh = (R["cen"] - tr) * 1e3, (R["par"] - tr) * 1e3, (R["hm"] - tr) * 1e3
            say(f"| {cname} | {pk} | {R['t_max']:.4f} | {R['cen']:.4f} | {dc:+.2f} +/- {R['cen_err']*1e3:.2f} | "
                f"{R['par']:.4f} | {dp:+.2f} +/- {R['par_err']*1e3:.2f} | {R['hm']:.4f} | "
                f"{dh:+.2f} +/- {R['hm_err']*1e3:.2f} | [{R['hm_win'][0]*1e3:.0f}, {R['hm_win'][1]*1e3:.0f}] |")
    say()
    say("## Literature (ms, X-ray minus the SAME radio pulse)")
    say()
    say("| source | P1 delay | P2 delay |")
    say("|---|---|---|")
    for k, (a, ea, b, eb) in LIT.items():
        say(f"| {k} | {a:+.2f} +/- {ea:.2f} | {b:+.2f} +/- {eb:.2f} |")
    say("(Ge+2023 second number 34.30 +/- 0.56 ms is w.r.t. the FIRST radio pulse; minus 28.97 ms -> +5.33 w.r.t. P2.)")
    say()

    # ---------------- figure ----------------
    Rall = {pk: res[("HXMT recovered (EVT+fill)", pk)] for pk in RADIO}
    Robs = {pk: res[("HXMT observed only (no fill)", pk)] for pk in RADIO}
    lo, hi = 0.395, 0.475
    x, raw, net = curve(D["hxmt_all"], lo, hi, bkg_h)
    xo, rawo, neto = curve(D["hxmt_obs"], lo, hi, bkg_h)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.step(x * 1e3, neto / BIN, where="mid", color="C0", lw=0.9, alpha=0.45, label="HXMT/HE observed only")
    ax.step(x * 1e3, net / BIN, where="mid", color="C0", lw=1.4, label="HXMT/HE recovered (EVT + fill)")
    ymax = net.max() / BIN * 1.15
    for pk, tr in RADIO.items():
        ax.axvline(tr * 1e3, color="C2", ls="--", lw=1.3)
        ax.text(tr * 1e3 - 0.6, ymax * 0.97, f"CHIME {pk}", color="C2", ha="right", va="top", fontsize=9)
        R = Rall[pk]
        ax.axvspan(R["cen_win"][0] * 1e3, R["cen_win"][1] * 1e3, color="0.85", alpha=0.35, lw=0)
        for key, col, lab, dy in (("cen", "C3", "centroid", 0.90), ("par", "C1", "parabola", 0.84),
                                  ("hm", "C4", "half-max", 0.78)):
            v, e = R[key] * 1e3, R[key + "_err"] * 1e3
            ax.errorbar(v, ymax * dy, xerr=e, fmt="o", color=col, ms=4, capsize=2, lw=1)
            ax.text(v + 0.5, ymax * dy, f"{lab} {v - tr*1e3:+.1f} ms", color=col, fontsize=7.5, va="center")
    ax.set_xlim(lo * 1e3, hi * 1e3)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("t - T0 (ms), geocentric, T0 = 14:34:24.011 UTC")
    ax.set_ylabel("net rate (events s$^{-1}$)")
    ax.set_title("FRB/XRB 200428: HXMT/HE recovered sub-peaks vs CHIME radio pulses (1 ms)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"xray_radio_200428.{ext}", dpi=200, bbox_inches="tight")
    say(f"Figure: {OUT}/xray_radio_200428.pdf/.png")
    with open(OUT / "RESULTS.md", "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
