#!/usr/bin/env python3
"""Cross-correlation time lags for GRB 221009A: Insight-HXMT/HE vs GECAM-C.

Event-level CCF. Both event lists are put on a common 1 ms grid; a trial lag
tau = k ms is an index offset; the CCF is the Pearson coefficient of the two
series re-summed into BIN-wide bins. The peak is refined by a least-squares
parabola over +-W around the 1 ms grid maximum.

Convention: lag(a, b) = t(a) - t(b) at maximum correlation (lag > 0: a arrives
later than b).

Statistical error (deterministic): first-order Poisson propagation through the
peak condition C'(tau*) = 0,
    var(tau*) = [sum x'_i^2 var(y_i)/s_y^2 + sum y'_i^2 var(x_i)/s_x^2] / (N C'')^2,
with x, y the BIN-binned series, s their standard deviations, and x', y' the
time derivatives of their 3-bin running means. Cross-check: lags from the
independent HE boxes (A, B, C) against independent GECAM-C detector halves
(GRD01-06, GRD07-12).

GECAM-C event data (all 12 GRDs) exist for T0+330..678 s only; the main
emission is covered only by the 50 ms GRD01 BTIME product (part C).

Parts:
  A  HE vs GECAM-C in the matching band in 5 s sub-windows across the flare
     region: timing zero point, and its dependence on the HE gap-fill fraction
     and live fraction. During the burst HE ran in its low-gain mode and ~88%
     of its excess events are CsI events, so the all-channel HE light curve
     samples ~0.1-3 MeV deposited energy; the matching GECAM-C band is LG
     0.4-3 MeV (GRD01-12). GECAM-C HG 200-350 keV and 50-100 keV are kept as
     comparison bands (they carry the source's own spectral lag).
  B  GECAM-C lag-energy spectrum (GRD01-12) and HE lag-energy in the flare.
  C  HE (gross, reconstructed) vs GECAM-C GRD01 BTIME LG (0.38-6.3 MeV) and
     HG on the main pulses, 50 ms; reported for completeness.

Outputs: data/p221009/timing/lag_results.json, lag_flare_subwindows.png, lag_energy.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from p221009_timing_common import OUT, he_events, eng_1hz, gecam_btime

FINE = 0.001
MAXLAG = 0.5


def load():
    t, fill, ch, box = he_events()
    keep = ~((ch == 0) & (np.abs(t - np.round(t)) < 1e-7))  # 1 Hz marker records
    he = dict(t=t[keep], fill=fill[keep], ch=ch[keep], box=box[keep])
    g = np.load(OUT / "gecam_c.npz")
    eb = np.load(OUT / "gecam_ebounds.npz")
    e = np.sqrt(eb["emin"] * eb["emax"])[g["pi"]]
    gc = dict(t=g["t"], det=g["det"], gain=g["gain"], e=e)
    return he, gc


def fine_cumsum(t, lo, hi, pad):
    edges = lo - pad + FINE * np.arange(int(round((hi - lo + 2 * pad) / FINE)) + 1)
    return np.concatenate([[0], np.cumsum(np.histogram(t, edges)[0])])


def binned(cs, pad_bins, nb, step, shift):
    j = np.arange(nb) * step + pad_bins - shift
    return (cs[j + step] - cs[j]).astype(float)


def ccf_lag(ta, tb, lo, hi, bin_w=0.02, W=0.06, return_curve=False):
    """lag = t(a) - t(b) and its analytic 1-sigma error."""
    step = int(round(bin_w / FINE))
    nb = int(round((hi - lo) / bin_w))
    pad = MAXLAG + 0.1
    pb = int(round(pad / FINE))
    sel_a = (ta > lo - pad) & (ta < hi + pad)
    sel_b = (tb > lo - pad) & (tb < hi + pad)
    ca = fine_cumsum(ta[sel_a], lo, hi, pad)
    cb = fine_cumsum(tb[sel_b], lo, hi, pad)
    y = binned(cb, pb, nb, step, 0)
    ys = (y - y.mean()) / y.std()
    ks = np.arange(-int(MAXLAG / FINE), int(MAXLAG / FINE) + 1)
    c = np.empty(len(ks))
    for i, k in enumerate(ks):
        # a sampled k ms later  <=>  a shifted earlier by k  <=>  lag(a-b) = +k
        x = binned(ca, pb, nb, step, -k)
        c[i] = np.mean((x - x.mean()) / x.std() * ys)
    lags = ks * FINE
    i0 = int(np.argmax(c))
    m = np.abs(lags - lags[i0]) <= W
    p = np.polyfit(lags[m] - lags[i0], c[m], 2)
    tau = lags[i0] - p[1] / (2 * p[0]) if p[0] < 0 else np.nan
    if not np.isfinite(tau) or abs(lags[i0]) >= MAXLAG - W or abs(tau) >= MAXLAG - W:
        out = (np.nan, np.nan, float(c.max()))
        return out + ((lags, c) if return_curve else ())
    x = binned(ca, pb, nb, step, -int(round(tau / FINE)))
    sx, sy = x.std(), y.std()
    k3 = np.ones(3) / 3
    xp = np.gradient(np.convolve((x - x.mean()) / sx, k3, "same"), bin_w)
    yp = np.gradient(np.convolve(ys, k3, "same"), bin_w)
    c2 = 2 * p[0]  # curvature in 1/s^2 (lags in s)
    var = (np.sum(xp**2 * y / sy**2) + np.sum(yp**2 * x / sx**2)) / (nb * c2) ** 2
    out = (float(tau), float(np.sqrt(var)), float(c.max()))
    return out + ((lags, c) if return_curve else ())


def flive_mean(et, fl, lo, hi):
    m = (et >= np.floor(lo)) & (et < hi)
    return float(np.mean(fl[m]))


def part_a(he, gc, et, fl):
    """Same-band HE vs GECAM-C in 5 s sub-windows, T0+440..600 s."""
    he_sel = he["ch"] >= 20
    g_sel = (gc["gain"] == 1) & (gc["e"] >= 400) & (gc["e"] < 3000)
    rows = []
    for lo in np.arange(440.0, 600.0, 5.0):
        hi = lo + 5.0
        w = (he["t"] >= lo) & (he["t"] < hi)
        tau, err, cmax = ccf_lag(he["t"][he_sel], gc["t"][g_sel], lo, hi)
        sp = []
        for b in range(3):
            for half in ((1, 6), (7, 12)):
                gm = g_sel & (gc["det"] >= half[0]) & (gc["det"] <= half[1])
                sp.append(ccf_lag(he["t"][he_sel & (he["box"] == b)], gc["t"][gm], lo, hi)[0])
        sp = np.array(sp)
        nfin = int(np.isfinite(sp).sum())
        rows.append(dict(lo=float(lo), hi=float(hi), lag=tau, err=err, cmax=cmax,
                         split_mean=float(np.nanmean(sp)) if nfin else np.nan,
                         split_sem=float(np.nanstd(sp, ddof=1) / np.sqrt(nfin)) if nfin > 1 else np.nan,
                         fill=float(he["fill"][w].mean()),
                         flive=flive_mean(et, fl, lo, hi),
                         he_rate=float(w.sum() / 5.0)))
        r = rows[-1]
        print(f"A {lo:5.0f}-{hi:5.0f}  lag {tau*1e3:+6.1f} ± {err*1e3:4.1f} ms "
              f"(split {r['split_mean']*1e3:+6.1f} ± {r['split_sem']*1e3:4.1f})  "
              f"cmax {cmax:.2f}  fill {r['fill']:.2f}  flive {r['flive']:.3f}  "
              f"HE {r['he_rate']:.0f}/s", flush=True)
    # zero point from windows whose CCF peak is well defined (cmax >= 0.9:
    # strong sub-second structure); smooth windows have flat CCFs and are not used
    good = [r for r in rows if r["cmax"] >= 0.9 and np.isfinite(r["lag"])]
    wts = np.array([1 / r["err"] ** 2 for r in good])
    lags = np.array([r["lag"] for r in good])
    zp = float(np.sum(wts * lags) / wts.sum())
    summary = dict(zero_point=zp, zero_point_err=float(1 / np.sqrt(wts.sum())),
                   n=len(good), chi2=float(np.sum(wts * (lags - zp) ** 2)))
    sat = [r for r in rows if r["fill"] > 0.05 and r["cmax"] >= 0.9 and np.isfinite(r["lag"])]
    if sat:
        ws = np.array([1 / r["err"] ** 2 for r in sat])
        ls = np.array([r["lag"] for r in sat])
        summary.update(sat_mean=float(np.sum(ws * ls) / ws.sum()),
                       sat_err=float(1 / np.sqrt(ws.sum())), sat_n=len(sat),
                       sat_windows=[(r["lo"], r["hi"]) for r in sat])
    comp = {"LG 0.4-3 MeV": g_sel,
            "HG 200-350 keV": (gc["gain"] == 0) & (gc["e"] >= 200) & (gc["e"] < 350),
            "HG 50-100 keV": (gc["gain"] == 0) & (gc["e"] >= 50) & (gc["e"] < 100)}
    for lo, hi in [(440.0, 500.0), (525.0, 600.0), (500.0, 525.0)]:
        w = (he["t"] >= lo) & (he["t"] < hi)
        for cn, cs in comp.items():
            tau, err, cmax = ccf_lag(he["t"][he_sel], gc["t"][cs], lo, hi)
            sp = np.array([ccf_lag(he["t"][he_sel & (he["box"] == b)],
                                   gc["t"][cs & (gc["det"] >= h0) & (gc["det"] <= h0 + 5)],
                                   lo, hi)[0] for b in range(3) for h0 in (1, 7)])
            summary[f"win_{lo:.0f}_{hi:.0f} | {cn}"] = dict(
                lag=tau, err=err, cmax=cmax, fill=float(he["fill"][w].mean()),
                split_mean=float(np.nanmean(sp)),
                split_sem=float(np.nanstd(sp, ddof=1) / np.sqrt(np.isfinite(sp).sum())))
    print("A summary", json.dumps(summary, indent=1), flush=True)
    return rows, summary


GE_BANDS = [("15-30", 15, 30, 0), ("30-50", 30, 50, 0), ("50-100", 50, 100, 0),
            ("100-200", 100, 200, 0), ("200-350", 200, 350, 0),
            ("400-1000", 400, 1000, 1), ("1000-6000", 1000, 6000, 1)]
HE_EB = [("20-50", 20, 43), ("50-100", 43, 81), ("100-250", 81, 200)]


def part_b(he, gc):
    """Lag-energy relative to GECAM-C 50-100 keV (and HE-internal vs HE 20-50)."""
    ref = (gc["gain"] == 0) & (gc["e"] >= 50) & (gc["e"] < 100)
    out = {}
    for lo, hi in [(500.0, 525.0), (440.0, 500.0), (525.0, 600.0)]:
        key = f"{lo:.0f}-{hi:.0f}"
        out[key] = {"gecam": {}, "he_vs_gecam": {}, "he_internal": {}}
        for nm, e1, e2, gn in GE_BANDS:
            s = (gc["gain"] == gn) & (gc["e"] >= e1) & (gc["e"] < e2)
            out[key]["gecam"][nm] = ccf_lag(gc["t"][s], gc["t"][ref], lo, hi)
        out[key]["he_all_vs_gecam"] = ccf_lag(he["t"][he["ch"] >= 20], gc["t"][ref], lo, hi)
        he_ref = (he["ch"] >= 20) & (he["ch"] < 43)
        for nm, c1, c2 in HE_EB:
            s = (he["ch"] >= c1) & (he["ch"] < c2)
            out[key]["he_vs_gecam"][nm] = ccf_lag(he["t"][s], gc["t"][ref], lo, hi)
            out[key]["he_internal"][nm] = ccf_lag(he["t"][s], he["t"][he_ref], lo, hi)
        print(f"B {key}")
        print("    he_all_vs_gecam", out[key]["he_all_vs_gecam"])
        for grp, d in out[key].items():
            if not isinstance(d, dict):
                continue
            print("   ", grp, "  ".join(f"{k}:{v[0]*1e3:+.1f}±{v[1]*1e3:.1f}(c{v[2]:.2f})"
                                       for k, v in d.items()), flush=True)
    return out


def part_c(he):
    """HE vs GECAM-C GRD01 BTIME (50 ms) on the main pulses (gross HE, no DT)."""
    out = {}
    wins = {"P1 rise 200-229": (200.0, 229.0), "P2 249.5-267": (249.5, 267.0),
            "decay 277.5-330": (277.5, 330.0)}
    for gn, chans in {"LG 0.38-3.2 MeV": range(9, 12), "HG 14-350 keV": range(1, 6)}.items():
        tg, cg, lg, _ = gecam_btime(1, chans)
        for wn, (lo, hi) in wins.items():
            m = (tg >= lo) & (tg + 0.05 <= hi)
            step, nb, e0 = 50, int(m.sum()), tg[m][0]
            yb = cg[m] / lg[m] * 0.05                 # dead-time corrected counts
            sel = (he["t"] > lo - 1) & (he["t"] < hi + 1)
            hb = np.histogram(he["t"][sel], e0 + FINE * np.arange(-600, nb * step + 601))[0]
            cs = np.concatenate([[0], np.cumsum(hb)])
            ks = np.arange(-500, 501)
            ys = (yb - yb.mean()) / yb.std()
            c = np.empty(len(ks))
            for i, k in enumerate(ks):
                j = np.arange(nb) * step + 600 + k
                x = (cs[j + step] - cs[j]).astype(float)
                c[i] = np.mean((x - x.mean()) / x.std() * ys)
            lags = ks * FINE
            i0 = int(np.argmax(c)); mm = np.abs(lags - lags[i0]) <= 0.1
            p = np.polyfit(lags[mm] - lags[i0], c[mm], 2)
            tau = lags[i0] - p[1] / (2 * p[0])
            edge = abs(lags[i0]) >= 0.45
            out[f"{wn} | {gn}"] = dict(lag=float(tau), cmax=float(c.max()), at_edge=bool(edge))
            print(f"C {wn:18s} {gn:16s} lag(HE-GECAM) {tau*1e3:+7.1f} ms  cmax {c.max():.3f}"
                  f"  {'(edge of search range)' if edge else ''}", flush=True)
    return out


def plots(rows, summ, b):
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    lo = np.array([r["lo"] for r in rows]) + 2.5
    lag = np.array([r["lag"] for r in rows]) * 1e3
    err = np.array([r["err"] for r in rows]) * 1e3
    fill = np.array([r["fill"] for r in rows])
    cm = np.array([r["cmax"] for r in rows])
    # windows without sub-second structure have flat CCFs; show only cmax >= 0.6
    lag = np.where(cm >= 0.6, lag, np.nan)
    unsat = fill == 0
    ax[0].errorbar(lo[unsat], lag[unsat], err[unsat], fmt="o", ms=3, color="C0",
                   label="no gap-filled events (CCF max >= 0.6)")
    ax[0].errorbar(lo[~unsat], lag[~unsat], err[~unsat], fmt="s", ms=3, color="C3",
                   label="window contains gap-filled events")
    ax[0].axhline(summ["zero_point"] * 1e3, color="C0", lw=0.8)
    ax[0].axhline(0, color="0.5", lw=0.5)
    ax[0].set_ylabel("lag t(HE) - t(GECAM-C LG 0.4-3 MeV) (ms)")
    ax[0].set_ylim(-100, 100)
    ax[0].legend(fontsize=8)
    ax[1].bar(lo, [r["he_rate"] for r in rows], width=5, color="0.75")
    ax[1].set_ylabel("HE gross evt/s")
    ax2 = ax[1].twinx()
    ax2.plot(lo, fill, "C3.-", label="gap-fill fraction")
    ax2.plot(lo, [1 - r["flive"] for r in rows], "C2.-", label=r"$1-\bar f_{\rm live}$")
    ax2.set_ylabel("fraction")
    ax2.legend(fontsize=8, loc="upper right")
    ax[1].set_xlabel("T - T0 (s), 5 s windows")
    fig.tight_layout()
    fig.savefig(OUT / "lag_flare_subwindows.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6, 4))
    emid = [np.sqrt(e1 * e2) for _, e1, e2, _ in GE_BANDS]
    d = b["500-525"]["gecam"]
    ax.errorbar(emid, [d[n][0] * 1e3 for n, *_ in GE_BANDS],
                [d[n][1] * 1e3 for n, *_ in GE_BANDS], fmt="o-", ms=4,
                label="GECAM-C GRD01-12, T0+500-525 s")
    hl, he_, _ = b["500-525"]["he_all_vs_gecam"]
    ax.axhspan((hl - he_) * 1e3, (hl + he_) * 1e3, color="C3", alpha=0.25, lw=0,
               label="HE all channels (low gain, ~0.1-3 MeV)")
    ax.axhline(hl * 1e3, color="C3", lw=1)
    ax.set_xscale("log"); ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("GECAM-C energy (keV)")
    ax.set_ylabel("lag vs GECAM-C 50-100 keV (ms)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "lag_energy.png", dpi=130)


def main():
    he, gc = load()
    et, fl, _ = eng_1hz()
    rows, summ = part_a(he, gc, et, fl)
    b = part_b(he, gc)
    c = part_c(he)
    plots(rows, summ, b)
    (OUT / "lag_results.json").write_text(json.dumps(
        dict(flare_subwindows=rows, zero_point=summ, lag_energy=b, main_btime=c), indent=1))


if __name__ == "__main__":
    main()
