#!/usr/bin/env python3
"""GRB 221009A v2 paper figures from the frozen v2 tables.

    ./.venv/bin/python scripts/p221009_v2_figs.py

Reads data/p221009/v2/{lc_221009a_v2_1s.csv, scale_flux_v2.csv, numbers_v2.csv}
(p221009_v2_numbers.py) and writes f1_lc_full_v2, f2_lc_main_v2, f3_lc_flare_v2,
f4_scale_flux_dt_v2 (.pdf + .png) to data/p221009/v2/. Single-column SCPMA width
(pubstyle.COL_W), no tight bbox, so the PDF is printed at 1:1.

Standard curves: HE net of the time-dependent background B(t); GECAM-C GRD01 LG
corrected for its own dead time, x200. Bands: DT-corrected rate +- stat
(dark) and +- (stat (+) sys) (light). HE-off bins (SAA-mode shut-off, mask of
p221009_v2_common.off_mask) are NaN and hatched.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import pubstyle  # noqa: E402
from p221009_v2_common import OUT, SCALE  # noqa: E402

C_OBS, C_REC, C_DT, C_G = pubstyle.C_OBS, pubstyle.C_RECON, "k", pubstyle.C_GBM
HE_LBL = "HE all events (CsI-dom., ~0.2–3 MeV)"
G_FLOOR = 50.0


def num(metric):
    v = NUM.loc[metric, "value"]
    try:
        return float(v)
    except ValueError:
        return v


def off_spans(lc):
    bad = ~lc.valid.to_numpy()
    t = lc.t.to_numpy()
    spans, i = [], 0
    while i < len(t):
        if bad[i]:
            j = i
            while j + 1 < len(t) and bad[j + 1]:
                j += 1
            spans.append((t[i], t[j] + 1))
            i = j + 1
        else:
            i += 1
    return spans


def shade_off(ax, lc, label=False):
    for k, (a, b) in enumerate(off_spans(lc)):
        ax.axvspan(a, b, facecolor="none", edgecolor="0.55", hatch="////", lw=0,
                   zorder=0, label="HE off (SAA mode)" if (label and k == 0) else None)


def step(ax, t, y, **kw):
    ax.step(np.append(t, t[-1] + 1), np.append(y, y[-1]), where="post", **kw)


def band(ax, t, y, s, **kw):
    tt = np.repeat(np.append(t, t[-1] + 1), 2)[1:-1]
    lo = np.repeat(y - s, 2)
    hi = np.repeat(y + s, 2)
    ax.fill_between(tt, lo, hi, lw=0, **kw)


def traces(ax, lc, lw=0.8, bands=True):
    t = lc.t.to_numpy()
    g = lc.gecam_net_dc.to_numpy() * SCALE
    # GECAM-C is in the SAA before T0+172 s; after T0+600 s its revisit-orbit
    # background is over-subtracted (net < 0), so neither range is drawn
    g = np.where((t >= 172) & (t < 600), g, np.nan)
    if bands:
        tot = np.hypot(lc.sig_stat, lc.sig_sys).to_numpy()
        band(ax, t, lc.dt.to_numpy(), tot, color="0.75", alpha=0.6, zorder=1)
        band(ax, t, lc.dt.to_numpy(), lc.sig_stat.to_numpy(), color="0.45", alpha=0.6, zorder=1)
    step(ax, t, g, color=C_G, lw=lw, zorder=2)
    step(ax, t, lc.obs.to_numpy(), color=C_OBS, lw=lw, zorder=4)
    step(ax, t, lc.rec.to_numpy(), color=C_REC, lw=lw, zorder=3)
    step(ax, t, lc.dt.to_numpy(), color=C_DT, lw=lw, ls=(0, (3, 1.5)), zorder=5)


def legend_handles(full=False):
    h = [Line2D([], [], color=C_OBS, lw=1, label="HE observed (CsI-dom., ~0.2–3 MeV)"),
         Line2D([], [], color=C_REC, lw=1, label="+ FIFO-gap reconstruction"),
         Line2D([], [], color=C_DT, lw=1, ls=(0, (3, 1.5)), label="+ dead-time correction"),
         Patch(facecolor="0.45", alpha=0.6, label=r"$\pm1\sigma$ stat / stat$\oplus$sys",
               edgecolor="0.75", lw=2),
         Line2D([], [], color=C_G, lw=1, label=r"GECAM-C LG $\times200$ (DT-corr.)"),
         Patch(facecolor="none", edgecolor="0.55", hatch="////", lw=0, label="HE off (SAA mode)")]
    return h


def top_legend(fig, handles):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=2,
               fontsize=6.3, frameon=False, columnspacing=0.8, handlelength=1.6)


def ratio_panel(ax, lc, lo, hi, text=None):
    t = lc.t.to_numpy()
    sel = (lc.valid & (t >= lo) & (t < hi) & (lc.gecam_net > G_FLOOR)).to_numpy()
    G = lc.gecam_net_dc.to_numpy() * SCALE
    sG = lc.gecam_sig_dc.to_numpy() * SCALE
    gb = lc.gecam_bsys.to_numpy() * SCALE
    for col, s_stat, s_sys, c, mk, lab in (
            ("rec", lc.sig_rec, None, C_REC, "o", "reconstruction"),
            ("dt", lc.sig_stat, lc.sig_sys, C_DT, "o", "+ dead-time corr.")):
        H = lc[col].to_numpy()
        r = H / G
        ss = np.hypot(s_stat.to_numpy() / H, sG / G)
        if s_sys is not None:
            ss = np.sqrt(ss ** 2 + (s_sys.to_numpy() / H) ** 2 + (gb / G) ** 2)
        else:
            ss = np.sqrt(ss ** 2 + (gb / G) ** 2)
        ax.errorbar(t[sel] + 0.5, r[sel], yerr=np.abs(r[sel]) * ss[sel], fmt=mk, ms=1.8,
                    mfc=c if col == "rec" else "white", mec=c, ecolor=c, elinewidth=0.5,
                    capsize=0, lw=0, zorder=3 if col == "dt" else 2, label=lab)
    ax.axhline(1, color="0.5", lw=0.5, ls="--", zorder=1)
    ax.set_ylabel("HE / GECAM")
    if text:
        ax.text(0.01, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="0.8", lw=0.4, pad=1.5))


def fig_full(lc):
    fig = plt.figure(figsize=(pubstyle.COL_W, 3.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.15], hspace=0.06,
                          left=0.155, right=0.975, top=0.855, bottom=0.10)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    traces(ax, lc, lw=0.6)
    shade_off(ax, lc)
    ax.axvspan(-50, 172, color="0.5", alpha=0.10, lw=0, zorder=0)
    ax.text(60, 4e5, "GECAM-C in SAA\n(no reference)", ha="center", va="center", fontsize=7,
            color="0.35", style="italic")
    ax.set_yscale("symlog", linthresh=1e3, linscale=0.6)
    ax.set_ylim(-4e2, 1.5e7)
    ax.set_yticks([0, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xlim(-50, 700)
    ax.text(650, 6e4, "no GECAM-C\nreference", ha="center", va="center", fontsize=6.3,
            color="0.35", style="italic")
    ax.set_ylabel("HE net rate (evt/s)")
    top_legend(fig, legend_handles())
    plt.setp(ax.get_xticklabels(), visible=False)
    ratio_panel(axr, lc, 172, 600)
    axr.set_ylim(0, 2.2)
    axr.set_xlabel(r"Time since $T_0$ (s)")
    shade_off(axr, lc)
    axr.axvspan(-50, 172, color="0.5", alpha=0.10, lw=0, zorder=0)
    return fig


def fig_main(lc):
    fig = plt.figure(figsize=(pubstyle.COL_W, 3.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.15], hspace=0.06,
                          left=0.155, right=0.975, top=0.855, bottom=0.10)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    traces(ax, lc, lw=0.8)
    shade_off(ax, lc)
    ax.set_xlim(170, 310)
    ax.set_ylim(-1e5, 9.0e6)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1e6:g}")
    ax.set_ylabel(r"HE net rate ($10^6$ evt/s)")
    top_legend(fig, legend_handles())
    for k in ("peak1", "peak2"):
        tp = num(f"{k}_time")
        ax.annotate(f"{num(k + '_rate_dt') / 1e6:.2f}" + r"$\times10^6$", xy=(tp + 0.5, num(k + "_rate_dt")),
                    xytext=(3, 2), textcoords="offset points", ha="left", fontsize=6.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
    axin = ax.inset_axes([0.065, 0.30, 0.285, 0.40])
    traces(axin, lc, lw=0.6)
    axin.set_xlim(175, 212)
    axin.set_ylim(-5e3, 9e4)
    axin.tick_params(labelsize=6, length=1.8)
    axin.set_yticks([0, 4e4, 8e4])
    axin.set_yticklabels(["0", "40k", "80k"])
    axin.set_title(r"175–212 s ($\times200$ fixed here)", fontsize=6.3, pad=2)
    plt.setp(ax.get_xticklabels(), visible=False)
    txt = (f"DT: {num('ratio_main_dt_gdc_med'):.2f}$\\pm${num('ratio_main_dt_gdc_rsig'):.2f}   "
           f"rec.: {num('ratio_main_raw_gdc_med'):.2f}$\\pm${num('ratio_main_raw_gdc_rsig'):.2f}   "
           f"({int(num('ratio_main_dt_gdc_n'))} bins, 180–300 s)")
    ratio_panel(axr, lc, 180, 300, text=txt)
    axr.set_ylim(0, 1.9)
    shade_off(axr, lc)
    axr.set_xlabel(r"Time since $T_0$ (s)")
    return fig


def fig_flare(lc):
    fig = plt.figure(figsize=(pubstyle.COL_W, 3.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.15], hspace=0.06,
                          left=0.155, right=0.975, top=0.855, bottom=0.10)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    traces(ax, lc, lw=0.8)
    t = lc.t.to_numpy(); y = lc.dt.to_numpy()
    a = [(400, 425), (615, 640)]
    m0 = (t >= a[0][0]) & (t < a[0][1]); m1 = (t >= a[1][0]) & (t < a[1][1])
    r0, r1 = np.nanmean(y[m0]), np.nanmean(y[m1])
    t0, t1 = t[m0].mean(), t[m1].mean()
    tt = np.array([440, 610])
    ax.plot(tt, r0 + (r1 - r0) * (tt - t0) / (t1 - t0), color="0.35", lw=0.7, ls=":", zorder=6)
    fs, fe = num("flare_start"), num("flare_end")
    a5, a95 = num("flare_t90_start"), num("flare_t90_end")
    ax.axvspan(a5, a95 + 1, color="C2", alpha=0.08, lw=0, zorder=0)
    for v in (fs, fe):
        ax.axvline(v, color="C2", lw=0.6, ls="--", zorder=0)
    ax.set_xlim(440, 610)
    ax.set_ylim(-1.5e4, 3.6e5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1e5:g}")
    ax.set_ylabel(r"HE net rate ($10^5$ evt/s)")
    h = legend_handles()[:5] + [Line2D([], [], color="0.35", lw=0.7, ls=":", label="local continuum"),
                                Line2D([], [], color="C2", lw=0.6, ls="--",
                                       label=f"flare {fs:.0f}–{fe:.0f} s (>3$\\sigma$)"),
                                Patch(facecolor="C2", alpha=0.15, label=f"$T_{{90}}$ = {num('flare_t90'):.0f} s")]
    top_legend(fig, h)
    plt.setp(ax.get_xticklabels(), visible=False)
    txt = (f"DT: {num('ratio_flare_dt_gdc_med'):.2f}$\\pm${num('ratio_flare_dt_gdc_rsig'):.2f}   "
           f"rec.: {num('ratio_flare_raw_gdc_med'):.2f}$\\pm${num('ratio_flare_raw_gdc_rsig'):.2f}   "
           f"({int(num('ratio_flare_dt_gdc_n'))} bins)")
    ratio_panel(axr, lc, 440, 610, text=txt)
    axr.set_ylim(0, 2.2)
    axr.set_xlabel(r"Time since $T_0$ (s)")
    return fig


def fig_scale(sf, lc):
    d = sf[sf.gecam == "gdc"]
    tau = num("tau_fit_gecam_units")
    RB = np.median(d.B_true) / SCALE
    fig = plt.figure(figsize=(pubstyle.COL_W, 3.6))
    gs = fig.add_gridspec(2, 1, hspace=0.06, left=0.165, right=0.975, top=0.985, bottom=0.105)
    axes = [fig.add_subplot(gs[0])]
    axes.append(fig.add_subplot(gs[1], sharex=axes[0]))
    dec = [30, 100, 300, 1000, 3000, 10000, 30000]
    cen = [np.sqrt(a * b) for a, b in zip(dec[:-1], dec[1:])]
    sat = d.obs_gross >= 3.5e4
    for ax, col, c, tag in ((axes[0], "scale_raw", pubstyle.C_RECON, "reconstruction only"),
                            (axes[1], "scale_dt", "k", "+ dead-time correction")):
        ax.scatter(d.G[~sat], d[col][~sat], s=4, c=c, alpha=0.5, lw=0, label="1-s bins")
        ax.scatter(d.G[sat], d[col][sat], s=9, c=c, marker="x", lw=0.6, alpha=0.8,
                   label=r"HE obs. gross $\geq3.5\times10^4$ evt/s")
        m, lo, hi = [], [], []
        for a, b in zip(dec[:-1], dec[1:]):
            v = d[col][(d.G >= a) & (d.G < b)]
            m.append(np.median(v)); q = np.percentile(v, [25, 75]); lo.append(q[0]); hi.append(q[1])
        m, lo, hi = map(np.array, (m, lo, hi))
        ax.errorbar(cen, m, yerr=[m - lo, hi - m], fmt="o", ms=3, color="C3", mfc="white",
                    elinewidth=0.8, capsize=1.5, zorder=10, label="decade median (IQR)")
        for x, y in zip(cen, m):
            ax.annotate(f"{y * 1e3:.1f}", xy=(x, y), xytext=(4, -7),
                        textcoords="offset points", ha="left", fontsize=6.3, color="C3")
        ax.axhline(1 / SCALE, color="0.4", lw=0.6, ls="--", label="1/200 (early main emission)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(25, 3.5e4)
        ax.set_ylim(1.5e-3, 1.0e-1)
        ax.set_ylabel(r"$s_i$ = GECAM / HE")
        ax.text(0.03, 0.06 if ax is axes[0] else 0.92, tag, transform=ax.transAxes,
                fontsize=7, va="bottom" if ax is axes[0] else "top")
    R = np.logspace(np.log10(25), np.log10(3.5e4), 300)
    axes[0].plot(R, (1 + tau * (R + RB)) / SCALE, color=C_G, lw=1.0, zorder=9,
                 label=(r"$\frac{1}{200}[1+\tau(R+R_B)]$, " +
                        rf"$\tau_{{\rm box}}={num('tau_box_rec_fit_us'):.1f}\,\mu$s/rec."))
    axes[0].legend(loc="upper left", fontsize=6.0, framealpha=0.92)
    plt.setp(axes[0].get_xticklabels(), visible=False)
    axes[1].set_xlabel(r"GECAM-C LG net rate $R$, dead-time corr. (cts/s)")
    return fig


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("saved", OUT / f"{stem}.pdf")


def main():
    global NUM
    pubstyle.apply()
    NUM = pd.read_csv(OUT / "numbers_v2.csv").set_index("metric")
    lc = pd.read_csv(OUT / "lc_221009a_v2_1s.csv")
    sf = pd.read_csv(OUT / "scale_flux_v2.csv")
    save(fig_full(lc), "f1_lc_full_v2")
    save(fig_main(lc), "f2_lc_main_v2")
    save(fig_flare(lc), "f3_lc_flare_v2")
    save(fig_scale(sf, lc), "f4_scale_flux_dt_v2")


if __name__ == "__main__":
    main()
