#!/usr/bin/env python3
"""GRB 221009A v4 figures: precursor (Fig P), quiescent period (Fig Q) and
MeV power-spectrum thresholds (Fig S).

    python p221009_v4_figs.py IN OUT          (farm: resp venv, matplotlib 3.9.4)

IN: prec_lc.csv (p221009_v4_lc.py), joint_fits.csv (gbm.md 5), quiet_he_1s.csv
and quiet_he_windows.csv (p221009_v4_he.py), quiet_lc_ctime_1s.csv and
quietspec_fits.csv (gbm.md 3b), gbm_qpo_sens.csv (scout.md 8.1); Fig Q round 5: r4_summary.txt
(resp.md 9.2), quietpred_v3.csv (gbm.md 9), qfold_II.csv (p221009_v4_qfold.py).
OUT: figP_precursor_v4, figQ_quiet_v4, figS_pds_v4 (.pdf + .png).
Style as p221009_v3_figs.py: pubstyle (STIX, 8 pt ticks/legend, 9 pt labels),
SCPMA widths, no tight bbox.  All times are relative to the paper T0 =
2022-10-09T13:17:00.000 UTC (HXMT MET 339945423.0; zero of the HE event cache and
of the v2/v3 1-s bins).  GBM times: t_paper = t_GBM - 0.0108 s (p221009_v4_ltt.py).
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

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
pubstyle.apply()

D_GBM = 0.0108                     # t_paper = t_GBM - 0.0108 s (trigger UTC -0.0125, light travel +0.0017)
C_HE = pubstyle.C_OBS
C_LIT = pubstyle.C_SAT
C_NAI = ["#fdb863", "#ff7f0e", "#b35806"]
C_BGO = ["#7d4fd0", "#3f1f8c"]
HE_BAND = (170.0, 3100.0)          # keV, CsI ch 20-255 (resp.md 2.6)
SEG_EDGES = np.array([-1, 0, 1, 2, 3, 4.5, 6.5, 8.576]) - D_GBM
AXEL = [(-0.69, 0.23, 15e3, 3e3, 3e3), (0.23, 1.00, 7e3, 2e3, 2e3), (1.00, 2.72, 6e3, 2e3, 2e3),
        (2.72, 7.00, 1.3e3, 1.2e3, 0.6e3), (7.00, 19.95, 330.0, 40.0, 20.0)]   # Ep, -err, +err (keV); precursor_lit.md 1.1
KEV2ERG = 1.602176634e-9
# HE quiet-period background (resp.md 8.2): bias of the blind LargeEvt extrapolations in quiet-like
# conditions, regression on dLE evaluated at the quiet period (dLE = +640): -18.5 counts/s -> +18.5
# correction (resp also quotes the stratum mean -12.6; +18.5 is the value behind resp's II +23); sys = 24
# counts/s (rms of the quiet-like stratum).  Applied from T0 on (12 s after the calibration window).
HE_BIAS = 18.5
HE_SYS = 24.0
# Round 5 (referee 2 round 2, R4; resp.md 9.2): panel (a) shows the uncorrected net rate; the systematic band is
# centred on the per-segment bias predicted by the blind-extrapolation regressions (mean of the dLE + dLE/dt and
# the dLE-only forms, both marked) with the regression scatter (21.4) as half-width; before 60 s the precursor tail
# (GBM time-resolved COMP fits folded into HE, x1.12) is added.  Values parsed from resp/out/r4_summary.txt.
R4_SEGS = ((12, 60), (60, 85), (85, 115), (115, 172))   # 115-172 drawn outside II only; II has its own band
HE_SYS_R4 = 21.0                   # annotation of II (resp.md 9.2: 21-24)
C_TAIL = "#d95f02"
# Panel (d) (R3): HE predictions in II, C over the current HE/GBM area-ratio range (resp.md 9.1), source
# suppression of the LargeEvt background 0.93 for the PL and Lesage shapes (gbm.md 9.1), 1 for the soft COMP/Band
C_RANGE = (0.96, 1.25)
C_NOM = 1.12
SUPP = 0.93
CONV = 7.6e8                      # CsI ch20-255 counts/s per erg cm-2 s-1 (0.3-3 MeV), resp.md 4.4
C_SH = "k"                         # shared-C joint Ep (main)
C_FREE = "#8fa8c8"                 # per-interval free-C joint Ep (comparison)
C_C1 = "#1f78b4"                   # joint Ep with C fixed at the independent calibration 1.00 (round 6)


def step(ax, lo, hi, y, **kw):
    y = np.asarray(y)
    ax.step(np.append(lo, hi[-1]), np.append(y, y[-1]), where="post", **kw)


def rebin(df, cols, k):
    n = len(df) // k * k
    out = {"t_lo": df.t_lo.to_numpy()[:n:k], "t_hi": df.t_hi.to_numpy()[k - 1:n:k]}
    for c in cols:
        out[c + "_net"] = (df[c + "_rate"] - df[c + "_bkg"]).to_numpy()[:n].reshape(-1, k).mean(1)
        out[c + "_err"] = np.sqrt((df[c + "_err"].to_numpy()[:n] ** 2).reshape(-1, k).sum(1)) / k
    return pd.DataFrame(out)


def seglines(ax):
    for x in SEG_EDGES:
        ax.axvline(x, color="0.6", lw=0.4, ls=":", zorder=0)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


# ================================================================ Fig P
# joint-fit table for panel (d): the DT_HE = 0.0108 s rerun (HE window simultaneous with GBM on the paper
# axis) when present, else the original joint2 table (HE window 0.049 s early)
JOINT = "joint_fits_T0paper.csv" if (IN / "joint_fits_T0paper.csv").exists() else "joint_fits.csv"


def load_joint(name):
    J = pd.read_csv(IN / name)
    return J[(J.model == "COMP") & (J.tag == "tr") & (J["var"] == "nominal") & (J.t2 <= 8.6)]


def fig_ep_compare():
    """Diagnostic (not for the paper): joint Ep, original vs DT_HE = 0.0108 s rerun."""
    fig, a = plt.subplots(figsize=(pubstyle.COL_W, 2.4))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.18)
    for name, col, off, lab in (("joint_fits.csv", "0.6", -0.1, "HE window 0.049 s early (joint2)"),
                                ("joint_fits_T0paper.csv", "k", 0.1, "HE window simultaneous (rerun)")):
        for _, r in load_joint(name)[lambda d: d.data == "GBM+HE"].iterrows():
            xm = 0.5 * (r.t1 + r.t2) - D_GBM + off
            if np.isfinite(r.Ep_hi1) and r.Ep < 1.9e5:
                a.errorbar(xm, r.Ep, yerr=[[r.Ep - r.Ep_lo1], [r.Ep_hi1 - r.Ep]], fmt="o", ms=2.8, color=col,
                           elinewidth=0.8, capsize=0)
        a.plot([], [], "o", color=col, ms=3, label=lab)
    a.set_yscale("log"); a.set_xlabel(r"Time since $T_0$ (s)"); a.set_ylabel(r"$E_{\rm p}$ (keV)")
    a.legend(loc="upper right", frameon=False)
    return fig


def fig_precursor():
    lc = rebin(pd.read_csv(IN / "prec_lc.csv"), ["he", "nai_lo", "nai_mid", "nai_hi", "bgo_lo", "bgo_hi"], 2)
    lo, hi = lc.t_lo.to_numpy(), lc.t_hi.to_numpy()
    tc = 0.5 * (lo + hi)
    fig = plt.figure(figsize=(pubstyle.COL_W, 5.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.45], hspace=0.07,
                          left=0.165, right=0.965, top=0.985, bottom=0.075)
    axs = [fig.add_subplot(gs[0])]
    for i in range(1, 4):
        axs.append(fig.add_subplot(gs[i], sharex=axs[0]))
    # (a) HE
    a = axs[0]
    a.fill_between(np.repeat(np.append(lo, hi[-1]), 2)[1:-1], np.repeat(lc.he_net - lc.he_err, 2),
                   np.repeat(lc.he_net + lc.he_err, 2), color=C_HE, alpha=0.25, lw=0)
    step(a, lo, hi, lc.he_net, color=C_HE, lw=0.8)
    a.text(0.97, 0.88, "HE/CsI 0.17–3.1 MeV", transform=a.transAxes, ha="right", va="top", fontsize=8)
    a.set_ylabel(r"cts s$^{-1}$")
    # (b) NaI
    a = axs[1]
    for c, lab, col in zip(["nai_lo", "nai_mid", "nai_hi"], ["8–50 keV", "50–300 keV", "0.3–0.9 MeV"], C_NAI):
        step(a, lo, hi, lc[c + "_net"], color=col, lw=0.8, label=lab)
    a.legend(loc="upper right", frameon=False, title="GBM NaI (n4+n8)", title_fontsize=8, handlelength=1.2)
    a.set_ylabel(r"cts s$^{-1}$")
    # (c) BGO
    a = axs[2]
    for c, lab, col in zip(["bgo_lo", "bgo_hi"], ["0.4–1.5 MeV", "1.5–38 MeV"], C_BGO):
        step(a, lo, hi, lc[c + "_net"], color=col, lw=0.8, label=lab)
    a.legend(loc="upper right", frameon=False, title="GBM BGO (b1)", title_fontsize=8, handlelength=1.2)
    a.set_ylabel(r"cts s$^{-1}$")
    for a in axs[:3]:
        a.axhline(0, color="0.5", lw=0.4)
        seglines(a)
        plt.setp(a.get_xticklabels(), visible=False)
    # (d) Ep(t)
    a = axs[3]
    J = load_joint(JOINT)
    a.axhspan(*HE_BAND, color=C_HE, alpha=0.10, lw=0, zorder=0)
    a.text(-4.7, 2600, "HE band", ha="left", va="top", fontsize=7, color=C_HE)
    for x0, x1, ep, el, eh in AXEL:
        x0, x1 = x0 - D_GBM, x1 - D_GBM
        xm = 0.5 * (x0 + x1)
        a.errorbar(xm, ep, xerr=[[xm - x0], [x1 - xm]], yerr=[[el], [eh]], fmt="s", ms=2.6, mfc="white",
                   mec=C_LIT, ecolor=C_LIT, elinewidth=0.7, capsize=0, zorder=3)
    def ep_point(xm, x0, x1, ep, lo, hi, col, mfc, ms, z):
        if hi is not None and np.isfinite(hi) and ep < 1.9e5:
            a.errorbar(xm, ep, xerr=[[xm - x0], [x1 - xm]], yerr=[[ep - lo], [hi - ep]], fmt="o", ms=ms, color=col,
                       mfc=mfc, mec=col, elinewidth=0.8, capsize=0, zorder=z)
        else:                                       # unbounded above: 1sigma lower limit
            a.errorbar(xm, lo, xerr=[[xm - x0], [x1 - xm]], yerr=[[0], [lo * 1.2]], lolims=True,
                       fmt="none", ecolor=col, elinewidth=0.8, capsize=1.8, zorder=z)

    for _, r in J[J.data == "GBM"].iterrows():
        x0, x1 = r.t1 - D_GBM, r.t2 - D_GBM
        ep_point(0.5 * (x0 + x1) - 0.15, x0, x1, r.Ep, r.Ep_lo1, r.Ep_hi1, "0.55", "white", 2.8, 3)
    # round 6 (gbm.md 9.4, joint5+joint6): shared C (main) and C fixed at the independent calibration 1.00
    SC = pd.read_csv(IN / "j6_sharedC_ep.csv")
    SC = SC[SC.sys == "nominal"]
    F1 = pd.read_csv(IN / "j6_fixedC_ep.csv")
    F1 = F1[np.isclose(F1.C, 1.0) & (F1.sys == "nominal")]
    for _, r in F1.iterrows():
        x0, x1 = r.t1 - D_GBM, r.t2 - D_GBM
        ep_point(0.5 * (x0 + x1) + 0.2, x0, x1, r.Ep, r.Ep_lo1, r.Ep_hi1 if r.Ep < 1.9e5 else np.nan, C_C1, "white",
                 2.8, 4)
    for _, r in SC.iterrows():
        x0, x1 = r.t1 - D_GBM, r.t2 - D_GBM
        ep_point(0.5 * (x0 + x1), x0, x1, r.Ep, r.Ep_lo1, r.Ep_hi1, C_SH, C_SH, 3.0, 5)
    a.set_yscale("log")
    a.set_ylim(100, 2.5e5)
    a.set_ylabel(r"$E_{\rm p}$ (keV)")
    a.set_xlabel(r"Time since $T_0$ (s)")
    a.set_xlim(-5, 20)
    seglines(a)
    h = [Line2D([], [], color="0.55", marker="o", mfc="white", ms=3, lw=0.8, label="GBM (n4+n8+b1)"),
         Line2D([], [], color=C_SH, marker="o", ms=3, lw=0.8, label=rf"GBM + HE, shared $C={SC.C_shared.iloc[0]:.2f}$"),
         Line2D([], [], color=C_C1, marker="o", mfc="white", ms=3, lw=0.8, label=r"GBM + HE, $C=1.00$ (calibration)"),
         Line2D([], [], color=C_LIT, marker="s", mfc="white", ms=3, lw=0.7, label="Axelsson+25 (GBM+LAT)")]
    a.legend(handles=h, loc="upper right", frameon=False, handlelength=1.4)
    for a, tag in zip(axs, "abcd"):
        a.text(0.015, 0.93 if tag != "d" else 0.95, f"({tag})", transform=a.transAxes, ha="left", va="top", fontsize=9)
    return fig


# ================================================================ Fig Q
def nufnu(model, e, s):          # (superseded by qcurves_II.csv from the gdt functions; kept for reference)
    if model == "PL":
        n = s.A * (e / 100.0) ** s["index"]
    elif model == "COMP":
        n = s.A * (e / 100.0) ** s["index"] * np.exp(-(2 + s["index"]) * e / s.Epeak)
    else:                                          # Band (gdt convention, pivot 100 keV)
        a, b, ep = s.alpha, s.beta, s.Epeak
        e0 = ep / (2 + a)
        eb = (a - b) * e0
        n = np.where(e < eb, s.A * (e / 100.0) ** a * np.exp(-e / e0),
                     s.A * ((a - b) * e0 / 100.0) ** (a - b) * np.exp(b - a) * (e / 100.0) ** b)
    return e ** 2 * n * KEV2ERG


def read_r4(path):
    """resp.md 9.2 (p221009_resp_r4.py): per-segment bias predictions and the GBM precursor-tail prediction."""
    import re
    seg, tail, scat = {}, [], None
    for line in open(path):
        m = re.search(r"\(scatter ([\d.]+)\); dLE only", line)
        if m:
            scat = float(m.group(1))
        m = re.search(r"GBM COMP \[([\d.]+),([\d.]+)\].*\(x1\.12 = (\d+)\)", line)
        if m:
            tail.append((float(m.group(1)) - D_GBM, float(m.group(2)) - D_GBM, float(m.group(3))))
        m = re.match(r"\s+([\d.]+)-([\d.]+)\s*: net\s+([+-][\d.]+) \+-\s+([\d.]+);.*bias\s+([+-][\d.]+) "
                     r"\(dLE-only\s+([+-][\d.]+)\).*GBM tail\s+([\d.]+)", line)
        if m:
            v = [float(x) for x in m.groups()]
            seg[(v[0], v[1])] = dict(net=v[2], stat=v[3], b2=v[4], b1=v[5], tail=v[6])
    return seg, tail, scat


def fig_quiet():
    K = 5
    fig = plt.figure(figsize=(pubstyle.FULL_W, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.08, wspace=0.42,
                          left=0.072, right=0.985, top=0.97, bottom=0.125)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    gr = gs[:, 1].subgridspec(2, 1, height_ratios=[1.55, 1], hspace=0.42)
    a3 = fig.add_subplot(gr[0]); a4 = fig.add_subplot(gr[1])
    # HE: external-slope LargeEvt background (d1008 slope), 5-s bins, no bias correction (the expected bias is the band)
    q = pd.read_csv(IN / "quiet_he_1s.csv")
    q = q[(q.t >= -50) & (q.t < 180)]
    n = len(q) // K * K
    t0 = q.t.to_numpy()[:n:K]
    net = (q.cnt - q.bkg_d1008).to_numpy()[:n].reshape(-1, K).mean(1)
    err = np.sqrt(q.cnt.to_numpy()[:n].reshape(-1, K).sum(1)) / K
    for a in (a1, a2):
        a.axvspan(115, 172, color="0.85", lw=0, zorder=0)
        a.axvspan(121.219 - D_GBM, 164.228 - D_GBM, color="0.72", lw=0, zorder=0)
        a.axhline(0, color="0.5", lw=0.4)
    seg, tail, scat = read_r4(IN / "r4_summary.txt")
    # interval II gets its own bias prediction (resp.md 9.2 row "II"; r4 used GBM T0 - 0.06 s, drawn here on the
    # paper-T0 window); the 115-172 s segment band is kept only outside II
    t2a, t2b = 121.219 - D_GBM, 164.228 - D_GBM
    kII = [k for k in seg if k[0] % 1][0]
    pieces = [((s0, s1), [(s0, s1)]) for s0, s1 in R4_SEGS[:-1]]
    pieces += [(R4_SEGS[-1], [(R4_SEGS[-1][0], t2a), (t2b, R4_SEGS[-1][1])]), (kII, [(t2a, t2b)])]
    for key, spans in pieces:
        r = seg[key]
        mid = 0.5 * (r["b2"] + r["b1"])
        for s0, s1 in spans:
            a1.fill_between([s0, s1], mid - scat, mid + scat, color=C_HE, alpha=0.16, lw=0, zorder=0.5)
            for bb in (r["b2"], r["b1"]):
                a1.plot([s0, s1], [bb, bb], color=C_HE, lw=0.5, ls=(0, (2, 1.5)), zorder=0.6)
    for s0, s1 in R4_SEGS:
        r = seg[(s0, s1)]
        mid = 0.5 * (r["b2"] + r["b1"])
        if r["tail"] > 0 and s1 <= 60:
            a1.fill_between([s0, s1], mid + r["tail"] - scat, mid + r["tail"] + scat, color=C_TAIL, alpha=0.18, lw=0,
                            zorder=0.55)
            tt = [(max(x0, s0), min(x1, s1), p) for x0, x1, p in tail if x1 > s0 and x0 < s1]
            if tt[-1][1] < s1:
                tt.append((tt[-1][1], s1, 0.0))
            a1.step([x[0] for x in tt] + [tt[-1][1]], [mid + x[2] for x in tt] + [mid + tt[-1][2]], where="post",
                    color=C_TAIL, lw=0.7, zorder=0.7)
    a1.errorbar(t0 + K / 2, net, yerr=err, fmt="o", ms=1.8, color=C_HE, elinewidth=0.6, capsize=0, zorder=3)
    a1.set_ylim(-140, 260)
    a1.set_ylabel(r"HE cts s$^{-1}$")
    a1.text(0.43, 0.95, "HE/CsI 0.17–3.1 MeV", transform=a1.transAxes, ha="center", va="top", fontsize=8)
    W = pd.read_csv(IN / "quiet_he_windows.csv")
    w = W[(W.ref == "d1008") & (W.window == "II_T0paper")].iloc[0]
    a1.text(143, 252, "II\n" + rf"${w.net_rate + HE_BIAS:+.0f}\pm{w.stat:.0f}\pm{HE_SYS_R4:.0f}$", ha="center",
            va="top", fontsize=7, linespacing=1.3)
    a1.legend(handles=[Patch(color=C_HE, alpha=0.3, lw=0, label=rf"bkg-model bias $\pm{scat:.0f}$"),
                       Patch(color=C_TAIL, alpha=0.3, lw=0, label="+ GBM precursor tail")],
              loc="lower left", bbox_to_anchor=(0.19, 0.0), frameon=False, fontsize=6, handlelength=1.2,
              labelspacing=0.2)
    plt.setp(a1.get_xticklabels(), visible=False)
    # GBM NaI n4+n8 27-293 keV (CTIME ch 2-4), orbital background x (a+bt) fitted on -400..-10 and 95..118 s
    # (gbm.md 3b "orb30fit"; points) and the +-15-orbit version (thin line)
    g = pd.read_csv(IN / "quiet_lc_ctime_1s.csv")
    g["t"] = g.t - D_GBM
    g = g[(g.t >= -50) & (g.t < 180)].reset_index(drop=True)
    tot = 0; b30 = 0; b15 = 0
    for d in ("n4", "n8"):
        for c in (2, 3, 4):
            tot = tot + g[f"{d}_ch{c}_rate"]; b30 = b30 + g[f"{d}_ch{c}_orb30fit"]; b15 = b15 + g[f"{d}_ch{c}_orb15fit"]
    n = len(g) // K * K
    tg = (g.t.to_numpy()[:n] - 0.512).reshape(-1, K)[:, 0]
    dur = 1.024 * K
    ng30 = (tot - b30).to_numpy()[:n].reshape(-1, K).mean(1)
    ng15 = (tot - b15).to_numpy()[:n].reshape(-1, K).mean(1)
    eg = np.sqrt(tot.to_numpy()[:n].reshape(-1, K).sum(1) * 1.024) / dur
    a2.errorbar(tg + dur / 2, ng30, yerr=eg, fmt="o", ms=1.8, color=pubstyle.C_GBM, elinewidth=0.6, capsize=0, zorder=3,
                label=r"orbital bkg ($\pm$30 orbits)")
    a2.step(np.append(tg, tg[-1] + dur), np.append(ng15, ng15[-1]), where="post", color="#b35806", lw=0.5, zorder=2,
            label=r"$\pm$15 orbits")
    a2.set_ylim(-120, 520)
    a2.set_ylabel(r"GBM cts s$^{-1}$")
    a2.text(0.43, 0.93, "GBM NaI n4+n8 27–290 keV", transform=a2.transAxes, ha="center", va="top", fontsize=8)
    a2.set_xlim(-50, 180)
    a2.set_xlabel(r"Time since $T_0$ (s)")
    # nuFnu in II: curves evaluated with the gdt functions of the GBM fits (p221009_v4_qcurves.py)
    QC = pd.read_csv(IN / "qcurves_II.csv")
    e = QC.E.to_numpy()
    a3.plot(e, QC.gbm_PL, color="k", ls="--", lw=0.9, label=r"GBM PL ($\Gamma=-2.06$)")
    a3.plot(e, QC.gbm_COMP, color=pubstyle.C_GBM, ls="-", lw=0.9, label=r"GBM CPL ($E_{\rm p}\approx54$ keV)")
    a3.plot(e, QC.gbm_Band, color="#b35806", ls=":", lw=0.9, label="GBM Band")
    a3.plot(e, QC.lesage_BBCOMP_orb30, color=pubstyle.C_EXT2, ls="-", lw=0.9,
            label=r"Lesage+23 BB+CPL ($E_{\rm p}=5.7$ MeV)")
    a3.plot(e, QC.lesage_BBCOMP_orb15, color=pubstyle.C_EXT2, ls="-", lw=0.5, alpha=0.7)
    a3.axvspan(20, 500, color="0.92", lw=0, zorder=0)
    a3.text(100, 6e-7, "NaI", ha="center", va="top", fontsize=7, color="0.4")
    a3.plot([170, 3100], [3.2e-7, 3.2e-7], color=C_HE, lw=2.0, solid_capstyle="butt")
    a3.text(730, 2.6e-7, "HE band", ha="center", va="top", fontsize=6.5, color=C_HE)
    a3.set_xscale("log"); a3.set_yscale("log")
    a3.set_xlim(10, 2e4); a3.set_ylim(1e-11, 8e-7)
    a3.set_xlabel("Energy (keV)", labelpad=1)
    a3.set_ylabel(r"$\nu F_\nu$ (erg cm$^{-2}$ s$^{-1}$)")
    a3.legend(loc="lower left", frameon=False, handlelength=2.0, fontsize=6, labelspacing=0.15,
              title=r"$T_0+121$–164 s (II)", title_fontsize=7, alignment="left")
    # (d) count space (R3): each model folded through the HE DRM (gbm/quietpred_v3.csv for PL and Lesage, the
    # numbers quoted in gbm.md 9; qfold_II.csv for COMP and Band) vs the measured II net rate
    QF = pd.read_csv(IN / "qfold_II.csv").set_index("model")
    QV = pd.read_csv(IN / "quietpred_v3.csv")
    QV = QV[(QV.C == 1.0) & (QV.suppression == 1.0)]

    def v3(bkg, model):
        s = QV[(QV.bkg == bkg) & (QV.model == model)].set_index("drm_gt3MeV_scale").he_pred_C1_noSupp
        return {k: s.loc[k] for k in (1.0, 0.7, 1.3, 0.0)}

    def qf(name):
        r = QF.loc[name]
        return {k: r[f"he_pred_C1_s3_{k:g}"] for k in (1.0, 0.7, 1.3, 0.0)}

    ROWS = (("GBM CPL", qf("gbm_COMP_orb30"), 1.0, pubstyle.C_GBM, False),
            ("GBM Band", qf("gbm_Band_orb30"), 1.0, "#b35806", False),
            (r"PL, $\pm$30 orb", v3("orb30", "PL"), SUPP, "k", False),
            (r"PL, $\pm$15 orb", v3("orb15", "PL"), SUPP, "k", False),
            (r"Lesage, $\pm$30 orb", v3("orb30", "L_BBCOMP"), SUPP, pubstyle.C_EXT2, True),
            (r"Lesage, $\pm$15 orb", v3("orb15", "L_BBCOMP"), SUPP, pubstyle.C_EXT2, True))
    meas, mtot = w.net_rate + HE_BIAS, np.hypot(w.stat, HE_SYS)
    a4.axvspan(meas - mtot, meas + mtot, color=C_HE, alpha=0.16, lw=0, zorder=0)
    a4.axvline(meas, color=C_HE, lw=0.8, zorder=0.5)
    a4.axvline(meas + 3 * mtot, color=C_HE, lw=0.8, ls="--", zorder=0.5)
    a4.text(meas + 3 * mtot + 3, -0.75, r"3$\sigma$", fontsize=6.5, color=C_HE, ha="left", va="center")
    a4.text(meas, -0.75, "HE", fontsize=6.5, color=C_HE, ha="center", va="center")
    for i, (lab, p, f, col, s0) in enumerate(ROWS):
        lo3 = min(p[s] * c * f for s in (0.7, 1.3) for c in C_RANGE)
        hi3 = max(p[s] * c * f for s in (0.7, 1.3) for c in C_RANGE)
        a4.plot([lo3, hi3], [i, i], color=col, lw=0.7, solid_capstyle="butt")
        a4.plot([p[1.0] * C_RANGE[0] * f, p[1.0] * C_RANGE[1] * f], [i, i], color=col, lw=2.6, solid_capstyle="butt")
        a4.plot(p[1.0] * C_NOM * f, i, "o", ms=3.2, color=col, mec="white", mew=0.5, zorder=4)
        if s0:
            a4.plot(p[0.0] * C_NOM * f, i, "o", ms=3.0, mfc="white", mec=col, mew=0.8, zorder=4)
    a4.set_yticks(range(len(ROWS)))
    a4.set_yticklabels([r[0] for r in ROWS], fontsize=6.5)
    a4.tick_params(axis="y", length=0)
    a4.set_ylim(len(ROWS) - 0.4, -1.3)
    a4.set_xlim(-10, 240)
    a4.set_xlabel(r"Predicted HE rate in II (cts s$^{-1}$)", labelpad=1)
    for a, tag, x in ((a1, "a", 0.008), (a2, "b", 0.008), (a3, "c", 0.03), (a4, "d", 0.93)):
        a.text(x, 0.93 if tag != "d" else 0.97, f"({tag})", transform=a.transAxes, ha="left", va="top", fontsize=9)
    return fig


# ================================================================ Fig S
def fig_pds():
    K = pd.read_csv(IN / "gbm_qpo_sens.csv")
    B = pd.read_csv(IN / "pds_qpo_bins.csv")
    f = np.array([3, 10, 30, 100, 300, 1000])
    fig, axs = plt.subplots(1, 2, figsize=(pubstyle.COL_W, 2.3), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.18, wspace=0.08)
    streams = (("HE (this run)", "HE/CsI 0.2–3 MeV", C_HE, "o"),
               ("GBM BGO b0+b1 0.2-3 MeV", "GBM BGO 0.2–3 MeV", C_BGO[0], "s"),
               ("GBM NaI n8+n4+n7+n3 8-900 keV", "GBM NaI 8–900 keV", pubstyle.C_GBM, "^"))
    for a, seg, title in zip(axs, ("stack_bright", "flare_rise"), ("191 bright s", r"$T_0+440$–503 s")):
        for stream, lab, col, mk in streams:
            r = K[(K.seg == seg) & (K.stream == stream)].iloc[0]
            a.plot(f, [100 * r[f"rms3_{x}Hz"] for x in f], marker=mk, ms=3, color=col, lw=0.9, label=lab,
                   mfc=col if stream.startswith("HE") else "white")
        if seg == "flare_rise":                      # GECAM-C LG 0.4-3 MeV (12 GRD, events; scout.md 3.1)
            g = B[(B.seg == seg) & (B.inst == "GECAM LG0.4-3")]
            y = [100 * g.iloc[int(np.argmin(np.abs(np.log(g.f.to_numpy() / x))))].rms3 for x in f]
            a.plot(f, y, marker="D", ms=2.8, color="#2e8b57", lw=0.9, mfc="white", label="GECAM-C LG 0.4–3 MeV")
        a.set_xscale("log")
        a.set_xlabel("Frequency (Hz)")
        a.text(0.5, 1.02, title, transform=a.transAxes, ha="center", va="bottom", fontsize=8)
    axs[0].set_ylabel(r"3$\sigma$ fractional rms (%)")
    axs[0].set_yscale("log")
    axs[0].set_ylim(1, 50)
    h1, l1 = axs[1].get_legend_handles_labels()
    axs[0].legend(h1, l1, loc="upper left", frameon=False, fontsize=6.5, handlelength=1.4)
    return fig


# ================================================================ Supplementary: HE vs GBM shape at 180-216 s
def fig_supp180():
    """resp.md 9.1 (R2), loss-free seconds 180-186 + 193-215 s: HE net counts vs the GBM Band fits folded into HE
    (p221009_v4_supp180.py, same computation as the 'clean' row of p221009_resp_r2.py), C fitted with 3% sys."""
    import re
    G = pd.read_csv(IN / "supp180_groups.csv")
    txt = open(IN / "supp180_summary.txt").read()
    expo = float(re.search(r"exposure ([\d.]+) s", txt).group(1))
    sysf = float(re.search(r"sys ([\d.]+)", txt).group(1))
    m = re.search(r"C = ([\d.]+)\s+chi2 = ([\d.]+)/(\d+)", txt)
    c, x2, ndf = float(m.group(1)), float(m.group(2)), int(m.group(3))
    m = re.search(r"delta = ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", txt)
    d0, dlo, dhi = (float(x) for x in m.groups())
    el, eh = G.E_lo.to_numpy(), G.E_hi.to_numpy()
    ec = 0.5 * (el + eh)
    de = eh - el
    net = (G.src - G.bkg).to_numpy()
    mu = G.mu_f.to_numpy()
    var = np.maximum(G.src.to_numpy(), 1) + G.bvar.to_numpy() + (sysf * mu) ** 2
    sig = np.sqrt(var)
    sst = np.sqrt(np.maximum(G.src.to_numpy(), 1) + G.bvar.to_numpy())
    fig = plt.figure(figsize=(pubstyle.COL_W, 3.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.06, left=0.17, right=0.97, top=0.975, bottom=0.12)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharex=a1)
    k = expo * de
    a1.errorbar(ec, net / k, xerr=[ec - el, eh - ec], yerr=sst / k, fmt="o", ms=2.2, color=C_HE, elinewidth=0.7,
                capsize=0, zorder=3, label="HE/CsI net (loss-free seconds)")
    step(a1, el, eh, c * mu / k, color=pubstyle.C_GBM, lw=0.9, label=rf"GBM Band $\times\,C={c:.3f}$")
    step(a1, el, eh, G.bkg.to_numpy() / k, color="0.6", lw=0.6, ls="--", label="HE background")
    a1.set_yscale("log"); a1.set_xscale("log")
    a1.set_ylim(0.1, 60)
    a1.set_ylabel(r"cts s$^{-1}$ keV$^{-1}$")
    a1.legend(loc="lower left", frameon=False, fontsize=6.5, handlelength=1.6)
    a1.text(0.97, 0.95, r"$T_0+180$–186, 193–215 s", transform=a1.transAxes, ha="right", va="top", fontsize=8)
    plt.setp(a1.get_xticklabels(), visible=False)
    a2.axhline(0, color="0.5", lw=0.5)
    for dd, ls, lw in ((0.03, "--", 0.8), (0.06, ":", 1.0)):
        for sgn in (1, -1):
            y = c * mu * ((ec / 1000.0) ** (sgn * dd) - 1) / sig
            step(a2, el, eh, y, color="#b35806" if sgn > 0 else "#5e3c99", ls=ls, lw=lw)
    a2.errorbar(ec, (net - c * mu) / sig, xerr=[ec - el, eh - ec], yerr=1.0, fmt="o", ms=2.2, color=C_HE,
                elinewidth=0.7, capsize=0, zorder=3)
    h = [Line2D([], [], color="#b35806", ls="--", lw=0.8, label=r"$\delta=+0.03$"),
         Line2D([], [], color="#b35806", ls=":", lw=1.0, label=r"$+0.06$"),
         Line2D([], [], color="#5e3c99", ls="--", lw=0.8, label=r"$\delta=-0.03$"),
         Line2D([], [], color="#5e3c99", ls=":", lw=1.0, label=r"$-0.06$")]
    a2.legend(handles=h, loc="upper right", ncol=2, frameon=False, fontsize=6.5, handlelength=1.6, columnspacing=0.8,
              labelspacing=0.15)
    a2.text(0.03, 0.05, rf"$\chi^2={x2:.1f}/{ndf}$ ({100 * sysf:.0f}% sys); tilt fit $\delta={d0:+.3f}$"
            rf"$\,[{dlo:+.3f},{dhi:+.3f}]$", transform=a2.transAxes, ha="left", va="bottom", fontsize=6.5)
    a2.set_ylim(-5.5, 5.5)
    a2.set_xlim(170, 3200)
    a2.set_xticks([200, 300, 500, 1000, 2000, 3000])
    a2.set_xticklabels(["200", "300", "500", "1000", "2000", "3000"])
    a2.set_xlabel("Deposited energy (keV)")
    a2.set_ylabel(r"(data$-$model)/$\sigma$")
    for a, tag in ((a1, "a"), (a2, "b")):
        a.text(0.015, 0.95 if tag == "a" else 0.9, f"({tag})", transform=a.transAxes, ha="left", va="top", fontsize=9)
    return fig


if __name__ == "__main__":
    save(fig_precursor(), "figP_precursor_v4")
    save(fig_quiet(), "figQ_quiet_v4")
    save(fig_pds(), "figS_pds_v4")
    if (IN / "supp180_groups.csv").exists():
        save(fig_supp180(), "figSupp_180216_v4")
    if JOINT == "joint_fits_T0paper.csv":
        save(fig_ep_compare(), "diag_ep_old_vs_new")
    print("panel (d) from", JOINT)
    print("done")
