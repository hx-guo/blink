#!/usr/bin/env python3
"""Tier 2: energy-resolved recovery through REAL FIFO saturation, checked
against Fermi/GBM (no FIFO, no gaps) in the same deposited-energy bands.

Two figures per burst (GRB 260226A, GRB 250919A), each with a CsI column
(70-150 / 150-300 / 300-700 keV, the bands in which these off-axis GRBs are
seen by HE) and an NaI column (20-50 / 50-100 / 100-250 keV):

  tier2_<burst>_bands_hr : band-resolved net light curves at 0.1 s through the
      saturated phase -- observed-only (1K-equivalent), recovered (obs + fillers,
      analytic +-1 sigma band from recovery_cov), Fermi/GBM scaled per band on
      filler-free bins; bottom row = hardness ratio (hard/soft band) of the
      three.  FIFO loss is energy-independent per box, so a hardness ratio built
      from observed-only counts is not biased by the gaps -- the test is that
      the recovery restores the per-band flux (GBM-checked) without inventing
      any hardness feature.

      The HR agreement on filler bins is reported two ways.  `*_fillerbins_pooled`
      sums the counts over the filler bins first and then forms the ratio; it is
      always defined and is the number to read.  `*_fillerbins` keeps the older
      bin-by-bin chi2, which needs every bin to pass a 3-sigma cut on the soft
      band -- fine for CsI (30/30 bins) but fatal for NaI, where 20-50 keV has
      too few net counts per 0.1 s bin and only 1 of 30 survives, so it carries
      a `usable` flag (n_bins >= 5) and its chi2 has been nan in the past.
      Sums over bins take the variance from the full covariance (1^T Cov 1), not
      from the diagonal: the diagonal is what the error bars on the light curves
      can show, but the k calibration is shared between gaps and couples bins
      across a segment (it adds 1-8% to sigma here).  The pooled HR error still
      treats the hard and soft bands as independent, which they are not -- both
      recover from the same reference events -- so it is conservative: the true
      error on a ratio of positively correlated counts is smaller.

  tier2_<burst>_segment_spectrum : counts spectrum accumulated over the
      saturated segment -- observed-only, recovered, and the fillers alone
      (spectrum of the lost particles); bottom = recovered/observed per energy
      bin (HXMT, analytic error) against the SAME ratio predicted by GBM from
      its own gap-free light curve: R_GBM(E) = 3 G_seg(E) / sum_box G_alive,box(E),
      where G_alive,box is GBM's net count in the segment with that box's gap
      intervals removed. This is response-free: it asks whether the particles
      HE lost in the gaps had the spectrum the recovery gave them.

Deposited-energy caveat: NaI(Tl)@GBM and CsI(Na)/NaI(Tl)@HE redistribute an
incident spectrum differently; the comparison is at deposited-energy level and
uses ratios/scales, not an incident-energy unfold.

Inputs: data/talk20_spec/recon_<burst>/{events,gapcov,gapbins}.csv from
  blink sat reconstruct <T0> --before B --after A --gapcov-out .. --gapbins-out ..
  (260226A: 2026-02-26T10:37:53 -8/+80; 250919A: 432865758 -30/+60)

Run from blink/:
  .venv/bin/python scripts/talk20_spec_tier2_real.py --burst 260226A
  .venv/bin/python scripts/talk20_spec_tier2_real.py --burst 250919A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "he_nai_cal"))
sys.path.insert(0, str(HERE.parent / "analysis"))
import recovery_cov as rc  # noqa: E402
from plot_hxmt_vs_ibis_bands import channel_to_kev_lut  # noqa: E402
from plot_hxmt_csi_multi import hxmt_met, load_gbm  # noqa: E402

ROOT = HERE.parent
OUT = Path("/Users/skyair/Developer/ihep/talk-20/figs/spectrum")
BIN = 0.1

CFG = {
    "260226A": dict(
        t0="2026-02-26T10:37:53", recon=ROOT / "data/talk20_spec/recon_260226a",
        csi_lut=ROOT / "data/hxmt_aux/csi_ch2e_260226.npy",
        gbm=dict(dir=str(ROOT / "data/fermi_gbm/bn260226443"), trig="bn260226443",
                 dets=["n0", "n3"], tmet=793795080.95811, tutc="2026-02-26T10:37:55.958"),
        bkg=[(-8, -3), (65, 80)], scale_range=(18, 44), xlim=(18.5, 23.5),
        segment=(19.1, 22.6), label="GRB 260226A"),
    "250919A": dict(
        t0="2025-09-19T00:29:15", recon=ROOT / "data/talk20_spec/recon_250919a",
        csi_lut=ROOT / "data/hxmt_aux/csi_ch2e_250919.npy",
        gbm=dict(dir=str(ROOT / "data/fermi_gbm/bn250919020"), trig="bn250919020",
                 dets=["n7", "n8"], tmet=779934537.28, tutc="2025-09-19T00:28:52.28"),
        bkg=[(-30, -20), (40, 60)], scale_range=(-2, 15), xlim=(5.5, 11.5),
        segment=(7.65, 8.05), label="GRB 250919A"),
}
VIEWS = {
    "csi": dict(pw=(90, 257), chmin=30, bands=[(70, 150), (150, 300), (300, 700)],
                ebins=np.geomspace(70, 800, 12), hr=((300, 700), (70, 150)), name="CsI"),
    "nai": dict(pw=(54, 70), chmin=20, bands=[(20, 50), (50, 100), (100, 250)],
                ebins=np.geomspace(20, 350, 10), hr=((100, 250), (20, 50)), name="NaI"),
}
C_OBS, C_REC, C_BAND, C_EXT, C_FILL = "#20347e", "#2f6db5", "#a9c8ea", "#e8792b", "#7fb2dd"


# ────────────────────────── loading ──────────────────────────

def load_hxmt(cfg):
    df = pd.read_csv(cfg["recon"] / "events.csv",
                     usecols=["box", "type", "met", "channel", "pulse_width"])
    df = df[np.isfinite(df["met"])].reset_index(drop=True)
    trig = hxmt_met(cfg["t0"])
    df["t"] = df["met"] - trig
    e_nai = channel_to_kev_lut()
    e_csi = np.load(cfg["csi_lut"])
    ch = np.clip(df["channel"].to_numpy(int), 0, 255)
    pw = df["pulse_width"].to_numpy(int)
    for view, v in VIEWS.items():
        e = (e_nai if view == "nai" else e_csi)[ch]
        ok = (pw >= v["pw"][0]) & (pw <= v["pw"][1]) & (ch >= v["chmin"]) & np.isfinite(e)
        df[f"e_{view}"] = np.where(ok, e, np.nan)
    blocks = rc.load_blocks(str(cfg["recon"] / "gapcov.csv"))
    bins = rc.load_bins(str(cfg["recon"] / "gapbins.csv"))
    gaps = {}
    for b in blocks:
        gaps.setdefault(b["target_box"], []).append((b["t_start"] - trig, b["t_stop"] - trig))
    return df, trig, blocks, bins, gaps


def ev_dict(df, mask):
    d = df[mask]
    return {"box": d["box"].to_numpy(str), "type": d["type"].to_numpy(str),
            "met": d["met"].to_numpy(float), "channel": d["channel"].to_numpy(int)}


def band_mask(df, view, lo, hi):
    e = df[f"e_{view}"].to_numpy()
    return (e >= lo) & (e < hi)


def linear_bkg(x, rate, bkgm):
    return np.polyval(np.polyfit(x[bkgm], rate[bkgm], 1), x)


def rec_counts_and_var(df, blocks, bins, trig, mask, edges_rel):
    """Recovered (obs+fill) counts per bin and analytic variance (recovery_cov)."""
    N, cov = rc.cov_matrix(ev_dict(df, mask), blocks, bins, np.asarray(edges_rel) + trig,
                           box=None, include_u=False)
    # 对角给误差棒（图上只能画对角），完整矩阵给求和量：把若干 bin 加起来时
    # 方差是 1ᵀCov1 而不是对角之和。实测非对角只占 1-3%（gap ~27ms 远窄于
    # 0.1s 的 bin，相关几乎都落在 bin 内部），但既然算出来了就别丢。
    return N, np.diag(cov), np.asarray(cov)


def pooled_var(cov_win, win_mask, sel):
    """窗内若干 bin 求和后的方差 = 该子块的全元素和（含非对角）。"""
    idx = np.nonzero((sel & win_mask)[win_mask])[0]
    if idx.size == 0:
        return 0.0
    return float(cov_win[np.ix_(idx, idx)].sum())


def interval_counts(t, intervals):
    """Number of t values inside the union of disjoint intervals."""
    n = 0
    for a, b in intervals:
        n += int(np.sum((t >= a) & (t < b)))
    return n


def live_fraction(edges, gaps):
    """Per-bin live fraction of the three-box sum: 1 - (summed gap overlap)/(3*width)."""
    dead = np.zeros(len(edges) - 1)
    for iv in gaps.values():
        for s, e in iv:
            i0 = max(int(np.searchsorted(edges, s, side="right")) - 1, 0)
            i1 = min(int(np.searchsorted(edges, e, side="left")), len(edges) - 1)
            for i in range(i0, i1):
                dead[i] += max(0.0, min(e, edges[i + 1]) - max(s, edges[i]))
    return 1.0 - dead / (len(gaps) * np.diff(edges))


def subtract(seg, gaps):
    """seg minus a list of (possibly overlapping) gap intervals -> list of intervals."""
    a, b = seg
    cur = [(a, b)]
    for s, e in gaps:
        nxt = []
        for x, y in cur:
            if e <= x or s >= y:
                nxt.append((x, y))
            else:
                if s > x:
                    nxt.append((x, s))
                if e < y:
                    nxt.append((e, y))
        cur = nxt
    return cur


# ────────────────────────── main ──────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", required=True, choices=list(CFG))
    ap.add_argument("--segment", type=float, nargs=2)
    args = ap.parse_args()
    cfg = CFG[args.burst]
    if args.segment:
        cfg["segment"] = tuple(args.segment)
    OUT.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.size": 11.5, "axes.labelsize": 12, "legend.fontsize": 9.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "pdf.fonttype": 42,
        "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
    })

    df, trig, blocks, bins, gaps = load_hxmt(cfg)
    is_evt = (df["type"] == "EVT").to_numpy()
    is_fill = (df["type"] == "FILL_GAP").to_numpy()
    t_all = df["t"].to_numpy()
    gt, ge = load_gbm(t0=cfg["t0"], **cfg["gbm"])
    print(f"{args.burst}: HXMT {is_evt.sum():,} obs + {is_fill.sum():,} fill; "
          f"GBM {len(gt):,} events; gaps per box "
          + ", ".join(f"{b}:{len(g)}" for b, g in sorted(gaps.items())), file=sys.stderr)

    tmin, tmax = np.floor(t_all.min()), np.ceil(t_all.max())
    edges = np.arange(tmin, tmax + BIN / 2, BIN)
    x = edges[:-1] + BIN / 2
    bkgm = np.zeros(len(x), bool)
    for a, b in cfg["bkg"]:
        bkgm |= (x >= a) & (x < b)
    s1, s2 = cfg["scale_range"]
    fill_bins = np.histogram(t_all[is_fill], bins=edges)[0] > 0
    scale_m = (x >= s1) & (x < s2) & ~fill_bins
    xl = cfg["xlim"]
    view_win = (x >= xl[0] - BIN) & (x <= xl[1] + BIN)
    seg = cfg["segment"]
    # observed-only light curves lose live time in the gaps: the background
    # they contain is bkg * live fraction, so subtract only that much
    # (otherwise the observed-only deficit is exaggerated in the gap bins).
    lf = live_fraction(edges, gaps)
    numbers = {"burst": args.burst, "segment": seg, "views": {}}
    res = {}

    # ═══════════ figure A: band light curves + hardness ratio ═══════════
    figA, axA = plt.subplots(4, 2, figsize=(13, 8.2), sharex=True,
                             gridspec_kw={"hspace": 0.07, "wspace": 0.16,
                                          "height_ratios": [1, 1, 1, 1.15]})
    # ═══════════ figure B: segment spectra ═══════════
    figB, axB = plt.subplots(2, 2, figsize=(12.5, 6.8), sharex="col",
                             gridspec_kw={"height_ratios": [2.4, 1.3], "hspace": 0.06,
                                          "wspace": 0.2})

    for j, (view, v) in enumerate(VIEWS.items()):
        nums = {"bands": {}, "hr": {}, "spectrum": {}}
        band_series = {}
        for i, (lo, hi) in enumerate(v["bands"]):
            m = band_mask(df, view, lo, hi)
            raw_obs = np.histogram(t_all[m & is_evt], bins=edges)[0].astype(float)
            # recovered counts + analytic variance only over the display window
            # (cov_matrix on the full window is unnecessary); elsewhere obs+fill.
            raw_rec = np.histogram(t_all[m], bins=edges)[0].astype(float)
            var_rec = raw_rec.copy()
            wi = np.nonzero(view_win)[0]
            e_win = np.append(edges[wi], edges[wi[-1] + 1])
            Nw, Vw, Cw = rec_counts_and_var(df, blocks, bins, trig, m, e_win)
            # same histogram up to float rounding of events sitting on a bin edge
            assert np.abs(Nw - raw_rec[wi]).max() <= 3, (view, lo, hi, np.abs(Nw - raw_rec[wi]).max())
            raw_rec[wi] = Nw
            var_rec[wi] = Vw
            gm = (ge >= lo) & (ge < hi)
            raw_gbm = np.histogram(gt[gm], bins=edges)[0].astype(float)
            r_obs, r_rec, r_gbm = raw_obs / BIN, raw_rec / BIN, raw_gbm / BIN
            n_obs = r_obs - linear_bkg(x, r_obs, bkgm) * lf
            n_rec = r_rec - linear_bkg(x, r_rec, bkgm)
            n_gbm = r_gbm - linear_bkg(x, r_gbm, bkgm)
            scale = n_rec[scale_m].sum() / n_gbm[scale_m].sum()
            band_series[(lo, hi)] = dict(n_obs=n_obs, n_rec=n_rec, n_gbm=n_gbm * scale,
                                         s_obs=np.sqrt(raw_obs) / BIN,
                                         s_rec=np.sqrt(var_rec) / BIN,
                                         s_gbm=np.sqrt(raw_gbm) / BIN * scale,
                                         cov_win=Cw, gbm_scale=scale)
            # segment sums (net counts) for the numbers table
            sm = (x >= seg[0]) & (x < seg[1])
            fb = sm & fill_bins
            nums["bands"][f"{lo}-{hi}"] = {
                "gbm_scale": float(scale),
                "seg_net_obs": float(n_obs[sm].sum() * BIN),
                "seg_net_rec": float(n_rec[sm].sum() * BIN),
                "seg_net_rec_sigma": float(np.sqrt(pooled_var(Cw, view_win, sm))),
                "seg_net_gbm_scaled": float((n_gbm * scale)[sm].sum() * BIN),
                "seg_net_gbm_scaled_sigma": float(np.sqrt(raw_gbm[sm].sum()) * scale),
                "fillerbins_obs_over_gbm": float(n_obs[fb].sum() / (n_gbm * scale)[fb].sum()),
                "fillerbins_rec_over_gbm": float(n_rec[fb].sum() / (n_gbm * scale)[fb].sum()),
                "fillerbins_rec_over_gbm_sigma": float(np.sqrt(pooled_var(Cw, view_win, fb) + (raw_gbm[fb].sum()) * scale ** 2 * (n_rec[fb].sum() / (n_gbm * scale)[fb].sum()) ** 2) / (n_gbm * scale)[fb].sum()),
                "n_filler_bins": int(fb.sum()),
            }
            ax = axA[i, j]
            ax.axvspan(*seg, color="tab:red", alpha=0.06, lw=0, zorder=0)
            bs = band_series[(lo, hi)]
            ax.fill_between(x, 0, bs["n_obs"], step="mid", color=C_OBS, alpha=0.35, lw=0, zorder=1)
            ax.fill_between(x, bs["n_rec"] - bs["s_rec"], bs["n_rec"] + bs["s_rec"], step="mid",
                            color=C_BAND, alpha=0.7, lw=0, zorder=2)
            ax.step(x, bs["n_obs"], where="mid", color=C_OBS, lw=1.0, label="HE observed only", zorder=3)
            ax.step(x, bs["n_rec"], where="mid", color=C_REC, lw=1.3, label="HE recovered (±1σ analytic)", zorder=4)
            ax.fill_between(x, bs["n_gbm"] - bs["s_gbm"], bs["n_gbm"] + bs["s_gbm"], step="mid",
                            color=C_EXT, alpha=0.22, lw=0, zorder=3)
            ax.step(x, bs["n_gbm"], where="mid", color=C_EXT, lw=1.1,
                    label=f"Fermi/GBM {'+'.join(cfg['gbm']['dets'])} ×{scale:.3f}", zorder=5)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_xlim(*xl)
            ax.set_ylim(0, 1.15 * np.nanmax(bs["n_rec"][view_win] + bs["s_rec"][view_win]))
            ax.text(0.015, 0.88, f"{v['name']} {lo}–{hi} keV", transform=ax.transAxes, fontweight="bold")
            if i == 0:
                ax.legend(loc="upper right", frameon=False, ncol=1)
            if j == 0:
                ax.set_ylabel("net counts / s")

        # hardness ratio
        (hlo, hhi), (slo, shi) = v["hr"]
        H, S = band_series[(hlo, hhi)], band_series[(slo, shi)]
        ax = axA[3, j]
        ax.axvspan(*seg, color="tab:red", alpha=0.06, lw=0, zorder=0)
        hr_series = {}
        for key, col, lab, z in (("obs", C_OBS, "observed only", 3), ("rec", C_REC, "recovered", 4),
                                 ("gbm", C_EXT, "Fermi/GBM", 5)):
            nh, sh = H[f"n_{key}"], H[f"s_{key}"]
            ns, ss = S[f"n_{key}"], S[f"s_{key}"]
            ok = view_win & (ns > 3 * ss) & (nh > 0)
            hr = np.where(ok, nh / np.where(ns > 0, ns, np.nan), np.nan)
            shr = hr * np.sqrt((sh / np.where(nh > 0, nh, np.nan)) ** 2 + (ss / np.where(ns > 0, ns, np.nan)) ** 2)
            hr_series[key] = (hr, shr)
            off = {"obs": -0.012, "rec": 0.0, "gbm": 0.012}[key]
            ax.errorbar(x[ok] + off, hr[ok], yerr=shr[ok], fmt="o", ms=3.2, color=col, ecolor=col,
                        elinewidth=0.8, capsize=0, label=lab, zorder=z, alpha=0.95)
        ax.set_xlim(*xl)
        ax.set_ylabel("hardness ratio")
        ymax_hr = np.nanmax([np.nanmax((hr_series[k][0] + hr_series[k][1])[view_win]) for k in hr_series])
        ax.set_ylim(0, 1.45 * ymax_hr)
        ax.text(0.015, 0.88, f"{v['name']} HR ({hlo}–{hhi})/({slo}–{shi})",
                transform=ax.transAxes, fontweight="bold")
        ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=9)
        ax.set_xlabel(f"time since T0 (s)   [T0 = {cfg['t0']} UTC]")
        # HR agreement on filler bins —— 先把计数加起来再做比。
        # 逐 bin 版（下面）要求每个 bin 的软段净计数 > 3σ 才算数，对 NaI 是
        # 致命的：软段 20-50 keV 每 0.1 s bin 的净计数太少，30 个 filler bin 里
        # recovered 只有 1 个过筛，chi2/mean_pull 就失去意义（曾出现过 nan）。
        # 合并版不需要逐 bin 卡阈，恒有定义；recovered 的求和方差用 1ᵀCov1。
        fbin = view_win & fill_bins
        pooled = {}
        for key in ("obs", "rec", "gbm"):
            nh = H[f"n_{key}"][fbin].sum() * BIN
            ns = S[f"n_{key}"][fbin].sum() * BIN
            if key == "rec":
                vh = pooled_var(H["cov_win"], view_win, fbin)
                vs = pooled_var(S["cov_win"], view_win, fbin)
            else:
                vh = float(np.sum((H[f"s_{key}"][fbin] * BIN) ** 2))
                vs = float(np.sum((S[f"s_{key}"][fbin] * BIN) ** 2))
            hr = nh / ns
            pooled[key] = (hr, hr * np.sqrt(vh / nh ** 2 + vs / ns ** 2))
        for key in ("obs", "rec"):
            d = pooled[key][0] - pooled["gbm"][0]
            sd = np.sqrt(pooled[key][1] ** 2 + pooled["gbm"][1] ** 2)
            nums["hr"][f"{key}_vs_gbm_fillerbins_pooled"] = {
                "n_bins": int(fbin.sum()),
                "hr_hxmt": float(pooled[key][0]), "sigma_hxmt": float(pooled[key][1]),
                "hr_gbm": float(pooled["gbm"][0]), "sigma_gbm": float(pooled["gbm"][1]),
                "pull": float(d / sd),
            }
        # 逐 bin 版保留作对照。n_bins 小于 5 时它不构成检验，标出来。
        okb = view_win & fill_bins & np.isfinite(hr_series["rec"][0]) & np.isfinite(hr_series["gbm"][0])
        for key in ("obs", "rec"):
            d = hr_series[key][0][okb] - hr_series["gbm"][0][okb]
            s = np.sqrt(hr_series[key][1][okb] ** 2 + hr_series["gbm"][1][okb] ** 2)
            nums["hr"][f"{key}_vs_gbm_fillerbins"] = {
                "n_bins": int(okb.sum()), "chi2": float(np.sum((d / s) ** 2)),
                "mean_pull": float(np.mean(d / s)) if okb.sum() else float("nan"),
                "mean_hr_hxmt": float(np.nanmean(hr_series[key][0][okb])) if okb.sum() else float("nan"),
                "mean_hr_gbm": float(np.nanmean(hr_series["gbm"][0][okb])) if okb.sum() else float("nan"),
                "usable": bool(okb.sum() >= 5),
            }
        # HR over the whole segment from summed net counts
        sm = (x >= seg[0]) & (x < seg[1])
        for key in ("obs", "rec", "gbm"):
            nh, ns = H[f"n_{key}"][sm].sum(), S[f"n_{key}"][sm].sum()
            if key == "rec":  # 求和 ⇒ 1ᵀCov1，不是对角之和
                sh = np.sqrt(pooled_var(H["cov_win"], view_win, sm)) / BIN
                ss = np.sqrt(pooled_var(S["cov_win"], view_win, sm)) / BIN
            else:
                sh = np.sqrt(np.sum(H[f"s_{key}"][sm] ** 2)); ss = np.sqrt(np.sum(S[f"s_{key}"][sm] ** 2))
            nums["hr"][f"segment_{key}"] = [float(nh / ns), float(nh / ns * np.sqrt((sh / nh) ** 2 + (ss / ns) ** 2))]

        # ═══════════ segment spectrum ═══════════
        eb = v["ebins"]; w = np.diff(eb); ctr = np.sqrt(eb[:-1] * eb[1:])
        e_all = df[f"e_{view}"].to_numpy()
        in_seg = (t_all >= seg[0]) & (t_all < seg[1])
        dur = seg[1] - seg[0]
        # per-energy-bin linear background (1 s bins over the whole window)
        e1 = np.arange(tmin, tmax + 0.5, 1.0); x1 = e1[:-1] + 0.5
        bm1 = np.zeros(len(x1), bool)
        for a, b in cfg["bkg"]:
            bm1 |= (x1 >= a) & (x1 < b)
        H_obs = np.zeros(len(w)); H_rec = np.zeros(len(w)); V_rec = np.zeros(len(w))
        B_hx = np.zeros(len(w)); H_gseg = np.zeros(len(w)); H_galive = np.zeros(len(w))
        B_gseg = np.zeros(len(w)); B_galive = np.zeros(len(w)); RAW_galive = np.zeros(len(w))
        alive = {b: subtract(seg, g) for b, g in gaps.items()}
        alive_dur = {b: sum(y - x_ for x_, y in iv) for b, iv in alive.items()}
        for k in range(len(w)):
            mk = (e_all >= eb[k]) & (e_all < eb[k + 1])
            H_obs[k] = np.sum(mk & is_evt & in_seg)
            N1, V1, _ = rec_counts_and_var(df, blocks, bins, trig, mk, np.array(seg))
            H_rec[k], V_rec[k] = N1[0], V1[0]
            # HXMT background: obs rate per 1 s bin, linear fit on bkg windows
            r1 = np.histogram(t_all[mk & is_evt], bins=e1)[0].astype(float)
            B_hx[k] = np.sum(linear_bkg(x1, r1, bm1)[(x1 >= seg[0]) & (x1 < seg[1])]) * (1.0) \
                if False else np.polyval(np.polyfit(x1[bm1], r1[bm1], 1), 0.5 * (seg[0] + seg[1])) * dur
            gk = (ge >= eb[k]) & (ge < eb[k + 1])
            g1 = np.histogram(gt[gk], bins=e1)[0].astype(float)
            gcoef = np.polyfit(x1[bm1], g1[bm1], 1)
            H_gseg[k] = np.sum(gk & (gt >= seg[0]) & (gt < seg[1]))
            B_gseg[k] = np.polyval(gcoef, 0.5 * (seg[0] + seg[1])) * dur
            for b, iv in alive.items():
                H_galive[k] += interval_counts(gt[gk], iv)
                B_galive[k] += sum(np.polyval(gcoef, 0.5 * (x_ + y)) * (y - x_) for x_, y in iv)
        RAW_galive = H_galive.copy()
        f_alive = np.mean([d / dur for d in alive_dur.values()])
        net_obs = H_obs - B_hx * f_alive          # observed-only carries background only while alive
        net_rec = H_rec - B_hx
        net_fill = H_rec - H_obs
        # HXMT ratio recovered/observed; GBM-predicted ratio 3*G_seg / sum_box G_alive
        with np.errstate(divide="ignore", invalid="ignore"):
            R_hx = net_rec / net_obs
            S_hx = np.sqrt(V_rec) / np.abs(net_obs)          # conservative: full analytic Var(rec)/obs
            G_seg = H_gseg - B_gseg; G_al = H_galive - B_galive
            R_g = 3 * G_seg / G_al
            S_g = R_g * np.sqrt(H_gseg / G_seg ** 2 + RAW_galive / G_al ** 2)
        good = (net_obs > 5 * np.sqrt(np.maximum(H_obs, 1))) & (G_al > 5 * np.sqrt(np.maximum(RAW_galive, 1)))

        def log_slope(R, S, ok):
            """Weighted LSQ slope d ln R / d ln E (+-1 sigma): the photon-index
            bias a fit to the observed-only spectrum would carry."""
            xx = np.log(ctr[ok]); yy = np.log(R[ok]); ww = (R[ok] / S[ok]) ** 2
            A = np.vstack([xx, np.ones_like(xx)]).T
            cov = np.linalg.inv(A.T @ (A * ww[:, None]))
            beta = cov @ (A.T @ (ww * yy))
            return float(beta[0]), float(np.sqrt(cov[0, 0]))

        slope_hx = log_slope(R_hx, S_hx, good)
        slope_g = log_slope(R_g, S_g, good)
        nums["spectrum"] = {
            "ratio_logslope_hxmt": slope_hx, "ratio_logslope_gbm_pred": slope_g,
            "ebins": eb.tolist(), "net_obs": net_obs.tolist(), "net_rec": net_rec.tolist(),
            "sigma_rec": np.sqrt(V_rec).tolist(), "fill": net_fill.tolist(),
            "ratio_hxmt": R_hx.tolist(), "sigma_ratio_hxmt": S_hx.tolist(),
            "ratio_gbm_pred": R_g.tolist(), "sigma_ratio_gbm_pred": S_g.tolist(),
            "good": good.tolist(),
            "total_ratio_hxmt": float(net_rec[good].sum() / net_obs[good].sum()),
            "total_ratio_gbm_pred": float(3 * G_seg[good].sum() / G_al[good].sum()),
            "chi2_hxmt_vs_gbm_pred": float(np.sum(((R_hx[good] - R_g[good]) / np.sqrt(S_hx[good] ** 2 + S_g[good] ** 2)) ** 2)),
            "dof": int(good.sum()),
            "alive_fraction_per_box": {b: float(d / dur) for b, d in alive_dur.items()},
        }
        ax = axB[0, j]
        ax.stairs(net_obs / w, eb, color=C_OBS, lw=1.6, label=f"observed only (net {net_obs.sum():,.0f})")
        ax.fill_between(ctr, (net_rec - np.sqrt(V_rec)) / w, (net_rec + np.sqrt(V_rec)) / w, step="mid",
                        color=C_BAND, alpha=0.6, lw=0)
        ax.stairs(net_rec / w, eb, color=C_REC, lw=1.8, label=f"recovered (net {net_rec.sum():,.0f}, ±1σ analytic)")
        ax.stairs(net_fill / w, eb, color=C_FILL, lw=1.3, ls="--", label=f"fillers = lost particles ({net_fill.sum():,.0f})")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(eb[0], eb[-1])
        ax.set_ylabel("net counts / keV")
        ax.set_title(f"{v['name']}, saturated segment T0+{seg[0]:.2f}…{seg[1]:.2f} s", fontsize=12)
        ax.legend(loc="lower left", frameon=False)
        ax = axB[1, j]
        ax.axhline(1, color="grey", lw=0.7)
        ax.errorbar(ctr[good], R_g[good], yerr=S_g[good], fmt="s", ms=5, color=C_EXT, ecolor=C_EXT,
                    capsize=2, label="GBM-predicted 3·G(seg) / Σ G(alive)", zorder=3)
        ax.errorbar(ctr[good] * 1.03, R_hx[good], yerr=S_hx[good], fmt="o", ms=4.5, color=C_REC, ecolor=C_REC,
                    capsize=2, label="HE recovered / observed", zorder=4)
        ax.set_xscale("log")
        ax.set_ylabel("recovered / observed")
        ax.set_xlabel("deposited energy (keV)")
        ax.legend(loc="upper left", frameon=False, ncol=1)
        chi2 = nums["spectrum"]["chi2_hxmt_vs_gbm_pred"]; dof = nums["spectrum"]["dof"]
        ax.text(0.98, 0.06, f"χ²/dof HE vs GBM-pred = {chi2:.1f}/{dof}", transform=ax.transAxes,
                ha="right", fontsize=10)
        lo_, hi_ = np.nanmin(np.r_[R_g[good] - S_g[good], R_hx[good] - S_hx[good]]), \
            np.nanmax(np.r_[R_g[good] + S_g[good], R_hx[good] + S_hx[good]])
        ax.set_ylim(max(0.8, lo_ - 0.05), hi_ + 0.12)
        numbers["views"][view] = nums
        res[view] = dict(band_series=band_series, hr_series=hr_series, hr_def=v["hr"], eb=eb, w=w, ctr=ctr,
                         net_obs=net_obs, net_rec=net_rec, V_rec=V_rec, net_fill=net_fill,
                         R_hx=R_hx, S_hx=S_hx, R_g=R_g, S_g=S_g, good=good)

    figA.suptitle(f"{cfg['label']}: band-resolved recovery through the FIFO-saturated phase "
                  f"(0.1 s bins; red band = segment used for the spectrum)", fontsize=12.5, y=0.995)
    figA.savefig(OUT / f"tier2_{args.burst}_bands_hr.pdf", bbox_inches="tight")
    figA.savefig(OUT / f"tier2_{args.burst}_bands_hr.png", dpi=170, bbox_inches="tight")
    figB.suptitle(f"{cfg['label']}: counts spectrum of the saturated segment — what was lost, what was put back",
                  fontsize=12.5, y=0.995)
    figB.savefig(OUT / f"tier2_{args.burst}_segment_spectrum.pdf", bbox_inches="tight")
    figB.savefig(OUT / f"tier2_{args.burst}_segment_spectrum.png", dpi=170, bbox_inches="tight")
    # ═══════════ CsI-only slide versions ═══════════
    r = res["csi"]; v = VIEWS["csi"]
    figC, axC = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.08, "wspace": 0.14})
    panels = [(0, 0, v["bands"][0]), (0, 1, v["bands"][1]), (1, 0, v["bands"][2])]
    for (pi, pj, (lo, hi)) in panels:
        ax = axC[pi, pj]; bs = r["band_series"][(lo, hi)]
        ax.axvspan(*seg, color="tab:red", alpha=0.06, lw=0, zorder=0)
        ax.fill_between(x, 0, bs["n_obs"], step="mid", color=C_OBS, alpha=0.35, lw=0, zorder=1)
        ax.fill_between(x, bs["n_rec"] - bs["s_rec"], bs["n_rec"] + bs["s_rec"], step="mid",
                        color=C_BAND, alpha=0.7, lw=0, zorder=2)
        ax.step(x, bs["n_obs"], where="mid", color=C_OBS, lw=1.0, label="HE observed only", zorder=3)
        ax.step(x, bs["n_rec"], where="mid", color=C_REC, lw=1.3, label="HE recovered (±1σ analytic)", zorder=4)
        ax.fill_between(x, bs["n_gbm"] - bs["s_gbm"], bs["n_gbm"] + bs["s_gbm"], step="mid",
                        color=C_EXT, alpha=0.22, lw=0, zorder=3)
        sc = numbers["views"]["csi"]["bands"][f"{lo}-{hi}"]["gbm_scale"]
        ax.step(x, bs["n_gbm"], where="mid", color=C_EXT, lw=1.1,
                label=f"Fermi/GBM {'+'.join(cfg['gbm']['dets'])} ×{sc:.2f}", zorder=5)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlim(*xl)
        ax.set_ylim(0, 1.18 * np.nanmax(bs["n_rec"][view_win] + bs["s_rec"][view_win]))
        ax.text(0.015, 0.88, f"CsI {lo}–{hi} keV", transform=ax.transAxes, fontweight="bold")
        if (pi, pj) == (0, 0):
            ax.legend(loc="upper right", frameon=False)
        if pj == 0:
            ax.set_ylabel("net counts / s")
        if pi == 1:
            ax.set_xlabel(f"time since T0 (s)   [T0 = {cfg['t0']} UTC]")
    ax = axC[1, 1]
    (hlo, hhi), (slo, shi) = r["hr_def"]
    ax.axvspan(*seg, color="tab:red", alpha=0.06, lw=0, zorder=0)
    for key, col, lab, z in (("obs", C_OBS, "observed only", 3), ("rec", C_REC, "recovered", 4),
                             ("gbm", C_EXT, "Fermi/GBM", 5)):
        hr, shr = r["hr_series"][key]
        ok = np.isfinite(hr) & view_win
        off = {"obs": -0.012, "rec": 0.0, "gbm": 0.012}[key]
        ax.errorbar(x[ok] + off, hr[ok], yerr=shr[ok], fmt="o", ms=3.2, color=col, ecolor=col,
                    elinewidth=0.8, capsize=0, label=lab, zorder=z)
    ymax_hr = np.nanmax([np.nanmax((r["hr_series"][k][0] + r["hr_series"][k][1])[view_win]) for k in r["hr_series"]])
    ax.set_ylim(0, 1.45 * ymax_hr); ax.set_xlim(*xl)
    ax.text(0.015, 0.88, f"HR ({hlo}–{hhi})/({slo}–{shi}) keV", transform=ax.transAxes, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=9)
    ax.set_xlabel(f"time since T0 (s)   [T0 = {cfg['t0']} UTC]")
    figC.suptitle(f"{cfg['label']}: HE CsI band-resolved recovery through the saturated phase (0.1 s bins)",
                  fontsize=12.5, y=0.995)
    figC.savefig(OUT / f"tier2_{args.burst}_csi_bands_hr.pdf", bbox_inches="tight")
    figC.savefig(OUT / f"tier2_{args.burst}_csi_bands_hr.png", dpi=170, bbox_inches="tight")

    figD, axD = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1.3], "hspace": 0.06})
    eb, w, ctr, good = r["eb"], r["w"], r["ctr"], r["good"]
    ax = axD[0]
    ax.stairs(r["net_obs"] / w, eb, color=C_OBS, lw=1.6, label=f"observed only (net {r['net_obs'].sum():,.0f})")
    ax.fill_between(ctr, (r["net_rec"] - np.sqrt(r["V_rec"])) / w, (r["net_rec"] + np.sqrt(r["V_rec"])) / w,
                    step="mid", color=C_BAND, alpha=0.6, lw=0)
    ax.stairs(r["net_rec"] / w, eb, color=C_REC, lw=1.8, label=f"recovered (net {r['net_rec'].sum():,.0f}, ±1σ analytic)")
    ax.stairs(r["net_fill"] / w, eb, color=C_FILL, lw=1.3, ls="--", label=f"fillers = lost particles ({r['net_fill'].sum():,.0f})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(eb[0], eb[-1])
    ax.set_ylabel("net counts / keV")
    ax.set_title(f"{cfg['label']} CsI, saturated segment T0+{seg[0]:.2f}…{seg[1]:.2f} s", fontsize=12)
    ax.legend(loc="lower left", frameon=False)
    ax = axD[1]
    ax.axhline(1, color="grey", lw=0.7)
    ax.errorbar(ctr[good], r["R_g"][good], yerr=r["S_g"][good], fmt="s", ms=5, color=C_EXT, ecolor=C_EXT,
                capsize=2, label="GBM-predicted 3·G(seg) / Σ G(alive)", zorder=3)
    ax.errorbar(ctr[good] * 1.03, r["R_hx"][good], yerr=r["S_hx"][good], fmt="o", ms=4.5, color=C_REC, ecolor=C_REC,
                capsize=2, label="HE recovered / observed", zorder=4)
    ax.set_xscale("log"); ax.set_ylabel("recovered / observed"); ax.set_xlabel("deposited energy (keV)")
    ax.legend(loc="upper left", frameon=False)
    sp = numbers["views"]["csi"]["spectrum"]
    ax.text(0.98, 0.06, f"χ²/dof HE vs GBM-pred = {sp['chi2_hxmt_vs_gbm_pred']:.1f}/{sp['dof']}",
            transform=ax.transAxes, ha="right", fontsize=10)
    lo_ = np.nanmin(np.r_[r["R_g"][good] - r["S_g"][good], r["R_hx"][good] - r["S_hx"][good]])
    hi_ = np.nanmax(np.r_[r["R_g"][good] + r["S_g"][good], r["R_hx"][good] + r["S_hx"][good]])
    ax.set_ylim(max(0.8, lo_ - 0.05), hi_ + 0.12)
    figD.savefig(OUT / f"tier2_{args.burst}_csi_segment_spectrum.pdf", bbox_inches="tight")
    figD.savefig(OUT / f"tier2_{args.burst}_csi_segment_spectrum.png", dpi=170, bbox_inches="tight")

    with open(OUT / f"tier2_{args.burst}_numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)
    for view, nums in numbers["views"].items():
        print(f"== {view}")
        for b, d in nums["bands"].items():
            print(f"  band {b}: scale {d['gbm_scale']:.3f}; filler bins {d['n_filler_bins']}: "
                  f"obs/GBM {d['fillerbins_obs_over_gbm']:.3f}  rec/GBM {d['fillerbins_rec_over_gbm']:.3f}"
                  f" ± {d['fillerbins_rec_over_gbm_sigma']:.3f}")
        for k, d in nums["hr"].items():
            print(f"  HR {k}: {d}")
        sp = nums["spectrum"]
        print(f"  spectrum: total rec/obs {sp['total_ratio_hxmt']:.3f} vs GBM-pred {sp['total_ratio_gbm_pred']:.3f}; "
              f"chi2/dof {sp['chi2_hxmt_vs_gbm_pred']:.1f}/{sp['dof']}; alive frac {sp['alive_fraction_per_box']}")
        print(f"  ratio log-slope dlnR/dlnE: HE {sp['ratio_logslope_hxmt'][0]:+.3f} ± {sp['ratio_logslope_hxmt'][1]:.3f}; "
              f"GBM-pred {sp['ratio_logslope_gbm_pred'][0]:+.3f} ± {sp['ratio_logslope_gbm_pred'][1]:.3f}")


if __name__ == "__main__":
    main()
