#!/usr/bin/env python3
"""Tier 1: energy-resolved injection validation of the FIFO-gap recovery.

Fake 30 ms gaps are injected on one box of GRB 250919A (`blink sat inject`,
233 gap centres from T0-4.8 s to T0+19.9 s at 0.1 s spacing, avoiding real
FIFO resets by >= 0.1 s), the other two boxes recover them, and the recovered
fillers (channel + pulse_width copied at even quantiles from the reference
boxes' in-gap distribution) are compared with the withheld truth events of
the target box, gap by gap and pooled over all gaps of all three boxes.

Three views of the same particles:
  nai : pulse_width in [54,70], channel >= 20, deposited keV from the CALDB
        NaI gain (18-unit average);  bands 20-50 / 50-100 / 100-250 keV
  csi : pulse_width in [90,257], channel >= 30, deposited keV from the
        per-burst CsI LUT (csi_ch2e_250919.npy); bands 70-150/150-300/300-700
  all : every particle, raw pulse-height channel (wrap ch<20 -> +256)

Errors: truth bins are Poisson. Recovered bins take the analytic covariance
of analysis/recovery_cov.py (S diag(C) S^T + k-term) evaluated on the
band-selected event stream, i.e. the reference boxes' Poisson scatter
propagated through the recovery, NOT sqrt(N_fill). Per-gap pulls add the
Poisson floor of the truth (include_u=True, same leg as injection_pull.py).

Inputs (produced by scripts/talks/talk20_spec_run_inject.sh):
  data/talk20_spec/inject_250919a/{events,gapcov,gapbins,truth}_{a,b,c}.csv
  data/recon_cache/250919A_recon.csv   (truth events with channel+pulse_width)

Run from blink/:
  .venv/bin/python scripts/talks/talk20_spec_tier1_injection.py
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../bursts",)]  # sibling and shared script modules
sys.path.insert(0, str(HERE.parent / "he_nai_cal"))
sys.path.insert(0, str(HERE.parent.parent / "analysis"))
import recovery_cov as rc  # noqa: E402
from plot_hxmt_vs_ibis_bands import channel_to_kev_lut  # noqa: E402

T0 = 432865758.0
INJ = HERE.parent.parent / "data/talk20_spec/inject_250919a"
RECON = HERE.parent.parent / "data/recon_cache/250919A_recon.csv"
CSI_LUT = HERE.parent.parent / "data/hxmt_aux/csi_ch2e_250919.npy"
OUT = Path("/Users/skyair/Developer/ihep/talk-20/figs/spectrum")
BOXES = ("a", "b", "c")
BURST_PHASE = (4.0, 12.0)   # s rel. T0: main emission (CsI net rate elevated)

NAI_PW, CSI_PW = (54, 70), (90, 257)
NAI_CH_MIN, CSI_CH_MIN = 20, 30
NAI_BANDS = [(20, 50), (50, 100), (100, 250)]
CSI_BANDS = [(70, 150), (150, 300), (300, 700)]
ALL_BANDS = [(20, 60), (60, 120), (120, 200), (200, 256)]   # channel (wrap region >= 256 excluded from bands)

C_TRUTH, C_REC, C_BAND = "#e8792b", "#2f6db5", "#9ec1e6"


# ────────────────────────── loading ──────────────────────────

def load_truth_intervals(box):
    rows = []
    with open(INJ / f"truth_{box}.csv") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["gap_id"]), float(r["t_start"]), float(r["t_stop"]),
                         int(r["n_truth"]), int(r["n_fill"])))
    return rows


def load_recon_box(box):
    """Observed (EVT) events of one box from the recon cache: met, channel, pw."""
    df = pd.read_csv(RECON, usecols=["box", "type", "met", "channel", "pulse_width"])
    df = df[(df["box"] == box.upper()) & (df["type"] == "EVT")]
    return (df["met"].to_numpy(float), df["channel"].to_numpy(int),
            df["pulse_width"].to_numpy(int))


def load_fill(box):
    df = pd.read_csv(INJ / f"events_{box}.csv",
                     usecols=["box", "type", "met", "channel", "pulse_width"])
    df = df[np.isfinite(df["met"])]
    fill = df[df["type"] == "FILL_GAP"]
    return df, (fill["met"].to_numpy(float), fill["channel"].to_numpy(int),
                fill["pulse_width"].to_numpy(int))


def events_dict(df, mask):
    """recovery_cov event dict from a (pre-filtered) DataFrame."""
    d = df[mask]
    return {"box": d["box"].to_numpy(str), "type": d["type"].to_numpy(str),
            "met": d["met"].to_numpy(float), "channel": d["channel"].to_numpy(int)}


# ────────────────────────── selections ──────────────────────────

E_NAI = channel_to_kev_lut()
E_CSI = np.load(CSI_LUT)


def select(ch, pw, view):
    """Return (mask, x) where x is the spectral coordinate of the view."""
    ch = np.asarray(ch); pw = np.asarray(pw)
    if view == "nai":
        m = (pw >= NAI_PW[0]) & (pw <= NAI_PW[1]) & (ch >= NAI_CH_MIN)
        return m, E_NAI[np.clip(ch, 0, 255)]
    if view == "csi":
        e = E_CSI[np.clip(ch, 0, 255)]
        m = (pw >= CSI_PW[0]) & (pw <= CSI_PW[1]) & (ch >= CSI_CH_MIN) & np.isfinite(e)
        return m, e
    if view == "all":
        return np.ones(len(ch), bool), np.where(ch < 20, ch + 256, ch).astype(float)
    raise ValueError(view)


def channel_range_for(view, lo, hi):
    """Raw-channel closed range whose spectral coordinate lies in [lo, hi)."""
    ch = np.arange(256)
    _, x = select(ch, np.full(256, 60 if view == "nai" else 150), view)
    if view == "all":
        chs = ch[(x >= lo) & (x < hi)]
        return int(chs.min()), int(chs.max())
    ok = np.isfinite(x) & (x >= lo) & (x < hi)
    chs = ch[ok]
    return int(chs.min()), int(chs.max())


VIEWS = {
    "nai": dict(bins=np.geomspace(20, 350, 15), bands=NAI_BANDS, pw=NAI_PW,
                label="NaI (pulse width 54–70), deposited keV"),
    "csi": dict(bins=np.geomspace(70, 800, 15), bands=CSI_BANDS, pw=CSI_PW,
                label="CsI (pulse width 90–257), deposited keV"),
    "all": dict(bins=np.arange(20, 276, 8), bands=ALL_BANDS, pw=None,
                label="all particles, raw pulse-height channel"),
}


# ────────────────────────── analytic fill variance ──────────────────────────

def fill_var_per_gap(df, blocks, bins, box, chlo, chhi, pw_range, gap_rows):
    """Diagonal Var(fill) per injected gap for one channel range (and optional
    pulse-width window), from recovery_cov on the band-selected stream.
    Returns (var_meas, n_fill) arrays aligned with gap_rows."""
    m = (df["channel"] >= chlo) & (df["channel"] <= chhi)
    if pw_range is not None:
        m &= (df["pulse_width"] >= pw_range[0]) & (df["pulse_width"] <= pw_range[1])
    ev = events_dict(df, m.to_numpy())
    edges = np.array(sorted({t for _, s, e, _, _ in gap_rows for t in (s, e)}))
    N, cov = rc.cov_matrix(ev, blocks, bins, edges, box=box.upper(), include_u=False)
    idx = np.searchsorted(edges, [s for _, s, _, _, _ in gap_rows])
    return np.diag(cov)[idx], N[idx]


# ────────────────────────── main ──────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10.5,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "pdf.fonttype": 42,
        "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
    })

    # per box: truth / fill particles tagged with gap id + gap-phase flag
    truth = {v: [] for v in VIEWS}      # x values
    fill = {v: [] for v in VIEWS}
    gap_meta = []                       # (box, gap_id, t_center_rel)
    per_gap = {v: {} for v in VIEWS}    # view -> band -> lists of (truth, fill, var)
    numbers = {}

    cache = INJ / "tier1_stage_cache.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            truth, fill, gap_meta, per_gap = pickle.load(f)
        print(f"loaded stage cache {cache}")
        boxes_to_run = ()
    else:
        boxes_to_run = BOXES
    for box in boxes_to_run:
        rows = load_truth_intervals(box)
        tmet, tch, tpw = load_recon_box(box)
        df, (fmet, fch, fpw) = load_fill(box)
        blocks = rc.load_blocks(str(INJ / f"gapcov_{box}.csv"))
        bins = rc.load_bins(str(INJ / f"gapbins_{box}.csv"))

        # truth particles inside each injected gap (from the recon cache, which
        # carries pulse_width; counts must equal inject's n_truth)
        gid_t = np.full(len(tmet), -1)
        gid_f = np.full(len(fmet), -1)
        n_mismatch = 0
        for gid, s, e, n_truth, n_fill in rows:
            a, b = np.searchsorted(tmet, (s, e))
            # inject counts the GPS-second marker slot as an event when an
            # integer MET falls in the gap; the recon cache may not carry it.
            assert abs((b - a) - n_truth) <= 1, (box, gid, b - a, n_truth)
            if b - a != n_truth:
                n_mismatch += 1
            gid_t[a:b] = gid
            gid_f[(fmet >= s) & (fmet < e)] = gid
            gap_meta.append((box, gid, 0.5 * (s + e) - T0))
        tm = (gid_t >= 0) & ~((tch == 0) & (tpw == 0))   # drop GPS-second marker slots
        assert (gid_f >= 0).all()
        if n_mismatch:
            print(f"  ({n_mismatch} gaps differ by one second-marker slot from inject's n_truth)")
        print(f"box {box}: {len(rows)} gaps, truth {tm.sum()}, fill {len(fmet)} "
              f"(fill/truth = {len(fmet) / tm.sum():.4f})")

        for view, cfg in VIEWS.items():
            mt, xt = select(tch[tm], tpw[tm], view)
            mf, xf = select(fch, fpw, view)
            truth[view].append((xt[mt], gid_t[tm][mt], np.full(mt.sum(), box)))
            fill[view].append((xf[mf], gid_f[mf], np.full(mf.sum(), box)))
            for lo, hi in cfg["bands"]:
                chlo, chhi = channel_range_for(view, lo, hi)
                var, nf = fill_var_per_gap(df, blocks, bins, box, chlo, chhi,
                                           cfg["pw"], rows)
                nt = np.array([np.sum((gid_t[tm][mt] == gid) & (xt[mt] >= lo) & (xt[mt] < hi))
                               for gid, *_ in rows])
                nf2 = np.array([np.sum((gid_f[mf] == gid) & (xf[mf] >= lo) & (xf[mf] < hi))
                                for gid, *_ in rows])
                # cov_matrix's N counts by raw channel range; the view's x-binning
                # is a deterministic function of channel, so the two agree up to
                # a boundary event sitting exactly on a gap edge.
                assert np.abs(nf.astype(int) - nf2).max() <= 1, (view, lo, hi)
                per_gap[view].setdefault((lo, hi), []).append(
                    (nt, nf2, var, np.array([0.5 * (s + e) - T0 for _, s, e, _, _ in rows])))

    if boxes_to_run:
        with open(cache, "wb") as f:
            pickle.dump((truth, fill, gap_meta, per_gap), f)

    # ── pooled spectra + ratio (figure 1: NaI + CsI; figure 1b: all-channel) ──
    def pooled(view, phase=None):
        cfg = VIEWS[view]
        xt = np.concatenate([x for x, _, _ in truth[view]])
        xf = np.concatenate([x for x, _, _ in fill[view]])
        if phase is not None:
            # keep particles whose gap centre lies in the phase window
            meta = {(b, g): t for b, g, t in gap_meta}
            keep_t = np.concatenate([
                np.array([phase[0] <= meta[(b, g)] < phase[1] for g, b in zip(gids, bx)], bool)
                for _, gids, bx in truth[view]])
            keep_f = np.concatenate([
                np.array([phase[0] <= meta[(b, g)] < phase[1] for g, b in zip(gids, bx)], bool)
                for _, gids, bx in fill[view]])
            xt, xf = xt[keep_t], xf[keep_f]
        ht, _ = np.histogram(xt, cfg["bins"])
        hf, _ = np.histogram(xf, cfg["bins"])
        return ht, hf

    def band_var(view, phase=None):
        """Summed analytic Var(fill) and counts per band over gaps (optionally
        restricted to a phase window)."""
        out = {}
        for band, lst in per_gap[view].items():
            nt = np.concatenate([a for a, _, _, _ in lst])
            nf = np.concatenate([a for _, a, _, _ in lst])
            var = np.concatenate([a for _, _, a, _ in lst])
            tc = np.concatenate([a for _, _, _, a in lst])
            if phase is not None:
                k = (tc >= phase[0]) & (tc < phase[1])
                nt, nf, var = nt[k], nf[k], var[k]
            out[band] = (nt.sum(), nf.sum(), var.sum())
        return out

    def ratio_err(ht, hf, view):
        """Per-bin sigma of hf/ht: truth Poisson + analytic fill variance scaled
        from the enclosing band (fill variance is close to Poisson-like in the
        reference counts, so var/N of the band is applied to each bin)."""
        cfg = VIEWS[view]
        bv = band_var(view)
        ctr = np.sqrt(cfg["bins"][:-1] * cfg["bins"][1:]) if view != "all" \
            else 0.5 * (cfg["bins"][:-1] + cfg["bins"][1:])
        vf = np.zeros_like(hf, float)
        for (lo, hi), (nt, nf, var) in bv.items():
            k = (ctr >= lo) & (ctr < hi)
            vf[k] = hf[k] * (var / nf if nf > 0 else 1.0)
        # bins outside all bands: fall back to the nearest band's var/N
        k0 = vf == 0
        if k0.any():
            vf[k0] = hf[k0] * np.mean([v / n for _, (t, n, v) in bv.items() if n > 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = hf / ht
            s = r * np.sqrt(vf / np.maximum(hf, 1) ** 2 + 1.0 / np.maximum(ht, 1))
        return r, s, vf

    numbers["pooled"] = {}
    for view in VIEWS:
        ht, hf = pooled(view)
        r, s, vf = ratio_err(ht, hf, view)
        ok = ht >= 20
        chi2 = float(np.sum(((r[ok] - 1) / s[ok]) ** 2))
        bv = band_var(view)
        bv_burst = band_var(view, BURST_PHASE)
        numbers["pooled"][view] = {
            "n_truth": int(ht.sum()), "n_fill": int(hf.sum()),
            "total_ratio": float(hf.sum() / ht.sum()),
            "chi2_ratio_vs_1": chi2, "dof": int(ok.sum()),
            "bands": {f"{lo}-{hi}": {
                "n_truth": int(nt), "n_fill": int(nf),
                "ratio": float(nf / nt), "sigma_ratio": float(np.sqrt(var + nt) / nt)}
                for (lo, hi), (nt, nf, var) in bv.items()},
            "bands_burst_phase": {f"{lo}-{hi}": {
                "n_truth": int(nt), "n_fill": int(nf),
                "ratio": float(nf / nt) if nt else float("nan"),
                "sigma_ratio": float(np.sqrt(var + nt) / nt) if nt else float("nan")}
                for (lo, hi), (nt, nf, var) in bv_burst.items()},
        }

    # ── per-gap pulls per band ──
    numbers["pulls"] = {}
    pulls = {v: {} for v in VIEWS}
    GROUP = {"nai": 10, "csi": 1, "all": 1}   # consecutive gaps summed per pull

    def grouped(a, g):
        n = (len(a) // g) * g
        return a[:n].reshape(-1, g).sum(axis=1)

    for view in VIEWS:
        numbers["pulls"][view] = {}
        g = GROUP[view]
        for (lo, hi), lst in per_gap[view].items():
            nt = np.concatenate([grouped(a, g) for a, _, _, _ in lst])
            nf = np.concatenate([grouped(a, g) for _, a, _, _ in lst])
            var = np.concatenate([grouped(a, g) for _, _, a, _ in lst])
            sig = np.sqrt(var + nf)            # include_u leg: + Poisson floor
            ok = (sig > 0) & (nt > 0)
            pull = (nf[ok] - nt[ok]) / sig[ok]
            numbers["pulls"][view][f"{lo}-{hi}"] = {
                "gaps_per_pull": g,
                "n_gaps": int(ok.sum()), "mean": float(pull.mean()),
                "std": float(pull.std(ddof=1)),
                "chi2_dof": float(np.mean(pull ** 2)),
                "median_truth_per_gap": float(np.median(nt[ok])),
                "bias_pct": float(100 * (nf[ok].sum() / nt[ok].sum() - 1)),
            }
            pulls[view][(lo, hi)] = (nt, nf, var, pull)

    # ── figure 1: NaI + CsI spectra with ratio panels ──
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.8), sharex="col",
                             gridspec_kw={"height_ratios": [2.6, 1.15], "hspace": 0.06,
                                          "wspace": 0.22})
    for j, view in enumerate(("nai", "csi")):
        cfg = VIEWS[view]
        ht, hf = pooled(view)
        r, s, vf = ratio_err(ht, hf, view)
        b = cfg["bins"]; w = np.diff(b); ctr = np.sqrt(b[:-1] * b[1:])
        ax = axes[0, j]
        ax.stairs(ht / w, b, color=C_TRUTH, lw=2.0, label=f"truth (deleted, n={ht.sum():,})")
        ax.fill_between(ctr, (hf - np.sqrt(vf)) / w, (hf + np.sqrt(vf)) / w,
                        step="mid", color=C_BAND, alpha=0.55, lw=0, label="recovered ±1σ (analytic)")
        ax.stairs(hf / w, b, color=C_REC, lw=1.6, label=f"recovered (n={hf.sum():,})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(b[0], b[-1])
        ax.set_ylabel("counts / keV")
        ax.set_title(cfg["label"], fontsize=12.5)
        ax.legend(loc="lower left", frameon=False)
        for lo, hi in cfg["bands"]:
            ax.axvline(lo, color="grey", lw=0.6, ls=":")
        ax.axvline(cfg["bands"][-1][1], color="grey", lw=0.6, ls=":")
        ax = axes[1, j]
        ax.axhspan(0.95, 1.05, color="grey", alpha=0.12, lw=0)
        ax.axhline(1, color="k", lw=0.8)
        ax.errorbar(ctr, r, yerr=s, fmt="o", ms=4, color=C_REC, ecolor=C_REC, capsize=2)
        for k, (lo, hi) in enumerate(cfg["bands"]):
            nt, nf, var = band_var(view)[(lo, hi)]
            rb = nf / nt; sb = np.sqrt(var + nt) / nt
            ax.hlines(rb, lo, hi, color=C_TRUTH, lw=2.2)
            ax.fill_between([lo, hi], rb - sb, rb + sb, color=C_TRUTH, alpha=0.18, lw=0)
            ax.text(np.sqrt(lo * hi), 1.175 if k % 2 == 0 else 1.115,
                    f"{lo}–{hi}: {rb:.3f}±{sb:.3f}", ha="center", fontsize=9.2, color=C_TRUTH)
        ax.set_ylim(0.78, 1.24)
        ax.set_xscale("log")
        ax.set_ylabel("recovered / truth")
        ax.set_xlabel("deposited energy (keV)")
        chi2 = numbers["pooled"][view]["chi2_ratio_vs_1"]; dof = numbers["pooled"][view]["dof"]
        ax.text(0.98, 0.06, f"χ²/dof vs 1 = {chi2:.1f}/{dof}", transform=ax.transAxes,
                ha="right", fontsize=10)
    fig.suptitle("GRB 250919A injection test: 699 fake 30 ms gaps (233 per box), recovered fillers vs withheld truth",
                 fontsize=12.5, y=0.995)
    fig.savefig(OUT / "tier1_injection_spectrum.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tier1_injection_spectrum.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ── figure 1b: all-particle channel spectrum ──
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.6, 1.15], "hspace": 0.06})
    cfg = VIEWS["all"]; ht, hf = pooled("all"); r, s, vf = ratio_err(ht, hf, "all")
    b = cfg["bins"]; w = np.diff(b); ctr = 0.5 * (b[:-1] + b[1:])
    ax = axes[0]
    ax.stairs(ht, b, color=C_TRUTH, lw=2.0, label=f"truth (deleted, n={ht.sum():,})")
    ax.fill_between(ctr, hf - np.sqrt(vf), hf + np.sqrt(vf), step="mid", color=C_BAND,
                    alpha=0.55, lw=0, label="recovered ±1σ (analytic)")
    ax.stairs(hf, b, color=C_REC, lw=1.6, label=f"recovered (n={hf.sum():,})")
    ax.set_yscale("log"); ax.set_ylabel("counts / 8 channels")
    ax.set_title(cfg["label"], fontsize=12.5); ax.legend(loc="upper right", frameon=False)
    ax.axvline(256, color="grey", lw=0.6, ls=":")
    ax.text(257, ax.get_ylim()[0] * 1.5, "wrap", fontsize=9, color="grey")
    ax = axes[1]
    ax.axhspan(0.95, 1.05, color="grey", alpha=0.12, lw=0); ax.axhline(1, color="k", lw=0.8)
    ax.errorbar(ctr, r, yerr=s, fmt="o", ms=4, color=C_REC, ecolor=C_REC, capsize=2)
    ax.set_ylim(0.78, 1.24); ax.set_ylabel("recovered / truth"); ax.set_xlabel("pulse-height channel")
    chi2 = numbers["pooled"]["all"]["chi2_ratio_vs_1"]; dof = numbers["pooled"]["all"]["dof"]
    ax.text(0.98, 0.06, f"χ²/dof vs 1 = {chi2:.1f}/{dof}", transform=ax.transAxes, ha="right", fontsize=10)
    fig.savefig(OUT / "tier1_injection_spectrum_allch.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tier1_injection_spectrum_allch.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ── figure 2: per-gap pulls per band ──
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), sharex=True, sharey="row")
    xx = np.linspace(-4, 4, 200)
    for i, view in enumerate(("nai", "csi")):
        for j, (lo, hi) in enumerate(VIEWS[view]["bands"]):
            nt, nf, var, pull = pulls[view][(lo, hi)]
            ax = axes[i, j]
            ax.hist(pull, bins=np.arange(-4.25, 4.5, 0.5), density=True, color=C_BAND,
                    edgecolor=C_REC, lw=0.8)
            ax.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color=C_TRUTH, lw=1.6)
            st = numbers["pulls"][view][f"{lo}-{hi}"]
            unit = "keV"
            grp = f"{st['n_gaps']} groups of {st['gaps_per_pull']} gaps" if st["gaps_per_pull"] > 1 \
                else f"{st['n_gaps']} gaps"
            ax.text(0.04, 0.95, f"{'NaI' if view == 'nai' else 'CsI'} {lo}–{hi} {unit}\n"
                    f"{grp}, median {st['median_truth_per_gap']:.0f} truth/pull\n"
                    f"pull mean {st['mean']:+.2f}, std {st['std']:.2f}\n"
                    f"bias {st['bias_pct']:+.1f}%",
                    transform=ax.transAxes, va="top", fontsize=9.8)
            ax.set_xlim(-4, 4)
            if i == 1:
                ax.set_xlabel("pull = (recovered − truth) / σ")
            if j == 0:
                ax.set_ylabel("density")
    fig.suptitle("Per-gap pulls per band: σ² = analytic recovery variance + Poisson floor of the truth",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "tier1_injection_pulls.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tier1_injection_pulls.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    with open(OUT / "tier1_numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
