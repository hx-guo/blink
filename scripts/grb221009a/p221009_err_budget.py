#!/usr/bin/env python3
"""GRB 221009A: per-1-s error budget of the dead-time-corrected HE light curve,
and the HXMT/GECAM ratio statistics with error bars.

Deterministic (no random numbers): every term is propagated analytically.
Run from the blink repo root after scripts/grb221009a/p221009_err_recvar.py:

    ./.venv/bin/python scripts/grb221009a/p221009_err_budget.py

Light curve definitions are those of scripts/talks/make_f10.py / numbers_221009a.py
(1-s bins, left edges -50..699 s rel. T0 = 339945423, gross background B =
mean of the bins before -10 s, R_src = (R_net + B)/f_live - B with
f_live = 18-detector mean of 1 - D/L_cyc, NaN where f_live <= 0.05); the
script asserts it reproduces their arrays before adding errors.

Error terms, all expressed on R_src (evt/s):
  statistical
    sig_obs    observed-event Poisson, sqrt(N_obs)/f
    sig_fill   gap-fill recovery, sqrt(Var_rec - N_obs)/f, Var_rec from the
               three-table model (S diag(C) S^T + k + r terms, include_u=False)
    sig_fq     f_live counter quantisation (D, L_cyc integer in 16 us units)
    sig_bstat  Poisson error of B (fully correlated across bins)
  systematic (reported separately, not added to the statistical error)
    sig_align  eng-frame / event-time alignment: frame k is measured from the
               SAA on/off edges to start at k + phi, phi in [0, PHI_MAX];
               term = |R_src(phi=PHI_MAX) - R_src(phi=0)|
    sig_det    detector averaging: |sum_box N_b/f_b - N/f_mean|
    sig_degsys degenerate (no cross-reference) gap ramp bias proxy,
               sum over gaps of sys_bias_scale * n_fill_in_bin / f
    sig_bsys   background drift: |B_orb(t) - B| with B_orb the orbital model of
               scripts/grb221009a/p221009_timing_bkg.py (gross rate linear in the
               engineering Cnt_LargeEvt, calibrated off-source); the 1K
               detector-pattern tracer (dets 2+9 vs the other 16) is kept as an
               independent cross-check where the source is negligible
GECAM-C GRD01 LG: Poisson of burst and revisited-orbit background counts, and
a background systematic from the burst-day vs revisit-day residual after a
comparable particle-region exit in the same file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../gecam", "../lib", "../talks",)]  # sibling and shared script modules
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION  # noqa: E402

T0 = 339945423.0
BEFORE, AFTER = 50, 700
OUT = Path("data/p221009/errors")
GECAM_SCALE = 200.0
FLIVE_MIN = 0.05
PHI_MAX = 0.16      # largest frame-start offset allowed by the SAA edges (s)
G_FLOOR = 50.0      # GECAM net cts/s significance cut (make_f10 / numbers)
WINDOWS = {"main": (170, 310), "flare": (440, 610)}
GECAM_BTIME = "data/gecam_c/gcg_btime_221009_13_v12.fits"
GECAM_BKG = "data/gecam_c/gcg_btime_221014_13_v06.fits"
HE_1K = "data/1K/Y202210/20221009-1943/HXMT_20221009T13_HE-Evt_FFFFFF_V1_1K.FITS"


# ───────────────────────────── HXMT events ─────────────────────────────

def load_counts(edges):
    ev = pd.read_csv(OUT / "events.csv", usecols=["box", "type", "met"],
                     dtype={"box": "category", "type": "category", "met": "float64"})
    out = {}
    for b in ("A", "B", "C"):
        m = ev["box"] == b
        t = ev["met"].to_numpy()[m]
        evt = (ev["type"] == "EVT").to_numpy()[m]
        out[b] = (np.histogram(t, bins=edges)[0].astype(float),
                  np.histogram(t[evt], bins=edges)[0].astype(float))
        out["fill_" + b] = np.sort(t[~evt])
    return out


def fill_per_gap(blocks, fills, edges):
    """Degenerate-gap bias proxy per bin: sum_gaps sys_bias_scale * fills_in_bin."""
    acc = np.zeros(len(edges) - 1)
    for _, g in blocks[blocks["type"] == "degenerate"].iterrows():
        ft = fills["fill_" + g["target_box"]]
        j0, j1 = np.searchsorted(ft, [g["t_start"], g["t_stop"] + 1e-9])
        if j1 > j0:
            h = np.histogram(ft[j0:j1], bins=edges)[0]
            acc += g["sys_bias_scale"] * h
    return acc


# ───────────────────────────── engineering ─────────────────────────────

def load_eng():
    """Per-frame D (18 dets), L_cyc, labelled by integer second rel. T0."""
    frames = {}
    for box, code in BOX_CODE.items():
        path = sorted(Path(f"data/1B/2022/20221009/{code}").glob(
            f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
        d = fits.open(path, memmap=True)["HE_Eng"].data
        off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met = d["Time"].astype(float) + off + MET_CORRECTION
        lab = np.round(met - T0).astype(int)
        L = d["Length_Time_Cycle"].astype(float)
        for j in range(6):
            g = BOX_OFFSET[box] + j
            D = d[f"DeadTime_PHODet_{g}"].astype(float)
            for k, dd, ll in zip(lab, D, L):
                frames.setdefault(k, {})[g] = (dd, ll)
    return frames


def eng_on_grid(frames, x):
    n = len(x)
    f_mean = np.full(n, np.nan)
    f_box = {b: np.full(n, np.nan) for b in BOX_CODE}
    sig_fq = np.full(n, np.nan)
    q = 1.0 / np.sqrt(12.0)  # uniform rounding of an integer counter
    for i, xx in enumerate(x):
        fr = frames.get(int(round(xx)))
        if not fr or len(fr) < 18:
            continue
        fi, var = [], []
        for g in range(18):
            D, L = fr[g]
            if L <= 0:
                fi.append(np.nan); var.append(np.nan); continue
            fi.append(1.0 - D / L)
            var.append(q * q * (1.0 / L**2 + D**2 / L**4))
        fi = np.array(fi)
        f_mean[i] = np.nanmean(fi)
        sig_fq[i] = np.sqrt(np.nansum(var)) / 18.0
        for b, o in BOX_OFFSET.items():
            f_box[b][i] = np.nanmean(fi[o:o + 6])
    return f_mean, f_box, sig_fq


def frame_offset_test(frames):
    """Frame-start offset phi from the SAA-mode shut-off (frame 229) and turn-on
    (frame 248) edges: on-fraction of the edge frame (D_edge / D_neighbour)
    against the event time at which each box stops / restarts. Returns rows."""
    ev = pd.read_csv(OUT / "events_saa_edges.csv")
    ev = ev[(ev["type"] == "EVT") & ~((ev["channel"] == 0) & (ev["met"] % 1 == 0))]
    rows = []
    for box in ("A", "B", "C"):
        t = ev.loc[ev["box"] == box, "met"].to_numpy() - T0
        dets = range(BOX_OFFSET[box], BOX_OFFSET[box] + 6)
        Dm = lambda k: np.mean([frames[k][g][0] for g in dets])
        Lc = lambda k: np.mean([frames[k][g][1] for g in dets]) * 16e-6
        pre = t[(t > 228.5) & (t < 230.5)]
        q_off = Dm(229) / Dm(228)
        for tag, t_off in (("last", pre.max()), ("q999", np.quantile(pre, 0.999))):
            rows.append((box, "off_" + tag, t_off, q_off, t_off - q_off * Lc(229) - 229))
        post = t[(t > 247.5) & (t < 250.0)]
        q_on = Dm(248) / Dm(249)
        rows.append((box, "on_first", post.min(), q_on,
                     post.min() - (1 - q_on) * Lc(248) - 248))
    return pd.DataFrame(rows, columns=["box", "edge", "t_event", "on_frac", "phi"])


def f_shift(f, phi):
    """Bin-averaged live fraction if frame k starts at k+phi: the first phi of
    bin k is covered by frame k-1 (harmonic weighting, as counts divide)."""
    fm1 = np.roll(f, 1)
    fm1[0] = np.nan
    inv = (1 - phi) / f + phi / fm1
    return 1.0 / inv


# ───────────────────────────── background tracer ─────────────────────────────

def background_tracer(width=30.0):
    """1K detector-pattern tracer of the gross background B_hat(t).

    Dets 2 and 9 see the burst ~20% harder than the other 16 (their share of
    the main-pulse counts is a = 0.136 versus k/(1+k) = 0.113 for the
    background), so S = (hot - k*oth)/(a - k(1-a)) and B_hat = tot - S.
    k is fixed on T0-990..-10 and T0+2200..+2580 (no burst), a on
    T0+200..230. Valid only where S is negligible against B, because the
    source share a depends on the burst spectrum (0.136 on the main pulse,
    0.153 on the flare): t < 150 s and t >= 570 s.
    """
    d = fits.open(HE_1K, memmap=True)[1].data
    t = d["Time"] - T0
    det = d["Det_ID"]
    edges = np.arange(-990, 2580 + width, width)
    H = np.array([np.histogram(t[det == k], bins=edges)[0] for k in range(18)], float)
    xc = edges[:-1] + width / 2
    hot = H[2] + H[9]
    oth = np.delete(H, [2, 9], axis=0).sum(0)
    tot = H.sum(0)
    ref = (edges[:-1] < -10) | (edges[:-1] >= 2200)
    k = hot[ref].sum() / oth[ref].sum()
    m = (edges[:-1] >= 200) & (edges[:-1] < 230)
    a = hot[m].sum() / tot[m].sum()
    den = a - k * (1 - a)
    S = (hot - k * oth) / den
    sS = np.sqrt(hot + k * k * oth) / abs(den)
    lo = edges[:-1]
    valid = (lo < 150 - width) | (lo >= 570)
    return pd.DataFrame({"t_mid": xc, "tot": tot / width, "S": S / width,
                         "sig_S": sS / width, "B_hat": (tot - S) / width,
                         "valid": valid}), k, a


# ───────────────────────────── GECAM ─────────────────────────────

def gecam(edges_rel):
    g0 = (Time("2022-10-09T13:17:00", scale="utc").tt
          - Time(59215 + 0.00080074074, format="mjd", scale="tt")).sec
    fb, fk = fits.open(GECAM_BTIME), fits.open(GECAM_BKG)
    b, k = fb["SPECTRUM01"].data, fk["SPECTRUM01"].data
    off = fk[0].header["TSTART"] - fb[0].header["TSTART"]
    tb = b["STARTTIME"]
    cb = b["COUNTS"][:, 7:13].sum(1).astype(float)
    ck_raw = k["COUNTS"][:, 7:13].sum(1).astype(float)
    tk = k["STARTTIME"] - off
    ck = np.interp(tb, tk, ck_raw, left=np.nan, right=np.nan)
    eb = b["EXPOSURE"][:, 1] / 0.05
    ek = np.interp(tb, tk, k["EXPOSURE"][:, 1] / 0.05, left=np.nan, right=np.nan)
    trel = tb - g0
    n = np.histogram(trel, bins=edges_rel)[0].astype(float)
    Cb = np.histogram(trel, bins=edges_rel, weights=cb)[0]
    Ck = np.histogram(trel, bins=edges_rel, weights=ck)[0]
    Cb_ec = np.histogram(trel, bins=edges_rel, weights=cb / eb)[0]
    Ck_ec = np.histogram(trel, bins=edges_rel, weights=ck / ek)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        T = n * 0.05
        rate = (Cb - Ck) / T
        sig = np.sqrt(Cb + Ck) / T
        rate_ec = (Cb_ec - Ck_ec) / T
        live = np.histogram(trel, bins=edges_rel, weights=eb)[0] / n

    # background systematic: burst-day minus revisit-day LG after the
    # particle-region passage at T0+860..975 (no burst signal there), as a
    # function of time since exit; applied to the exit at T0+168.
    def r40(tt, cc, lo):
        mm = (tt >= lo) & (tt < lo + 40)
        return cc[mm].sum() / (mm.sum() * 0.05)
    tt_k = tk - g0
    dts, res = [], []
    for lo in np.arange(980, 1500, 40):
        dts.append(lo + 20 - 975.0)
        res.append(r40(trel, cb, lo) - r40(tt_k, ck_raw, lo))
    dts, res = np.array(dts), np.array(res)
    floor = np.std([r40(trel, cb, lo) - r40(tt_k, ck_raw, lo)
                    for lo in np.arange(-1100, -580, 40)])
    x = edges_rel[:-1] + 0.5
    sig_bsys = np.where(x > 168, np.interp(x - 168, dts, np.abs(res)), np.nan)
    sig_bsys = np.maximum(sig_bsys, floor)
    return rate, sig, rate_ec, live, sig_bsys, (dts, res, floor)


# ───────────────────────────── ratio statistics ─────────────────────────────

def wstats(r, s):
    """Inverse-variance mean, chi2/dof about it, and the extra scatter s_int
    that brings chi2/dof to 1 (solved by bisection, deterministic)."""
    w = 1 / s**2
    mu = np.sum(w * r) / np.sum(w)
    chi2 = np.sum((r - mu) ** 2 * w)
    dof = len(r) - 1

    def excess(si):
        ww = 1 / (s**2 + si**2)
        m = np.sum(ww * r) / np.sum(ww)
        return np.sum((r - m) ** 2 * ww) - dof
    if excess(0.0) <= 0:
        s_int = 0.0
    else:
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
        s_int = 0.5 * (lo + hi)
    ww = 1 / (s**2 + s_int**2)
    mu_int = np.sum(ww * r) / np.sum(ww)
    emu_int = 1 / np.sqrt(np.sum(ww))
    return dict(n=len(r), mu=mu, emu=1 / np.sqrt(np.sum(w)), chi2=chi2, dof=dof,
                s_int=s_int, mu_int=mu_int, emu_int=emu_int,
                med=np.median(r),
                rsig=np.subtract(*np.percentile(r, [75, 25])) / 1.349,
                med_err=np.median(s))


def main():
    edges = T0 + np.arange(-BEFORE, AFTER + 1, 1.0)
    x = edges[:-1] - T0
    rv = pd.read_csv(OUT / "recvar_1s.csv")
    assert np.allclose(rv["t"].to_numpy(), x)
    cnt = load_counts(edges)
    n_all = sum(cnt[b][0] for b in "ABC")
    n_obs = sum(cnt[b][1] for b in "ABC")
    assert np.array_equal(n_all, rv["n_all"].to_numpy())
    pre = x < -10
    B = n_all[pre].mean()
    sig_bstat = np.sqrt(n_all[pre].sum()) / pre.sum()

    frames = load_eng()
    f, f_box, sig_fq_f = eng_on_grid(frames, x)
    ok = f > FLIVE_MIN
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = n_all - B
        obs = n_obs - B
        dt = np.where(ok, n_all / f - B, np.nan)

    # cross-check against the paper's pipeline arrays
    from make_f10 import load_hxmt, load_eng_dtau, to_grid
    x_p, _, obs_p, allr_p, _, B_p = load_hxmt(BEFORE, AFTER, 1.0)
    et, _, flive_p, _, _ = load_eng_dtau(BEFORE, AFTER)
    fp = to_grid(x_p, et, flive_p)
    assert np.allclose(x_p, x) and abs(B_p - B) < 1e-6
    assert np.allclose(allr_p, rec) and np.allclose(fp, f, equal_nan=True)

    # statistical terms
    var_fill = np.clip(rv["var_rec"].to_numpy() - n_obs, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_obs = np.sqrt(n_obs) / f
        s_fill = np.sqrt(var_fill) / f
        s_fq = n_all / f**2 * sig_fq_f
    s_bstat = np.full_like(x, sig_bstat)
    s_stat = np.sqrt(s_obs**2 + s_fill**2 + s_fq**2 + s_bstat**2)

    # systematic terms
    offs = frame_offset_test(frames)
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_phi = n_all / f_shift(f, PHI_MAX) - B
        s_align = np.abs(dt_phi - dt)
        dt_box = sum(cnt[b][0] / f_box[b] for b in "ABC") - B
        s_det = np.abs(dt_box - dt)
    blocks = pd.read_csv(OUT / "gapcov.csv")
    blocks["t_start"] -= 0.0
    degsys = fill_per_gap(blocks, cnt, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_degsys = degsys / f
    trc, k_hot, a_hot = background_tracer()
    from p221009_timing_bkg import background_clean
    tl_b, _, B_orb_all, _, _, _ = background_clean()
    B_hat = np.interp(x, tl_b, B_orb_all)
    s_bsys = np.abs(B_hat - B)
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_borb = np.where(ok, n_all / f - B_hat, np.nan)
    s_sys = np.sqrt(np.nan_to_num(s_align)**2 + np.nan_to_num(s_det)**2
                    + s_degsys**2 + s_bsys**2)

    # GECAM
    edges_rel = np.arange(-BEFORE, AFTER + 1, 1.0)
    g_rate, g_sig, g_rate_ec, g_live, g_bsys, g_bsys_tab = gecam(edges_rel)
    from plot_hxmt_vs_gecam import compute_time_offset, load_gecam_btime
    g_met, _, _ = compute_time_offset()
    _, g_ref, _ = load_gecam_btime(g_met, BEFORE, AFTER, 1.0, "lg", True)
    good_g = np.isfinite(g_ref)
    assert np.allclose(g_rate[good_g], g_ref[good_g], rtol=1e-6, atol=1e-6)
    g_rate = np.nan_to_num(g_rate, nan=0.0)

    lc = pd.DataFrame({
        "t": x, "obs": obs, "rec": rec, "dt_corr": dt, "f_live": f,
        "n_obs": n_obs, "n_fill": n_all - n_obs, "var_rec": rv["var_rec"],
        "sig_obs": s_obs, "sig_fill": s_fill, "sig_fq": s_fq, "sig_bstat": s_bstat,
        "sig_stat": s_stat,
        "sig_align": s_align, "sig_det": s_det, "sig_degsys": s_degsys,
        "sig_bsys": s_bsys, "sig_sys": s_sys, "B_hat": B_hat, "dt_corr_Borb": dt_borb,
        "dt_corr_phi": dt_phi, "dt_corr_box": dt_box,
        "gecam_net": g_rate, "gecam_sig": g_sig, "gecam_bsys": g_bsys,
        "gecam_net_ec": g_rate_ec, "gecam_live": g_live,
    })
    lc.to_csv(OUT / "lc_221009a_1s_errors.csv", index=False, float_format="%.6g")
    offs.to_csv(OUT / "frame_offset_saa_edges.csv", index=False, float_format="%.4f")
    trc.to_csv(OUT / "bkg_tracer_1k_30s.csv", index=False, float_format="%.1f")

    # ── typical relative errors ──
    rep = []
    def rel(name, sel):
        d = lc[sel & np.isfinite(lc["dt_corr"]) & (lc["dt_corr"] > 0)]
        cols = ["sig_obs", "sig_fill", "sig_fq", "sig_bstat", "sig_stat",
                "sig_align", "sig_det", "sig_degsys", "sig_bsys", "sig_sys"]
        r = {c: np.median(d[c] / d["dt_corr"]) for c in cols}
        r.update(name=name, n=len(d), rate=np.median(d["dt_corr"]),
                 f=np.median(d["f_live"]))
        rep.append(r)
    rel("peak T0+227", lc["t"] == 227)
    rel("peak T0+262", lc["t"] == 262)
    rel("main 180-300 (all finite)", (lc["t"] >= 180) & (lc["t"] < 300))
    rel("flare peak T0+510", lc["t"] == 510)
    rel("flare 442-593", (lc["t"] >= 442) & (lc["t"] < 593))
    rel("tail 600-700", lc["t"] >= 600)
    rel("precursor/quiet -10..170", (lc["t"] >= -10) & (lc["t"] < 170))
    rep = pd.DataFrame(rep).set_index("name")
    rep.to_csv(OUT / "rel_errors_by_phase.csv", float_format="%.4g")

    # ── ratio statistics ──
    gs = g_rate * GECAM_SCALE
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        win = (x >= lo) & (x < hi)
        sig = win & (g_rate > G_FLOOR) & np.isfinite(rec) & (rec > 0)
        for kind, H, sH_stat, sH_sys in (
                ("rec", rec, np.sqrt(rv["var_rec"].to_numpy() + sig_bstat**2),
                 np.sqrt(np.nan_to_num(s_degsys * f)**2 + s_bsys**2)),
                ("dt", dt, s_stat, s_sys)):
            m = sig & np.isfinite(H)
            r = H[m] / gs[m]
            sG = g_sig[m] / g_rate[m]
            s_st = np.abs(r) * np.sqrt((sH_stat[m] / H[m])**2 + sG**2)
            s_gb = np.abs(r) * g_bsys[m] / g_rate[m]
            s_all = np.sqrt(s_st**2 + (np.abs(r) * sH_sys[m] / H[m])**2 + s_gb**2)
            for tag, ss, rr in (("stat", s_st, r), ("stat+sys", s_all, r)):
                st = wstats(rr, ss)
                st.update(window=wname, kind=kind, errors=tag)
                rows.append(st)
            if kind == "dt":
                # GECAM LG exposure (dead-time) corrected reference
                gec = g_rate_ec[m] * GECAM_SCALE
                r2 = H[m] / gec
                st = wstats(r2, np.abs(r2) * np.sqrt((sH_stat[m] / H[m])**2 + sG**2))
                st.update(window=wname, kind="dt/GECAM_expcorr", errors="stat")
                rows.append(st)
                # whole-frame misalignment (f from frame k-1 / k+1)
                for sh in (-1, 1):
                    fs = np.roll(f, sh)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        Hs = np.where(fs > FLIVE_MIN, n_all / fs - B, np.nan)
                    mm = m & np.isfinite(Hs)
                    rs = Hs[mm] / gs[mm]
                    ss = np.abs(rs) * np.sqrt((s_stat[mm] * f[mm] / fs[mm] / Hs[mm])**2
                                              + (g_sig[mm] / g_rate[mm])**2)
                    st = wstats(rs, ss)
                    st.update(window=wname, kind=f"dt_frame{sh:+d}", errors="stat")
                    rows.append(st)
                # per-bin table for the record
                pd.DataFrame({"t": x[m], "ratio": r, "sig_stat": s_st,
                              "sig_all": s_all, "f_live": f[m],
                              "gecam_net": g_rate[m], "gecam_live": g_live[m],
                              "pull_stat": (r - np.median(r)) / s_st}).to_csv(
                    OUT / f"ratio_bins_{wname}.csv", index=False, float_format="%.5g")
    rt = pd.DataFrame(rows)[["window", "kind", "errors", "n", "med", "rsig", "mu", "emu",
                             "chi2", "dof", "s_int", "mu_int", "emu_int", "med_err"]]
    rt.to_csv(OUT / "ratio_stats.csv", index=False, float_format="%.4g")

    pd.set_option("display.width", 200)
    print(f"B = {B:.1f} +- {sig_bstat:.1f} evt/s;  tracer k={k_hot:.4f} a={a_hot:.4f}")
    print(offs.to_string())
    print(rep.to_string())
    print(rt.to_string())
    print("GECAM bkg residual template (dt, resid cts/s):",
          np.round(g_bsys_tab[0]), np.round(g_bsys_tab[1], 1), "floor", round(g_bsys_tab[2], 1))


if __name__ == "__main__":
    main()
