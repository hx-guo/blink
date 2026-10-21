#!/usr/bin/env python3
"""GRB 221009A v2 frozen numbers: light curve with errors, HXMT/GECAM ratios,
peaks, fill-mode decomposition, tau, scale-flux decades, flare.

Run from the blink repo root after p221009_v2_bkg.py:
    ./.venv/bin/python scripts/grb221009a/p221009_v2_numbers.py

Inputs reused from the first-wave groups (read only):
  data/p221009/errors/lc_221009a_1s_errors.csv   per-bin error terms (errors group)
  data/p221009/audit/reconstruct.err, detect_?.txt  fill branch per gap (audit group)
  data/p221009/v2/bkg_model_1s.csv               B(t) (p221009_v2_bkg.py)

Outputs (data/p221009/v2/):
  numbers_v2.csv            metric,value,unit,note
  lc_221009a_v2_1s.csv      1-s light curve with errors and the GECAM reference
  ratio_stats_v2.csv        all ratio subsets x HE background x GECAM correction
  ratio_bins_v2.csv         per-bin ratios with errors (standard selection)
  scale_flux_v2.csv         per-bin scale-flux points and decade medians
  numbers_v2_log.txt        printed tables
Deterministic; every exclusion is written to the CSV notes.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from p221009_v2_common import (OUT, X, EDGES, SCALE, he_counts, he_events, is_marker,  # noqa: E402
                               he_off_intervals, off_mask, eng_on_grid, gecam_lg, robust)
from p221009_err_budget import wstats, gecam as err_gecam  # noqa: E402
from engineering_prediction import L_CYC_TO_SEC  # noqa: E402

FLIVE_MIN = 0.05
G_FLOOR = 50.0          # GECAM LG net cts/s (uncorrected), ratio selection
G_BRIGHT = 1000.0       # bright-bin subset
MAIN = (180, 300)
MAIN_ALT = (170, 310)
CAL = (175, 212)        # early main emission used for the x200 scale
P1 = (213, 229)         # P1 as seen by HE (rise to the shut-off)
P1PK = (226, 229)       # the three brightest complete P1 seconds
P2 = (249, 266)         # P2 as seen by HE (turn-on to the second shut-off)
FLARE_WIN = (440, 610)
ANCH = ((400, 425), (615, 640))
rows, log = [], []


def emit(metric, value, unit="", note=""):
    rows.append((metric, value, unit, note))


def say(s=""):
    print(s)
    log.append(s)


def fmt(v, p=3):
    return f"{v:.{p}g}" if np.isfinite(v) else "nan"


# ───────────────────────────── load ─────────────────────────────

def load():
    c = he_counts()
    eng = eng_on_grid(X)
    f = eng["f"]
    g = gecam_lg()
    _, _, _, _, g_bsys, _ = err_gecam(EDGES)
    bad = off_mask()
    bk = pd.read_csv(OUT / "bkg_model_1s.csv")
    bk = bk[(bk.t >= X[0]) & (bk.t <= X[-1])].reset_index(drop=True)
    assert np.allclose(bk.t.to_numpy(), X)
    er = pd.read_csv("data/p221009/errors/lc_221009a_1s_errors.csv")
    assert np.allclose(er.t, X) and np.array_equal(er.n_obs, c["n_obs"])
    assert np.array_equal(er.n_fill, c["n_all"] - c["n_obs"])
    assert np.nanmax(np.abs(er.f_live - f)) < 1e-5

    # background live fraction: 1/f - 1 = cb * B_true on the source-free calibration seconds
    full = pd.read_csv(OUT / "bkg_model_1s.csv")
    m = (full.cal_alt.to_numpy() & np.isfinite(full.f_live)).astype(bool)
    cb = np.sum((1 / full.f_live[m] - 1) * full.n_over_f[m]) / np.sum(full.n_over_f[m] ** 2)

    N, No = c["n_all"], c["n_obs"]
    Bc_obs = N[X < -10].mean()                  # paper's B (observed gross)
    Bt = bk.B_std.to_numpy()                     # B_true(t)
    Bt_obs = Bt / (1 + cb * Bt)                  # observed-gross background
    ok = f > FLIVE_MIN
    with np.errstate(divide="ignore", invalid="ignore"):
        d = dict(
            N=N, No=No, f=f, bad=bad, g=g, g_bsys=g_bsys, Bt=Bt, Bt_obs=Bt_obs, Bc=Bc_obs,
            Bsys=bk.B_sys.to_numpy(), cb=cb, eng=eng, er=er, c=c,
            # standard: time-dependent background
            obs=No - Bt_obs, rec=N - Bt_obs, dt=np.where(ok, N / f - Bt, np.nan),
            # old convention: constant observed-gross B
            obs_c=No - Bc_obs, rec_c=N - Bc_obs, dt_c=np.where(ok, N / f - Bc_obs, np.nan),
            dt_box=sum(c["all_" + b] / eng[b]["f"] for b in "ABC") - Bt,
            B_alt={k: bk[k].to_numpy() for k in ("B_acd", "B_acd_alt", "B_acd_pflare", "B_le_alt",
                                                  "B_timing_pre", "B_timing_occ")},
            sigB=bk.sig_B.to_numpy(),
        )
    # HE-off bins are not measurements
    for k in ("obs", "rec", "dt", "obs_c", "rec_c", "dt_c", "dt_box"):
        d[k] = np.where(bad, np.nan, d[k])
    # errors: statistical terms from the errors group, B term replaced
    d["sig_stat"] = np.sqrt(er.sig_obs ** 2 + er.sig_fill ** 2 + er.sig_fq ** 2
                            + bk.sig_B ** 2).to_numpy()
    d["sig_stat_c"] = er.sig_stat.to_numpy()
    d["sig_sys"] = np.sqrt(np.nan_to_num(er.sig_align) ** 2 + np.nan_to_num(er.sig_det) ** 2
                           + er.sig_degsys ** 2 + d["Bsys"] ** 2).to_numpy()
    d["sig_sys_c"] = np.sqrt(np.nan_to_num(er.sig_align) ** 2 + np.nan_to_num(er.sig_det) ** 2
                             + er.sig_degsys ** 2 + (Bt - Bc_obs) ** 2).to_numpy()
    d["sig_rec"] = np.sqrt(er.var_rec + bk.sig_B ** 2).to_numpy()
    return d


# ───────────────────────────── sections ─────────────────────────────

def sec_background(d):
    say("== background ==")
    emit("B_evt", f"{d['Bc']:.0f}", "evt/s",
         "OLD convention kept for reference: observed gross background, mean of T0-50..-10 (1B obs+filler)")
    Bt0 = np.mean(d["N"][X < -10] / d["f"][X < -10])
    emit("B_true_pre", f"{Bt0:.0f}", "evt/s",
         "dead-time-corrected gross background, mean of N/f over T0-50..-10; = B_evt/f_bkg")
    for lo, hi in ((-50, -10), (180, 300), (300, 440), (440, 610), (600, 700)):
        w = (X >= lo) & (X < hi)
        emit(f"B_t_mean_{lo}_{hi}", f"{np.mean(d['Bt'][w]):.0f}", "evt/s",
             "standard B_true(t): timing-group LargeEvt model (6065+1.084 LE, bridged 174-300/498-540), "
             "dead-time corrected; window mean; "
             f"model systematic {np.mean(d['Bsys'][w]):.0f} evt/s")
    emit("B_tail_rise", f"{np.mean(d['Bt'][(X >= 600) & (X < 700)]) / Bt0:.2f}", "",
         "B_true(600-700)/B_true(pre): the 600-700 s rise is orbital background (see bkg_check.txt)")
    for lo, hi in ((600, 650), (650, 700)):
        w = (X >= lo) & (X < hi)
        alts = [np.mean(d["N"][w] / d["f"][w] - d["B_alt"][k][w]) for k in d["B_alt"]]
        emit(f"tail_source_{lo}_{hi}", f"{np.nanmean(d['dt'][w]):.0f}", "evt/s",
             f"HE source rate after B(t) (afterglow/tail); alternative B models give "
             f"{min(alts):.0f}..{max(alts):.0f}; with constant B: {np.nanmean(d['dt_c'][w]):.0f}")
    emit("f_bkg_coeff", f"{d['cb']:.3e}", "1/(evt/s)",
         "1/f-1 = c*B_true on source-free seconds; B_obs(t)=B_true/(1+c B_true) used for the obs/rec curves")


def sec_mask(d):
    per_box, union, inter = he_off_intervals()
    for b in "ABC":
        for k, (a, z) in enumerate(per_box[b]):
            emit(f"he_off{k+1}_box{b}", f"{a:.3f}..{z:.3f}", "s",
                 "last observed event before / first after the SAA-mode shut-off (markers excluded)")
    for k in range(2):
        emit(f"he_off{k+1}_all_boxes_off", f"{inter[k][0]:.2f}..{inter[k][1]:.2f}", "s",
             "all three boxes without events (intersection)")
        emit(f"he_off{k+1}_any_box_off", f"{union[k][0]:.2f}..{union[k][1]:.2f}", "s",
             "at least one box without events (union) -- defines the mask")
    bins = X[d["bad"]].astype(int)
    runs = []
    for v in bins:
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    emit("he_off_bins", " ".join(f"{a}-{b}" for a, b in runs), "s",
         f"{len(bins)} 1-s bins overlapping the union by >10 ms; NaN in every curve, excluded "
         "from ratios and integrals (old pipeline kept them at -B and filtered by allr>0)")
    emit("he_off_nbins", str(len(bins)), "bins", "")


def sec_peaks(d):
    say("== peaks ==")
    f, N, No, dt = d["f"], d["N"], d["No"], d["dt"]
    emit("flive_min", f"{np.nanmin(np.where((X >= 170) & (X < 310) & ~d['bad'], f, np.nan)):.3f}",
         "", "deepest 3-box mean live fraction in the main window (valid bins)")
    i0 = np.nanargmin(np.where((X >= 170) & (X < 310) & ~d["bad"], f, np.nan))
    emit("flive_min_time", f"{X[i0]:.0f}", "s", "")
    for nm, (lo, hi) in (("peak1", (180, 240)), ("peak2", (240, 300))):
        w = (X >= lo) & (X < hi)
        i = np.nanargmax(np.where(w, dt, np.nan))
        emit(f"{nm}_time", f"{X[i]:.0f}", "s",
             "brightest RECOVERABLE (valid) 1-s bin of the DT-corrected curve -- not the burst peak")
        emit(f"{nm}_rate_dt", f"{dt[i]:.3e}", "evt/s",
             f"DT-corrected net rate; stat {d['sig_stat'][i]:.2e}, sys {d['sig_sys'][i]:.2e}; "
             f"with constant B {d['dt_c'][i]:.3e}")
        emit(f"{nm}_flive", f"{f[i]:.3f}", "", "")
        emit(f"{nm}_fill_frac", f"{(N[i] - No[i]) / N[i]:.3f}", "",
             f"filler / (observed+filler) recorded events in the bin (n_obs={No[i]:.0f}, n_fill={N[i]-No[i]:.0f})")
        emit(f"{nm}_obs_share", f"{No[i] / (N[i] / f[i]):.4f}", "",
             "directly observed events / dead-time-corrected gross events in the bin")
        emit(f"{nm}_dt_share", f"{1 - f[i]:.3f}", "",
             "share of the corrected gross added by the live-fraction division (1-f)")
        emit(f"{nm}_gecam200_dc", f"{SCALE * d['g']['net_dc'][i]:.3e}", "evt/s",
             "GECAM LG (dead-time corrected) x200 in the same bin")
    w = (X >= 180) & (X < 300)
    for col, tag in (("net_dc", "dc"), ("net", "raw")):
        gg = np.where(w, d["g"][col], np.nan)
        j = np.nanargmax(gg)
        emit(f"gecam_peak_time_{tag}", f"{X[j]:.0f}", "s",
             f"brightest GECAM LG second in 180-300 ({'dead-time corrected' if tag == 'dc' else 'uncorrected'}); "
             f"HE off: {bool(d['bad'][j])}")
        emit(f"gecam_peak_x200_{tag}", f"{SCALE * gg[j]:.3e}", "evt/s", "")
    # precursor and flare peaks (Table 1)
    w = (X >= -10) & (X < 20)
    i = np.nanargmax(np.where(w, dt, np.nan))
    emit("precursor_peak_time", f"{X[i]:.0f}", "s", "brightest DT-corrected bin in T0-10..+20")
    emit("precursor_peak_rate_dt", f"{dt[i]:.2e}", "evt/s",
         f"B(t) background; 1-s rms of the pre-burst bins {np.std(dt[X < -10]):.0f} evt/s; GECAM-C in SAA")
    w = (X >= 440) & (X < 610)
    i = np.nanargmax(np.where(w, dt, np.nan))
    emit("flare_peak_rate_dt", f"{dt[i]:.2e}", "evt/s", f"at T0+{X[i]:.0f}; f_live {f[i]:.3f}")
    # FIFO ceiling
    w = (X >= 170) & (X < 310) & ~d["bad"]
    emit("obs_gross_max", f"{np.max(No[w]):.0f}", "evt/s", "max observed gross 1-s rate, main window (FIFO-limited)")
    emit("obs_net_max", f"{np.nanmax(d['obs'][w]):.3e}", "evt/s",
         "max observed rate net of B(t), main window (old peak_obs_ceiling 3.075e4 was net of constant B)")
    gb = w & (d["g"]["net"] > 2000)
    emit("obs_gross_median_bright", f"{np.median(No[gb]):.0f}", "evt/s",
         f"median observed gross rate on the {gb.sum()} bins with GECAM > 2000 cts/s (FIFO plateau)")
    emit("rec_peak", f"{np.nanmax(d['rec'][w]):.3e}", "evt/s", "max reconstructed (obs+filler) net rate, main window")
    for t in (227, 262):
        j = int(t - X[0])
        emit(f"dt_over_obs_{t}", f"{dt[j] / d['obs'][j]:.0f}", "", "DT-corrected / observed net")


def sec_integral(d):
    say("== integral ==")
    w = (X >= MAIN[0]) & (X < MAIN[1])
    vb = w & ~d["bad"]
    I = np.nansum(d["dt"][vb])
    Ic = np.nansum(d["dt_c"][vb])
    g = d["g"]
    G_dc = SCALE * np.where(np.isfinite(g["net_dc"]), g["net_dc"], 0.0)
    G_raw = SCALE * np.where(np.isfinite(g["net"]), g["net"], 0.0)
    emit("mainpulse_window", f"{MAIN[0]}..{MAIN[1]}", "s", "integration window")
    emit("mainpulse_counts_dt", f"{I:.3e}", "counts",
         f"LOWER LIMIT: DT-corrected net counts over the {vb.sum()} valid bins; HE-off bins NaN; "
         f"stat {np.sqrt(np.nansum(d['sig_stat'][vb]**2)):.1e}")
    emit("mainpulse_counts_dt_constB", f"{Ic:.3e}", "counts", "same with the old constant B (old value 4.075e7)")
    emit("mainpulse_counts_B_change", f"{(I - Ic) / Ic * 100:.2f}", "%", "effect of B(t) vs constant B on the integral")
    emit("mainpulse_nan_bins", str(int((w & d["bad"]).sum())), "bins", "HE-off bins inside the window (NaN)")
    for tag, G in (("dc", G_dc), ("raw", G_raw)):
        emit(f"mainpulse_masked_frac_gecam_{tag}", f"{G[w & d['bad']].sum() / G[w].sum():.3f}", "",
             f"share of the GECAM LG x200 counts in {MAIN[0]}-{MAIN[1]} s falling in the HE-off bins "
             f"({'LG dead-time corrected' if tag == 'dc' else 'uncorrected'})")
    emit("mainpulse_gecam_x200_total_dc", f"{G_dc[w].sum():.3e}", "counts", "GECAM LG dc x200 over the window")
    emit("mainpulse_counts_filled_estimate", f"{I + G_dc[w & d['bad']].sum():.3e}", "counts",
         "model-dependent: HE valid-bin integral + GECAM dc x200 in the HE-off bins")
    emit("mainpulse_valid_vs_gecam", f"{I / G_dc[vb].sum():.3f}", "",
         "HE integral / GECAM dc x200 integral over the same valid bins")


def sec_fillmode(d):
    say("== fill modes ==")
    D = Path("data/p221009/audit")
    pat = re.compile(r"^gap\[(\d+)\]: ([\d.]+)s\s+n_lost=(\d+)\s+(cross-ref|FALLBACK|NO-REF)")
    recs, box_i, last = [], -1, None
    for ln in (D / "reconstruct.err").read_text().splitlines():
        m = pat.match(ln)
        if not m:
            continue
        i = int(m.group(1))
        if i == 0 or last is None or i < last:
            box_i += 1
        last = i
        recs.append(("ABC"[box_i], i, float(m.group(2)), int(m.group(3)), m.group(4)))
    df = pd.DataFrame(recs, columns=["box", "idx", "dur", "n_lost", "mode"])
    for b in "ABC":
        det = pd.read_csv(D / f"detect_{b.lower()}.txt").sort_values("start_met").reset_index(drop=True)
        s = df[df.box == b]
        assert len(det) == len(s)
        df.loc[s.index, "t"] = det.start_met.values[s.idx.values] - 339945423.0
    tot_fill = (d["N"] - d["No"]).sum()
    emit("fill_total_events", f"{df.n_lost.sum():.4e}", "events",
         f"all gaps of the run (T0-50..700); the cache holds {tot_fill:.4e} filler events")
    df["cls"] = np.where(df["mode"] == "cross-ref", "crossref", "fallback")
    for tag, sel in (("all", np.ones(len(df), bool)),
                     ("main", ((df.t >= MAIN[0]) & (df.t < MAIN[1])).to_numpy()),
                     ("p1", ((df.t >= 213) & (df.t < 230)).to_numpy()),
                     ("p2", ((df.t >= 248) & (df.t < 268)).to_numpy()),
                     ("flare", ((df.t >= 440) & (df.t < 610)).to_numpy())):
        s = df[sel]
        ng, ne = len(s), s.n_lost.sum()
        for cls in ("crossref", "fallback"):
            q = s[s.cls == cls]
            emit(f"fill_{tag}_{cls}_gap_frac", f"{len(q) / ng:.3f}", "",
                 f"{len(q)} of {ng} gaps starting in the window")
            emit(f"fill_{tag}_{cls}_event_frac", f"{q.n_lost.sum() / ne:.3f}", "",
                 f"{q.n_lost.sum()} of {ne} filler events")
        q = s[s["mode"] == "NO-REF"]
        emit(f"fill_{tag}_noref_gap_frac", f"{len(q) / ng:.3f}", "",
             "no reference box at all (subset of fallback) -- the old '~8%' matches this, but the "
             "linear-ramp fallback is also used when the reference covers <30% of the gap")
        say(f"  {tag}: {ng} gaps, {ne} events; crossref {np.mean(s.cls=='crossref'):.3f} of gaps")
    s = df[(df.t >= MAIN[0]) & (df.t < MAIN[1])]
    for cls in ("crossref", "fallback"):
        q = s[s.cls == cls].dur * 1e3
        p10, p50, p90 = np.percentile(q, [10, 50, 90])
        emit(f"fill_main_{cls}_dur_ms", f"{p50:.1f}", "ms",
             f"median gap duration; p10 {p10:.1f}, p90 {p90:.1f}, max {q.max():.1f} ms (n={len(q)})")
    q = s.dur * 1e3
    emit("fill_main_dur_ms", f"{np.median(q):.1f}", "ms",
         f"all main-window gaps: p10 {np.percentile(q,10):.1f}, p90 {np.percentile(q,90):.1f}, "
         f"max {q.max():.1f} ms; total gap time {q.sum()/1e3:.2f} s over 3 boxes")
    w = (X >= MAIN[0]) & (X < MAIN[1]) & ~d["bad"]
    emit("fill_main_share_of_recorded", f"{(d['N'][w] - d['No'][w]).sum() / d['N'][w].sum():.3f}", "",
         "filler / (observed+filler) events summed over the valid main-window bins")
    emit("fill_main_share_of_corrected", f"{(d['N'][w] - d['No'][w]).sum() / (d['N'][w] / d['f'][w]).sum():.3f}", "",
         "filler events / DT-corrected gross events, valid main-window bins")
    emit("obs_main_share_of_corrected", f"{d['No'][w].sum() / (d['N'][w] / d['f'][w]).sum():.3f}", "",
         "observed events / DT-corrected gross events, valid main-window bins")


def ratio_block(d, sel, Hkey, Gkey):
    """Per-bin ratio arrays for a selection; H in {'dt','dt_c','rec','rec_c'}, G in {'net','net_dc'}."""
    H = d[Hkey][sel]
    G = d["g"][Gkey][sel] * SCALE
    sG = d["g"]["sig_dc" if Gkey == "net_dc" else "sig"][sel] * SCALE
    if Hkey in ("dt", "dt_c"):
        sHs = d["sig_stat" if Hkey == "dt" else "sig_stat_c"][sel]
        sHy = d["sig_sys" if Hkey == "dt" else "sig_sys_c"][sel]
    else:
        sHs = d["sig_rec"][sel]
        er = d["er"]
        sHy = np.sqrt((er.sig_degsys.to_numpy() * d["f"])[sel] ** 2 + d["Bsys"][sel] ** 2)
    r = H / G
    s_st = np.abs(r) * np.sqrt((sHs / H) ** 2 + (sG / G) ** 2)
    s_all = np.sqrt(s_st ** 2 + (r * sHy / H) ** 2 + (r * d["g_bsys"][sel] * SCALE / G) ** 2)
    return r, s_st, s_all


def sec_ratios(d):
    say("== ratios ==")
    g = d["g"]
    base = ~d["bad"] & np.isfinite(g["net"]) & (g["net"] > G_FLOOR) & np.isfinite(d["dt"])

    def win(lo, hi):
        return (X >= lo) & (X < hi)
    subsets = {
        "main": base & win(*MAIN),
        "main170_310": base & win(*MAIN_ALT),
        "main_bright": base & win(*MAIN) & (g["net"] > G_BRIGHT),
        "main_faint": base & win(*MAIN) & (g["net"] <= G_BRIGHT),
        "main_noncal": base & win(213, MAIN[1]),
        "main_deep": base & win(*MAIN) & (d["f"] < 0.5),
        "P1": base & win(*P1),
        "P1peak": base & win(*P1PK),
        "P2": base & win(*P2),
        "flare": base & win(*FLARE_WIN),
        "flare_interval": base & win(*d["flare_iv"]),
    }
    rules = {
        "main": f"valid bins in {MAIN[0]}-{MAIN[1]} s with GECAM LG net (uncorrected) > {G_FLOOR:.0f} cts/s",
        "main170_310": "as main, 170-310 s (old pipeline window)",
        "main_bright": f"main and GECAM net > {G_BRIGHT:.0f} cts/s",
        "main_faint": f"main and GECAM net <= {G_BRIGHT:.0f} cts/s",
        "main_noncal": "main and t >= 213 s (drops the 175-212 s bins that fix x200)",
        "main_deep": "main and f_live < 0.5",
        "P1": f"valid bins {P1[0]}-{P1[1]-1} s (P1 as seen by HE), GECAM > {G_FLOOR:.0f}",
        "P1peak": f"bins {P1PK[0]}-{P1PK[1]-1} s",
        "P2": f"valid bins {P2[0]}-{P2[1]-1} s, GECAM > {G_FLOOR:.0f}",
        "flare": f"bins {FLARE_WIN[0]}-{FLARE_WIN[1]} s with GECAM > {G_FLOOR:.0f} (old flare window)",
        "flare_interval": "bins inside the v2 flare interval, GECAM > 50",
    }
    out = []
    for name, sel in subsets.items():
        for Hkey, Htag in (("dt", "dt_Bt"), ("dt_c", "dt_Bconst"), ("rec", "rec_Bt"), ("rec_c", "rec_Bconst")):
            s2 = sel & np.isfinite(d[Hkey])
            for Gkey, Gtag in (("net_dc", "gecam_dc"), ("net", "gecam_raw")):
                r, s_st, s_all = ratio_block(d, s2, Hkey, Gkey)
                st = wstats(r, s_st)
                sa = wstats(r, s_all)
                out.append(dict(subset=name, he=Htag, gecam=Gtag, n=st["n"], med=st["med"],
                                rsig=st["rsig"], mu=st["mu"], emu=st["emu"], chi2=st["chi2"],
                                dof=st["dof"], s_int=st["s_int"], mu_int=st["mu_int"],
                                emu_int=st["emu_int"], chi2dof_statsys=sa["chi2"] / max(sa["dof"], 1),
                                mu_statsys=sa["mu"], emu_statsys=sa["emu"], rule=rules[name]))
    rt = pd.DataFrame(out)
    rt.to_csv(OUT / "ratio_stats_v2.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    say(rt[rt.he.isin(["dt_Bt", "rec_Bt"])].drop(columns="rule").to_string())

    for name in subsets:
        if name == "flare_interval":      # identical bin set to "flare"; kept in ratio_stats_v2.csv only
            continue
        for gtag, gnote in (("gecam_dc", "GECAM LG dead-time corrected (standard)"),
                            ("gecam_raw", "GECAM LG uncorrected")):
            for he, hnote in (("dt_Bt", "HE DT-corrected, B(t)"), ("rec_Bt", "HE reconstruction only, B(t)")):
                q = rt[(rt.subset == name) & (rt.gecam == gtag) & (rt.he == he)].iloc[0]
                key = f"ratio_{name}_{'dt' if he == 'dt_Bt' else 'raw'}_{'gdc' if gtag == 'gecam_dc' else 'graw'}"
                emit(key + "_n", str(int(q.n)), "bins", rules[name])
                emit(key + "_med", f"{q.med:.3f}", "", f"median; {hnote}; {gnote}")
                emit(key + "_rsig", f"{q.rsig:.3f}", "", "IQR/1.349")
                if he == "dt_Bt":
                    emit(key + "_wmean", f"{q.mu:.3f}", "", f"inverse-variance mean (stat errors), +- {q.emu:.3f}")
                    emit(key + "_chi2dof", f"{q.chi2:.1f}/{int(q.dof)}", "", "chi2/dof about the weighted mean, stat errors")
                    emit(key + "_sint", f"{q.s_int:.3f}", "",
                         f"intrinsic scatter for chi2/dof=1; mean with it {q.mu_int:.3f} +- {q.emu_int:.3f}; "
                         f"chi2/dof with stat+sys errors {q.chi2dof_statsys:.2f}")
    # flare: B sensitivity
    for gtag in ("gecam_dc", "gecam_raw"):
        q = rt[(rt.subset == "flare") & (rt.gecam == gtag) & (rt.he == "dt_Bconst")].iloc[0]
        emit(f"ratio_flare_dt_{'gdc' if gtag == 'gecam_dc' else 'graw'}_constB_med", f"{q.med:.3f}", "",
             f"flare ratio with the OLD constant B (n={int(q.n)}, rsig {q.rsig:.3f}) -- B sensitivity")
    for name in ("main", "main_bright", "P1peak", "P2"):
        q = rt[(rt.subset == name) & (rt.gecam == "gecam_dc") & (rt.he == "dt_Bconst")].iloc[0]
        emit(f"ratio_{name}_dt_gdc_constB_med", f"{q.med:.3f}", "", "same with the old constant B (B sensitivity)")
    # per-bin table of the standard selection
    rows_b = []
    for name in ("main", "flare"):
        sel = subsets[name]
        r1, s1, a1 = ratio_block(d, sel, "dt", "net_dc")
        r2, s2, a2 = ratio_block(d, sel, "dt", "net")
        rows_b.append(pd.DataFrame(dict(subset=name, t=X[sel], ratio_dt_gdc=r1, sig_stat_gdc=s1, sig_all_gdc=a1,
                                        ratio_dt_graw=r2, sig_stat_graw=s2, sig_all_graw=a2,
                                        f_live=d["f"][sel], gecam_net=d["g"]["net"][sel],
                                        gecam_net_dc=d["g"]["net_dc"][sel], gecam_live=d["g"]["live"][sel])))
    pd.concat(rows_b).to_csv(OUT / "ratio_bins_v2.csv", index=False, float_format="%.5g")
    return rt


def sec_scale(d):
    say("== x200 refit on 175-212 s ==")
    g = d["g"]
    w = (X >= CAL[0]) & (X < CAL[1]) & ~d["bad"] & (g["net"] > G_FLOOR)
    nofill = w & (d["N"] == d["No"])
    res = {}
    for tag, sel in (("nofill", nofill), ("all", w)):
        for Hkey in ("dt", "dt_c"):
            for Gkey in ("net_dc", "net"):
                H, G = d[Hkey][sel], g[Gkey][sel]
                sH = d["sig_stat"][sel]
                sG = np.hypot(g["sig_dc" if Gkey == "net_dc" else "sig"][sel], d["g_bsys"][sel])
                s = H.sum() / G.sum()
                for _ in range(50):      # weighted LSQ through the origin, errors in both
                    wt = 1 / (sH ** 2 + s ** 2 * sG ** 2)
                    s = np.sum(wt * H * G) / np.sum(wt * G * G)
                es = 1 / np.sqrt(np.sum(wt * G * G))
                chi = np.sum(wt * (H - s * G) ** 2) / (sel.sum() - 1)
                res[(tag, Hkey, Gkey)] = (s, es, chi, np.median(H / G), H.sum() / G.sum(), sel.sum())
                say(f"  {tag:6s} {Hkey:4s} {Gkey:6s} n={sel.sum():2d}  wLSQ {s:6.1f} +- {es:4.1f} (chi2/dof {chi:.2f})"
                    f"  median {np.median(H / G):6.1f}  sum-ratio {H.sum() / G.sum():6.1f}")
    s, es, chi, med, sr, n = res[("nofill", "dt", "net_dc")]
    emit("scale_fit_wlsq", f"{s:.1f}", "",
         f"HE DT (B(t)) / GECAM LG dc on {CAL[0]}-{CAL[1]} s, {n} valid bins with GECAM>{G_FLOOR:.0f} and no FIFO "
         f"gap (bins {' '.join(str(int(v)) for v in X[nofill])}); weighted LSQ through origin, errors: HE stat, "
         f"GECAM Poisson + background systematic; +- {es:.1f} (stat), chi2/dof {chi:.2f}")
    emit("scale_fit_median", f"{med:.1f}", "", "median of per-bin HE/GECAM, same bins")
    emit("scale_fit_sumratio", f"{sr:.1f}", "", "sum HE / sum GECAM, same bins")
    s2 = res[("nofill", "dt_c", "net_dc")]
    emit("scale_fit_wlsq_constB", f"{s2[0]:.1f}", "", f"same with the old constant B (+- {s2[1]:.1f})")
    s3 = res[("all", "dt", "net_dc")]
    emit("scale_fit_wlsq_incl_recon", f"{s3[0]:.1f}", "",
         f"all {s3[5]} valid bins with GECAM>50 in 175-212 incl. the reconstructed 186-192 s (+- {s3[1]:.1f})")
    s4 = res[("nofill", "dt", "net")]
    emit("scale_fit_wlsq_gecam_raw", f"{s4[0]:.1f}", "", f"GECAM uncorrected (+- {s4[1]:.1f})")
    emit("scale_adopted", "200", "", "kept: see freeze.md for the decision")
    return res


def sec_tau(d):
    say("== tau ==")
    eng, f, c = d["eng"], d["f"], d["c"]
    t, fill, ch, pw, box = he_events()
    mk = is_marker(t, fill, ch, pw)
    rec = {b: c["all_" + b] - np.histogram(t[mk & (box == i)], EDGES)[0] for i, b in enumerate("ABC")}
    good = ~d["bad"]
    phases = {"pre": (-50, -10), "tail300_440": (300, 440), "flare442_593": (442, 593), "tail600_700": (600, 700)}
    for nm, (lo, hi) in phases.items():
        trig, trigp, recd = [], [], []
        for b in "ABC":
            e = eng[b]
            Dsec = e["D"] * L_CYC_TO_SEC
            s = good & (X >= lo) & (X < hi) & np.isfinite(e["f"])
            unw = s & (e["Pmax"] < 15000) & (1.45 * rec[b] / e["f"] / 6 < 10000)
            trig.append(np.median(Dsec[unw] / e["P"][unw]) * 1e6)
            trigp.append(np.median(Dsec[unw] / (e["P"][unw] * e["f"][unw])) * 1e6)
            recd.append(np.median(Dsec[s] / rec[b][s]) * 1e6)
        emit(f"tau_box_trig_{nm}_us", f"{np.median(trigp):.2f}", "us",
             f"counter: D/(sum6 Cnt_PHODet * f) per trigger, median over boxes {np.round(trigp, 2).tolist()}; "
             f"D/sumP {np.round(trig, 2).tolist()}")
        emit(f"tau_box_rec_{nm}_us", f"{np.median(recd):.2f}", "us",
             f"counter: D / recorded events (obs+filler, markers excluded) per box, {np.round(recd, 2).tolist()}")
    # bright bins: counter per record
    deep = good & (X >= 170) & (X < 310) & (f < 0.5)
    recd = [np.median((eng[b]["D"] * L_CYC_TO_SEC)[deep] / rec[b][deep]) * 1e6 for b in "ABC"]
    emit("tau_box_rec_deep_us", f"{np.median(recd):.2f}", "us",
         f"counter per recorded event on the {deep.sum()} valid main bins with f<0.5, boxes {np.round(recd, 2).tolist()}")
    # k: trigger / record ratio of the burst excess (unwrapped frames)
    ks = []
    for b in "ABC":
        e = eng[b]
        s = good & np.isfinite(e["f"]) & (e["Pmax"] < 15000)
        rt = rec[b] / e["f"]
        pre = s & (X < -10)
        kb = np.median(e["P"][pre] / rt[pre])
        burst = s & (X >= 170) & (rt - np.median(rt[pre]) > 2000) & (kb * rt / 6 < 10000)
        dP = e["P"][burst] - np.median(e["P"][pre])
        dR = rt[burst] - np.median(rt[pre])
        ks.append((kb, np.sum(dP * dR) / np.sum(dR * dR)))
    ks = np.array(ks)
    emit("k_trig_per_rec_bkg", f"{np.median(ks[:, 0]):.2f}", "", f"boxes {np.round(ks[:, 0], 2).tolist()}")
    emit("k_trig_per_rec_burst", f"{np.median(ks[:, 1]):.2f}", "",
         f"through-origin slope of excess triggers vs excess true records, unwrapped frames; boxes {np.round(ks[:, 1], 2).tolist()}")
    # per-bin fit of the non-paralyzable model against GECAM
    g = d["g"]
    out = {}
    for Gkey in ("net_dc", "net"):
        G = g[Gkey]
        sel = good & (d["rec"] > 5e3) & (G > 30) & np.isfinite(f) & np.isfinite(G)
        n_true = (SCALE * G[sel] + d["Bt"][sel]) / 3          # per box, records/s
        (tb,), cov = curve_fit(lambda n, tau: 1 / (1 + tau * n), n_true, f[sel], p0=[4e-6])
        y = 1 / f[sel] - 1
        tb2 = np.sum(y * n_true) / np.sum(n_true ** 2)
        med = np.median(y / n_true)
        meddeep = np.median((y / n_true)[f[sel] < 0.5])
        out[Gkey] = (tb, np.sqrt(cov[0, 0]), tb2, med, meddeep, sel.sum(), (f[sel] < 0.5).sum())
        say(f"  GECAM {Gkey}: tau_box fit {tb*1e6:.3f} +- {np.sqrt(cov[0,0])*1e6:.3f} us/record "
            f"(1/f-1 lin {tb2*1e6:.3f}; per-bin median {med*1e6:.3f}; f<0.5 median {meddeep*1e6:.3f}; n={sel.sum()})")
    tb, etb, tb2, med, meddeep, n, nd = out["net_dc"]
    emit("tau_box_rec_fit_us", f"{tb*1e6:.2f}", "us",
         f"per-bin LSQ of f=1/(1+tau n_true), n_true=(200 G_dc + B_true(t))/3 per box; {n} valid bins with "
         f"HE rec net>5e3 and GECAM dc>30 (HE-off bins excluded); +- {etb*1e6:.2f} (stat)")
    emit("tau_box_rec_fit_lin_us", f"{tb2*1e6:.2f}", "us", "through-origin LSQ of 1/f-1 on n_true, same bins")
    emit("tau_box_rec_perbin_median_us", f"{med*1e6:.2f}", "us",
         f"median of (1/f-1)/n_true; on the {nd} f<0.5 bins: {meddeep*1e6:.2f}")
    t0 = out["net"]
    emit("tau_box_rec_fit_gecam_raw_us", f"{t0[0]*1e6:.2f}", "us", "same fit with uncorrected GECAM (audit: 4.02)")
    tau_g = tb * SCALE / 3
    emit("tau_fit_gecam_units", f"{tau_g:.3e}", "(cts/s)^-1",
         "same fit in the old f4 units: f = 1/(1+tau(R+R_B)), tau = 3 tau_box/200 (old decade-median fit 2.09e-4)")
    trig_c = float([r for r in rows if r[0] == "tau_box_trig_flare442_593_us"][0][1])
    rec_deep = float([r for r in rows if r[0] == "tau_box_rec_deep_us"][0][1])
    k_b = np.median(ks[:, 1])
    emit("tau_diff_rec_fit_vs_counter", f"{(meddeep * 1e6 / rec_deep - 1) * 100:.1f}", "%",
         f"per-record: GECAM per-bin median on f<0.5 bins ({meddeep*1e6:.2f}) vs counter on the same bins "
         f"({rec_deep:.2f}); global fit vs counter: {(tb * 1e6 / rec_deep - 1) * 100:.1f}%")
    emit("tau_diff_trig_fit_vs_counter", f"{(tb * 1e6 / k_b / trig_c - 1) * 100:.1f}", "%",
         f"per-trigger: fit {tb*1e6:.2f}/k_burst {k_b:.2f} = {tb*1e6/k_b:.2f} us vs counter {trig_c:.2f} us (flare phase)")
    emit("tau_det_eng_us", "deprecated", "us",
         "old 17.7 us = box-shared D / single-detector Cnt_PHODet (per-detector equivalent, = 6x box value); "
         "replaced by tau_box_trig_*")
    return tau_g


def sec_scaleflux(d, tau_g):
    say("== scale-flux decades ==")
    g = d["g"]
    dec = [30, 100, 300, 1000, 3000, 10000, 30000]
    pts = []
    for Gkey, gtag in (("net_dc", "gdc"), ("net", "graw")):
        G = g[Gkey]
        sig = ~d["bad"] & (d["rec"] > 5e3) & (G > 30) & np.isfinite(G)
        sr = np.where(sig, G / d["rec"], np.nan)
        sd = np.where(sig, G / d["dt"], np.nan)
        for lo, hi in zip(dec[:-1], dec[1:]):
            w = sig & (G >= lo) & (G < hi)
            mr, md = np.median(sr[w]) * 1e3, np.median(sd[w]) * 1e3
            fl = np.median(d["f"][w])
            R = np.sqrt(lo * hi)
            RB = np.median(d["Bt"][w]) / SCALE
            model = (1 + tau_g * (R + RB)) / SCALE * 1e3
            emit(f"scaleflux_{gtag}_{lo}_{hi}_raw", f"{mr:.2f}", "1e-3",
                 f"median GECAM/HE(rec, B(t)); n={w.sum()}; median f {fl:.3f}; model at decade centre {model:.2f}")
            emit(f"scaleflux_{gtag}_{lo}_{hi}_dt", f"{md:.2f}", "1e-3", "median GECAM/HE(DT, B(t))")
            say(f"  {gtag} {lo:>5}-{hi:<6} raw {mr:6.2f} dt {md:5.2f} n={w.sum():3d} f {fl:.3f} model {model:5.2f}")
        pts.append(pd.DataFrame(dict(gecam=gtag, t=X[sig], G=G[sig], scale_raw=sr[sig], scale_dt=sd[sig],
                                     f_live=d["f"][sig], obs_gross=d["No"][sig], B_true=d["Bt"][sig])))
    emit("scaleflux_selection", "HE rec net>5e3 & GECAM>30 & valid", "",
         "HE-off bins (incl. partial 229, 248, 266, 267, 276) excluded; decades on the respective GECAM rate")
    pd.concat(pts).to_csv(OUT / "scale_flux_v2.csv", index=False, float_format="%.6g")
    # brightest-decade excess over the model
    G = g["net_dc"]
    sig = ~d["bad"] & (d["rec"] > 5e3) & (G > 30)
    w = sig & (G >= 10000)
    model_bin = (1 + tau_g * (G[w] + d["Bt"][w] / SCALE)) / SCALE
    ex = (G[w] / d["rec"][w]) / model_bin
    emit("scaleflux_bright_excess_over_model", f"{np.median(ex):.2f}", "",
         f"per-bin raw scale / model (per-bin tau fit) on the {w.sum()} bins with GECAM dc>1e4; "
         f"range {ex.min():.2f}-{ex.max():.2f} (old: '~1.7x')")


def flare_numbers(x, y, prefix, note, extra=True, anchors=ANCH, nsig=3.0):
    """Local-continuum flare definition (as numbers_221009a.py) on curve y."""
    pre = (x >= anchors[0][0]) & (x < anchors[0][1])
    post = (x >= anchors[1][0]) & (x < anchors[1][1])
    r_pre, t_pre = np.nanmean(y[pre]), np.mean(x[pre])
    r_post, t_post = np.nanmean(y[post]), np.mean(x[post])
    cont = r_pre + (r_post - r_pre) * (x - t_pre) / (t_post - t_pre)
    sc = np.nanstd(np.concatenate([y[pre] - r_pre, y[post] - r_post]))
    search = (x >= 425) & (x < 610)
    exc = np.where(search, y - cont, np.nan)
    idx = np.where(search & np.isfinite(exc) & (exc > nsig * sc))[0]
    fs, fe = x[idx[0]], x[idx[-1]] + 1
    sel = (x >= fs) & (x < fe) & np.isfinite(exc)
    e, ts = exc[sel], x[sel]
    cum = np.cumsum(np.clip(e, 0, None)) / np.sum(np.clip(e, 0, None))
    t05, t95 = ts[np.searchsorted(cum, 0.05)], ts[np.searchsorted(cum, 0.95)]
    res = dict(start=fs, end=fe, t90=t95 - t05, t05=t05, t95=t95, peak=ts[np.nanargmax(e)],
               excess=np.nansum(e), sigc=sc, nbelow=int(np.sum(e <= nsig * sc)))
    return res, cont


def sec_flare(d):
    say("== flare ==")
    res_all = {}
    for curve, ckey in (("dt_Bt", "dt"), ("dt_Bconst", "dt_c"), ("rec_Bt", "rec"), ("rec_Bconst", "rec_c")):
        r, _ = flare_numbers(X, d[ckey], curve, "")
        res_all[curve] = r
        say(f"  {curve:11s} {r['start']:.0f}-{r['end']:.0f}  T90 {r['t90']:.0f} ({r['t05']:.0f}-{r['t95']:.0f}) "
            f"peak {r['peak']:.0f}  excess {r['excess']:.3e}  sigma_cont {r['sigc']:.0f}  below-3sig {r['nbelow']}")
    # alternative B models on the DT curve
    for k, B in d["B_alt"].items():
        y = np.where(d["bad"], np.nan, d["N"] / d["f"] - B)
        r, _ = flare_numbers(X, y, k, "")
        res_all["dt_" + k] = r
        say(f"  dt_{k:11s} {r['start']:.0f}-{r['end']:.0f}  T90 {r['t90']:.0f}  peak {r['peak']:.0f}  excess {r['excess']:.3e}")
    # definition sensitivity on the standard curve: 5 sigma; zero baseline over the old search window
    r5, _ = flare_numbers(X, d["dt"], "", "", nsig=5.0)
    res_all["dt_Bt_5sig"] = r5
    w = (X >= 425) & (X < 610)
    e = np.clip(d["dt"][w], 0, None)
    cum = np.cumsum(e) / e.sum()
    z05, z95 = X[w][np.searchsorted(cum, 0.05)], X[w][np.searchsorted(cum, 0.95)]
    s = res_all["dt_Bt"]
    emit("flare_start", f"{s['start']:.0f}", "s",
         "first 1-s bin >3 sigma above the local linear continuum (anchors 400-425 & 615-640 s) of the "
         "DT-corrected B(t) curve (old: 442 on the rec curve with constant B)")
    emit("flare_end", f"{s['end']:.0f}", "s", f"end of the last >3 sigma bin; {s['nbelow']} bins inside fall below 3 sigma")
    emit("flare_t90", f"{s['t90']:.0f}", "s", "5%..95% of the cumulative continuum-subtracted counts")
    emit("flare_t90_start", f"{s['t05']:.0f}", "s", "")
    emit("flare_t90_end", f"{s['t95']:.0f}", "s", "")
    emit("flare_peak_time", f"{s['peak']:.0f}", "s", "brightest continuum-subtracted bin")
    emit("flare_excess_counts", f"{s['excess']:.3e}", "counts", "continuum-subtracted DT-corrected counts in the interval")
    fm = (X >= s["start"]) & (X < s["end"])
    emit("flare_recovered_events", f"{np.sum((d['N'] - d['No'])[fm]):.3e}", "counts",
         "filler events across the flare interval (B-independent)")
    emit("flare_dt_added_events", f"{np.nansum((d['N'] / d['f'] - d['N'])[fm]):.3e}", "counts",
         "events added by the live-fraction division across the flare interval")
    emit("flare_flive_median", f"{np.median(d['f'][fm]):.3f}", "",
         f"median f_live in the interval; min {np.min(d['f'][fm]):.3f}")
    for k, r in res_all.items():
        if k == "dt_Bt":
            continue
        emit(f"flare_sens_{k}", f"{r['start']:.0f}-{r['end']:.0f}|T90 {r['t90']:.0f} ({r['t05']:.0f}-{r['t95']:.0f})"
             f"|peak {r['peak']:.0f}|excess {r['excess']:.3e}", "",
             "sensitivity: interval|T90|peak|excess for another curve/background/threshold")
    emit("flare_sens_zero_baseline_t90", f"{z95 - z05:.0f}", "s",
         f"no local continuum: T90 of the B(t)-subtracted DT curve over 425-610 s ({z05:.0f}-{z95:.0f}); "
         "shows how much of T90 is set by the continuum definition")
    # continuum-subtracted ratio (HE and GECAM both local-continuum subtracted)
    g = d["g"]["net_dc"] * SCALE
    for tag, ck in (("Bt", "dt"), ("Bconst", "dt_c")):
        _, ch = flare_numbers(X, d[ck], "", "")
        _, cg = flare_numbers(X, np.where(np.isfinite(g), g, np.nan), "", "")
        sel = fm & np.isfinite(d[ck]) & (d["g"]["net"] > G_FLOOR)
        rr = (d[ck][sel] - ch[sel]) / (g[sel] - cg[sel])
        tot = np.sum(d[ck][sel] - ch[sel]) / np.sum(g[sel] - cg[sel])
        m, sgm, n = robust(rr)
        emit(f"ratio_flare_excess_{tag}", f"{tot:.3f}", "",
             f"sum(HE - continuum)/sum(GECAM dc x200 - continuum) over {n} interval bins; per-bin median "
             f"{m:.2f} +- {sgm:.2f}")
    return s


def sec_perpeak(d):
    """Per-second table for the P1 peak (226-228) and P2 (255-266) seconds."""
    say("== per-second peak table ==")
    g = d["g"]
    rows_t = []
    for t in list(range(226, 229)) + list(range(255, 267)):
        j = int(t - X[0])
        N, No, f = d["N"][j], d["No"][j], d["f"][j]
        valid = not d["bad"][j]
        dt = N / f - d["Bt"][j]
        r_dc = dt / (SCALE * g["net_dc"][j])
        r_raw = dt / (SCALE * g["net"][j])
        obs_net = No - d["Bt_obs"][j]
        rows_t.append(dict(t=t, valid=valid, n_obs=No, n_fill=N - No, n_rec=N, f_live=f,
                           n_dtcorr=N / f, R_dt=dt, fill_frac=(N - No) / N,
                           obs_over_rec=No / N, obs_over_dtcorr=No / (N / f),
                           obsnet_over_Rdt=obs_net / dt,
                           ratio_dt_gdc=r_dc, ratio_dt_graw=r_raw,
                           gecam_live=g["live"][j], gecam_net=g["net"][j], gecam_net_dc=g["net_dc"][j]))
        note = "" if valid else "PARTIAL HE-off second (excluded everywhere else); f_live not that of the counted events"
        emit(f"sec{t}_ratio_dt_graw", f"{r_raw:.3f}", "", f"HE DT (B(t)) / GECAM LG uncorrected x200 {note}")
        emit(f"sec{t}_ratio_dt_gdc", f"{r_dc:.3f}", "", "HE DT (B(t)) / GECAM LG dead-time corrected x200")
        emit(f"sec{t}_flive", f"{f:.3f}", "", "HE 3-box mean live fraction")
        emit(f"sec{t}_fill_frac", f"{(N - No) / N:.3f}", "", "filler / (observed + filler) recorded events")
        emit(f"sec{t}_gecam_live", f"{g['live'][j]:.3f}", "", "GECAM LG live fraction (BTIME EXPOSURE[:,1], mean of 20 rows)")
    tab = pd.DataFrame(rows_t)
    tab.to_csv(OUT / "peak_seconds_v2.csv", index=False, float_format="%.6g")
    say(tab.to_string(float_format=lambda v: f"{v:.4g}"))
    p1 = tab[tab.t.isin([226, 227, 228])]
    for _, q in p1.iterrows():
        t = int(q.t)
        emit(f"sec{t}_n_obs", f"{q.n_obs:.0f}", "events", "observed (recorded, not filled) events in the second, incl. background")
        emit(f"sec{t}_n_fill", f"{q.n_fill:.0f}", "events", "gap-filler events")
        emit(f"sec{t}_n_dtcorr", f"{q.n_dtcorr:.4e}", "events", "(observed+filler)/f_live, gross")
        emit(f"sec{t}_obs_over_rec", f"{q.obs_over_rec:.4f}", "", "observed / (observed+filler)  [eband-group footing]")
        emit(f"sec{t}_obs_over_dtcorr", f"{q.obs_over_dtcorr:.4f}", "", "observed / ((observed+filler)/f)  [gross]")
        emit(f"sec{t}_obsnet_over_Rdt", f"{q.obsnet_over_Rdt:.4f}", "",
             "(observed - B_obs) / R_dt  [audit-group footing: net observed over net DT-corrected]")
    S = p1[["n_obs", "n_fill", "n_rec", "n_dtcorr"]].sum()
    emit("P1peak3s_n_obs", f"{S.n_obs:.0f}", "events", "sum over 226-228 s")
    emit("P1peak3s_obs_over_rec", f"{S.n_obs / S.n_rec:.4f}", "", "3-s sum: observed/(observed+filler)")
    emit("P1peak3s_obs_over_dtcorr", f"{S.n_obs / S.n_dtcorr:.4f}", "", "3-s sum: observed/DT-corrected gross")
    emit("pileup_loss_bound", "P1 <~10-20%, P2 <~10%", "",
         "eband group (eband.md par.12): loss not restored by f_live, bounded by the GECAM ratio; "
         "discussion-only upper limit, not applied as a correction")


def sec_flare_bsys(d):
    """Flare ratio and numbers under each alternative background model."""
    say("== flare vs background model ==")
    g = d["g"]
    sel = ~d["bad"] & (X >= FLARE_WIN[0]) & (X < FLARE_WIN[1]) & (g["net"] > G_FLOOR)
    meds = {}
    for k, B in list(d["B_alt"].items()) + [("B_std", d["Bt"])]:
        H = d["N"] / d["f"] - B
        meds[k] = np.median(H[sel] / (SCALE * g["net_dc"][sel]))
        r, _ = flare_numbers(X, np.where(d["bad"], np.nan, H), "", "")
        say(f"  {k:14s} flare ratio (gdc) {meds[k]:.3f}   interval {r['start']:.0f}-{r['end']:.0f} "
            f"T90 {r['t90']:.0f} excess {r['excess']:.3e}")
    lo, hi = min(meds.values()), max(meds.values())
    emit("ratio_flare_dt_gdc_Bmodel_range", f"{lo:.3f}..{hi:.3f}", "",
         "median flare ratio over the background models (standard LargeEvt; LargeEvt with pre-only / "
         "occulted-only calibration; independent ACD-tag proxy with three calibrations; LargeEvt quadratic): "
         "the extrapolation/proxy systematic of the flare ratio")
    emit("ratio_flare_dt_gdc_Bsys", f"{(hi - lo) / 2:.3f}", "", "half the model range, quoted as background systematic")
    w = (X >= 570) & (X < 700)
    diff = np.mean(np.abs(d["B_alt"]["B_acd"][w] - d["Bt"][w]))
    emit("B_proxy_diff_570_700", f"{diff:.0f}", "evt/s",
         "mean |B_ACD - B_LargeEvt| on 570-700 s (two independent proxies, beyond the LargeEvt calibrated range)")


TIMING_ROWS = [
    # metric, value, unit, note (source)
    ("timing_precursor_t90", "10.2+-0.4", "s", "p221009_timing_precursor.py (precursor.json dur_proxy/dur_quad); spread of two background models"),
    ("timing_precursor_t50", "3.0+-0.2", "s", "p221009_timing_precursor.py"),
    ("timing_precursor_onset", "-0.90", "s", "first Bayesian-block change point (0.05 s bins, p0=0.01); timing.md par.1"),
    ("timing_precursor_peak_0p1s", "3.6+-0.4e3 @0.45", "evt/s @ s", "precursor.json peak_0.1s (ch>=20, 1K)"),
    ("timing_precursor_peak_1s", "2.96+-0.11e3", "evt/s", "precursor.json peak_1.0s (ch>=20, 1K); the v2 light-curve bin [0,1) all channels gives precursor_peak_rate_dt"),
    ("timing_precursor_HR1", "0.68+-0.06 vs 0.868+-0.011", "", "ch53-99/ch30-53, precursor vs T0+176-186 s; precursor.json band_counts"),
    ("timing_quiescent_excess_20_170", "-12+-8(stat)+-10(bkg)", "evt/s", "precursor.json quiescent_excess_20_170; < 30 evt/s (~1% of precursor peak)"),
    ("timing_main_onset", "174.2", "s", "Bayesian-block change point; timing.md par.1"),
    ("timing_quiescent_length", "164-167", "s", "timing.md par.0 item 3"),
    ("timing_pulse187_fwhm", "6.9 (185.75-192.69)", "s", "pulses.json empirical.pulse_187"),
    ("timing_P2_fwhm", "8.0 (257.12-265.16)", "s", "pulses.json empirical.P2"),
    ("timing_P2_plateau_ratio", "195-203", "", "HE_DT / GECAM LG per second on 258-264 s (timing.md item 7; their GECAM LG 0.4-3 MeV)"),
    ("timing_flare_spike_fwhm", "2.5 (509.74-512.21)", "s", "pulses.json empirical.flare_spike (1 s bins)"),
    ("timing_flare_spike_peak_0p1s", "510.65", "s", "timing.md item 10; same 0.1 s bin as GECAM LG"),
    ("timing_gecam_lg_brightest_second", "230-231", "s", "pulses.json brightest_second: LG 32704 cts/s vs 26847 at 227 s (x1.22)"),
    ("timing_light_travel_ms", "5.5-6.0", "ms", "GECAM-C later than HXMT; geometry.json"),
    ("timing_lag_he_minus_lg", "-7.0+-6.6(stat)+-8(sys)", "ms", "T0+500-525 s, HE all channels vs GECAM LG 0.4-3 MeV (lag_results.json); |dt| <~ 15 ms"),
    ("timing_lag_subsamples", "-6.8+-4.3", "ms", "6 independent subsamples (3 HE boxes x 2 GECAM groups)"),
    ("timing_mvt_flare_unsat", "0.128", "s", "Haar structure function, T0+440-500 s (no filler); mvt.json; not quoted in gap-filled intervals"),
    ("timing_flare_t90_alt", "166 (411.5-577.5)", "s", "tail.json flare_t90.tail_powerlaw: orbital background + extrapolated tail continuum (definition-dependent)"),
]


def sec_timing():
    for r in TIMING_ROWS:
        emit(*r)


def write_lc(d):
    g = d["g"]
    lc = pd.DataFrame(dict(
        t=X, valid=~d["bad"], n_obs=d["No"], n_fill=d["N"] - d["No"], f_live=d["f"],
        f_box_A=d["eng"]["A"]["f"], f_box_B=d["eng"]["B"]["f"], f_box_C=d["eng"]["C"]["f"],
        B_true=d["Bt"], B_obs=d["Bt_obs"], B_sys=d["Bsys"],
        obs=d["obs"], rec=d["rec"], dt=d["dt"], sig_stat=np.where(d["bad"], np.nan, d["sig_stat"]),
        sig_sys=np.where(d["bad"], np.nan, d["sig_sys"]),
        sig_obs=d["er"].sig_obs, sig_fill=d["er"].sig_fill, sig_align=d["er"].sig_align,
        sig_det=d["er"].sig_det, sig_degsys=d["er"].sig_degsys,
        sig_rec=np.where(d["bad"], np.nan, d["sig_rec"]),
        dt_box=d["dt_box"], obs_constB=d["obs_c"], rec_constB=d["rec_c"], dt_constB=d["dt_c"],
        gecam_net=g["net"], gecam_sig=g["sig"], gecam_net_dc=g["net_dc"], gecam_sig_dc=g["sig_dc"],
        gecam_bsys=d["g_bsys"], gecam_live=g["live"]))
    lc.to_csv(OUT / "lc_221009a_v2_1s.csv", index=False, float_format="%.6g")
    w = ~d["bad"] & np.isfinite(d["dt_box"]) & (X >= 170) & (X < 310)
    rel = np.abs(d["dt_box"][w] - d["dt"][w]) / np.abs(d["dt"][w])
    emit("box_vs_mean_correction_max", f"{np.nanmax(rel) * 100:.2f}", "%",
         "max |sum_b N_b/f_b - N/f_mean| / R over valid main-window bins (the light curve uses N/f_mean)")


def main():
    d = load()
    sec_background(d)
    sec_mask(d)
    sec_peaks(d)
    sec_integral(d)
    sec_fillmode(d)
    s = sec_flare(d)
    d["flare_iv"] = (s["start"], s["end"])
    sec_ratios(d)
    sec_scale(d)
    tau_g = sec_tau(d)
    sec_scaleflux(d, tau_g)
    sec_perpeak(d)
    sec_flare_bsys(d)
    sec_timing()
    write_lc(d)
    for old in ("ratio_main_raw_med", "ratio_main_raw_sig", "ratio_main_raw_n", "ratio_main_dt_med",
                "ratio_main_dt_sig", "ratio_main_dt_n", "ratio_flare_raw_med", "ratio_flare_raw_sig",
                "ratio_flare_raw_n", "ratio_flare_dt_med", "ratio_flare_dt_sig", "ratio_flare_dt_n"):
        emit(old, "deprecated", "", "replaced by ratio_<subset>_<dt|raw>_<gdc|graw>_* (see freeze.md table)")
    emit("peak_obs_ceiling", "deprecated", "evt/s", "replaced by obs_gross_max / obs_net_max / obs_gross_median_bright")
    pd.DataFrame(rows, columns=["metric", "value", "unit", "note"]).to_csv(OUT / "numbers_v2.csv", index=False)
    (OUT / "numbers_v2_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
