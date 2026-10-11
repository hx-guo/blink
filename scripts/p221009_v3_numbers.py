#!/usr/bin/env python3
"""GRB 221009A v3 frozen numbers (v2 + referee revision): light curve with errors, HXMT/GECAM ratios,
peaks, fill-mode decomposition, tau, scale-flux decades, flare.

Run from the blink repo root after p221009_v3_bkg.py (on the farm: jobs/run_v3.sh):
    python scripts/p221009_v3_numbers.py

Derived from p221009_v2_numbers.py (unchanged, so v2 stays reproducible); v3 adds
the sections marked "v3" at the end and unifies the constant-B comparison on
B_true = mean N/f before T0-10 s (9464; 9332 is the observed gross value).

Inputs reused from the first-wave groups (read only):
  data/p221009/errors/lc_221009a_1s_errors.csv   per-bin error terms (errors group)
  data/p221009/audit/reconstruct.err, detect_?.txt  fill branch per gap (audit group)
  data/p221009/v3/bkg_model_1s.csv, bkg_bands_1s.csv   B(t) (p221009_v3_bkg.py)

Outputs (data/p221009/v3/):
  numbers_v3.csv            metric,value,unit,note
  lc_221009a_v3_1s.csv      1-s light curve with errors and the GECAM reference
  ratio_stats_v3.csv        all ratio subsets x HE background x GECAM correction
  ratio_bins_v3.csv         per-bin ratios with errors (standard selection)
  scale_flux_v3.csv         per-bin scale-flux points and decade medians
  numbers_v3_log.txt        printed tables
Deterministic; every exclusion is written to the CSV notes.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
from p221009_v2_common import (X, EDGES, SCALE, he_counts, he_events, is_marker,  # noqa: E402
                               he_off_intervals, off_mask, eng_on_grid, gecam_lg, robust)
from p221009_err_budget import wstats, gecam as err_gecam  # noqa: E402
from engineering_prediction import L_CYC_TO_SEC  # noqa: E402

OUT = Path("data/p221009/v3")
OUT.mkdir(parents=True, exist_ok=True)

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
    m = (full.cal_std.to_numpy() & np.isfinite(full.f_live)).astype(bool)
    cb = np.sum((1 / full.f_live[m] - 1) * full.n_over_f[m]) / np.sum(full.n_over_f[m] ** 2)

    N, No = c["n_all"], c["n_obs"]
    Bc_obs = N[X < -10].mean()                  # observed gross pre-burst background (9332)
    Bc_true = np.mean(N[X < -10] / f[X < -10])  # dead-time-corrected pre-burst background (9464)
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
            # constant-B comparison: observed curves minus the observed 9332, the
            # dead-time-corrected curve minus the dead-time-corrected 9464 (v2 used 9332 here)
            obs_c=No - Bc_obs, rec_c=N - Bc_obs, dt_c=np.where(ok, N / f - Bc_true, np.nan), Bc_true=Bc_true,
            dt_box=sum(c["all_" + b] / eng[b]["f"] for b in "ABC") - Bt,
            B_alt={k: bk[k].to_numpy() for k in ("B_timing_occ1750", "B_timing_occ2000",
                                                  "B_timing_pre_only", "B_timing_occ_only", "B_acd",
                                                  "B_acd_desc", "B_le_quad")},
            B_hour13=bk["B_timing_hour13"].to_numpy(),
            B_v2cal=bk["B_timing_v2cal"].to_numpy(),
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
                             + er.sig_degsys ** 2 + (Bt - Bc_true) ** 2).to_numpy()
    d["sig_rec"] = np.sqrt(er.var_rec + bk.sig_B ** 2).to_numpy()
    return d


# ───────────────────────────── sections ─────────────────────────────

def sec_background(d):
    say("== background ==")
    emit("B_obs_pre", f"{d['Bc']:.0f}", "evt/s",
         "pre-burst background, OBSERVED gross rate (mean of 1B obs+filler over T0-50..-10); the v1 'B'. "
         "Only subtracted from the observed / reconstructed curves (constant-B comparison)")
    Bt0 = np.mean(d["N"][X < -10] / d["f"][X < -10])
    emit("B_true_pre", f"{Bt0:.0f}", "evt/s",
         "pre-burst background, DEAD-TIME-CORRECTED (mean of N/f over T0-50..-10) = B_obs_pre/f_bkg; "
         "the constant B of every dt_constB number and curve in v3 (v2 subtracted 9332 there)")
    for lo, hi in ((-50, -10), (180, 300), (300, 440), (440, 610), (600, 700)):
        w = (X >= lo) & (X < hi)
        emit(f"B_t_mean_{lo}_{hi}", f"{np.mean(d['Bt'][w]):.0f}", "evt/s",
             "standard B_true(t): linear Cnt_LargeEvt model refitted on T0-4220..-10 + T0+1930..(egress-50) "
             "(bridged 174-300/498-540), "
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
    for lo, hi in ((600, 650), (650, 700)):
        w = (X >= lo) & (X < hi)
        emit(f"tail_source_{lo}_{hi}_v2cal", f"{np.nanmean(d['N'][w] / d['f'][w] - d['B_v2cal'][w]):.0f}", "evt/s",
             "same with the v2 calibration (T0-1020..-10 + 1780..2580 s)")


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
    rt.to_csv(OUT / "ratio_stats_v3.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    say(rt[rt.he.isin(["dt_Bt", "rec_Bt"])].drop(columns="rule").to_string())

    for name in subsets:
        if name == "flare_interval":      # identical bin set to "flare"; kept in ratio_stats_v3.csv only
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
    pd.concat(rows_b).to_csv(OUT / "ratio_bins_v3.csv", index=False, float_format="%.5g")
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
            rr_ = np.median(d["rec"][w] / (SCALE * G[w]))
            rd_ = np.median(d["dt"][w] / (SCALE * G[w]))
            Rm = np.median(G[w])
            tau_box = tau_g * 3 / SCALE                 # per recorded event, per box (s)
            fmod = 1 / (1 + tau_box * (SCALE * Rm + np.median(d["Bt"][w])) / 3)
            emit(f"rhoflux_{gtag}_{lo}_{hi}_raw", f"{rr_:.3f}", "",
                 f"median HE_rec/(200 R) (HE/GECAM convention); n={w.sum()}; median R {Rm:.0f} cts/s; "
                 f"model 1/[1+tau_box(200R+B)/3] at the median R and B: {fmod:.3f}; median measured f {fl:.3f}")
            emit(f"rhoflux_{gtag}_{lo}_{hi}_dt", f"{rd_:.3f}", "", "median HE_DT/(200 R)")
            say(f"  {gtag} {lo:>5}-{hi:<6} raw {mr:6.2f} dt {md:5.2f} n={w.sum():3d} f {fl:.3f} model {model:5.2f}")
        pts.append(pd.DataFrame(dict(gecam=gtag, t=X[sig], G=G[sig], scale_raw=sr[sig], scale_dt=sd[sig],
                                     f_live=d["f"][sig], obs_gross=d["No"][sig], B_true=d["Bt"][sig])))
    emit("scaleflux_selection", "HE rec net>5e3 & GECAM>30 & valid", "",
         "HE-off bins (incl. partial 229, 248, 266, 267, 276) excluded; decades on the respective GECAM rate")
    pd.concat(pts).to_csv(OUT / "scale_flux_v3.csv", index=False, float_format="%.6g")
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
    tab.to_csv(OUT / "peak_seconds_v3.csv", index=False, float_format="%.6g")
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
    for lo, hi in ((300, 440), (440, 498), (545, 600), (600, 700)):
        w = (X >= lo) & (X < hi)
        emit(f"B_std_minus_ACD_{lo}_{hi}", f"{np.mean(d['Bt'][w] - d['B_alt']['B_acd'][w]):+.0f}", "evt/s",
             "mean B_LargeEvt - B_ACD (ACD-tag proxy corrected for source tagging) in the window")
    t90s = []
    for k, B in list(d["B_alt"].items()) + [("B_std", d["Bt"])]:
        r, _ = flare_numbers(X, np.where(d["bad"], np.nan, d["N"] / d["f"] - B), "", "")
        t90s.append(r["t90"])
    emit("flare_t90_Bmodel_range", f"{min(t90s):.0f}..{max(t90s):.0f}", "s",
         "flare T90 (local-continuum definition) over all background models")


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
    lc.to_csv(OUT / "lc_221009a_v3_1s.csv", index=False, float_format="%.6g")
    w = ~d["bad"] & np.isfinite(d["dt_box"]) & (X >= 170) & (X < 310)
    rel = np.abs(d["dt_box"][w] - d["dt"][w]) / np.abs(d["dt"][w])
    emit("box_vs_mean_correction_max", f"{np.nanmax(rel) * 100:.2f}", "%",
         "max |sum_b N_b/f_b - N/f_mean| / R over valid main-window bins (the light curve uses N/f_mean)")


# ═════════════════════════════ v3 sections ═════════════════════════════

AN23 = {  # An+2023 Methods figures, read by the literature group (lit.md par.10.3), +-10%
    "csi_0p4_2p95": [(600, 610, 1200), (695, 705, 720), (795, 805, 650), (890, 900, 550)],
    "csi_0p6_3": [(625, 700, 490), (795, 805, 400), (890, 900, 350)],
}


def sec_afterglow(d):
    """v3 item 2: net rates 600-700 / 700-900 s and the An+2023 HE/CsI comparison."""
    say("== afterglow ==")
    hb = pd.read_csv(OUT / "bkg_hours_1s.csv")
    bb = pd.read_csv(OUT / "bkg_bands_1s.csv")
    t = hb.t.to_numpy()
    nf = hb.n_over_f.to_numpy()
    alts = [c for c in hb.columns if c.startswith("B_") and c not in ("B_std", "B_sys", "B_l", "B_timing_hour13",
                                                                        "B_timing_v2cal")]
    for lo, hi in ((600, 700), (700, 900), (600, 650), (650, 700)):
        w = (t >= lo) & (t < hi)
        S = np.mean(nf[w] - hb.B_std[w])
        st = np.sqrt(np.sum(hb.n1k[w] / hb.f_live[w] ** 2)) / w.sum()
        vals = {c: np.mean(nf[w] - hb[c][w]) for c in alts}
        lo_v, hi_v = min(vals.values()), max(vals.values())
        emit(f"net_all_{lo}_{hi}", f"{S:.0f}", "evt/s",
             f"all events, DT-corrected, minus v3 B(t); stat {st:.0f}; over the background models "
             f"{lo_v:.0f}..{hi_v:.0f} ({min(vals, key=vals.get)} .. {max(vals, key=vals.get)}); "
             f"v2 calibration {np.mean(nf[w] - hb.B_timing_v2cal[w]):.0f}")
    f = bb.f_live.to_numpy()
    for band in ("csi_0p4_2p95", "csi_0p6_3", "csi_all"):
        n = bb[band + "_n"].to_numpy()
        for lo, hi in ((600, 700), (700, 900)):
            w = (t >= lo) & (t < hi)
            vals = {k: np.mean(n[w] / f[w] - bb[band + "_B" + k][w]) for k in ("", "_hour13", "_occ2000", "_acd")}
            emit(f"net_{band}_{lo}_{hi}", f"{vals['']:.0f}", "cts/s",
                 f"1K CsI (pw>75) band, DT-corrected, minus the band background (LargeEvt, v3 windows); "
                 f"hour13-only {vals['_hour13']:.0f}, occ2000 {vals['_occ2000']:.0f}, ACD proxy {vals['_acd']:.0f}; "
                 f"stat {np.sqrt(np.sum(n[w] / f[w] ** 2)) / w.sum():.0f}")
        for lo, hi, an in AN23.get(band, []):
            w = (t >= lo) & (t < hi)
            vals = [np.mean(n[w] / f[w] - bb[band + "_B" + k][w]) for k in ("", "_occ2000", "_acd")]
            h13 = np.mean(n[w] / f[w] - bb[band + "_B_hour13"][w])
            emit(f"an23_{band}_{lo}_{hi}", f"{vals[0]:.0f}", "cts/s",
                 f"this work (range over the v3 background models {min(vals):.0f}..{max(vals):.0f}; "
                 f"one-hour (v2-like) calibration {h13:.0f}; stat "
                 f"{np.sqrt(np.sum(n[w] / f[w] ** 2)) / w.sum():.0f}) vs An+2023 read-off {an} cts/s")
            say(f"  {band} {lo}-{hi}: {vals[0]:.0f} (range {min(vals):.0f}..{max(vals):.0f}) vs An+2023 {an}")
    emit("an23_band_note", "approx", "",
         "low-gain CsI channel ranges ch43-255 / ch61-255 with pulse width > 75 stand for 0.4-2.95 / 0.6-3 MeV "
         "(E ~ 0.011 ch - 0.07 MeV from eband.md; ch255 ~ 2.7 MeV); no response matrix, no PSD as in An+2023")


def sec_tau_sys(d):
    """v3 item 5: fitting-method spread of the GECAM-based tau."""
    say("== tau method systematics ==")
    g, f = d["g"], d["f"]
    G = g["net_dc"]
    sel = ~d["bad"] & (d["rec"] > 5e3) & (G > 30) & np.isfinite(f) & np.isfinite(G)
    n = (SCALE * G[sel] + d["Bt"][sel]) / 3
    y = 1 / f[sel] - 1
    est = {}
    (tb,), _ = curve_fit(lambda x, tau: 1 / (1 + tau * x), n, f[sel], p0=[4e-6])
    est["LSQ on f, per bin"] = tb
    est["LSQ on 1/f-1, through origin"] = np.sum(y * n) / np.sum(n * n)
    est["median of (1/f-1)/n"] = np.median(y / n)
    est["median of (1/f-1)/n, f<0.5"] = np.median((y / n)[f[sel] < 0.5])
    sG = np.hypot(g["sig_dc"][sel], d["g_bsys"][sel]) / G[sel] * (SCALE * G[sel] / (3 * n))
    w = 1 / (sG ** 2 + 1e-6)
    est["weighted LSQ on 1/f-1 (GECAM errors)"] = np.sum(w * y / n * n * n) / np.sum(w * n * n)
    dec = [30, 100, 300, 1000, 3000, 10000, 30000]
    cen, fm = [], []
    for lo, hi in zip(dec[:-1], dec[1:]):
        m = (G[sel] >= lo) & (G[sel] < hi)
        cen.append((SCALE * np.sqrt(lo * hi) + np.median(d["Bt"][sel][m])) / 3)
        fm.append(np.median(f[sel][m]))
    (td,), _ = curve_fit(lambda x, tau: 1 / (1 + tau * x), np.array(cen), np.array(fm), p0=[4e-6])
    est["decade medians vs decade centre"] = td
    for k, v in est.items():
        say(f"  {k:42s} {v * 1e6:.2f} us/record")
    vals = np.array(list(est.values())) * 1e6
    lo, hi = vals.min(), vals.max()
    emit("tau_box_rec_methods_range", f"{lo:.2f}..{hi:.2f}", "us",
         "per-recorded-event box tau from GECAM over fitting methods: " +
         "; ".join(f"{k} {v * 1e6:.2f}" for k, v in est.items()))
    cnt = float([r for r in rows if r[0] == "tau_box_rec_deep_us"][0][1])
    trig = float([r for r in rows if r[0] == "tau_box_trig_flare442_593_us"][0][1])
    k = float([r for r in rows if r[0] == "k_trig_per_rec_burst"][0][1])
    emit("tau_counter_vs_fit_rec", f"{(lo / cnt - 1) * 100:+.0f}..{(hi / cnt - 1) * 100:+.0f}", "%",
         f"per recorded event: fit methods {lo:.2f}-{hi:.2f} us vs counter {cnt:.2f} us on the deep bins")
    emit("tau_counter_vs_fit_trig", f"{(lo / k / trig - 1) * 100:+.0f}..{(hi / k / trig - 1) * 100:+.0f}", "%",
         f"per trigger: fit methods / k_burst {k:.2f} = {lo / k:.2f}-{hi / k:.2f} us vs counter {trig:.2f} us; "
         "the method spread is the systematic of the fit")
    emit("tau_scale_dependence", f"{(200 / 181.4 - 1) * 100:+.0f}", "%",
         "tau from the GECAM fit scales ~1/(x200 scale) at high rate: with x181 instead of x200 the fitted tau "
         "rises by ~10% (n_true is proportional to the scale)")


def sec_norm(d):
    """v3 item 6: what the x200 normalization does and does not affect."""
    say("== normalization ==")
    s_fit, s_fit17 = 181.4, 193.7
    for name in ("main", "main_bright", "main_deep", "P1peak", "P2", "flare"):
        med = float([r for r in rows if r[0] == f"ratio_{name}_dt_gdc_med"][0][1])
        emit(f"ratio_{name}_dt_gdc_med_scale181", f"{med * 200 / s_fit:.3f}", "",
             f"same median with x{s_fit} (10-bin weighted fit) instead of x200; with x{s_fit17} (17-bin fit): "
             f"{med * 200 / s_fit17:.3f}")
    emit("norm_uncertainty", f"{27 / 181.4 * 100:.0f} (10 bins) / {13 / 193.7 * 100:.0f} (17 bins)", "%",
         "fractional uncertainty of the x200 normalization; it multiplies every HE/GECAM ratio and nothing else")
    j = int(227 - X[0])
    er = d["er"]
    comp = dict(align=er.sig_align[j], degsys=er.sig_degsys[j], det=er.sig_det[j], bkg=d["Bsys"][j])
    emit("peak1_err_stat", f"{d['sig_stat'][j]:.3g}", "evt/s",
         f"statistical (observed Poisson + gap-fill + f quantisation + B), {d['sig_stat'][j] / d['dt'][j] * 100:.2f}%")
    for k, v in comp.items():
        emit(f"peak1_err_sys_{k}", f"{v:.3g}", "evt/s", f"{v / d['dt'][j] * 100:.2f}% of the T0+227 s rate")
    d["peak1_sys_comp"] = comp


def sec_p1p2(d):
    """v3 item 7: Peak-1/Peak-2 from HE and GECAM on the same bins."""
    say("== P1/P2 ==")
    g = d["g"]
    i1, i2 = int(227 - X[0]), int(262 - X[0])
    he = d["dt"][i1] / d["dt"][i2]
    emit("p1p2_he_227_262", f"{he:.3f}", "", "HE DT-corrected, T0+227 / T0+262 s")
    for k, tag in (("net_dc", "gdc"), ("net", "graw")):
        gg = g[k][i1] / g[k][i2]
        emit(f"p1p2_gecam_{tag}_227_262", f"{gg:.3f}", "",
             f"GECAM LG {'dead-time corrected' if tag == 'gdc' else 'uncorrected'}, same bins; HE/GECAM {he / gg:.3f}")
        w1 = (X >= 180) & (X < 240)
        w2 = (X >= 240) & (X < 300)
        m1, m2 = np.nanmax(g[k][w1]), np.nanmax(g[k][w2])
        emit(f"p1p2_gecam_{tag}_maxima", f"{m1 / m2:.3f}", "",
             f"brightest second of each pulse (T0+{X[w1][np.nanargmax(g[k][w1])]:.0f} / "
             f"{X[w2][np.nanargmax(g[k][w2])]:.0f} s; the first inside the HE-off interval)")
    w1 = (X >= 226) & (X < 229)
    w2 = (X >= 255) & (X < 266)
    emit("p1p2_he_max_visible", f"{np.nanmax(d['dt'][w1]) / np.nanmax(d['dt'][w2]):.3f}", "",
         "HE: brightest valid second of P1 (226-228) / of P2 (255-265)")


def gecam_rows():
    """50-ms GECAM LG rows: t (rel T0), net dead-time-corrected rate (cts/s)."""
    from astropy.io import fits
    from p221009_v2_common import GECAM_BTIME, GECAM_BKG, GECAM_T0
    fb, fk = fits.open(GECAM_BTIME), fits.open(GECAM_BKG)
    b, k = fb["SPECTRUM01"].data, fk["SPECTRUM01"].data
    off = fk[0].header["TSTART"] - fb[0].header["TSTART"]
    tb = b["STARTTIME"] - GECAM_T0
    tk = k["STARTTIME"] - off - GECAM_T0
    cb = b["COUNTS"][:, 9:13].sum(1).astype(float)
    ck = np.interp(tb, tk, k["COUNTS"][:, 9:13].sum(1).astype(float))
    eb = b["EXPOSURE"][:, 1] / 0.05
    ek = np.interp(tb, tk, k["EXPOSURE"][:, 1] / 0.05)
    return tb, (cb / eb - ck / ek) / 0.05


def sec_jensen(d):
    """v3 item 8: sub-second (Jensen) bias of the 1-Hz live-fraction correction.

    Per 50-ms row i of second k: GECAM LG dead-time-corrected net rate r_i
    gives the HE true record rate per box n_i = (200 r_i + B_true)/3 and the
    trigger rate T_i = (k_burst 200 r_i + k_bkg B_true)/3; a non-paralyzable
    box chain with tau_trig has live fraction f_i = 1/(1 + tau T_i). The
    counter measures fbar = <f_i>; the recorded count is sum f_i n_i, and the
    1-Hz correction returns sum f_i n_i / fbar instead of sum n_i.
    bias = (<f n>/<f> - <n>)/(<n> - B/3), relative to the net rate (< 0). Model A: tau = counter value; model B:
    tau rescaled per second so that fbar equals the measured f. Variability
    below 50 ms is not resolved, so |bias| is a lower limit. Deterministic.
    """
    say("== Jensen ==")
    tb, r = gecam_rows()
    tau = float([x for x in rows if x[0] == "tau_box_trig_flare442_593_us"][0][1]) * 1e-6
    kb = float([x for x in rows if x[0] == "k_trig_per_rec_burst"][0][1])
    kg = float([x for x in rows if x[0] == "k_trig_per_rec_bkg"][0][1])
    out = []
    for k in X[(X >= 180) & (X < 300) & ~d["bad"]].astype(int):
        m = (tb >= k - 1e-6) & (tb < k + 1 - 1e-6)
        ri = np.clip(r[m], 0, None)
        j = int(k - X[0])
        B = d["Bt"][j]
        n = (SCALE * ri + B) / 3
        T = (kb * SCALE * ri + kg * B) / 3

        def bias(tt):
            fi = 1 / (1 + tt * T)
            # relative to the NET rate: (sum f n / fbar - sum n) / (sum n - B)
            return (np.mean(fi * n) / np.mean(fi) - np.mean(n)) / (np.mean(n) - B / 3), np.mean(fi)
        bA, fA = bias(tau)
        lo, hi = 1e-8, 1e-3
        for _ in range(100):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if bias(mid)[1] > d["f"][j] else (lo, mid)
        bB, fB = bias(np.sqrt(lo * hi))
        cv = np.std(ri) / max(np.mean(ri), 1e-9)
        out.append(dict(t=k, f_meas=d["f"][j], f_modelA=fA, biasA=bA, tau_B_us=np.sqrt(lo * hi) * 1e6,
                        biasB=bB, gecam_cv50ms=cv))
    tab = pd.DataFrame(out)
    tab.to_csv(OUT / "jensen_v3.csv", index=False, float_format="%.5g")
    say(tab.to_string(float_format=lambda v: f"{v:.4g}"))
    for t in (226, 227, 228, 262):
        q = tab[tab.t == t].iloc[0]
        emit(f"jensen_{t}", f"{q.biasB * 100:+.2f}", "%",
             f"1-Hz correction minus 50-ms correction, model B (tau matched to the measured f); "
             f"model A (tau={tau * 1e6:.2f} us/trigger, k={kb}/{kg}) {q.biasA * 100:+.2f}%; GECAM 50-ms CV {q.gecam_cv50ms:.2f}")
    emit("jensen_main_median", f"{np.median(tab.biasB) * 100:+.2f}", "%",
         f"median over {len(tab)} valid main-window seconds; min {tab.biasB.min() * 100:+.2f}% at T0+"
         f"{tab.t[tab.biasB.idxmin()]:.0f} s (model B)")
    deep = tab[tab.f_meas < 0.5]
    emit("jensen_deep_median", f"{np.median(deep.biasB) * 100:+.2f}", "%",
         f"median over the {len(deep)} seconds with f<0.5 (model B); lower limit (no sub-50-ms variability)")
    q1, q2 = tab[tab.t == 227].iloc[0], tab[tab.t == 262].iloc[0]
    emit("jensen_p1p2_effect", f"{((1 + q2.biasB) / (1 + q1.biasB) - 1) * 100:+.2f}", "%",
         "change of the HE 227/262 ratio if the 1-Hz bias were removed")
    pk = tab[tab.t.isin([226, 227, 228])]
    med = float([x for x in rows if x[0] == "ratio_P1peak_dt_gdc_med"][0][1])
    emit("ratio_P1peak_dt_gdc_med_jensen_corr", f"{med / (1 + np.median(pk.biasB)):.3f}", "",
         "P1 peak ratio after removing the (model B) Jensen bias of 226-228 s")
    d["jensen"] = tab


def sec_lcyc(d):
    """v3 item 9: amplification of the D and L_cyc precision at low live fraction."""
    say("== L_cyc ==")
    eng = d["eng"]
    allL = np.concatenate([eng[b]["L"][np.isfinite(eng[b]["L"])] for b in "ABC"])
    emit("Lcyc_units", f"{np.median(allL):.0f}", "16-us ticks",
         f"Length_Time_Cycle: median {np.median(allL):.0f} = {np.median(allL) * 16e-6:.4f} s, range "
         f"{allL.min():.0f}-{allL.max():.0f} over T0-50..700 (integer count of the same 16-us clock as D)")
    Dmax = np.nanmax([np.nanmax(eng[b]["D"][~d["bad"]]) for b in "ABC"])
    emit("D_max_units", f"{Dmax:.0f}", "16-us ticks",
         f"largest box-mean DeadTime_PHODet in valid bins = {Dmax / 65535 * 100:.0f}% of the 16-bit range, "
         f"D/L = {Dmax / np.median(allL):.3f}")
    for t in (227, 262):
        j = int(t - X[0])
        amps, dq = [], []
        for b in "ABC":
            D, L, fb = eng[b]["D"][j], eng[b]["L"][j], eng[b]["f"][j]
            A = (1 - fb) / fb
            # +-1 tick in D (box mean of 6: 1/sqrt(6)) and in L, uniform rounding rms 1/sqrt(12)
            dfrel = np.sqrt((1 / (np.sqrt(6) * D)) ** 2 + (1 / L) ** 2) / np.sqrt(12) * A
            amps.append(A); dq.append(dfrel)
        A = np.mean(amps)
        emit(f"Lcyc_amplification_{t}", f"{A:.1f}", "",
             f"(1-f)/f: relative error of f per relative error of D/L at T0+{t} s (f={d['f'][j]:.3f})")
        emit(f"Lcyc_quantisation_err_{t}", f"{np.mean(dq) * 100:.4f}", "%",
             "relative error of f from the integer D and L counters (rms of rounding), amplified")
        emit(f"Lcyc_hypothetical_0p5pct_{t}", f"{A * 0.5:.1f}", "%",
             "error of f if L_cyc were uncertain by 0.5% (the referee's example); L_cyc is counted on the "
             "same clock as D, so a scale error of the clock cancels in D/L")


def sec_peak_budget(d):
    """v3 item 6 (cont.): suggested statistical / systematic split of the T0+227 s rate."""
    j = int(227 - X[0])
    R = d["dt"][j]
    comp = dict(d["peak1_sys_comp"])
    jb = d["jensen"]
    comp["jensen"] = abs(jb[jb.t == 227].biasB.iloc[0]) * R
    comp["lcyc"] = float([x for x in rows if x[0] == "Lcyc_quantisation_err_227"][0][1]) / 100 * R
    tot = np.sqrt(sum(v ** 2 for v in comp.values()))
    emit("peak1_err_sys_total", f"{tot:.3g}", "evt/s",
         "quadrature sum of " + ", ".join(f"{k} {v / R * 100:.2f}%" for k, v in comp.items())
         + f" = {tot / R * 100:.1f}%; none of these depends on the x200 normalization")
    emit("peak1_quote", f"({R / 1e6:.2f} +- {d['sig_stat'][j] / 1e6:.2f} (stat) +- {tot / 1e6:.2f} (sys))e6", "evt/s",
         "suggested quote; the HE/GECAM ratio at this bin carries in addition the x200 normalization "
         "(+-7-15%) and the GECAM LG dead-time correction")


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
    sec_afterglow(d)
    sec_tau_sys(d)
    sec_norm(d)
    sec_p1p2(d)
    sec_jensen(d)
    sec_lcyc(d)
    sec_peak_budget(d)
    write_lc(d)
    for old in ("ratio_main_raw_med", "ratio_main_raw_sig", "ratio_main_raw_n", "ratio_main_dt_med",
                "ratio_main_dt_sig", "ratio_main_dt_n", "ratio_flare_raw_med", "ratio_flare_raw_sig",
                "ratio_flare_raw_n", "ratio_flare_dt_med", "ratio_flare_dt_sig", "ratio_flare_dt_n"):
        emit(old, "deprecated", "", "replaced by ratio_<subset>_<dt|raw>_<gdc|graw>_* (see freeze.md table)")
    emit("peak_obs_ceiling", "deprecated", "evt/s", "replaced by obs_gross_max / obs_net_max / obs_gross_median_bright")
    pd.DataFrame(rows, columns=["metric", "value", "unit", "note"]).to_csv(OUT / "numbers_v3.csv", index=False)
    (OUT / "numbers_v3_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
