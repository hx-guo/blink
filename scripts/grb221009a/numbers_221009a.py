#!/usr/bin/env python3
"""Quantitative numbers for the GRB 221009A HE light-curve paper.

Recomputes every number quoted in the paper from the corrected pipeline
(gross/f_live - B dead-time correction) and prints a CSV to stdout.
Run from the blink repo root:
    ./.venv/bin/python scripts/grb221009a/numbers_221009a.py > ../paper-hxmt-221009a/numbers.csv
"""
import sys
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../gecam", "../talks",)]  # sibling and shared script modules
import numpy as np

from make_f10 import (load_hxmt, load_eng_dtau, to_grid, GECAM_SCALE)
from plot_hxmt_vs_gecam import compute_time_offset, load_gecam_btime

BEFORE, AFTER, BIN = 50.0, 700.0, 1.0

rows = []
def emit(metric, value, unit="", note=""):
    rows.append((metric, value, unit, note))

# ---- load ----
x, edges, obs, allr, n_fill, B_evt = load_hxmt(BEFORE, AFTER, BIN)
et, dtau, flive, tau_us, B_eng = load_eng_dtau(BEFORE, AFTER)
g_met, _, _ = compute_time_offset()
_, g_rate, _ = load_gecam_btime(g_met, BEFORE, AFTER, BIN, "lg", True)
g_rate = np.nan_to_num(g_rate, nan=0.0)
gecam_s = g_rate * GECAM_SCALE

flive_g = to_grid(x, et, flive)
with np.errstate(divide="ignore", invalid="ignore"):
    all_dt = np.where(flive_g > 0.05, (allr + B_evt) / flive_g - B_evt, np.nan)

emit("B_evt", f"{B_evt:.0f}", "evt/s", "gross event-level background, mean of T0-50..-10")
emit("tau_det_eng_us", f"{np.nanmean(tau_us):.1f}", "us",
     "per-detector per-event processing time, D/Cnt_PHODet tail T0+300..700")

# ---- live fraction extremes ----
win_main = (et >= 170) & (et < 310)
i_min = np.nanargmin(np.where(win_main, flive, np.nan))
emit("flive_min", f"{flive[i_min]:.3f}", "", "deepest 18-det mean live fraction, main window")
emit("flive_min_time", f"{et[i_min]:.0f}", "s", "time of deepest live fraction")

# ---- main-pulse peaks (DT-corrected), split at T0+240 ----
m1 = (x >= 170) & (x < 240)
m2 = (x >= 240) & (x < 310)
for name, m in (("peak1", m1), ("peak2", m2)):
    r = np.where(m, all_dt, np.nan)
    i = np.nanargmax(r)
    emit(f"{name}_time", f"{x[i]:.0f}", "s", f"{name} time of DT-corrected maximum")
    emit(f"{name}_rate_dt", f"{r[i]:.3e}", "evt/s", f"{name} DT-corrected net rate")
emit("peak_obs_ceiling", f"{np.nanmax(np.where((x>=170)&(x<310), obs, np.nan)):.3e}",
     "evt/s", "max observed (FIFO-limited) rate in main window")

# ---- main-pulse integrated counts ----
mwin = (x >= 180) & (x < 300)
vals = all_dt[mwin]
emit("mainpulse_window", "180..300", "s", "integration window")
emit("mainpulse_counts_dt", f"{np.nansum(vals)*BIN:.3e}", "counts",
     "DT-corrected net counts, NaN bins excluded")
emit("mainpulse_nan_bins", f"{int(np.sum(~np.isfinite(vals)))}", "bins",
     "bins excluded (SAA-mode shutoff / f_live<=0.05 / no eng frame)")

# ---- flare: excess above the local decaying-tail continuum ----
# The post-main-pulse flare sits on the slowly decaying main-pulse tail, so its
# duration is defined relative to a LOCAL continuum anchored on the tail just
# outside the flare bump, NOT the pre-trigger background.
pre_a = (x >= 400) & (x < 425)
post_a = (x >= 615) & (x < 640)
r_pre, t_pre = np.nanmean(allr[pre_a]), np.nanmean(x[pre_a])
r_post, t_post = np.nanmean(allr[post_a]), np.nanmean(x[post_a])
cont = r_pre + (r_post - r_pre) * (x - t_pre) / (t_post - t_pre)
sig_cont = np.nanstd(np.concatenate([allr[pre_a] - r_pre, allr[post_a] - r_post]))
search = (x >= 425) & (x < 610)
excess = np.where(search, allr - cont, np.nan)
flare_bins = search & np.isfinite(excess) & (excess > 3 * sig_cont)
idx_f = np.where(flare_bins)[0]
if len(idx_f) == 0:
    for _k in ("flare_start", "flare_end", "flare_t90", "flare_t90_start",
               "flare_t90_end", "flare_peak_time", "flare_excess_counts",
               "flare_recovered_events"):
        emit(_k, "nan", "", "no 1-s bin exceeded 3sigma above local tail continuum")
else:
    f_start, f_end = x[idx_f[0]], x[idx_f[-1]] + BIN
    emit("flare_start", f"{f_start:.0f}", "s",
         "first 1-s bin >3sigma above local tail continuum (400-425 & 615-640 anchors)")
    emit("flare_end", f"{f_end:.0f}", "s",
         "last 1-s bin >3sigma above local tail continuum")
    sel = (x >= f_start) & (x < f_end) & np.isfinite(excess)
    exc, t_sel = excess[sel], x[sel]
    cexc = np.clip(exc, 0, None)
    cum = np.cumsum(cexc) / np.sum(cexc)
    t05, t95 = t_sel[np.searchsorted(cum, 0.05)], t_sel[np.searchsorted(cum, 0.95)]
    emit("flare_t90", f"{t95 - t05:.0f}", "s", "5%..95% cumulative continuum-subtracted counts")
    emit("flare_t90_start", f"{t05:.1f}", "s", "")
    emit("flare_t90_end", f"{t95:.1f}", "s", "")
    emit("flare_peak_time", f"{t_sel[np.nanargmax(exc)]:.0f}", "s",
         "brightest continuum-subtracted 1-s bin")
    emit("flare_excess_counts", f"{np.nansum(exc)*BIN:.2e}", "counts",
         "continuum-subtracted net counts in the flare interval")
    fill_mask = (x >= f_start) & (x < f_end)
    emit("flare_recovered_events", f"{np.nansum((allr - obs)[fill_mask])*BIN:.2e}", "counts",
         "reconstructed-minus-observed events across the flare interval")

# ---- ratio stats (same significance cut as make_f10 main mode) ----
def robust(rr):
    v = rr[np.isfinite(rr)]
    if len(v) == 0:
        return None
    med = np.median(v)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    return med, iqr / 1.349, len(v)

g_net = gecam_s / GECAM_SCALE
# main window (170..310) and flare window (440..610); same window-local
# G_FLOOR=50 cts/s significance cut make_f10 uses (H_FLOOR=0 off the full burst).
for win_tag, (wlo, whi) in (("main", (170, 310)), ("flare", (440, 610))):
    win = (x >= wlo) & (x < whi)
    sig = win & np.isfinite(g_net) & (g_net > 50.0) & np.isfinite(allr) & (allr > 0.0)
    for kind, num in (("raw", allr), ("dt", all_dt)):
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(sig & np.isfinite(num), num / gecam_s, np.nan)
        st = robust(rr)
        med, s, n = st if st else ("nan", "nan", 0)
        tag = f"ratio_{win_tag}_{kind}"
        emit(f"{tag}_med", f"{med:.2f}" if st else "nan", "", "HXMT/GECAM median")
        emit(f"{tag}_sig", f"{s:.2f}" if st else "nan", "", "IQR-derived robust sigma")
        emit(f"{tag}_n", f"{n}", "bins", "significant 1-s bins (GECAM net > 50 cts/s)")

import csv
w = csv.writer(sys.stdout)
w.writerow(["metric", "value", "unit", "note"])
for r in rows:
    w.writerow([str(c) for c in r])
