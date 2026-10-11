#!/usr/bin/env python3
"""Audit of the numbers quoted in the GRB 221009A HE light-curve paper.

Re-derives, from the same inputs as numbers_221009a.py / make_f10.py /
plot_scale_flux_dt.py, the quantities the paper quotes that are NOT in
numbers.csv, and lists the bins behind every ratio statistic so that no
selection is implicit. Writes text + CSV tables to data/p221009/audit/.

Run from the blink repo root:
    ./.venv/bin/python scripts/p221009_audit_numbers.py
"""
import csv, sys
from pathlib import Path
import numpy as np
from astropy.io import fits

sys.path.insert(0, "scripts")
from make_f10 import load_hxmt, load_eng_dtau, to_grid, GECAM_SCALE
from plot_hxmt_vs_gecam import compute_time_offset, load_gecam_btime
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION, L_CYC_TO_SEC

OUT = Path("data/p221009/audit")
OUT.mkdir(parents=True, exist_ok=True)
BEFORE, AFTER, BIN = 50.0, 700.0, 1.0
TRIGGER_MET_ASTROPY = 339945423.0
TRIGGER_MET_PY = 339945422.0

lines = []
def say(s=""):
    print(s)
    lines.append(s)

def robust(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return (np.nan, np.nan, 0)
    q75, q25 = np.percentile(v, [75, 25])
    return np.median(v), (q75 - q25) / 1.349, len(v)

# ---------------------------------------------------------------- inputs
x, edges, obs, allr, n_fill, B = load_hxmt(BEFORE, AFTER, BIN)
obs_gross = obs + B
all_gross = allr + B
et, dtau, flive, tau_us, B_eng = load_eng_dtau(BEFORE, AFTER)
g_met, _, _ = compute_time_offset()
_, g_raw, _ = load_gecam_btime(g_met, BEFORE, AFTER, BIN, "lg", True)
_, g_nobkg, _ = load_gecam_btime(g_met, BEFORE, AFTER, BIN, "lg", False)
g = np.nan_to_num(g_raw, nan=0.0)
flive_g = to_grid(x, et, flive)
with np.errstate(divide="ignore", invalid="ignore"):
    all_dt = np.where(flive_g > 0.05, (allr + B) / flive_g - B, np.nan)

say(f"B (gross bkg, mean T0-50..-10, obs+fill) = {B:.1f} evt/s; B/200 = {B/200:.2f}")
say(f"tau_i per detector (tail 300..700): mean {np.nanmean(tau_us):.2f} us, "
    f"median {np.nanmedian(tau_us):.2f}, range {np.nanmin(tau_us):.1f}..{np.nanmax(tau_us):.1f}")

# ------------------------------------------------- per-second eng counters
# raw per-box engineering frames: does the eng table go to zero in the SAA
# segments, and which seconds have no frame at all?
eng = {}
for box, code in BOX_CODE.items():
    fn = sorted(Path(f"data/1B/2022/20221009/{code}").glob(
        f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
    with fits.open(fn, memmap=True) as fe:
        d = fe["HE_Eng"].data
        off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met = d["Time"].astype(float) + off + MET_CORRECTION
        m = (met >= TRIGGER_MET_PY - BEFORE) & (met <= TRIGGER_MET_PY + AFTER)
        lc = d["Length_Time_Cycle"].astype(float)[m]
        dts = np.sum([d[f"DeadTime_PHODet_{BOX_OFFSET[box]+k}"].astype(float)[m]
                      for k in range(6)], axis=0)
        pho = np.sum([d[f"Cnt_PHODet_{BOX_OFFSET[box]+k}"].astype(float)[m]
                      for k in range(6)], axis=0)
        for t, a, b_, c in zip(met[m], lc, dts, pho):
            eng.setdefault(int(round(t - TRIGGER_MET_ASTROPY)), {})[box] = (a, b_, c)

# --------------------------------------------------------- SAA segments
say("\n== SAA detector-off segments (main window 200..300) ==")
say("t  obs_gross  all_gross  flive  all_dt  eng(A,B,C: Lcyc/sumD/sumPHO)")
rows_saa = []
for i, t in enumerate(x):
    if 225 <= t <= 282:
        e = eng.get(int(t), {})
        es = " ".join(f"{bx}:{int(v[0])}/{int(v[1])}/{int(v[2])}" for bx, v in sorted(e.items()))
        say(f"{t:5.0f} {obs_gross[i]:9.0f} {all_gross[i]:9.0f} {flive_g[i]:6.3f} "
            f"{all_dt[i]:11.0f}  {es}")
        rows_saa.append((t, obs_gross[i], all_gross[i], flive_g[i], all_dt[i], es))
with open(OUT / "saa_window_bins.csv", "w") as f:
    w = csv.writer(f); w.writerow(["t", "obs_gross", "all_gross", "flive", "all_dt", "eng"])
    w.writerows(rows_saa)
zero = (obs_gross == 0) & (x >= 170) & (x < 310)
partial = (obs_gross > 0) & (obs_gross < 0.2 * np.nanmedian(obs_gross[(x >= 200) & (x < 300)])) & (x >= 170) & (x < 310)
def runs(mask):
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    out, s = [], idx[0]
    for a, b_ in zip(idx[:-1], idx[1:]):
        if b_ != a + 1:
            out.append((x[s], x[a])); s = b_
    out.append((x[s], x[idx[-1]]))
    return out
say(f"bins with ZERO observed events (gross) in 170..310: runs {runs(zero)} "
    f"(n={zero.sum()})")
say(f"bins with partial (<20% of window median) observed events: runs {runs(partial)}")
saa = zero
say(f"all_dt in zero-event bins: unique values {np.unique(np.round(all_dt[saa]))}")
say(f"flive in zero-event bins: {np.unique(np.round(flive_g[saa], 3))}")

# ------------------------------------------------------ main-pulse integral
mwin = (x >= 180) & (x < 300)
I_all = np.nansum(all_dt[mwin])
I_nosaa = np.nansum(np.where(saa, 0.0, all_dt)[mwin])
say("\n== main-pulse integral 180..300 ==")
say(f"sum(all_dt) as in numbers.csv = {I_all:.4e}; NaN bins = {np.sum(~np.isfinite(all_dt[mwin]))}")
say(f"same with SAA zero bins set to 0 instead of {np.unique(np.round(all_dt[saa]))} : {I_nosaa:.4e}")
say(f"SAA zero-event bins inside 180..300: {np.sum(saa & mwin)}")
# linear-interpolation estimate of what the SAA bins would have held (GECAM x200)
gs = g * GECAM_SCALE
say(f"GECAM x200 integral over SAA bins (what HE missed, scale-based) = {np.sum(gs[saa & mwin]):.3e}")
say(f"GECAM x200 integral over 180..300 = {np.sum(gs[mwin]):.3e}")

# ----------------------------------------------------- ratio bin selection
say("\n== ratio bin selection (numbers_221009a / make_f10) ==")
for tag, (lo, hi) in (("main", (170, 310)), ("main180_300", (180, 300)), ("flare", (440, 610))):
    win = (x >= lo) & (x < hi)
    g_ok = win & (g > 50.0)
    h_ok = win & np.isfinite(allr) & (allr > 0)
    sig = g_ok & h_ok
    lost_g = win & ~(g > 50.0)
    lost_h = win & (g > 50.0) & ~(allr > 0)
    rr_raw = np.where(sig, allr / gs, np.nan)
    rr_dt = np.where(sig, all_dt / gs, np.nan)
    say(f"[{tag} {lo}..{hi}] window bins={win.sum()}  GECAM<=50: {lost_g.sum()}  "
        f"HXMT net<=0 (with GECAM>50): {lost_h.sum()} -> runs {runs(lost_h)}  "
        f"selected={sig.sum()}")
    say(f"    GECAM<=50 bins: runs {runs(lost_g)}")
    say(f"    raw {robust(rr_raw)}  dt {robust(rr_dt)}")
    with open(OUT / f"ratio_bins_{tag}.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["t", "selected", "gecam_net", "hxmt_obs_net", "hxmt_rec_net",
                    "flive", "hxmt_dt", "ratio_raw", "ratio_dt"])
        for i in np.where(win)[0]:
            w.writerow([f"{x[i]:.0f}", int(sig[i]), f"{g[i]:.1f}", f"{obs[i]:.0f}",
                        f"{allr[i]:.0f}", f"{flive_g[i]:.4f}", f"{all_dt[i]:.0f}",
                        f"{rr_raw[i]:.4f}", f"{rr_dt[i]:.4f}"])

# -------------------------------------------- ceilings and recon peak
mw = (x >= 170) & (x < 310)
say("\n== ceilings / peaks ==")
say(f"max observed NET rate 170..310 = {np.max(obs[mw]):.0f}; GROSS = {np.max(obs_gross[mw]):.0f}")
say(f"median observed GROSS rate over bins with GECAM>2000 (clipped regime) = "
    f"{np.median(obs_gross[mw & (g > 2000) & (obs_gross > 0)]):.0f}")
say(f"bins with observed gross >= 35000 in whole window: {np.sum(obs_gross >= 35000)}; "
    f">= 38000: {np.sum(obs_gross >= 38000)}")
i_rec = np.argmax(np.where(mw, allr, -np.inf))
say(f"max reconstructed (no DT) net = {allr[i_rec]:.3e} at T0+{x[i_rec]:.0f}")
for t0 in (227, 262):
    i = int(np.where(x == t0)[0][0])
    say(f"  T0+{t0}: obs_net {obs[i]:.3e}  rec_net {allr[i]:.3e}  flive {flive_g[i]:.3f}  "
        f"dt {all_dt[i]:.3e}  GECAMx200 {gs[i]:.3e}  dt/obs {all_dt[i]/obs[i]:.0f}")
i_g = np.argmax(np.where((x >= 172) & (x < 310), gs, -np.inf))
say(f"GECAM x200 max in 172..310 = {gs[i_g]:.3e} at T0+{x[i_g]:.0f}")

# --------------------------------------------- x200 scale window check
say("\n== x200 scale check (HXMT obs net / GECAM net) ==")
for lo, hi in ((175, 212), (180, 186), (193, 200), (180, 200)):
    m = (x >= lo) & (x < hi) & (g > 50)
    r = obs[m] / g[m]
    rr = allr[m] / g[m]
    say(f"  {lo}..{hi}: n={m.sum()} median obs/G = {np.median(r):.0f}, rec/G = {np.median(rr):.0f}; "
        f"obs gross max {np.max(obs_gross[m]):.0f}; flive min {np.nanmin(flive_g[m]):.3f}")
say("GECAM-C net (bkg-subtracted) vs raw, early window -50..175 (SAA check):")
for lo, hi in ((-50, -10), (-10, 10), (10, 100), (100, 172), (172, 180)):
    m = (x >= lo) & (x < hi)
    say(f"  {lo:4d}..{hi:4d}: GECAM net median {np.nanmedian(g_raw[m]):8.0f}  raw median "
        f"{np.nanmedian(g_nobkg[m]):8.0f}  HXMT rec net median {np.median(allr[m]):7.0f}")

# --------------------------------------------- decade medians + tau
say("\n== scale-flux decade medians (plot_scale_flux_dt logic, but with to_grid flive) ==")
sigs = (allr > 5e3) & (g > 30)
for name, num in (("raw", allr), ("dt", all_dt)):
    meds = []
    for a, b_ in zip([30, 100, 300, 1000, 3000, 10000], [100, 300, 1000, 3000, 10000, 30000]):
        mm = sigs & (g >= a) & (g < b_) & np.isfinite(num)
        meds.append((np.median(g[mm] / num[mm]) * 1e3, mm.sum()))
    say(f"  {name}: " + ", ".join(f"{m:.1f}(n={n})" for m, n in meds))
tau = 2.088e-4
say(f"tau = {tau:.3e} (GECAM cts/s)^-1 -> aggregate {tau/200*1e6:.3f} us/evt; "
    f"x18 det = {tau/200*18*1e6:.2f} us/evt/det")
Rc = np.sqrt(1e4 * 3e4)
model = (1 + tau * (Rc + B / 200)) / 200
say(f"model scale at brightest-decade centre R={Rc:.0f}: {model*1e3:.2f}e-3; "
    f"26.8/model = {26.8e-3/model:.2f}; 27.1/model = {27.1e-3/model:.2f}")
m_top = sigs & (g >= 1e4) & (g < 3e4)
say(f"brightest decade: median flive {np.nanmedian(flive_g[m_top]):.3f}, "
    f"(1/200)/flive = {0.005/np.nanmedian(flive_g[m_top])*1e3:.2f}e-3")
# Cnt_PHODet vs science event rate on the tail: is tau per processed event
# the same as per science event?
tail = (x >= 300) & (x < 700)
pho_rate = np.array([sum(v[2] for v in eng.get(int(t), {}).values()) for t in x])
say(f"tail 300..700: sum Cnt_PHODet (18 det) median {np.median(pho_rate[tail]):.0f}/s, "
    f"event-level gross (obs+fill) median {np.median(all_gross[tail]):.0f}/s, "
    f"ratio {np.median(pho_rate[tail] / all_gross[tail]):.2f}")
pre = x < -10
say(f"pre-burst: sum Cnt_PHODet median {np.median(pho_rate[pre]):.0f}/s vs event-level gross "
    f"{np.median(all_gross[pre]):.0f}/s; eng D/tau gross bkg {B_eng:.0f}")

# --------------------------------------------- flare definition stability
say("\n== flare interval sensitivity ==")
def flare(series, k=3.0, a1=(400, 425), a2=(615, 640)):
    pa = (x >= a1[0]) & (x < a1[1]); po = (x >= a2[0]) & (x < a2[1])
    r1, t1 = np.mean(series[pa]), np.mean(x[pa]); r2, t2 = np.mean(series[po]), np.mean(x[po])
    cont = r1 + (r2 - r1) * (x - t1) / (t2 - t1)
    sc = np.std(np.concatenate([series[pa] - r1, series[po] - r2]))
    s = (x >= 425) & (x < 610)
    ex = np.where(s, series - cont, np.nan)
    idx = np.where(s & (ex > k * sc))[0]
    fs, fe = x[idx[0]], x[idx[-1]] + 1
    sel = (x >= fs) & (x < fe)
    c = np.cumsum(np.clip(ex[sel], 0, None)); c /= c[-1]
    ts = x[sel]
    t05, t95 = ts[np.searchsorted(c, 0.05)], ts[np.searchsorted(c, 0.95)]
    gaps = len(idx) - (idx[-1] - idx[0] + 1)
    return fs, fe, t95 - t05, t05, t95, ts[np.nanargmax(ex[sel])], sc, gaps
for lbl, ser, kw in (("rec 3sig", allr, {}), ("dt 3sig", all_dt, {}),
                     ("rec 5sig", allr, {"k": 5}), ("rec anchors 390-420/620-650", allr,
                      {"a1": (390, 420), "a2": (620, 650)})):
    fs, fe, t90, a, b_, tp, sc, gp = flare(ser, **kw)
    say(f"  {lbl:28s} start {fs:.0f} end {fe:.0f} T90 {t90:.0f} ({a:.0f}-{b_:.0f}) peak {tp:.0f} "
        f"sig_cont {sc:.0f} sub-threshold bins inside [start,end) {-gp if gp<0 else gp}")

# --------------------------------------------- SAA by engineering counters
say("\n== SAA defined by zero engineering counters (sum D == 0 and sum PHO == 0 on all boxes) ==")
eng_off = np.array([len(eng.get(int(t), {})) == 3 and
                    all(v[1] == 0 and v[2] == 0 for v in eng[int(t)].values()) for t in x])
part = np.isin(x, [229, 248, 267, 276])
say(f"eng-off runs: {runs(eng_off)} (n={eng_off.sum()}); partial seconds 229/248/267/276")
say(f"observed gross in eng-off bins: {np.unique(obs_gross[eng_off])} evt/s "
    f"(channel-0, pulse-width-0 markers at integer seconds, one per box)")
say(f"all_dt in eng-off bins: {np.unique(np.round(all_dt[eng_off]))} (flive = 1 because D = 0)")
I_off0 = np.nansum(np.where(eng_off | part, 0.0, all_dt)[mwin])
say(f"main integral 180..300: as quoted {I_all:.4e}; eng-off bins set to 0: "
    f"{np.nansum(np.where(eng_off, 0.0, all_dt)[mwin]):.4e}; eng-off+partial set to 0: {I_off0:.4e}")
g_off = np.sum(gs[(eng_off | part) & mwin]); g_tot = np.sum(gs[mwin])
say(f"GECAM x200 counts in eng-off+partial bins = {g_off:.3e} of {g_tot:.3e} over 180..300 "
    f"({100*g_off/g_tot:.0f}%); HE integral + GECAM-in-gap = {I_off0 + g_off:.3e}")
i_gm = np.argmax(np.where(mwin, gs, -np.inf))
say(f"GECAM x200 peak in 180..300: {gs[i_gm]:.3e} at T0+{x[i_gm]:.0f} "
    f"(eng-off: {bool(eng_off[i_gm])})")

# ratio without the normalisation bins and without partial SAA seconds
say("\n== main-pulse ratio robustness ==")
win = (x >= 170) & (x < 310)
base = win & (g > 50.0) & np.isfinite(allr) & (allr > 0)
variants = {
    "as in paper (170..310)": base,
    "180..300 (stated window)": base & (x >= 180) & (x < 300),
    "drop partial SAA seconds": base & ~part,
    "drop normalisation bins <213": base & (x >= 213),
    "drop <213 and partial": base & (x >= 213) & ~part,
    "only flive<0.5 bins": base & (flive_g < 0.5),
}
for k, m in variants.items():
    a = robust(np.where(m, allr / np.where(gs > 0, gs, np.nan), np.nan))
    b_ = robust(np.where(m, all_dt / np.where(gs > 0, gs, np.nan), np.nan))
    say(f"  {k:32s} raw {a[0]:.2f}±{a[1]:.2f} (n={a[2]})  dt {b_[0]:.2f}±{b_[1]:.2f} (n={b_[2]})")

(OUT / "audit_numbers.txt").write_text("\n".join(lines) + "\n")
print(f"\nwrote {OUT/'audit_numbers.txt'}")

# --------------------------------------------- table fill-ins
say("\n== candidate table entries (precursor / flare peak) ==")
for lbl, lo, hi in (("precursor", -5, 20), ("flare", 442, 593)):
    m = (x >= lo) & (x < hi)
    i = np.nanargmax(np.where(m, all_dt, -np.inf))
    j = np.nanargmax(np.where(m, allr, -np.inf))
    say(f"  {lbl}: DT peak {all_dt[i]:.3e} at T0+{x[i]:.0f} (flive {flive_g[i]:.3f}); "
        f"rec peak {allr[j]:.3e} at T0+{x[j]:.0f}; obs net there {obs[j]:.3e}")
m = (x >= -5) & (x < 20)
say("  precursor bins: " + ", ".join(f"{x[k]:.0f}:{allr[k]:.0f}" for k in np.where(m)[0]))
say(f"  pre-burst rms of net 1-s rate (T0-50..-10): {np.std(allr[x < -10]):.0f}")
(OUT / "audit_numbers.txt").write_text("\n".join(lines) + "\n")
