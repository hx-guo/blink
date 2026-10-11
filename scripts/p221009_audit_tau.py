#!/usr/bin/env python3
"""Box-level dead-time constant of HE during GRB 221009A, on one footing.

Non-paralyzable, box-shared chain:  1 - f_b = tau * n_proc_b,
n_proc_b = f_b * n_true_b  =>  tau = (1/f_b - 1) / n_true_b = (1 - f_b) / n_proc_b.
The constant depends on which event class n counts, so it is estimated
per class:
  * per trigger   : n = Cnt_PHODet summed over the 6 detectors of a box
                    (discriminator counter, dead-time free per the companion
                    paper -> n_true; 14-bit, so only unwrapped frames used)
  * per record    : n_proc = recorded 1B events of the box (observed + filler),
                    i.e. the events that went through the chain into FIFO A
  * per record, GECAM : n_true = 200 (R + R_B) / 3, the GECAM-scaled HE
                    record rate the scale-flux fit uses
and the trigger/record ratio k of the BURST excess (not the background) is
measured on the frames where Cnt_PHODet is unwrapped.

Run from the blink repo root:
    ./.venv/bin/python scripts/p221009_audit_tau.py
"""
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

sys.path.insert(0, "scripts")
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION, L_CYC_TO_SEC
from make_f10 import load_hxmt, GECAM_SCALE
from plot_hxmt_vs_gecam import compute_time_offset, load_gecam_btime
import pandas as pd

T0 = 339945423.0
OUT = Path("data/p221009/audit")
lines = []
def say(s=""):
    print(s); lines.append(s)

x, edges, obs, allr, n_fill, B = load_hxmt(50, 700, 1.0)
g_met, _, _ = compute_time_offset()
_, g, _ = load_gecam_btime(g_met, 50, 700, 1.0, "lg", True)
g = np.nan_to_num(g, nan=0.0)

# per-box recorded events per 1-s bin (obs + filler), without the channel-0 markers
df = pd.read_csv("data/cache_221009a_reconstruct.csv", usecols=["box", "met", "channel", "pulse_width"])
df = df[~((df.channel == 0) & (df.pulse_width == 0) & (df.met % 1 == 0))]
rec = {b: np.histogram(df[df.box == b].met.values - T0, bins=edges)[0].astype(float) for b in "ABC"}

eng = {}
for box, code in BOX_CODE.items():
    fn = sorted(Path(f"data/1B/2022/20221009/{code}").glob(f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
    with fits.open(fn, memmap=True) as fe:
        d = fe["HE_Eng"].data
        off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        t = d["Time"].astype(float) + off + MET_CORRECTION - T0
        lc = d["Length_Time_Cycle"].astype(float)
        D = np.mean([d[f"DeadTime_PHODet_{BOX_OFFSET[box]+k}"].astype(float) for k in range(6)], 0)
        P = np.array([d[f"Cnt_PHODet_{BOX_OFFSET[box]+k}"].astype(float) for k in range(6)])
    m = {int(round(tt)): (dd, ll, pp) for tt, dd, ll, pp in zip(t, D, lc, P.T)}
    f = np.full(len(x), np.nan); Dsec = np.full(len(x), np.nan)
    Psum = np.full(len(x), np.nan); Pmax = np.full(len(x), np.nan)
    for i, xx in enumerate(x):
        v = m.get(int(round(xx)))
        if v is None or v[1] <= 0:
            continue
        f[i] = 1 - v[0] / v[1]; Dsec[i] = v[0] * L_CYC_TO_SEC
        Psum[i] = v[2].sum(); Pmax[i] = v[2].max()
    eng[box] = (f, Dsec, Psum, Pmax)

part = np.isin(x, [229, 248, 267, 276])
off = (x >= 230) & (x < 248) | (x >= 268) & (x < 276)
good = ~part & ~off

say("== per-class busy time per event (box level, median over boxes) ==")
say(f"L_CYC_TO_SEC = {L_CYC_TO_SEC*1e6:.0f} us; Lcyc ~ {np.nanmedian([np.nanmedian(eng[b][1]) for b in 'ABC']):.3f} s busy at median")
phases = {"pre -50..-10": (x >= -50) & (x < -10), "tail 300..440": (x >= 300) & (x < 440),
          "tail 600..700": (x >= 600) & (x < 700), "flare 442..593": (x >= 442) & (x < 593),
          "main f<0.5": (x >= 170) & (x < 310)}
for name, sel in phases.items():
    out = []
    for b in "ABC":
        f, Dsec, Psum, Pmax = eng[b]
        s = sel & good & np.isfinite(f)
        if name == "main f<0.5":
            s &= f < 0.5
        unw = s & (Pmax < 15000) & (1.45 * rec[b] / np.where(f > 0, f, np.nan) / 6 < 10000)
        t_trig = np.nanmedian(Dsec[unw] / Psum[unw]) * 1e6 if unw.any() else np.nan
        t_trig_proc = np.nanmedian(Dsec[unw] / (Psum[unw] * f[unw])) * 1e6 if unw.any() else np.nan
        t_rec = np.nanmedian(Dsec[s] / rec[b][s]) * 1e6
        out.append((t_trig, t_trig_proc, t_rec, s.sum(), unw.sum()))
    o = np.array(out)
    say(f"  {name:15s} D/P {np.round(o[:,0],2).tolist()}  D/(P f) {np.round(o[:,1],2).tolist()}  "
        f"D/rec {np.round(o[:,2],2).tolist()} us   (n bins {int(o[0,3])}, unwrapped {int(o[0,4])})")

# trigger/record ratio for background and for the burst excess
say("\n== trigger/record ratio k = P_true / rec_true (rec_true = rec / f) ==")
for b in "ABC":
    f, Dsec, Psum, Pmax = eng[b]
    s = good & np.isfinite(f) & (Pmax < 15000)
    rt = rec[b] / f
    pre = s & (x < -10)
    kb = np.median(Psum[pre] / rt[pre])
    # Cnt_PHODet is 14-bit: keep frames whose expected per-detector count
    # (k_bkg * rec_true / 6) stays well below 16384, so no wrap is possible
    burst = (s & (x >= 170) & (x < 700) & (rt - np.median(rt[pre]) > 2000)
             & (kb * rt / 6 < 10000))
    dP = Psum[burst] - np.median(Psum[pre]); dR = rt[burst] - np.median(rt[pre])
    slope = np.sum(dP * dR) / np.sum(dR * dR)
    say(f"  box {b}: background k = {kb:.2f};  burst-excess k (through-origin fit of dP vs dRec_true, "
        f"{burst.sum()} unwrapped bins, dRec_true up to {dR.max():.0f}/s) = {slope:.2f}; "
        f"median dP/dRec {np.median(dP/dR):.2f}")

# GECAM-scaled per-bin estimate and a per-bin refit of the scale-flux model
say("\n== GECAM-scaled record rate (the scale-flux fit's footing) ==")
fbar = np.nanmean([eng[b][0] for b in "ABC"], 0)
sig = good & (allr > 5e3) & (g > 30) & np.isfinite(fbar)
R_B = B / GECAM_SCALE
tauG_bin = (1 / fbar[sig] - 1) * 3 / (GECAM_SCALE * (g[sig] + R_B))
bright = sig & (fbar < 0.5)
tauG_bright = (1 / fbar[bright] - 1) * 3 / (GECAM_SCALE * (g[bright] + R_B))
(tau_fit,), cov = curve_fit(lambda R, t: 1 / (1 + t * (R + R_B)), g[sig], fbar[sig], p0=[2e-4])
say(f"  decade-median fit (plot_scale_flux_dt): tau = 2.088e-4 -> box {3*2.088e-4/200*1e6:.2f} us/record")
say(f"  per-bin least-squares fit ({sig.sum()} bins): tau = {tau_fit:.3e} -> box "
    f"{3*tau_fit/200*1e6:.2f} us/record (+-{3*np.sqrt(cov[0,0])/200*1e6:.2f})")
say(f"  per-bin (1/f-1)/n_true: median all sig {np.median(tauG_bin)*1e6:.2f} us, "
    f"f<0.5 bins {np.median(tauG_bright)*1e6:.2f} us (n={bright.sum()})")
recsum = rec["A"] + rec["B"] + rec["C"]
tauR_bright = 3 * (1 - fbar[bright]) / recsum[bright]
say(f"  same f<0.5 bins with HE recorded rate instead of GECAM: (1-f)/n_rec = "
    f"{np.median(tauR_bright)*1e6:.2f} us/record")
say(f"  ratio HE_DT/GECAMx200 on those bins: "
    f"{np.median((recsum[bright]/fbar[bright]-B)/(GECAM_SCALE*g[bright])):.2f}")
(OUT / "audit_tau.txt").write_text("\n".join(lines) + "\n")
