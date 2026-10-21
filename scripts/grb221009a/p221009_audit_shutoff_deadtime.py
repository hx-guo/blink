#!/usr/bin/env python3
"""GRB 221009A: exact HE shut-off edges and box-level dead-time sharing.

1. Shut-off edges: last observed event before and first after each HE
   detector-off interval, per box (T0 = 13:17:00.000 UTC = MET 339945423,
   the value the light-curve pipeline uses; the 13:17:00.050 alternative
   is shown too).
2. Data completeness of the 225-230 s bins: per-box event counts, gap
   time, engineering live fraction per box.
3. Dead-time sharing: within each box, how close are the six detectors'
   DeadTime_PHODet values; per-box tau = D_box / sum_6 Cnt_PHODet on the
   tail (one shared processing chain per box) versus the per-detector
   D_i / Cnt_PHODet_i used in make_f10.

Run from the blink repo root:
    ./.venv/bin/python scripts/grb221009a/p221009_audit_shutoff_deadtime.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
import sys
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION, L_CYC_TO_SEC

T0 = 339945423.0
OUT = Path("data/p221009/audit")
lines = []
def say(s=""):
    print(s); lines.append(s)

# ------------------------------------------------------------- 1, 2
df = pd.read_csv("data/cache_221009a_reconstruct.csv",
                 usecols=["box", "type", "met", "channel", "pulse_width"])
df["t"] = df.met - T0
real = df[~((df.channel == 0) & (df.pulse_width == 0) & (df.met % 1 == 0))]
ev = real[real.type == "EVT"]
say("== shut-off edges (observed events, excluding the integer-second channel-0 markers) ==")
for lo, hi in ((225, 235), (262, 272)):
    for b in "ABC":
        e = ev[(ev.box == b) & (ev.t >= lo) & (ev.t < hi + 20)].t.values
        e.sort()
        dt = np.diff(e)
        k = np.argmax(dt)
        say(f"  box {b}: last event before off T0+{e[k]:.4f}, first after T0+{e[k+1]:.4f} "
            f"(off {dt[k]:.3f} s); with T0=13:17:00.050: {e[k]-0.05:.4f} / {e[k+1]-0.05:.4f}")
say("\n== bins 224..230: per-box observed / filled events and largest intra-bin hole ==")
for t in range(224, 231):
    parts = []
    for b in "ABC":
        s = real[(real.box == b) & (real.t >= t) & (real.t < t + 1)]
        o = s[s.type == "EVT"].t.values; o.sort()
        hole = np.max(np.diff(np.r_[t, o, t + 1])) if len(o) else 1.0
        parts.append(f"{b}: obs {len(o)} fill {int((s.type=='FILL_GAP').sum())} maxhole {hole:.3f}s")
    say(f"  T0+{t}: " + " | ".join(parts))

# ------------------------------------------------------------- 3
say("\n== dead-time sharing within boxes ==")
for box, code in BOX_CODE.items():
    fn = sorted(Path(f"data/1B/2022/20221009/{code}").glob(
        f"HXMT_1B_{code}_20221009T130000*.fits"))[0]
    with fits.open(fn, memmap=True) as fe:
        d = fe["HE_Eng"].data
        off = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        t = d["Time"].astype(float) + off + MET_CORRECTION - T0
        lc = d["Length_Time_Cycle"].astype(float)
        D = np.array([d[f"DeadTime_PHODet_{BOX_OFFSET[box]+k}"].astype(float) for k in range(6)])
        P = np.array([d[f"Cnt_PHODet_{BOX_OFFSET[box]+k}"].astype(float) for k in range(6)])
    for lbl, m in (("pre -50..-10", (t >= -50) & (t < -10)), ("tail 300..700", (t >= 300) & (t < 700)),
                   ("peak 225..228", (t >= 225) & (t < 229))):
        Dm = D[:, m]
        spread = np.nanmedian((Dm.max(0) - Dm.min(0)) / np.where(Dm.mean(0) > 0, Dm.mean(0), np.nan))
        say(f"  box {box} {lbl:14s}: D per det (median over s) "
            f"{np.round(np.median(Dm, 1)).astype(int).tolist()}  max-min spread/mean {spread:.3%}  "
            f"Cnt_PHODet per det {np.round(np.median(P[:, m], 1)).astype(int).tolist()}  "
            f"f_live box {np.median(1 - Dm.mean(0) / lc[m]):.3f}")
    tail = (t >= 300) & (t < 700) & (P.max(0) < 16383)
    tau_det = np.nanmedian(D[:, tail] * L_CYC_TO_SEC / P[:, tail], axis=1) * 1e6
    tau_box = np.nanmedian(D[:, tail].mean(0) * L_CYC_TO_SEC / P[:, tail].sum(0)) * 1e6
    say(f"  box {box}: per-det D_i/P_i tail = {np.round(tau_det, 1).tolist()} us; "
        f"box-level D_box/sum6(P) = {tau_box:.2f} us per triggered event")

(OUT / "audit_shutoff_deadtime.txt").write_text("\n".join(lines) + "\n")
