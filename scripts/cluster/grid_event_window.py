"""导出某一时刻前后若干秒的逐事例数据（时刻、PI、探头号），附 EBOUNDS，供讲稿画单个 TGF 用。

事例准入与 grid_gap_lightcurve.py 相同（EVT_TYPE==1、PI∈[1,n_ch)、E_MIN≥30 keV）。
时刻为相对给定 ISO 时刻的秒数，MET 零点 2018-01-01T00:00:00 UTC。

用法: python3 grid_event_window.py <卫星> <ISO 时刻> <半宽 s> <输出 npz>
"""
import datetime as dt, os, sys
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_gap_lightcurve import pass_files, ETH

sat, iso, half, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
head, _, frac = iso.rstrip("Z").partition(".")
t0 = (dt.datetime.fromisoformat(head).replace(tzinfo=dt.timezone.utc) - REF).total_seconds() + (float("0." + frac) if frac else 0.0)
for path in pass_files(sat, iso[:10]):
    with fits.open(path, memmap=False) as h:
        g = h["GTI"].data
        if not (float(g["START"][0]) <= t0 <= float(g["STOP"][0])):
            continue
        eb = h["EBOUNDS"].data
        emin = np.asarray(eb["E_MIN"], float); emax = np.asarray(eb["E_MAX"], float)
        T, P, D = [], [], []
        for k in range(4):
            e = h[f"EVENTS{k}"].data
            t = np.asarray(e["TIME"], float); pi = np.asarray(e["PI"], int); ty = np.asarray(e["EVT_TYPE"], int)
            ok = (pi >= 1) & (pi < emin.size)
            en = np.zeros(t.size); en[ok] = emin[pi[ok] - 1]
            keep = (ty == 1) & ok & (en >= ETH) & (np.abs(t - t0) <= half)
            T.append(t[keep] - t0); P.append(pi[keep]); D.append(np.full(int(keep.sum()), k))
        t = np.concatenate(T); o = np.argsort(t, kind="stable")
        np.savez_compressed(out, t=t[o], pi=np.concatenate(P)[o], det=np.concatenate(D)[o], emin=emin, emax=emax, t0=np.array([t0]))
        print(os.path.basename(path), o.size)
        break
