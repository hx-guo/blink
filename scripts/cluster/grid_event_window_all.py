"""导出某一时刻前后若干秒 GRID 某星的全部事例（不做能量准入、不筛 EVT_TYPE），给单路丢包的例子图用。
打包按全部事例计数，所以数“两次沉默之间的事例数”必须用这个口径（与 grid_channel_silences.py 一致）。

用法: python3 grid_event_window_all.py <卫星> <ISO 时刻> <半宽 s> <输出 npz>
"""
import datetime as dt, os, sys
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_gap_lightcurve import pass_files

sat, iso, half, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
t0 = (dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc) - REF).total_seconds()
for path in pass_files(sat, iso[:10]):
    with fits.open(path, memmap=False) as h:
        g = h["GTI"].data
        if not (float(g["START"][0]) <= t0 <= float(g["STOP"][0])):
            continue
        T, D = [], []
        for k in range(4):
            t = np.asarray(h[f"EVENTS{k}"].data["TIME"], float)
            m = np.abs(t - t0) <= half
            T.append(t[m] - t0); D.append(np.full(int(m.sum()), k))
        t = np.concatenate(T); o = np.argsort(t, kind="stable")
        np.savez_compressed(out, t=t[o], det=np.concatenate(D)[o])
        print(os.path.basename(path), o.size); break
