"""讲稿用标定脉冲的数据：取某一时刻前后 2 s 的 GRID-03B 逐事例时刻、探头号。

事例准入与 grid_gap_lightcurve.py 相同。MET 以 2018-01-01T00:00:00 UTC 为零点（与 grid_features.py 一致）。

用法: python3 grid_pulser_slide_data.py <卫星> <ISO 时刻> <输出 npz>
"""
import datetime as dt, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_gap_lightcurve import pass_files, load

sat, iso, out = sys.argv[1], sys.argv[2], sys.argv[3]
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
t0 = (dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc) - REF).total_seconds()
for path in pass_files(sat, iso[:10]):
    gs, ge, t, d = load(path)
    if gs <= t0 <= ge:
        w = (t > t0 - 2) & (t < t0 + 2)
        np.savez_compressed(out, t=t[w] - t0, det=d[w], t0=np.array([t0]))
        print(os.path.basename(path), int(w.sum()))
        break
else:
    print("no file covers", iso)
