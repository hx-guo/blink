"""脉冲开着的时段，某一路缺一发时，那一路在那个时刻到底记下了什么（全部事例，不做准入）。

用法: python3 grid_pulser_missing.py <卫星> <ISO 时刻> <半宽 s>
"""
import datetime as dt, os, sys
from collections import Counter
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_gap_lightcurve import pass_files

TICK = 1 / 4194304
sat, iso, half = sys.argv[1], sys.argv[2], float(sys.argv[3])
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
t0 = (dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc) - REF).total_seconds()
for path in pass_files(sat, iso[:10]):
    with fits.open(path, memmap=False) as h:
        g = h["GTI"].data
        if not (float(g["START"][0]) <= t0 <= float(g["STOP"][0])):
            continue
        emin = np.asarray(h["EBOUNDS"].data["E_MIN"], float)
        T, D, PI, TY = [], [], [], []
        for k in range(4):
            e = h[f"EVENTS{k}"].data
            t = np.asarray(e["TIME"], float); m = np.abs(t - t0) <= half
            T.append(t[m]); D.append(np.full(int(m.sum()), k)); PI.append(np.asarray(e["PI"], int)[m]); TY.append(np.asarray(e["EVT_TYPE"], int)[m])
        t, d, pi, ty = (np.concatenate(x) for x in (T, D, PI, TY))
        tk = np.round(t / TICK).astype(np.int64)
        print(os.path.basename(path), "events", t.size, "EVT_TYPE counts", Counter(ty.tolist()))
        u, cnt = np.unique(tk, return_counts=True)
        cl = u[cnt >= 3]
        full = 0; kinds = Counter(); exam = []
        for c in cl:
            s = tk == c
            if set(d[s]) == {0, 1, 2, 3}:
                full += 1
                continue
            for m_ in {0, 1, 2, 3} - set(d[s]):
                near = (d == m_) & (np.abs(tk - c) <= 200)
                if not near.any():
                    kinds["nothing within 48us"] += 1
                else:
                    j = np.where(near)[0][np.argmin(np.abs(tk[near] - c))]
                    kinds["offset %+d ticks" % (tk[j] - c) if abs(tk[j] - c) <= 8 else "other event %.1f us away" % ((tk[j] - c) * TICK * 1e6)] += 1
                    if len(exam) < 12: exam.append((m_, int(tk[j] - c), int(pi[j]), int(ty[j])))
        print("clusters", cl.size, "all four", full)
        for k_, v in kinds.most_common(12): print(" ", k_, v)
        print("examples (det, dtick, PI, EVT_TYPE):", exam)
        # energy/type of pulse events themselves
        s = np.isin(tk, cl)
        print("pulse events EVT_TYPE", Counter(ty[s].tolist()), "PI range", np.percentile(pi[s], [1, 50, 99]), "E_MIN of PI", emin[np.clip(np.percentile(pi[s], [1, 50, 99]).astype(int) - 1, 0, emin.size - 1)])
        break
