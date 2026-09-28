"""grid_pulser_missing.py 的细分：缺的那一路最近的事例离脉冲时刻多远、PI 多少；脉冲事例本身的 PI 分布。

用法: python3 grid_pulser_missing2.py <卫星> <ISO 时刻> <半宽 s>
"""
import datetime as dt, os, sys
from collections import Counter
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_gap_lightcurve import pass_files

TICK = 1 / 4194304
sat, iso, half = sys.argv[1], sys.argv[2], float(sys.argv[3])
t0 = (dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc) - dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)).total_seconds()
for path in pass_files(sat, iso[:10]):
    with fits.open(path, memmap=False) as h:
        g = h["GTI"].data
        if not (float(g["START"][0]) <= t0 <= float(g["STOP"][0])):
            continue
        eb = h["EBOUNDS"].data
        print("EBOUNDS cols", eb.columns.names, "n", len(eb), "row0", eb[0], "row26", eb[26])
        T, D, PI = [], [], []
        for k in range(4):
            e = h[f"EVENTS{k}"].data
            if k == 0:
                print("EVENTS cols", e.columns.names)
            t = np.asarray(e["TIME"], float); m = np.abs(t - t0) <= half
            T.append(t[m]); D.append(np.full(int(m.sum()), k)); PI.append(np.asarray(e["PI"], int)[m])
        t, d, pi = (np.concatenate(x) for x in (T, D, PI))
        tk = np.round(t / TICK).astype(np.int64)
        u, cnt = np.unique(tk, return_counts=True); cl = u[cnt >= 3]
        cat = Counter(); dts = []; pis = []
        for c in cl:
            s = tk == c
            for m_ in {0, 1, 2, 3} - set(d[s].tolist()):
                near = (d == m_) & (np.abs(tk - c) <= 200)
                if not near.any():
                    cat["none within 48us"] += 1; continue
                j = np.where(near)[0][np.argmin(np.abs(tk[near] - c))]; dt_ = tk[j] - c
                dts.append(dt_); pis.append(pi[j])
                cat["|dt|<=8 ticks" if abs(dt_) <= 8 else ("9..25 ticks early" if -25 <= dt_ < -8 else "other")] += 1
        print("clusters", cl.size, "missing slots", sum(cat.values()), dict(cat))
        dts, pis = np.array(dts), np.array(pis)
        for lo, hi in ((-8, 8), (-25, -9)):
            s = (dts >= lo) & (dts <= hi)
            print("dt in [%d,%d]: n=%d, PI median %s, PI 10/90 %s" % (lo, hi, s.sum(), np.median(pis[s]), np.percentile(pis[s], [10, 90])))
        print("dt histogram (ticks) -30..8:", sorted(Counter(dts[(dts >= -30) & (dts <= 8)].tolist()).items()))
        s = np.isin(tk, cl)
        print("pulse PI hist", sorted(Counter(pi[s].tolist()).items()))
        print("background PI median", np.median(pi[~s]))
        break

# 周围什么都没有的那些：看这一路在脉冲时刻所处的空档有多长，以及空档两侧之间少了多少事例
if len(sys.argv) > 4:
    rate = {k: (d == k).sum() / (2 * half) for k in range(4)}
    L = []
    for c in cl:
        s = tk == c
        for m_ in {0, 1, 2, 3} - set(d[s].tolist()):
            if ((d == m_) & (np.abs(tk - c) <= 200)).any():
                continue
            tm = np.sort(tk[d == m_]); i = np.searchsorted(tm, c)
            if 0 < i < tm.size:
                L.append((tm[i] - tm[i - 1]) * TICK * rate[m_])
    L = np.array(L)
    print("rate per det", {k: round(v) for k, v in rate.items()})
    print("gap containing the missing pulse, in units of mean event spacing: pct 10/50/90", np.percentile(L, [10, 50, 90]).round(1),
          "fraction > 10:", (L > 10).mean().round(3))

# 讲稿图那 7 毫秒（t0 之后第一个脉冲前 0.3 ms 起）里缺的那几发，各自是哪种情况
if len(sys.argv) > 5:
    c0 = cl[np.searchsorted(cl, round(t0 / TICK))]
    for c in cl[(cl >= c0) & (cl < c0 + 7e-3 / TICK)]:
        s = tk == c
        for m_ in {0, 1, 2, 3} - set(d[s].tolist()):
            near = (d == m_) & (np.abs(tk - c) <= 400)
            print("pulse at %.3f ms, det %d missing; det %d events within ±95us (dtick, PI):" % ((c - c0) * TICK * 1e3 + 0.3, m_, m_),
                  list(zip((tk[near] - c).tolist(), pi[near].tolist())))
            tm = np.sort(tk[d == m_]); i = np.searchsorted(tm, c)
            print("   gap around it = %.1f mean spacings" % ((tm[i] - tm[i - 1]) * TICK * (d == m_).sum() / (2 * half)))
