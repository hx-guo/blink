#!/usr/bin/env python3
"""一阶谐波是真的还是最小二乘误差估得不对：置换检验 + 看相位。

置换 = 把 phi 随机打乱，其余不动，重算振幅。谐波若是真的方位效应，实测振幅
应当落在置换分布之外。相位若三箱相差约 120 deg，才像"三个箱绕指向轴摆开"。
"""
import csv
import numpy as np

D = os.environ.get('HXMT_AUDIT_DIR', '.').rstrip('/') + '/'
rows = []
for r in csv.DictReader(open(D + 'det_nadir.csv')):
    if not r['nadir_theta'] or not r['det_window'] or r['assoc'] != '1':
        continue
    w = np.array([int(x) for x in r['det_window'].split('|')], float)
    b = np.array([int(x) for x in r['det_baseline'].split('|')], float)
    if w.sum() == 0 or b.sum() == 0:
        continue
    W = np.array([w[0:6].sum(), w[6:12].sum(), w[12:18].sum()])
    B = np.array([b[0:6].sum(), b[6:12].sum(), b[12:18].sum()])
    if B.max() / B.sum() > 0.9:
        continue
    rows.append(dict(day=int(r['date']), phi=float(r['nadir_phi']), theta=float(r['nadir_theta']),
                     n=W.sum(), d=W / W.sum() - B / B.sum()))
print("闪电关联 TGF（三箱齐）N=%d" % len(rows))

phi = np.radians(np.array([r['phi'] for r in rows]))
n = np.array([r['n'] for r in rows], float)
day = np.array([r['day'] for r in rows])
Y = np.stack([r['d'] for r in rows])


def harmonic(p, y, wt):
    X = np.stack([np.ones_like(p), np.cos(p), np.sin(p)], 1)
    A = (X * wt[:, None]).T @ X
    c = np.linalg.solve(A, (X * wt[:, None]).T @ y)
    return np.hypot(c[1], c[2]), np.degrees(np.arctan2(c[2], c[1])) % 360


rng = np.random.default_rng(20260911)
print("\n%-6s %10s %10s   %s" % ("箱", "振幅", "相位", "置换分布(1000 次)：中位 / p95 / p99 / 超过实测的次数"))
for c, nm in enumerate('ABC'):
    y = Y[:, c]
    a, ph = harmonic(phi, y, n)
    sims = np.array([harmonic(rng.permutation(phi), y, n)[0] for _ in range(1000)])
    print("  %-4s %10.5f %9.1f°   %.5f / %.5f / %.5f / %d"
          % (nm, a, ph, np.median(sims), np.percentile(sims, 95), np.percentile(sims, 99),
             (sims >= a).sum()))

print("\n对照：把 phi 换成 theta 的一阶谐波（天底张角，不是绕轴方位）")
th = np.radians(np.array([r['theta'] for r in rows]))
for c, nm in enumerate('ABC'):
    a, ph = harmonic(th, Y[:, c], n)
    sims = np.array([harmonic(rng.permutation(th), Y[:, c], n)[0] for _ in range(500)])
    print("  Box %s 振幅 %.5f  相位 %.1f°  置换 p95 %.5f  超过实测 %d/500"
          % (nm, a, ph, np.percentile(sims, 95), (sims >= a).sum()))

print("\n对照：按天整体置换 phi（同一天的候选共用一个 phi 偏移），排除日期共线")
uniq = np.unique(day)
sims_by_day = {c: [] for c in range(3)}
for _ in range(500):
    shift = dict(zip(uniq, rng.uniform(0, 2 * np.pi, len(uniq))))
    p2 = phi + np.array([shift[d] for d in day])
    for c in range(3):
        sims_by_day[c].append(harmonic(p2, Y[:, c], n)[0])
for c, nm in enumerate('ABC'):
    a, _ = harmonic(phi, Y[:, c], n)
    s = np.array(sims_by_day[c])
    print("  Box %s 实测 %.5f  按天置换 p95 %.5f  超过实测 %d/500" % (nm, a, np.percentile(s, 95), (s >= a).sum()))
