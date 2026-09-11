#!/usr/bin/env python3
"""方位谐波的复现性 + REP 上的同一量。

分半复现：把闪电关联 TGF 按年份分两半，各自拟一阶谐波，比相位。
REP：同一套量，零假设用"按天整体置换 phi"（保留日期结构）。
"""
import os, csv, collections
import numpy as np

D = os.environ.get('HXMT_AUDIT_DIR', '.').rstrip('/') + '/'
STORM = {20230424, 20240428, 20240813, 20240918, 20241009, 20241010, 20241023, 20241024}
CHECK = 20250930

rows = []
for r in csv.DictReader(open(D + 'det_nadir.csv')):
    if not r['nadir_theta'] or not r['det_window']:
        continue
    w = np.array([int(x) for x in r['det_window'].split('|')], float)
    b = np.array([int(x) for x in r['det_baseline'].split('|')], float)
    if w.sum() == 0 or b.sum() == 0:
        continue
    W = np.array([w[0:6].sum(), w[6:12].sum(), w[12:18].sum()])
    B = np.array([b[0:6].sum(), b[6:12].sum(), b[12:18].sum()])
    if B.max() / B.sum() > 0.9:
        continue
    fa = float(r['false_positive_per_year']); assoc = r['assoc'] == '1'
    rows.append(dict(day=int(r['date']), assoc=assoc, sel=(fa < 1e-5) or (fa < 1 and assoc),
                     phi=np.radians(float(r['nadir_phi'])), n=W.sum(),
                     d=W / W.sum() - B / B.sum()))


def harmonic(p, y, wt):
    X = np.stack([np.ones_like(p), np.cos(p), np.sin(p)], 1)
    A = (X * wt[:, None]).T @ X
    c = np.linalg.solve(A, (X * wt[:, None]).T @ y)
    return np.hypot(c[1], c[2]), np.degrees(np.arctan2(c[2], c[1])) % 360


def fit(g, label, nsim=500, seed=1):
    if len(g) < 60:
        print("  %-28s N=%d 太少，跳过" % (label, len(g)))
        return
    phi = np.array([r['phi'] for r in g]); n = np.array([r['n'] for r in g], float)
    day = np.array([r['day'] for r in g]); Y = np.stack([r['d'] for r in g])
    rng = np.random.default_rng(seed)
    uniq = np.unique(day)
    out = []
    sims = {c: [] for c in range(3)}
    for _ in range(nsim):
        shift = dict(zip(uniq, rng.uniform(0, 2 * np.pi, len(uniq))))
        p2 = phi + np.array([shift[d] for d in day])
        for c in range(3):
            sims[c].append(harmonic(p2, Y[:, c], n)[0])
    for c, nm in enumerate('ABC'):
        a, ph = harmonic(phi, Y[:, c], n)
        s = np.array(sims[c])
        out.append("%s: %.4f @ %5.1f° (按天置换 p95 %.4f, 超 %d/%d)"
                   % (nm, a, ph, np.percentile(s, 95), (s >= a).sum(), nsim))
    print("  %-28s N=%4d  %s" % (label, len(g), out[0]))
    for line in out[1:]:
        print("  %-28s %5s  %s" % ("", "", line))


tgf = [r for r in rows if r['assoc']]
print("=== 分半复现（闪电关联 TGF）===")
early = [r for r in tgf if r['day'] < 20210101]
late = [r for r in tgf if r['day'] >= 20210101]
fit(tgf, "全部 TGF")
fit(early, "TGF 2017–2020")
fit(late, "TGF 2021–2024")

print("\n=== 同一量在 REP 上 ===")
rep = [r for r in rows if r['sel'] and r['day'] in STORM and not r['assoc']]
chk = [r for r in rows if r['sel'] and r['day'] == CHECK and not r['assoc']]
fit(rep, "certified REP (风暴日)")
fit(chk, "2025-09-30 (单一时段)")
other = [r for r in rows if not r['assoc'] and r['day'] not in STORM and r['day'] != CHECK and r['sel']]
fit(other, "其余判选未关联候选")
