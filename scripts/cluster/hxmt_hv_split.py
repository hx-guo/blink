#!/usr/bin/env python3
"""机箱级不均匀分两类：高压瞬变 vs 高压正常的整箱掉数。"""
import os, csv, collections, math
import numpy as np

D = os.environ.get('HXMT_AUDIT_DIR', '.').rstrip('/') + '/'
rows = []
for r in csv.DictReader(open(D + 'det_hv.csv')):
    if not r['hv_drop'] or not r['det_window']:
        continue
    w = np.array([int(x) for x in r['det_window'].split('|')], float)
    b = np.array([int(x) for x in r['det_baseline'].split('|')], float)
    if w.sum() == 0 or b.sum() == 0:
        continue
    W = np.array([w[0:6].sum(), w[6:12].sum(), w[12:18].sum()])
    B = np.array([b[0:6].sum(), b[6:12].sum(), b[12:18].sum()])
    sb = B / B.sum(); exp = W.sum() * sb
    chi2 = ((W - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
    flag = 'degraded' if sb.max() > 0.9 else ('switch' if chi2 > 30 else 'normal')
    rows.append(dict(start=r['start'], day=int(r['date']), assoc=r['assoc'] == '1',
                     flag=flag, W=W, B=B, n=W.sum(), chi2=chi2,
                     drop=float(r['hv_drop']), slope=float(r['hv_slope'])))

cat = {c['start'] for c in csv.DictReader(open(D + 'catalog_v6.csv'))}


def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k / n; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - h), 100 * (c + h))


print("机箱级不均匀的候选 (degraded + switch)，按高压状态拆：")
bad = [r for r in rows if r['flag'] != 'normal']
cls = collections.Counter()
for r in bad:
    if r['drop'] < 0.5:
        cls['高压已关（|HV| < 工作点一半）'] += 1
    elif r['slope'] > 20:
        cls['高压正在升/降（±3 s 跳变 > 20 V）'] += 1
    else:
        cls['高压全程正常 —— 整箱掉数'] += 1
for k, v in cls.most_common():
    print("   %-34s %4d  (%.1f%%)" % (k, v, 100 * v / len(bad)))
print("   合计 %d" % len(bad))

print("\n'高压全程正常的整箱掉数'那一类按天:")
g = [r for r in bad if r['drop'] >= 0.5 and r['slope'] <= 20]
for d, k in collections.Counter(r['day'] for r in g).most_common(10):
    print("   %s  %d" % (d, k))
print("   进目录的 %d 个" % sum(r['start'] in cat for r in g))

base = [r for r in rows if r['flag'] == 'normal']
kb = sum(r['slope'] > 20 for r in base)
lo, hi = wilson(kb, len(base))
print("\n偶然参照（箱分布正常的候选坐在高压瞬变上的比例）: %d/%d = %.2f%% [%.2f–%.2f%%]"
      % (kb, len(base), 100 * kb / len(base), lo, hi))
for nm in ('degraded', 'switch'):
    g2 = [r for r in rows if r['flag'] == nm]
    k2 = sum(r['slope'] > 20 or r['drop'] < 0.5 for r in g2)
    l2, h2 = wilson(k2, len(g2))
    print("  %-9s 高压异常（瞬变或已关）%3d/%3d = %.1f%% [%.1f–%.1f%%]  相对偶然 %.0f 倍"
          % (nm, k2, len(g2), 100 * k2 / len(g2), l2, h2, (k2 / len(g2)) / (kb / len(base))))
