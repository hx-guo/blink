#!/usr/bin/env python3
"""机箱级判据的两条腿合起来，看目录影响；顺带查小时边界。

腿 1（仪器在降级状态运行）：基线单箱占比 > 0.9
腿 2（机箱在候选处换了状态）：窗内箱分布相对"按自身基线分摊"的多项 chi2 > 30
   —— 同 n 的多项模拟在 7089 个平衡基线候选里给 0 个，所以 30 是物理边界不是统计阈
"""
import os, csv, collections
import numpy as np

D = os.environ.get('HXMT_AUDIT_DIR', '.').rstrip('/') + '/'
det = {}
for r in csv.DictReader(open(D + 'det_all.csv')):
    w = np.array([int(x) for x in r['det_window'].split('|')], float)
    b = np.array([int(x) for x in r['det_baseline'].split('|')], float)
    if w.sum() == 0 or b.sum() == 0:
        continue
    W = np.array([w[0:6].sum(), w[6:12].sum(), w[12:18].sum()])
    B = np.array([b[0:6].sum(), b[6:12].sum(), b[12:18].sum()])
    sb = B / B.sum()
    exp = W.sum() * sb
    det[r['start']] = dict(W=W, B=B, sw=W / W.sum(), sb=sb, n=W.sum(),
                           chi2=((W - exp) ** 2 / np.maximum(exp, 1e-9)).sum(),
                           assoc=r['assoc'] == '1')

leg1 = lambda d: d['sb'].max() > 0.9
leg2 = lambda d: d['chi2'] > 30.0
both = lambda d: leg1(d) or leg2(d)

allr = list(det.values())
tgf = [d for d in allr if d['assoc']]
cat = list(csv.DictReader(open(D + 'catalog_v6.csv')))

print("%-34s %-22s %-22s %s" % ("判据", "审计池 7544", "闪电关联 2547", "v6 目录 5421"))
for name, f in (("腿1 基线单箱 > 0.9（降级运行）", leg1),
                ("腿2 箱分布多项 chi2 > 30（换状态）", leg2),
                ("两腿并集", both)):
    a = sum(f(d) for d in allr)
    t = sum(f(d) for d in tgf)
    c = [x for x in cat if f(det[x['start']])]
    print("%-34s %5d (%.2f%%)        %4d (%.2f%%)          %4d (%.2f%%)"
          % (name, a, 100 * a / len(allr), t, 100 * t / len(tgf), len(c), 100 * len(c) / len(cat)))

hit = [x for x in cat if both(det[x['start']])]
print("\n并集命中的 %d 行目录: tier %s, 关联 %d 个"
      % (len(hit), dict(collections.Counter(x['tier'] for x in hit)),
         sum(x['associated'] == '1' for x in hit)))
print("  按天:", collections.Counter(x['date'] for x in hit).most_common(10))

print("\n=== 小时边界 ===")
def sec_in_hour(s):
    return int(s[14:16]) * 60 + float(s[17:26])
edge = [x for x in cat if min(sec_in_hour(x['start']), 3600 - sec_in_hour(x['start'])) < 1.0]
print("  目录里落在整点 ±1 s 的 %d / %d（均匀期望 %.1f）"
      % (len(edge), len(cat), 2 * len(cat) / 3600))
for x in edge[:10]:
    d = det[x['start']]
    print("    %s fa=%s W=%s chi2=%.0f" % (x['start'][:23], x['false_positive_per_year'],
                                           '/'.join('%d' % y for y in d['W']), d['chi2']))
