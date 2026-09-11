#!/usr/bin/env python3
"""最干净的一问：仪器三箱都正常的时候，有没有过"只有一个箱亮"的候选？

把"基线三箱平衡"定义死（三个占比都落在 0.28–0.39），再看窗内的箱分布偏离
按基线分摊的期望有多远。偶然期望并排给：多项分布下同样 n 的模拟。
"""
import csv
import numpy as np

D = os.environ.get('HXMT_AUDIT_DIR', '.').rstrip('/') + '/'
rows = []
for r in csv.DictReader(open(D + 'det_all.csv')):
    w = np.array([int(x) for x in r['det_window'].split('|')], float)
    b = np.array([int(x) for x in r['det_baseline'].split('|')], float)
    if w.sum() == 0 or b.sum() == 0:
        continue
    W = np.array([w[0:6].sum(), w[6:12].sum(), w[12:18].sum()])
    B = np.array([b[0:6].sum(), b[6:12].sum(), b[12:18].sum()])
    rows.append(dict(start=r['start'], day=int(r['date']), assoc=r['assoc'] == '1',
                     n=W.sum(), W=W, B=B, sw=W / W.sum(), sb=B / B.sum()))

bal = [r for r in rows if r['sb'].min() > 0.28 and r['sb'].max() < 0.39]
print("全部 %d 行，其中基线三箱平衡（占比全在 0.28–0.39）的 %d 行 (%.1f%%)"
      % (len(rows), len(bal), 100 * len(bal) / len(rows)))
print("  这批里闪电关联 %d，窗内计数中位 %.0f" % (sum(r['assoc'] for r in bal), np.median([r['n'] for r in bal])))

# 多项 chi2（dof=2），并排给同样 n 下的随机期望
rng = np.random.default_rng(20260911)
chi_obs, chi_sim = [], []
for r in bal:
    exp = r['n'] * r['sb']
    chi_obs.append(((r['W'] - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    sim = rng.multinomial(int(r['n']), r['sb'])
    chi_sim.append(((sim - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
chi_obs = np.array(chi_obs); chi_sim = np.array(chi_sim)
print("\n  多项 chi2 (dof=2):  实测 中位 %.2f p90 %.2f p99 %.2f 最大 %.1f"
      % (np.median(chi_obs), np.percentile(chi_obs, 90), np.percentile(chi_obs, 99), chi_obs.max()))
print("                      模拟 中位 %.2f p90 %.2f p99 %.2f 最大 %.1f"
      % (np.median(chi_sim), np.percentile(chi_sim, 90), np.percentile(chi_sim, 99), chi_sim.max()))
for t in (9.21, 13.8, 20.0, 30.0):
    print("    chi2 > %-5.1f : 实测 %4d / %d   模拟 %4d" % (t, (chi_obs > t).sum(), len(bal), (chi_sim > t).sum()))

print("\n  基线平衡的候选里，窗内单箱占比 > 0.9 的: %d / %d"
      % (sum(r['sw'].max() > 0.9 for r in bal), len(bal)))
print("  窗内单箱占比 > 0.7 的: %d / %d（同 n 的多项模拟给 %d）"
      % (sum(r['sw'].max() > 0.7 for r in bal), len(bal),
         sum((rng.multinomial(int(r['n']), r['sb']) / r['n']).max() > 0.7 for r in bal)))

print("\n  实测 chi2 最大的 10 个（基线平衡组）：")
idx = np.argsort(-chi_obs)[:10]
for i in idx:
    r = bal[i]
    print("    %s  n=%4d  W=%s  sb=%s  chi2=%.1f  assoc=%d"
          % (r['start'][:23], r['n'], '/'.join('%d' % x for x in r['W']),
             '/'.join('%.3f' % x for x in r['sb']), chi_obs[i], r['assoc']))
