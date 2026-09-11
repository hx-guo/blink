#!/usr/bin/env python3
"""单箱占比这个判据，剥掉"另两箱被压低"之后还剩多少分离度。

负样本 = 29 个缺口边缘候选（本底率被压低那批，上一轮定为开关机假信号）
正样本 = 120 个闪电关联 TGF + 120 个 r 正常的普通显著候选

三个量并排：
  raw      窗内单箱占比 max(sw)                         —— 上一轮报 AUC 0.9307
  excess   max(sw - sb)，拿同一候选自己的基线当分母     —— 免疫"另两箱被压低"
  dev      多项分布偏离：窗内计数相对"按基线分摊"的期望，取最大单箱的泊松显著度
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
    det[r['start']] = dict(W=W, B=B, sw=W / W.sum(), sb=B / B.sum(), n=W.sum(), nbg=B.sum(),
                           day=int(r['date']), assoc=r['assoc'] == '1',
                           fa=float(r['false_positive_per_year']))

lab = collections.defaultdict(list)
for r in csv.DictReader(open(D + 'cov_list.csv')):
    if r['start'] in det:
        lab[r['label']].append(det[r['start']])
print("gapedge %d, tgf %d, sig %d" % (len(lab['gapedge']), len(lab['tgf']), len(lab['sig'])))


def raw(d):
    return d['sw'].max()


def excess(d):
    return (d['sw'] - d['sb']).max()


def dev(d):
    """按基线分摊的期望下，最亮那箱的泊松显著度（单边，用 Wilson-Hilferty 近似）。"""
    exp = d['n'] * d['sb']
    z = (d['W'] - exp) / np.sqrt(np.maximum(exp, 1e-9))
    return z.max()


def auc(pos, neg, f):
    """neg 是要抓的假信号，取正方向：值越大越像假信号。"""
    a = np.array([f(x) for x in neg])
    b = np.array([f(x) for x in pos])
    # 秩和法，含并列各算一半
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # 并列平均秩
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = len(a)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(b))


pos = lab['tgf'] + lab['sig']
neg = lab['gapedge']
print("\n%-10s %-8s %-8s  %s" % ("量", "AUC", "", "中位值 (缺口边缘 / TGF / 普通显著)"))
for name, f in (("raw", raw), ("excess", excess), ("dev", dev)):
    print("  %-8s %.4f            %.3f / %.3f / %.3f"
          % (name, auc(pos, neg, f),
             np.median([f(x) for x in lab['gapedge']]),
             np.median([f(x) for x in lab['tgf']]),
             np.median([f(x) for x in lab['sig']])))

print("\n=== 各判据在整个审计池（7544 行）里的误伤 ===")
allr = list(det.values())
tgf_all = [d for d in allr if d['assoc']]
for name, f, thr in (("raw > 0.9", raw, 0.9), ("excess > 0.5", excess, 0.5), ("dev > 8", dev, 8.0)):
    cut_neg = sum(f(d) > thr for d in neg)
    cut_all = sum(f(d) > thr for d in allr)
    cut_tgf = sum(f(d) > thr for d in tgf_all)
    print("  %-13s 抓住缺口边缘 %2d/%d (%.0f%%)  切掉全池 %4d/%d (%.2f%%)  切掉闪电关联 TGF %3d/%d (%.2f%%)"
          % (name, cut_neg, len(neg), 100 * cut_neg / len(neg),
             cut_all, len(allr), 100 * cut_all / len(allr),
             cut_tgf, len(tgf_all), 100 * cut_tgf / len(tgf_all)))

print("\n=== OPEN-QUESTIONS 第 10 条建议的联合条件 ===")


def joint(d):
    return d['sw'].max() > 0.9 and d['B'].min() == 0


def joint_soft(d):
    return d['sw'].max() > 0.9 and d['sb'].max() > 0.9


for name, f in (("单箱>0.9 且 基线有箱为零", joint), ("单箱>0.9 且 基线也单箱>0.9", joint_soft)):
    print("  %-26s 抓住缺口边缘 %2d/%d  切掉全池 %4d/%d (%.2f%%)  切掉闪电关联 TGF %d/%d"
          % (name, sum(f(d) for d in neg), len(neg),
             sum(f(d) for d in allr), len(allr), 100 * sum(f(d) for d in allr) / len(allr),
             sum(f(d) for d in tgf_all), len(tgf_all)))
