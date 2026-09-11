#!/usr/bin/env python3
"""把"单箱占比高"拆成两类：机箱本来就没开（基线也单箱）vs 真的照亮不均匀。"""
import os, csv, collections
import numpy as np
P = os.environ.get('HXMT_AUDIT_DIR', '.').rstrip('/') + '/det_all.csv'
STORM={20230424,20240428,20240813,20240918,20241009,20241010,20241023,20241024}; CHECK=20250930
rec=[]
for r in csv.DictReader(open(P)):
    w=np.array([int(x) for x in r['det_window'].split('|')],float)
    b=np.array([int(x) for x in r['det_baseline'].split('|')],float)
    if w.sum()==0 or b.sum()==0: continue
    Wb=np.array([w[0:6].sum(),w[6:12].sum(),w[12:18].sum()])
    Bb=np.array([b[0:6].sum(),b[6:12].sum(),b[12:18].sum()])
    fa=float(r['false_positive_per_year']); assoc=r['assoc']=='1'; day=int(r['date'])
    rec.append(dict(day=day,start=r['start'],fa=fa,assoc=assoc,
                    sel=(fa<1e-5) or (fa<1.0 and assoc),train=r['is_train']=='1',
                    n=w.sum(),nbg=b.sum(),W=Wb,B=Bb,sw=Wb/Wb.sum(),sb=Bb/Bb.sum()))

def cls(r):
    hw = r['sw'].max()>0.9; hb = r['sb'].max()>0.9
    if hw and hb: return 'box-off (窗+基线同一箱独占)' if r['sw'].argmax()==r['sb'].argmax() else 'box-off (但换了箱)'
    if hw and not hb: return '真不均匀 (基线三箱齐, 窗内单箱)'
    if hb and not hw: return '基线单箱但窗内散开'
    return '正常'

print("=== 全部 7544 审计行的分类 ===")
c=collections.Counter(cls(r) for r in rec)
for k,v in c.most_common(): print(f"  {k:<32} {v:5d}  ({100*v/len(rec):.2f}%)")

print("\n=== '真不均匀' 那批是谁 ===")
g=[r for r in rec if cls(r).startswith('真不均匀')]
print(f"  N={len(g)}  按天:", collections.Counter(r['day'] for r in g).most_common(10))
print(f"  占优箱:", collections.Counter('ABC'[r['sw'].argmax()] for r in g))
print(f"  窗内计数中位 {np.median([r['n'] for r in g]):.0f}  assoc {sum(r['assoc'] for r in g)}")
print(f"  偶然期望: 窗内计数 k 个事例全落一箱的概率 3*(1/3)^k —— 逐个算")
pexp=[3*(1/3.)**r['n'] for r in g]
print(f"  这 {len(g)} 个的偶然期望之和 {sum(pexp):.4g}（即随机涨落做不出来）")

print("\n=== 三箱齐全的候选里，占优箱的分布（问 REP 是否真偏 C）===")
ok=[r for r in rec if r['sb'].max()<=0.9 and r['sw'].max()<=0.9]
def dom(g,nm):
    if not g: return
    sw=np.array([r['sw'] for r in g]); sb=np.array([r['sb'] for r in g]); d=sw-sb
    dd=d.argmax(1); n=len(g)
    from math import sqrt
    print(f"  {nm:<34} N={n:5d}  差值占优 A/B/C = {(dd==0).sum()}/{(dd==1).sum()}/{(dd==2).sum()}"
          f"  (随机各 {n/3:.0f}±{sqrt(n*2/9):.0f})")
    print(f"       {'':<34} 差值均值 dA {d[:,0].mean():+.4f}  dB {d[:,1].mean():+.4f}  dC {d[:,2].mean():+.4f}")
dom([r for r in ok if r['assoc']],'certified TGF')
dom([r for r in ok if r['sel'] and r['day'] in STORM and not r['assoc']],'certified REP (风暴日,三箱齐)')
dom([r for r in ok if r['sel'] and r['day']==CHECK and not r['assoc']],'2025-09-30 (对照日)')
dom([r for r in ok if not r['assoc'] and r['day'] not in STORM and r['day']!=CHECK],'其余未关联候选')
