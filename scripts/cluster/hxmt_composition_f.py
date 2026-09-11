#!/usr/bin/env python3
"""gbm 提的构成法：不碰 LST、不碰模板，只用星下点落在哪一类。

纯信号端点 = 闪电关联候选；纯本底端点 = fa > 1 池；待测 = 显著未关联。
**警告**：HXMT 的"认证 TGF"只能由 WWLLN 给，而 WWLLN 的探测效率有空间不均
（台站在陆上、远洋弱），所以纯信号端点很可能偏陆。方向单向：端点偏陆 ⇒
待测样本显得比端点"不够陆" ⇒ f 被低估。**所以这个数是下限。**
"""
import csv, sys
import numpy as np
from global_land_mask import globe

R = 6371.0
rng = np.random.default_rng(3)


def classify(lat, lon):
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    if bool(globe.is_land(lat, lon180)):
        return 0
    d = np.degrees(300.0 / R)
    la, lo = [], []
    for k in range(16):
        a = 2 * np.pi * k / 16
        la.append(np.clip(lat + d * np.cos(a), -89.9, 89.9))
        lo.append((((lon180 + d * np.sin(a) / max(np.cos(np.radians(lat)), 0.2)) + 180) % 360) - 180)
    return 1 if globe.is_land(np.array(la), np.array(lo)).any() else 2


def comp(labels):
    h = np.bincount(np.asarray(labels), minlength=3).astype(float)
    return h, h / h.sum()


cat = list(csv.DictReader(open(sys.argv[1])))
lat = np.array([float(c['latitude']) for c in cat]); lon = np.array([float(c['longitude']) for c in cat])
assoc = np.array([c['associated'] == '1' for c in cat])
lab = np.array([classify(a, b) for a, b in zip(lat, lon)])

bl, bla = [], []
with open(sys.argv[2]) as f:
    r = csv.reader(f); next(r)
    for i, row in enumerate(r):
        start, fpy, blon_, blat_, ass, nbn, train = row
        if float(fpy) > 1 and train == '0' and i % 6 == 0:
            bl.append(classify(float(blat_), float(blon_))); bla.append(abs(float(blat_)))
bl = np.array(bl); bla = np.array(bla)

for cut, tag in ((44.0, "|lat| <= 43（全覆盖）"), (26.0, "|lat| <= 26（与 GBM 同带）")):
    S = comp(lab[assoc & (np.abs(lat) < cut)])
    D = comp(lab[(~assoc) & (np.abs(lat) < cut)])
    B = comp(bl[bla < cut])
    print("=== %s ===" % tag)
    print("  %-24s %-6s %-8s %-8s %s" % ("人群", "n", "陆地", "近岸", "远洋"))
    for nm, (h, p) in (("纯信号端点 闪电关联", S), ("待测 显著未关联", D), ("纯本底端点 fa>1", B)):
        print("  %-24s %-6d %-8s %-8s %s" % (nm, int(h.sum()), *["%.1f%%" % (100 * v) for v in p]))
    inside = all(min(S[1][k], B[1][k]) <= D[1][k] <= max(S[1][k], B[1][k]) for k in range(3))
    print("  待测每一格都落在两端之间：%s" % ("是" if inside else "否"))
    fs = np.linspace(-0.5, 1.5, 4001)
    obs = D[0]
    mix = np.outer(fs, S[1]) + np.outer(1 - fs, B[1])
    nll = -(obs * np.log(np.maximum(mix, 1e-12))).sum(axis=1)
    i = nll.argmin(); ok = fs[nll <= nll[i] + 1.92]
    mu = obs.sum() * mix[i]
    d0 = 2 * (nll[np.argmin(np.abs(fs - 0.0))] - nll[i])
    print("  两成分多项式拟合: f = %.3f [%.3f, %.3f]，chi2 = %.1f/1，对 f=0 的 Delta = %.1f"
          % (fs[i], ok.min(), ok.max(), ((obs - mu) ** 2 / mu).sum(), d0))
    print("")
