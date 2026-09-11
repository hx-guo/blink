#!/usr/bin/env python3
"""逐候选分类、按类各拟一次：A 的加权问题自动消失，还白得一个自洽检验。

非循环的做法只有一种：**模板与数据必须是不同的时段**。
  模板 = pre-2025 该类的显著未关联候选
  数据 = post-2024 该类的显著未关联候选
  本底 = 留出的 fa > 1，同样按类取（本底率与 LST 覆盖都可能分类而异）
拟出来的是**该类的相对纯度** f_post/f_pre。三类的 f 应当相等——纯度不该依赖
候选落在陆上还是海上。不相等就是还有东西没弄干净。

用法: hxmt_lst_perclass.py <pool_lst.csv>
"""
import csv, sys, collections
import numpy as np
from global_land_mask import globe

NBIN = 8
CENTRES = np.arange(NBIN) * 3.0 + 1.5
R = 6371.0
rng = np.random.default_rng(20260911)


def lst_hours(iso, lon):
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def classify(lat, lon, ring=600.0):
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    d = np.degrees(ring / R)
    la = [lat]; lo = [lon180]
    for az in range(0, 360, 45):
        a = np.radians(az)
        la.append(np.clip(lat + d * np.cos(a), -89.9, 89.9))
        lo.append((((lon180 + d * np.sin(a) / max(np.cos(np.radians(lat)), 0.2)) + 180) % 360) - 180)
    f = globe.is_land(np.array(la), np.array(lo))
    return 'land' if f.all() else ('ocean' if not f.any() else 'coast')


def hist(x):
    return np.histogram(np.asarray(x), bins=NBIN, range=(0, 24))[0].astype(float)


def fit(obs, T, B):
    N = obs.sum()
    if N == 0 or T.sum() == 0 or B.sum() == 0:
        return (float('nan'),) * 4
    T = T / T.sum(); B = B / B.sum()
    fs = np.linspace(-1.0, 2.0, 3001)
    nll = np.array([-(obs * np.log(np.maximum(N * (f * T + (1 - f) * B), 1e-12))
                      - N * (f * T + (1 - f) * B)).sum() for f in fs])
    i = nll.argmin(); ok = fs[nll <= nll[i] + 1.92]
    mu = N * (fs[i] * T + (1 - fs[i]) * B)
    return fs[i], ok.min(), ok.max(), ((obs - mu) ** 2 / np.maximum(mu, 1e-9)).sum()


def dayboot(lst, day, T, B, n=300):
    by = collections.defaultdict(list)
    for a, d in zip(lst, day):
        by[d].append(a)
    ks = list(by)
    out = []
    for _ in range(n):
        s = np.concatenate([by[ks[j]] for j in rng.choice(len(ks), len(ks))])
        v = fit(hist(s), T, B)[0]
        if np.isfinite(v):
            out.append(v)
    return np.percentile(out, [2.5, 50, 97.5]) if out else (float('nan'),) * 3


def main():
    pre, post, bg = [], [], []
    with open(sys.argv[1]) as f:
        r = csv.reader(f); next(r)
        for i, (start, fpy, lon, lat, assoc, nb, train) in enumerate(r):
            fa = float(fpy)
            if fa < 1e-5 and train == '0' and assoc == '0':
                (pre if start[:4] < '2025' else post).append(
                    (start[:10], lst_hours(start, float(lon)), float(lat), float(lon)))
            elif fa > 1 and train == '0' and i % 12 == 0:
                bg.append((lst_hours(start, float(lon)), float(lat), float(lon)))
    print("pre-2025 %d，post-2024 %d，本底抽样 %d" % (len(pre), len(post), len(bg)))

    def split(rows, has_day):
        out = collections.defaultdict(lambda: ([], []))
        for x in rows:
            k = classify(x[-2], x[-1])
            out[k][0].append(x[1] if has_day else x[0])
            out[k][1].append(x[0] if has_day else '')
        return out

    P = split(pre, True); Q = split(post, True); Bc = split(bg, False)

    print("\n=== 两段的陆/海/近岸构成（构成若变，曝光或轨道变了）===")
    print("  %-6s %14s %14s" % ("类", "pre-2025", "post-2024"))
    for k in ('land', 'coast', 'ocean'):
        a, b = len(P[k][0]), len(Q[k][0])
        print("  %-6s %6d (%4.1f%%) %6d (%4.1f%%)"
              % (k, a, 100 * a / len(pre), b, 100 * b / len(post)))

    print("\n=== 按类各拟一次：模板 = pre-2025 同类，数据 = post-2024 同类 ===")
    print("  %-6s %6s %6s  %-28s %-8s %s" % ("类", "N模板", "N数据", "f = f_post/f_pre 95%", "chi2/6", "按天 bootstrap 95%"))
    fs = []
    for k in ('land', 'coast', 'ocean'):
        T = hist(P[k][0]); B = hist(Bc[k][0]); obs = hist(Q[k][0])
        b, lo, hi, c2 = fit(obs, T, B)
        q = dayboot(np.array(Q[k][0]), np.array(Q[k][1]), T, B)
        fs.append((b, lo, hi))
        print("  %-6s %6d %6d  %.3f [%.3f, %.3f]%s  %-8.1f [%.3f, %.3f]"
              % (k, len(P[k][0]), len(Q[k][0]), b, lo, hi, " " * 6, c2, q[0], q[2]))
    # 三个 f 相等吗：用各自的 95% 区间半宽当 sigma 做一个粗的一致性检验
    v = np.array([x[0] for x in fs])
    s = np.array([(x[2] - x[1]) / 2 / 1.96 for x in fs])
    w = 1 / s ** 2
    mean = (v * w).sum() / w.sum()
    chi2 = (((v - mean) / s) ** 2).sum()
    print("\n  三类一致性：加权均值 f = %.3f ± %.3f，chi2 = %.2f (dof=2)，p99 = 9.2 -> %s"
          % (mean, 1 / np.sqrt(w.sum()), chi2, "一致" if chi2 < 9.2 else "不一致，有东西没弄干净"))

    print("\n=== 各类模板本身的幅度 A（给别的仪器按自己的陆海比例加权用）===")
    for k in ('land', 'coast', 'ocean'):
        T = hist(P[k][0]); B = hist(Bc[k][0])
        print("  %-6s 模板 A = %.2f（N=%d）；同类本底 A = %.3f（N=%d）"
              % (k, (T.max() - T.min()) / 2 / T.mean(), int(T.sum()),
                 (B.max() - B.min()) / 2 / B.mean(), int(B.sum())))


if __name__ == "__main__":
    main()
