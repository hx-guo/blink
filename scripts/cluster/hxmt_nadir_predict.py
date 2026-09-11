#!/usr/bin/env python3
"""把方位谐波当成预言来检验，而不是只报一个振幅。

谐波拟合给出三个机箱在本体系里的方位（Box B ≈ 3°、C ≈ 121°、A ≈ 241°）。
那就有一条可证伪的预言：**天底方位角 phi 最靠近哪个箱，哪个箱的超出份额就该最大**。
命中率必须并排给偶然期望 1/3。

相位是在同一批数据上拟出来的，所以用留一半的办法去循环：相位用 2017–2020 拟，
命中率在 2021–2024 上数；再反过来一次。

用法: hxmt_nadir_predict.py <det_nadir.csv>
"""
import csv, sys
import numpy as np

rng = np.random.default_rng(20260911)


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        if not r['nadir_theta'] or not r['det_window']:
            continue
        w = np.array([int(x) for x in r['det_window'].split('|')], float)
        b = np.array([int(x) for x in r['det_baseline'].split('|')], float)
        if w.sum() == 0 or b.sum() == 0:
            continue
        W = np.array([w[0:6].sum(), w[6:12].sum(), w[12:18].sum()])
        B = np.array([b[0:6].sum(), b[6:12].sum(), b[12:18].sum()])
        if B.max() / B.sum() > 0.9:                # 只用三箱都活着的
            continue
        out.append(dict(day=int(r['date']), assoc=r['assoc'] == '1',
                        phi=float(r['nadir_phi']), n=W.sum(),
                        d=W / W.sum() - B / B.sum()))
    return out


def phases(rows):
    phi = np.radians(np.array([r['phi'] for r in rows]))
    n = np.array([r['n'] for r in rows], float)
    X = np.stack([np.ones_like(phi), np.cos(phi), np.sin(phi)], 1)
    A = (X * n[:, None]).T @ X
    out = []
    for c in range(3):
        y = np.array([r['d'][c] for r in rows])
        co = np.linalg.solve(A, (X * n[:, None]).T @ y)
        out.append(np.degrees(np.arctan2(co[2], co[1])) % 360.0)
    return np.array(out)


def hit_rate(rows, ph):
    """phi 最靠近哪个箱的方位，该箱的超出份额是不是最大。"""
    hit = 0
    for r in rows:
        sep = np.abs(((r['phi'] - ph + 180) % 360) - 180)
        if sep.argmin() == np.argmax(r['d']):
            hit += 1
    return hit, len(rows)


def null_rate(rows, ph, n_perm=500):
    """偶然期望不是 1/3。phi 的分布不均匀、三个箱当"最大超出"的次数也不等长，
    两个边缘分布一乘，朴素的 1/3 就不成立。把 phi 在候选之间置换（两个边缘
    分布都原样保留），重算命中率，这才是这个问题真正的偶然线。"""
    phi = np.array([r['phi'] for r in rows])
    arg = np.array([np.argmax(r['d']) for r in rows])
    out = []
    for _ in range(n_perm):
        p2 = rng.permutation(phi)
        sep = np.abs(((p2[:, None] - ph[None, :] + 180) % 360) - 180)
        out.append((sep.argmin(1) == arg).mean())
    return np.mean(out), np.percentile(out, [2.5, 97.5])


def wilson(k, n):
    p = k / n; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z / (1 + z * z / n) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h), 100 * (c + h)


rows = load(sys.argv[1])
tgf = [r for r in rows if r['assoc']]
print("闪电关联 TGF（三箱齐）%d" % len(tgf))
early = [r for r in tgf if r['day'] < 20210101]
late = [r for r in tgf if r['day'] >= 20210101]

for nm, fit_on, test_on in (("相位拟 2017–2020，命中率数 2021–2024", early, late),
                            ("相位拟 2021–2024，命中率数 2017–2020", late, early)):
    ph = phases(fit_on)
    k, n = hit_rate(test_on, ph)
    lo, hi = wilson(k, n)
    e, ei = null_rate(test_on, ph)
    print("\n  %s" % nm)
    print("    相位 A %.1f° B %.1f° C %.1f°（相邻间隔 %.1f° / %.1f°）"
          % (ph[0], ph[1], ph[2],
             (ph[2] - ph[1]) % 360, (ph[0] - ph[2]) % 360))
    print("    命中 %d/%d = %.1f%% [%.1f–%.1f%%]，置换偶然线 %.1f%% [%.1f–%.1f%%]"
          % (k, n, 100 * k / n, lo, hi, 100 * e, 100 * ei[0], 100 * ei[1]))

ph_all = phases(tgf)
other = [r for r in rows if not r['assoc']]
k, n = hit_rate(other, ph_all)
lo, hi = wilson(k, n)
e, ei = null_rate(other, ph_all)
print("\n  用全体 TGF 的相位，在未关联候选上数（独立人群）：")
print("    命中 %d/%d = %.1f%% [%.1f–%.1f%%]" % (k, n, 100 * k / n, lo, hi))
print("    置换偶然线 %.1f%% [%.1f–%.1f%%] —— 不是 1/3，phi 与占优箱的边缘分布都不均匀"
      % (100 * e, 100 * ei[0], 100 * ei[1]))
k2, n2 = hit_rate(other, (ph_all + 60.0) % 360.0)
e2, _ = null_rate(other, (ph_all + 60.0) % 360.0)
print("    对照：三个相位整体转 60° -> 命中 %.1f%%，其置换偶然线 %.1f%%（差值应当归零）"
      % (100 * k2 / n2, 100 * e2))
