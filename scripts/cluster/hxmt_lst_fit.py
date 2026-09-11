#!/usr/bin/env python3
"""当地太阳时（LST）两成分分解：一批候选里有多少是真 TGF，不用闪电台网。

四条硬要求逐条对应：
 1. 模板来自外部真值样本 —— 用闪电关联候选（不在被检验的样本里）。
 2. 本底模板留出 —— 用 fa > 1 的候选，与被检验的显著候选不重叠。
 3. 先验证本底 LST 是平的 —— 打印本底直方图与平坦性检验。
 4. 判别量与选样无关 —— LST 只由 UTC 与经度算出，选样只看显著性。

用法: lst.py <pool_lst.csv> <catalog_v6.csv>
"""
import csv, sys, collections, math
from datetime import datetime, timezone
import numpy as np

NBIN = 8            # 3 小时一格，与 GBM 同口径
rng = np.random.default_rng(20260911)


def lst_hours(iso, lon):
    """平太阳时：UTC 小时 + 经度/15。时差方程幅度 ±0.27 h，小于 3 h 的格宽。"""
    h = int(iso[11:13]) + int(iso[14:16]) / 60.0 + float(iso[17:19]) / 3600.0
    return (h + lon / 15.0) % 24.0


def hist(lst):
    h, _ = np.histogram(lst, bins=NBIN, range=(0, 24))
    return h.astype(float)


def flatness(h, name):
    n = h.sum()
    exp = n / len(h)
    chi2 = ((h - exp) ** 2 / exp).sum()
    frac = 100 * h / n
    print("  %-26s N=%7d  每格 %s %%" % (name, int(n), " ".join("%.1f" % x for x in frac)))
    print("  %-26s chi2=%.1f (dof=%d)，均匀的 p99 = %.1f"
          % ("", chi2, len(h) - 1, 18.5 if len(h) == 8 else float('nan')))
    return chi2


def fit_fraction(obs, tpl_tgf, tpl_bg):
    """二成分分箱泊松似然：obs ~ N*[f*T + (1-f)*B]，profile 出 f 的 68/95 区间。"""
    N = obs.sum()
    T = tpl_tgf / tpl_tgf.sum()
    B = tpl_bg / tpl_bg.sum()

    def nll(f):
        mu = N * (f * T + (1 - f) * B)
        mu = np.maximum(mu, 1e-12)
        return -(obs * np.log(mu) - mu).sum()

    fs = np.linspace(-0.5, 1.5, 2001)
    vals = np.array([nll(f) for f in fs])
    i = vals.argmin()
    best = fs[i]
    lo68, hi68 = interval(fs, vals, vals[i] + 0.5)
    lo95, hi95 = interval(fs, vals, vals[i] + 1.92)
    return best, (lo68, hi68), (lo95, hi95)


def interval(fs, vals, level):
    below = fs[vals <= level]
    return (below.min(), below.max()) if len(below) else (float('nan'), float('nan'))


def day_bootstrap(lst, days, tpl_tgf, tpl_bg, n=400):
    """按天重抽：同一天的候选高度相关，逐候选 bootstrap 会低估误差。"""
    byday = collections.defaultdict(list)
    for x, d in zip(lst, days):
        byday[d].append(x)
    keys = list(byday)
    out = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        sample = np.concatenate([byday[keys[k]] for k in pick])
        out.append(fit_fraction(hist(sample), tpl_tgf, tpl_bg)[0])
    return np.percentile(out, [2.5, 16, 50, 84, 97.5])


def main():
    pool_path, cat_path = sys.argv[1], sys.argv[2]

    cat = list(csv.DictReader(open(cat_path)))
    tgf = [c for c in cat if c['associated'] == '1']
    tgf_lst = np.array([lst_hours(c['start'], float(c['longitude'])) for c in tgf])
    tgf_day = np.array([c['date'] for c in tgf])

    pool_bg, pool_sig = [], []
    with open(pool_path) as f:
        r = csv.reader(f); next(r)
        for start, fpy, lon, lat, assoc, nb, train in r:
            fa = float(fpy)
            if fa > 1.0 and train == '0':
                pool_bg.append((start, float(lon)))
            elif fa < 1e-5 and train == '0' and assoc == '0':
                pool_sig.append((start, float(lon), float(lat)))
    print("本底池 (fa > 1, 非列车) %d；被检验样本 (fa < 1e-5, 非列车, 未关联) %d"
          % (len(pool_bg), len(pool_sig)))

    bg_lst = np.array([lst_hours(s, l) for s, l in pool_bg])
    sig_lst = np.array([lst_hours(s, l) for s, l, _ in pool_sig])
    sig_day = np.array([s[:10] for s, _, _ in pool_sig])

    print("\n=== 要求 3：本底的 LST 分布平不平 ===")
    flatness(hist(bg_lst), "本底 fa > 1")
    print("\n=== 真值模板（闪电关联 TGF）===")
    flatness(hist(tgf_lst), "闪电关联 TGF")

    print("\n=== 被检验样本的分解 ===")
    obs = hist(sig_lst)
    flatness(obs, "显著未关联")
    best, i68, i95 = fit_fraction(obs, hist(tgf_lst), hist(bg_lst))
    print("  f_TGF = %.3f  68%% [%.3f, %.3f]  95%% [%.3f, %.3f]  -> 真 TGF 约 %.0f 个"
          % (best, *i68, *i95, best * len(sig_lst)))
    q = day_bootstrap(sig_lst, sig_day, hist(tgf_lst), hist(bg_lst))
    print("  按天 bootstrap: 中位 %.3f，68%% [%.3f, %.3f]，95%% [%.3f, %.3f]"
          % (q[2], q[1], q[3], q[0], q[4]))

    print("\n=== 负对照：列车候选（REP）===")
    rep = []
    with open(pool_path) as f:
        r = csv.reader(f); next(r)
        for start, fpy, lon, lat, assoc, nb, train in r:
            if float(fpy) < 1e-5 and train == '1' and assoc == '0':
                rep.append((start, float(lon), float(lat)))
    rep_lst = np.array([lst_hours(s, l) for s, l, _ in rep])
    rep_day = np.array([s[:10] for s, _, _ in rep])
    print("  N=%d，独立日期 %d 个" % (len(rep), len(set(rep_day))))
    flatness(hist(rep_lst), "列车/REP 候选")
    b2, i2, i95b = fit_fraction(hist(rep_lst), hist(tgf_lst), hist(bg_lst))
    print("  f_TGF = %.3f  68%% [%.3f, %.3f]  95%% [%.3f, %.3f]" % (b2, *i2, *i95b))
    q2 = day_bootstrap(rep_lst, rep_day, hist(tgf_lst), hist(bg_lst))
    print("  按天 bootstrap: 中位 %.3f，95%% [%.3f, %.3f]" % (q2[2], q2[0], q2[4]))

    print("\n=== post-2024 单腿目录（WWLLN 到期之后）===")
    post = [(s, l) for s, l, _ in pool_sig if s[:4] >= '2025']
    if post:
        p_lst = np.array([lst_hours(s, l) for s, l in post])
        p_day = np.array([s[:10] for s, _ in post])
        print("  N=%d，独立日期 %d" % (len(post), len(set(p_day))))
        flatness(hist(p_lst), "post-2024 显著未关联")
        b3, i3, i95c = fit_fraction(hist(p_lst), hist(tgf_lst), hist(bg_lst))
        print("  f_TGF = %.3f  68%% [%.3f, %.3f]  95%% [%.3f, %.3f]" % (b3, *i3, *i95c))
        q3 = day_bootstrap(p_lst, p_day, hist(tgf_lst), hist(bg_lst))
        print("  按天 bootstrap: 中位 %.3f，95%% [%.3f, %.3f]" % (q3[2], q3[0], q3[4]))

    print("\n=== 口径检验：暗的三分之一 TGF 模板形状变不变 ===")
    cnt = np.array([int(c['count']) for c in tgf])
    thr = np.percentile(cnt, 33.3)
    dim = tgf_lst[cnt <= thr]; bright = tgf_lst[cnt > np.percentile(cnt, 66.7)]
    hd, hb = hist(dim), hist(bright)
    print("  暗 1/3 (count <= %.0f) N=%d: %s" % (thr, len(dim), " ".join("%.1f" % x for x in 100 * hd / hd.sum())))
    print("  亮 1/3               N=%d: %s" % (len(bright), " ".join("%.1f" % x for x in 100 * hb / hb.sum())))
    pooled = (hd / hd.sum() + hb / hb.sum()) / 2
    chi2 = (((hd / hd.sum() - hb / hb.sum()) ** 2) / np.maximum(pooled * (1 / hd.sum() + 1 / hb.sum()), 1e-12)).sum()
    print("  两形状的 chi2 = %.1f (dof=7)，p99 = 18.5" % chi2)
    b4 = fit_fraction(obs, hd, hist(bg_lst))[0]
    b5 = fit_fraction(obs, hb, hist(bg_lst))[0]
    print("  用暗模板拟 f_TGF = %.3f，用亮模板 %.3f（全样本模板 %.3f）" % (b4, b5, best))

    print("\n=== 供给：2547 个证实 TGF 的 LST 模板（3 h 一格，0–24 h）===")
    h = hist(tgf_lst)
    print("  计数: " + ", ".join("%d" % x for x in h))
    print("  归一: " + ", ".join("%.4f" % x for x in h / h.sum()))
    print("  纬度带: |lat| <= %.0f deg（HXMT 倾角 43）" % max(abs(float(c['latitude'])) for c in tgf))


if __name__ == "__main__":
    main()
