"""GECAM-A：把 `ga_window.py` 的逐候选表汇成四张表——对账、f₃、量程上限、组合。

**每一条判据都并排报「切前 / 切后 / 该条件下的偶然期望」**，不报单边数字。

`fa` 的重算不需要反解试验数：`fa = sf(mean, count) × (365.25 天 / bin_size_best)`，
分母是确定的（见 `poisson::false_positive_per_year`），所以准入上界改口径之后
只要重新数窗内与本底的存活事例，就能把 `fa` 精确地重算一遍。这与第 17 条那次
「反标定 T」的近似不是一回事，那里是因为没有落盘的 `sf` 才反解的。

用法: python3 ga_report.py <w01.csv> [w02.csv ...] [--log <win_h*.log> ...]
"""

import csv
import sys

import numpy as np
from scipy.stats import beta, poisson

YEAR = 3600.0 * 24.0 * 365.25
FPY = 20.0            # SearchConfig::false_positive_per_year
MIN_NUMBER = 8        # SearchConfig::min_number
EXPOSURE = 7050.0     # 2024-01-11 三个小时的 GTI 活时间（hours.json）
DAYS = 183            # A 星落在 WWLLN 覆盖内的天数（2022-10 .. 2024-12）
BINS = [0.0, 1.0, 10.0, 50.0, 200.0, 1e9]


def interval(k, n):
    """Clopper–Pearson 95% 区间，小样本比例一律带区间。"""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def load(paths):
    rows = []
    for path in paths:
        rows.extend(csv.DictReader(open(path)))
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            out[key] = np.array([r[key] for r in rows])
    return out


def strata(bin_us):
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        yield f"{lo:g}–{hi:g}" if hi < 1e8 else f"≥{lo:g}", (bin_us >= lo) & (bin_us < hi)


def main():
    paths = [a for a in sys.argv[1:] if not a.endswith(".log")]
    d = load(paths)
    n = len(d["start"])
    bin_s = d["bin_ns"] * 1e-9
    bin_us = d["bin_ns"] * 1e-3
    count, mean, fa, sf = d["count"], d["mean"], d["fa"], d["sf"]

    print("=" * 78)
    print("零、对账与公式自检")
    print("=" * 78)
    ok_count = int((d["n_core"] == count).sum())
    ok_mean = int((np.abs(d["mean_chk"] - mean) <= 1e-9 * np.maximum(mean, 1e-300)).sum())
    print(f"候选 {n} 个（2024-01-11，曝光 {EXPOSURE:.0f} s，率 {n / EXPOSURE:.3f} /s）")
    print(f"  窗内计数 n_core == count : {ok_count}/{n} = {ok_count / n * 100:.2f}%")
    print(f"  本底 mean 逐条重算一致    : {ok_mean}/{n} = {ok_mean / n * 100:.2f}%")
    sf_re = poisson.sf(count, mean)
    fa_re = sf_re * (YEAR / bin_s)
    good = np.isfinite(sf_re) & (sf_re > 0)
    rel = np.abs(fa_re[good] - fa[good]) / fa[good]
    print(f"  fa = sf(mean,count)·(年/bin) 复现：可算的 {good.sum()}/{n}，"
          f"相对差中位 {np.median(rel):.2e}、p99 {np.percentile(rel, 99):.2e}")

    print()
    print("=" * 78)
    print("一、f₃（≥3 重同戳簇里的计数占比），按窗长分层")
    print("=" * 78)
    f3, f2 = d["f3"], d["f2"]
    print(f"全样本 f₂ 中位 {np.median(f2):.3f}、f₃ 中位 {np.median(f3):.3f}、"
          f"f₃>0 占 {(f3 > 0).mean() * 100:.2f}%、max_d 中位 {np.median(d['max_d']):.0f}")
    print(f"{'窗长 µs':<10}{'n':>7}{'占比%':>7}{'f₃>0 %':>9}{'f₃中位':>8}{'f₂中位':>8}"
          f"{'三重偶然期望中位':>18}{'偶然 f₃>0 期望个数':>20}")
    for name, sel in strata(bin_us):
        if sel.sum() == 0:
            continue
        chance = 1.0 - np.exp(-d["exp_trip"][sel])
        print(f"{name:<10}{sel.sum():>7}{sel.mean() * 100:>7.1f}{(f3[sel] > 0).mean() * 100:>9.2f}"
              f"{np.median(f3[sel]):>8.3f}{np.median(f2[sel]):>8.3f}"
              f"{np.median(d['exp_trip'][sel]):>18.3g}{chance.sum():>20.2f}")

    long = bin_us >= 10.0
    short = ~long
    print()
    print(f"窗 ≥ 10 µs：{long.sum()} 个（{long.mean() * 100:.1f}%）。"
          f"f₃ 在这一档零参数可用：三重偶然期望中位 {np.median(d['exp_trip'][long]):.2g}，"
          f"整档偶然误否决期望 {(1 - np.exp(-d['exp_trip'][long])).sum():.2f} 个")
    keep_long = long & (f3 == 0)
    lo, hi = interval(int(keep_long.sum()), int(long.sum()))
    print(f"  `f₃ == 0` 留下 {keep_long.sum()} 个 = {keep_long.sum() / long.sum() * 100:.2f}% "
          f"[{lo * 100:.2f}, {hi * 100:.2f}]")
    print(f"  再加 `f₂ ≤ 0.3` 留下 {(keep_long & (f2 <= 0.3)).sum()} 个")
    print()
    print(f"窗 < 10 µs：{short.sum()} 个（{short.mean() * 100:.1f}%）。"
          f"f₃ 在这一档**不能用**：三重偶然期望中位 {np.median(d['exp_trip'][short]):.3g}")
    print(f"  `f₂ ≤ 0.3` 的绝对个数 {int((short & (f2 <= 0.3)).sum())} 个"
          f"（占该档 {(f2[short] <= 0.3).mean() * 100:.3f}%）；f₂ 中位 {np.median(f2[short]):.3f}")
    print(f"  「窗 < 10 µs 且 f₂ > 0.3」整段扣掉 = 扣 {int((short & (f2 > 0.3)).sum())} 个，"
          f"留 {int((short & (f2 <= 0.3)).sum())} 个")

    print()
    print("=" * 78)
    print("二、量程上限的准入口径（纯准入修正，不是新判据）")
    print("=" * 78)
    for tag, label in (("cut", "自适应包起点"), ("m5", "edge−5"), ("m10", "edge−10"), ("m15", "edge−15")):
        nc, bk = d["n_" + tag], d["b_" + tag]
        keep = nc >= MIN_NUMBER
        mean2 = np.where(d["n_bkg"] > 0, mean * bk / np.maximum(d["n_bkg"], 1), 0.0)
        fa2 = poisson.sf(nc, mean2) * (YEAR / bin_s)
        alive = keep & (fa2 <= FPY)
        print(f"{label:<14} 窗内事例存活 {nc.sum() / count.sum() * 100:5.1f}%  "
              f"本底事例存活 {bk.sum() / d['n_bkg'].sum() * 100:5.1f}%  "
              f"count ≥ 8 的 {int(keep.sum()):>6}  再要 fa ≤ 20 的 {int(alive.sum()):>6}"
              f"（{alive.mean() * 100:.2f}%）")

    print()
    hi_frac_win = d["n_hi"].sum() / count.sum()
    hi_frac_bkg = 1.0 - d["b_cut"].sum() / d["n_bkg"].sum()
    print(f"超量程包里的事例：候选窗内占 {hi_frac_win * 100:.2f}%，本底窗内占 {hi_frac_bkg * 100:.2f}%，"
          f"富集 {hi_frac_win / hi_frac_bkg:.2f} 倍")
    m2, m2hi, tot, hi_n = d["n_m2"].sum(), d["n_m2_hi"].sum(), count.sum(), d["n_hi"].sum()
    p_in = m2hi / m2 if m2 else float("nan")
    p_out = (hi_n - m2hi) / (tot - m2) if tot > m2 else float("nan")
    print(f"事例级 2×2（窗内，{int(tot)} 条）：在 ≥2 重同戳簇里的 {int(m2)} 条中 "
          f"{p_in * 100:.2f}% 落在包里；不在簇里的 {int(tot - m2)} 条中 {p_out * 100:.2f}%；"
          f"比值 {p_in / p_out:.3f}")
    print("候选级：按 bump_frac 分组看同戳量")
    bf = d["n_hi"] / count
    print(f"{'bump_frac':<12}{'n':>7}{'f₂中位':>8}{'f₃中位':>8}{'f₃>0 %':>9}{'窗长中位µs':>12}")
    for a, b in ((0, 1e-9), (1e-9, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)):
        sel = (bf >= a) & (bf < b)
        if sel.sum() < 5:
            continue
        print(f"{f'{a:g}–{b:g}':<12}{sel.sum():>7}{np.median(f2[sel]):>8.3f}{np.median(f3[sel]):>8.3f}"
              f"{(f3[sel] > 0).mean() * 100:>9.1f}{np.median(bin_us[sel]):>12.3f}")

    print()
    print("=" * 78)
    print("三、组合（双增益去重已在二进制里）")
    print("=" * 78)
    nc, bk = d["n_cut"], d["b_cut"]
    mean2 = np.where(d["n_bkg"] > 0, mean * bk / np.maximum(d["n_bkg"], 1), 0.0)
    fa2 = poisson.sf(nc, mean2) * (YEAR / bin_s)
    step1 = (nc >= MIN_NUMBER) & (fa2 <= FPY)
    f2c, f3c = d["f2_cut"], d["f3_cut"]
    step2 = step1 & np.where(long, f3c == 0, f2c <= 0.3)
    step3 = step2 & (f2c <= 0.3)
    for label, sel in (("① 原始（已去重）", np.ones(n, bool)),
                       ("② + 准入上界 edge−margin", step1),
                       ("③ + f₃==0(≥10µs) / f₂≤0.3(<10µs)", step2),
                       ("④ + f₂≤0.3 全档", step3)):
        k = int(sel.sum())
        print(f"{label:<34} {k:>7} 个  {k / EXPOSURE:8.4f} /s  {k / EXPOSURE * 86400:10.0f} /天  "
              f"183 天外推 {k / EXPOSURE * 86400 * DAYS / 1e6:7.3f} M")
    chance3 = (1 - np.exp(-d["exp_trip"][step1 & long])).sum()
    print(f"偶然期望：③ 里 `f₃>0` 这一刀在 ≥10 µs 档的偶然误否决期望 {chance3:.2f} 个／天")
    print(f"验收线 GECAM-B 0.54 个/天：④ 还差 {step3.sum() / EXPOSURE * 86400 / 0.54:.3g} 倍 "
          f"≈ {np.log10(max(step3.sum() / EXPOSURE * 86400 / 0.54, 1e-9)):.1f} 个数量级")

    print()
    print("幸存者形态（④）：")
    if step3.sum():
        s = step3
        print(f"  n={int(s.sum())}  count中位 {np.median(nc[s]):.0f}  窗长中位 {np.median(bin_us[s]):.2f} µs  "
              f"fa中位 {np.median(fa2[s]):.3g}  |lat|中位 {np.median(np.abs(d['lat'][s])):.1f}°  "
              f"n_det中位 {np.median(d['n_det_cut'][s]):.0f}")
        print(f"  其中 fa ≤ 0.01 的 {int((s & (fa2 <= 0.01)).sum())} 个；窗 ≥ 10 µs 的 {int((s & long).sum())} 个")


if __name__ == "__main__":
    main()
