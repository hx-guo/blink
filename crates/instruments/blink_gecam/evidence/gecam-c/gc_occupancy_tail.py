"""GECAM-C：实测占据数分布的尾概率 ÷ 泊松给出的尾概率。

**这条把"标称误报率作废"从定性的话变成逐仪器可量的数**：`fa` 是按泊松算的，
而事例流不泊松，于是同一个 `k` 上实测的尾概率比泊松大 N 倍——**论文里
"我们的 `fa` 是标称值乘以约 N" 要写得实，就得有这个 N。**

参照 GECAM-B 的那一点：`W = 10 µs`、λ = 0.101、k = 8 时**实测 80,639 格 vs
泊松 3.2e−4 ⇒ 2.5e8**。

三条口径（统筹定）：

* **报曲线不报单点**——比值随 `k` 剧烈变化；
* **至少报"候选 `k` 分布的中位处"那个值**，那才是对目录实际生效的数；
* **窗宽要标**，并给出比值对 `W` 的依赖。

**事例口径与搜索一致**：逐文件推准入道号窗 → 按 GTI 过滤 → 逐探头双增益去重。
**格子只在 GTI 段内切**，段尾不足一格的那一截丢掉（否则末格计数偏低、把尾压平）。

用法: gc_occupancy_tail.py <YYYY-MM-DD> <hour> [...]
"""

import glob
import os
import sys

import numpy as np
from astropy.io import fits
from scipy import stats

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_ENERGY_KEV, NORMAL = 40.0, 1
# 候选窗宽横跨五个数量级，所以比值要逐窗宽报。中位 bin_size_best 约 90 µs。
WIDTHS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
# C 星候选的 count 中位是 8–9（`min_number = 8` 是搜索的下限）。
MEDIAN_K = 9
# 逐段算 Fano 的段长。**整小时算出来的 Fano 掺着轨道尺度的速率漂移**，
# 而跨仪器比 Fano 时段长必须一致，否则比的不是同一个量。GECAM-B 用的是 10 s。
SEGMENT_SECONDS = float(os.environ.get("SEGMENT_SECONDS", "10"))


def ladder(hdus, min_energy_kev=MIN_ENERGY_KEV):
    eb = hdus["EBOUNDS"].data
    e_min = np.asarray(eb["E_MIN"], float)
    e_max = np.asarray(eb["E_MAX"], float)
    broken = np.flatnonzero(np.abs(e_min[1:] - e_max[:-1]) > e_max[:-1] * 1e-4)
    length = int(broken[0]) + 1 if broken.size else e_min.size
    above = np.flatnonzero(e_max[:length] > min_energy_kev)
    return (int(above[0]) if above.size else length), length


def newest(pattern):
    best = None
    for path in glob.glob(pattern):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) >= 5 and parts[-1].startswith("v"):
            try:
                v = int(parts[-1][1:])
            except ValueError:
                continue
            if best is None or v > best[0]:
                best = (v, path)
    return best[1] if best else None


def dedupe_mask(time, gain, dead_s):
    """一路探头内按死时间口径去重，返回"留下哪些"的布尔掩模。

    与 `gc_dedupe.adjacent` 同规则：只认时间序上相邻的一对，连续成段时从左到右
    贪心取的就是段起点开始隔一个的那些。丢掉高增益那条（`gain` 小的）。
    """
    keep = np.ones(time.size, bool)
    if time.size < 2:
        return keep
    ok = (gain[1:] != gain[:-1]) & (time[1:] - time[:-1] <= dead_s[:-1])
    if not ok.any():
        return keep
    idx = np.flatnonzero(ok)
    start = np.empty(idx.size, bool)
    start[0] = True
    start[1:] = idx[1:] != idx[:-1] + 1
    seg = np.maximum.accumulate(np.where(start, np.arange(idx.size), -1))
    picked = idx[(np.arange(idx.size) - seg) % 2 == 0]
    # 每对里丢掉高增益那条（gain 0），留低增益（gain 1）
    drop = np.where(gain[picked] < gain[picked + 1], picked, picked + 1)
    keep[drop] = False
    return keep


def load_hour(path):
    """与搜索同口径的事例流：准入 → 逐探头去重 → 合并排序。同时给出 GTI。"""
    times = []
    with fits.open(path, memmap=True) as hdus:
        min_channel, ladder_length = ladder(hdus)
        gti = hdus["GTI"].data
        gti_a = np.asarray(gti["START"], float)
        gti_b = np.asarray(gti["STOP"], float)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            if data is None or len(data) == 0:
                continue
            pi = np.asarray(data["PI"]).astype(np.int32)
            evt = np.asarray(data["EVT_TYPE"]).astype(np.int8)
            keep = (evt == NORMAL) & (pi >= min_channel) & (pi < ladder_length)
            if not keep.any():
                continue
            t = np.asarray(data["TIME"], float)[keep]
            order = np.argsort(t, kind="stable")
            t = t[order]
            g = np.asarray(data["GAIN_TYPE"])[keep][order].astype(np.int8)
            d = np.asarray(data["DEAD_TIME"])[keep][order].astype(float) * 1e-6
            times.append(t[dedupe_mask(t, g, d)])
    if not times:
        return None
    time = np.concatenate(times)
    time.sort()
    return time, gti_a, gti_b


def occupancy(time, gti_a, gti_b, width):
    """在 GTI 段内切等宽格，返回**占据数直方**（下标 = 每格的计数）与总格数。

    **不能把每格的计数实体化**：`W = 0.1 µs` 时 2,707 s 活时间就是 2.7e10 格，
    一个 int64 数组要 61 GB。而尾概率只需要两样东西——**非空格的占据数分布**
    （长度 = 事例数量级）与**总格数**（一个标量），空格子只进 `hist[0]`。

    段尾不足一格的那一截丢掉：否则末格计数偏低，会把尾压平。
    """
    hist = np.zeros(1, np.int64)
    total_bins = 0
    for a, b in zip(gti_a, gti_b):
        n_bins = int((b - a) // width)
        if n_bins < 1:
            continue
        total_bins += n_bins
        lo = np.searchsorted(time, a, "left")
        hi = np.searchsorted(time, a + n_bins * width, "left")
        if hi <= lo:
            continue
        idx = ((time[lo:hi] - a) / width).astype(np.int64)
        # idx 已按时间有序，相邻相等即同格 —— 直接数每段连续相等的长度
        _, per_bin = np.unique(idx, return_counts=True)
        segment = np.bincount(per_bin)
        if segment.size > hist.size:
            grown = np.zeros(segment.size, np.int64)
            grown[: hist.size] = hist
            hist = grown
        hist[: segment.size] += segment
        # 这一段里空着的格
        hist[0] += n_bins - per_bin.size
    return hist, total_bins


def tail_from_hist(hist, k):
    """直方给出的 `P(X >= k)` 的分子（格数）。"""
    return int(hist[k:].sum()) if k < hist.size else 0


def segmented_tail(time, gti_a, gti_b, width, segment_seconds, k_max=40):
    """逐段用**该段自己的 λ** 算泊松期望，再把各段加起来。

    **整小时一个 λ 是错的口径**：搜索用的是候选两侧各 0.5 s 的本底窗，λ 是
    **局部的**；而一小时里速率有轨道尺度的漂移，**拿全小时的 λ 去比，会把
    速率漂移算成过离散**，把尾比系统性高估。分段之后每段内速率近似平稳，
    剩下的才是真正的成团。

    返回 `(实测 ≥k 的格数, 局部泊松期望的格数, 总格数)` 三个数组，下标即 `k`。
    """
    observed = np.zeros(k_max + 1)
    expected = np.zeros(k_max + 1)
    total_bins = 0
    ks = np.arange(k_max + 1)
    for a, b in zip(gti_a, gti_b):
        start = a
        while start + segment_seconds <= b:
            n_bins = int(segment_seconds // width)
            if n_bins >= 2:
                total_bins += n_bins
                lo = np.searchsorted(time, start, "left")
                hi = np.searchsorted(time, start + n_bins * width, "left")
                lam = (hi - lo) / n_bins
                if hi > lo:
                    idx = ((time[lo:hi] - start) / width).astype(np.int64)
                    _, per_bin = np.unique(idx, return_counts=True)
                    hist = np.bincount(per_bin, minlength=k_max + 1)[: k_max + 1]
                    observed += hist[::-1].cumsum()[::-1]
                expected += n_bins * stats.poisson.sf(ks - 1, lam)
            start += segment_seconds
    return observed, expected, total_bins


def segmented_fano(time, gti_a, gti_b, width, segment_seconds):
    """把活时间切成 `segment_seconds` 的段，逐段算 Fano（方差/均值），返回中位。

    **整小时算出来的 Fano 不能直接当"成团强度"**：一小时里速率有轨道尺度的漂移，
    **速率非平稳本身就会抬高 Fano，而且越往大 `W` 越抬**。复合泊松在
    `W ≫ 簇宽`（实测簇宽 p50 = 60 ns / p99 = 149 ns）之后 Fano 应当进平台、
    不该一路涨；**涨了就要先排除段长**。

    Fano 只需要一阶与二阶矩，不必把每格的计数实体化：
    `mean = S1/n`、`var = S2/n − mean²`，其中 `S2` 只有非空格有贡献。
    """
    values = []
    for a, b in zip(gti_a, gti_b):
        start = a
        while start + segment_seconds <= b:
            stop = start + segment_seconds
            n_bins = int(segment_seconds // width)
            if n_bins >= 2:
                lo = np.searchsorted(time, start, "left")
                hi = np.searchsorted(time, start + n_bins * width, "left")
                s1 = hi - lo
                if s1 > 0:
                    idx = ((time[lo:hi] - start) / width).astype(np.int64)
                    _, per_bin = np.unique(idx, return_counts=True)
                    s2 = float((per_bin.astype(float) ** 2).sum())
                    mean = s1 / n_bins
                    var = s2 / n_bins - mean * mean
                    if mean > 0:
                        values.append(var / mean)
            start = stop
    return (float(np.median(values)), len(values)) if values else (float("nan"), 0)


def main():
    args = sys.argv[1:]
    for day, hour in zip(args[::2], args[1::2]):
        path = newest(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/gcg_evt_*_{hour}_v*.fits")
        if path is None:
            print(f"{day} {hour}h 无文件")
            continue
        loaded = load_hour(path)
        if loaded is None:
            print(f"{day} {hour}h 无事例")
            continue
        time, gti_a, gti_b = loaded
        live = float(np.clip(gti_b - gti_a, 0, None).sum())
        print(f"\n===== {os.path.basename(path)}  去重后事例 {time.size:,}  "
              f"GTI {live:.0f} s  全局率 {time.size / live:.0f} c/s =====")
        for width in WIDTHS:
            hist, total_bins = occupancy(time, gti_a, gti_b, width)
            if total_bins == 0:
                continue
            k_values = np.arange(hist.size)
            lam = float((hist * k_values).sum()) / total_bins
            variance = float((hist * (k_values - lam) ** 2).sum()) / total_bins
            fano_whole = variance / max(lam, 1e-12)
            fano_seg, n_seg = segmented_fano(time, gti_a, gti_b, width, SEGMENT_SECONDS)
            print(f"\n  窗宽 W = {width * 1e6:g} µs   格数 {total_bins:,}   λ = {lam:.4g}")
            print(f"    Fano：整小时 {fano_whole:.3f}   "
                  f"逐 {SEGMENT_SECONDS:g} s 段的中位 {fano_seg:.3f}"
                  f"（{n_seg} 段）   泊松应为 1")
            obs_seg, exp_seg, bins_seg = segmented_tail(
                time, gti_a, gti_b, width, SEGMENT_SECONDS
            )
            print("    k    实测 ≥k 的格数   全小时单一 λ 的比值   "
                  f"逐 {SEGMENT_SECONDS:g} s 局部 λ 的比值")
            shown = 0
            for k in range(2, 41):
                n_ge = tail_from_hist(hist, k)
                if n_ge == 0:
                    break
                whole = (n_ge / total_bins) / float(stats.poisson.sf(k - 1, lam))
                local = (
                    obs_seg[k] / exp_seg[k]
                    if k < exp_seg.size and exp_seg[k] > 0
                    else float("nan")
                )
                mark = "  ← 候选 k 中位" if k == MEDIAN_K else ""
                print(f"   {k:3d}   {n_ge:14,}   {whole:19.3g}   {local:19.3g}{mark}")
                shown += 1
                if shown >= 12 and k >= MEDIAN_K:
                    break


if __name__ == "__main__":
    main()
