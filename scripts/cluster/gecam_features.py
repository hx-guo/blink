"""GECAM 候选的事例级特征：同戳簇集、探头铺开程度、能谱、以及多个宽度下的 CPD 符合。

搜索一天出几千个候选，而配置的假阳性率是每年 20 个，差了四个数量级——多出来的
不是统计涨落，是泊松独立性被破坏了。天格上同一个现象的根因是带电粒子穿过整星：
各路探测器在同一个时间戳上各留一个计数，八个"独立"计数其实是一次事件。这里量
的就是这件事，量完再决定要不要立判据。

事例口径必须与搜索一致：先准入、后合并双增益重复（见 `read_events`）。任何
"回到事例流重算候选窗内的量"的脚本，第一件事都是把重算的计数与搜索报的
`count` 逐条对账，不到 100% 就先修口径，别信重算出来的任何数。

CPD 用多个窗宽量，是因为候选窗常常只有十几微秒：GECAM-C 两路 CPD 合计约 300 c/s，
10 µs 里期望 0.003 个，看见 0 个什么也说明不了。窗要宽到期望计数有意义为止。

用法: python3 gecam_features.py <signals.json> <输出 CSV> [卫星=GECAM-C]
"""

import csv
import glob
import json
import os
import sys
import datetime as dt

import numpy as np
from astropy.io import fits

ARCHIVE = {
    "GECAM-A": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_A", "GRD_evt", "CPD_evt", "gag", "gac"),
    "GECAM-B": ("/gecamfs/Archived-DATA/GSDC/LEVEL1/daily", "GECAM_B", "GRD_evt", "CPD_evt", "gbg", "gbc"),
    "GECAM-C": ("/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily", None, "GRD_EVT", "CPD_EVT", "gcg", "gcc"),
}
EPOCH = {"GECAM-A": (2019, 1, 1), "GECAM-B": (2019, 1, 1), "GECAM-C": (2021, 1, 1)}
# 事例准入，与 Rust 侧 `Event::keep` 一致
MIN_CHANNEL, OVERFLOW_CHANNEL, NORMAL_EVT_TYPE = 54, 448, 1
# CPD 符合的几个窗宽（秒），从候选窗两端各外扩这么多
CPD_HALF_WIDTHS = (1e-5, 1e-4, 1e-3, 1e-2)


def met(iso, satellite):
    """ISO 时刻折成 MET。

    小数秒必须整取，不能截到微秒。GECAM 的时戳分辨率是 0.03 µs，候选窗最短
    也是 0.03 µs，`datetime` 只到微秒，截断一次的误差就能有窗宽的几十倍——
    窗口整个错位，窗内事例捞不全（实测截断版重算的计数 99.2% 少于搜索报的，
    1–10 µs 的窗只捞回 62%）。所以整秒部分交给 `datetime`，小数部分单独按
    浮点加回去。
    """
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH[satellite], tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def hour_file(satellite, iso, kind):
    root, sat_dir, grd, cpd, grd_prefix, cpd_prefix = ARCHIVE[satellite]
    day = iso[:10].replace("-", "/")
    subdir, prefix = (grd, grd_prefix) if kind == "grd" else (cpd, cpd_prefix)
    directory = os.path.join(root, day, sat_dir, subdir) if sat_dir else os.path.join(root, day, subdir)
    stem = f"{prefix}_evt_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}_v"
    files = sorted(f for f in glob.glob(f"{directory}/{stem}*.fits"))
    return files[-1] if files else None


def read_events(path, want_pi):
    """把一个小时里各路事例合成按时间排好的 (时刻, 能道, 探头号)。

    **顺序是先准入、后合并双增益，不能反过来。** 同一个物理光子会被 ADC 的
    两个增益支路各写一行（见 crate 的 OPEN-QUESTIONS 第 17 条），搜索侧在
    准入之后把同探头同时戳的两条并成一条。这里必须照同一个顺序做，否则重算
    的窗内计数比搜索报的 count 系统性偏多——实测 A 星 2024-01-11 不合并时
    对账只有 12.83%（差值中位 +2、从不为负），先合并后准入是 99.90%，
    **先准入后合并才是 100.00%**，8,343 个候选逐条相等。

    一个 EVENTS 表就是一路探头，所以表内的同时戳天然就是同探头同时戳。
    保留高 `GAIN_TYPE`（低增益）那条，与 Rust 侧 `dedupe_gain_pairs` 同规则：
    高增益支路会饱和，低增益那条永远是有效测量。
    """
    times, channels, detectors = [], [], []
    with fits.open(path) as hdus:
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            keep = np.asarray(data["EVT_TYPE"]) == NORMAL_EVT_TYPE
            pi = np.asarray(data["PI"]) if want_pi else None
            if want_pi:
                keep &= (pi >= MIN_CHANNEL) & (pi < OVERFLOW_CHANNEL)
            time = np.asarray(data["TIME"], float)[keep]
            channel = pi[keep] if want_pi else np.zeros(int(keep.sum()), int)
            # CPD 没有 GAIN_TYPE 这一列，也没有双增益支路，不做合并
            gain = np.asarray(data["GAIN_TYPE"])[keep] if "GAIN_TYPE" in data.names else None
            if gain is not None and time.size:
                # 先按时戳排、同时戳内按增益档从高到低排，每个时戳留第一条
                order = np.lexsort((-gain.astype(np.int16), time))
                time, channel = time[order], channel[order]
                duplicate = np.zeros(time.size, bool)
                duplicate[1:] = time[1:] == time[:-1]
                time, channel = time[~duplicate], channel[~duplicate]
            times.append(time)
            channels.append(channel)
            detectors.append(np.full(time.size, int(hdu.name[-2:])))
    if not times:
        return None
    time = np.concatenate(times)
    order = np.argsort(time)
    return time[order], np.concatenate(channels)[order], np.concatenate(detectors)[order]


def span(time, low, high):
    """已排序时刻里落在 [low, high] 的下标区间。

    候选一多，布尔掩模就不像话了：A 星一天十万个候选 × 一小时四千万个事例
    是 1e13 次元素操作，B 星单小时 6.6 亿事例更甚。二分是一样的语义。
    """
    return int(np.searchsorted(time, low, "left")), int(np.searchsorted(time, high, "right"))


def main():
    signals_glob, out_path = sys.argv[1], sys.argv[2]
    satellite = sys.argv[3] if len(sys.argv) > 3 else "GECAM-C"

    signals = []
    for path in sorted(glob.glob(signals_glob)):
        signals.extend(json.load(open(path)))
    # 按时刻排一下，一小时的候选才会连着走，事例缓存只需留当前小时
    signals.sort(key=lambda signal: signal["start"])
    print(f"候选 {len(signals)} 个")

    writer = csv.writer(open(out_path, "w", newline=""))
    writer.writerow(
        ["start", "fa", "count", "mean", "bin_us", "lon", "lat", "n_core", "n_det_hit",
         "det_frac_max", "max_mult", "simul_frac", "n_multiplets", "multiplet_frac",
         "min_dt_us", "pi_core_med", "pi_bkg_med", "pi_ratio", "rate_win"]
        + [f"cpd_{int(w * 1e6)}us" for w in CPD_HALF_WIDTHS]
    )

    grd_cache, cpd_cache = {}, {}
    written = 0
    for signal in signals:
        iso = signal["start"]
        t0 = met(iso, satellite) + signal["delay"]
        t1 = t0 + signal["bin_size_best"]
        key = iso[:13]

        if key not in grd_cache:
            path = hour_file(satellite, iso, "grd")
            grd_cache[key] = read_events(path, want_pi=True) if path else None
            path = hour_file(satellite, iso, "cpd")
            cpd_cache[key] = read_events(path, want_pi=False) if path else None
        grd, cpd = grd_cache[key], cpd_cache[key]
        if grd is None:
            continue
        time, channel, detector = grd

        first, last = span(time, t0, t1)
        n_core = last - first
        if n_core == 0:
            continue
        core = slice(first, last)
        # 同戳簇。天格只有 4 路，一个粒子把 4 路全点亮，"最长一串"就够用；GECAM
        # 有 12 路，一个候选窗里往往有好几个各自独立的小簇（实测某候选在
        # 58.71 µs 上两个、58.81 µs 上两个），最长串只有 2，除以 8 就成了 0.25，
        # 看着像"不同戳"。所以要看的是**参与任何同戳簇的事例占多少**，
        # 而不是最长的那一串。
        #
        # `read_events` 已经把同探头的双增益重复并掉了，所以这里剩下的同戳
        # **只有跨探头的**——那才是"一个粒子穿过整台仪器"该留下的签名。合并
        # 之前算出来的 `multiplet_frac` 把双增益重复也算了进去，系统性偏高，
        # 两个口径的数不能混用。
        _, multiplicity = np.unique(time[core], return_counts=True)
        max_mult = int(multiplicity.max())
        n_multiplets = int((multiplicity >= 2).sum())
        multiplet_frac = float(multiplicity[multiplicity >= 2].sum()) / n_core
        _, det_counts = np.unique(detector[core], return_counts=True)
        gaps = np.diff(time[core])
        positive = gaps[gaps > 0]
        min_dt_us = positive.min() * 1e6 if positive.size else 0.0

        far_first, far_last = span(time, t0 - 1.0, t1 + 1.0)
        hole_first, hole_last = span(time, t0 - 0.01, t1 + 0.01)
        near = np.concatenate(
            (channel[far_first:hole_first], channel[hole_last:far_last])
        )
        pi_bkg = float(np.median(near)) if near.size else np.nan
        pi_core = float(np.median(channel[core]))
        window_first, window_last = span(time, t0 - 0.5, t1 + 0.5)

        cpd_counts = []
        for half in CPD_HALF_WIDTHS:
            if cpd is None:
                cpd_counts.append("")
                continue
            lower, upper = span(cpd[0], t0 - half, t1 + half)
            cpd_counts.append(upper - lower)

        writer.writerow(
            [iso[:23], f"{signal['false_positive_per_year']:.3e}", signal["count"],
             f"{signal['mean']:.5f}", f"{signal['bin_size_best'] * 1e6:.2f}",
             f"{signal['position']['longitude']:.3f}", f"{signal['position']['latitude']:.3f}",
             n_core, len(det_counts), f"{det_counts.max() / n_core:.3f}", max_mult,
             f"{max_mult / n_core:.3f}", n_multiplets, f"{multiplet_frac:.3f}",
             f"{min_dt_us:.3f}", f"{pi_core:.0f}",
             f"{pi_bkg:.0f}" if np.isfinite(pi_bkg) else "",
             f"{pi_core / pi_bkg:.3f}" if np.isfinite(pi_bkg) and pi_bkg > 0 else "",
             f"{(window_last - window_first) / (1.0 + signal['bin_size_best']):.0f}"]
            + cpd_counts
        )
        written += 1
        # 一小时的事例几千万，缓存只留当前小时
        if len(grd_cache) > 1:
            grd, cpd = grd_cache[key], cpd_cache.get(key)
            grd_cache.clear()
            cpd_cache.clear()
            grd_cache[key], cpd_cache[key] = grd, cpd

    print("写出", written)


if __name__ == "__main__":
    main()
