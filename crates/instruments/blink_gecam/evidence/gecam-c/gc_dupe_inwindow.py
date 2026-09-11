"""GECAM-C：双增益去重与跨探头合并在**候选窗内**的偶然配对率。

全小时的偶然配对率是 4.16%（低增益支路整体平移的对照量出来的）。**那个数在
候选窗里用不了**：候选窗是按"事例挤在一起"选出来的，窗内密度比全小时高几个
数量级，而偶然配对率随密度走。窗内才是"去重把多少真事例误并了"的所在。

**平移对照在窗内失效，不是精度问题是系统问题。** 平移 0.1–1 s 之后，候选窗里
剩下的是暴发的高增益那一半加上别处来的低增益本底——窗内密度被平移本身破坏了，
量到的是本底密度下的偶然率。所以这里换成**窗内均匀重抽**：

* **真**：取窗内通过准入的原始事例，按死时间口径配对（与 `gc_dedupe.adjacent`
  同规则，逐探头做），数两条都落在窗内的对；
* **偶然**：把这些事例的时戳在同一个窗内**均匀重抽**，标签（探头、增益、
  死时间）原样跟着事例走，再配一次。事例数、窗长、探头与增益的边缘分布全部
  原样保留，**只有时间上的符合结构被打掉**，配到的对因此全是偶然的。

**这个偶然数是下界，方向是单向的**：重抽假定事例在窗内均匀，而候选窗是最佳格、
窗内本身可能还有更陡的结构；真结构越陡，偶然配对只会更多。下界有多松，用
**密度对照**量出来而不是猜：同一批事例改撒在半个窗（密度 ×2）与十个窗
（密度 ÷10）里各跑一遍。十倍那一档同时是这个估计量的灵敏度检验——**偶然数
不随密度动的话，这个测量根本没有分辨力**。

同一套口径顺带量跨探头合并：**τ = 150 ns 固定**（底下是双增益两支路之间一个
约 100 ns 的电子学固定延迟，量化步 q 只决定能不能分辨开，所以 τ 不随 q 缩）。

用法: gc_dupe_inwindow.py <signals.json> [重抽次数=200] [小时数上限] [能阈道,梯长]

最后一个参数是**准入口径的覆盖**，只用来做对照：把它设成 `54,448` 在 896 道
纪元的小时上跑，对账会掉下来——那正是"照搬道号"错在哪里的正面证据。
"""

import datetime as dt
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = (2021, 1, 1)
# 能阈存能量、切的时候折成道号，折算逐文件做——见 `ladder()`。写死道号不行：
# 归档里有两把能量梯，同一个道号在它们上面差一倍能量。
MIN_ENERGY_KEV, NORMAL = 40.0, 1
# 跨探头合并窗。**常数，不随量化步 q 变**：底下是一个约 100 ns 的电子学固定
# 延迟，q 只决定它落在哪一格（C 星最细那段 q = 7.45 ns 上是 96.9 / 104.3 ns
# 相邻两格）。若让 τ 随 q 缩到 30–60 ns，连那条线都盖不住。
CROSS_DETECTOR_TAU = 150e-9


def ladder(hdus, min_energy_kev=MIN_ENERGY_KEV):
    """从这个文件自己的 EBOUNDS 推出 (能阈道, 梯长)，与 Rust 侧同规则。

    **梯长 = 从 ch0 起 `E_MAX[k-1] == E_MIN[k]` 连续到断开为止的长度**（470 行
    那版在 ch448 断，896 行那版一路到表尾），**能阈道 = 梯上第一个上边界越过
    `min_energy_kev` 的道**（470 版 ch54、896 版 ch109）。两个都不能写死：
    GECAM-C 2022-08-03 .. 10-15 那 1,121 小时是细梯，同一个道号差一倍能量。
    """
    eb = hdus["EBOUNDS"].data
    e_min = np.asarray(eb["E_MIN"], float)
    e_max = np.asarray(eb["E_MAX"], float)
    broken = np.flatnonzero(np.abs(e_min[1:] - e_max[:-1]) > e_max[:-1] * 1e-4)
    length = int(broken[0]) + 1 if broken.size else e_min.size
    above = np.flatnonzero(e_max[:length] > min_energy_kev)
    return (int(above[0]) if above.size else length), length


def met(iso):
    """ISO → MET。小数秒整取，不能截到微秒——窗最短只有 0.015 µs。"""
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def newest(pattern):
    best = None
    for path in glob.glob(pattern):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) >= 5 and parts[-1].startswith("v"):
            try:
                version = int(parts[-1][1:])
            except ValueError:
                continue
            if best is None or version > best[0]:
                best = (version, path)
    return best[1] if best else None


def pairs_one_detector(time, gain, dead_s):
    """一路探头内按死时间口径配对，返回配到的对数。

    与 `gc_dedupe.adjacent` 同规则：只认时间序上相邻的一对，一段连续的候选对里
    从左到右贪心取的就是起点开始隔一个的那些。**同探头**是硬前提——Rust 侧
    `dedupe_gain_pairs` 要求 `detector_id` 相同，所以这个函数一次只喂一路。
    """
    if time.size < 2:
        return 0
    ok = (gain[1:] != gain[:-1]) & (time[1:] - time[:-1] <= dead_s[:-1])
    if not ok.any():
        return 0
    idx = np.flatnonzero(ok)
    start = np.empty(idx.size, bool)
    start[0] = True
    start[1:] = idx[1:] != idx[:-1] + 1
    seg = np.maximum.accumulate(np.where(start, np.arange(idx.size), -1))
    return int(((np.arange(idx.size) - seg) % 2 == 0).sum())


def merges_in_window(time, detector, gain, dead_s):
    """窗内（逐探头）配到的对数。输入必须已按时间排好。"""
    total = 0
    for unit in np.unique(detector):
        pick = detector == unit
        total += pairs_one_detector(time[pick], gain[pick], dead_s[pick])
    return total


def cross_detector_pairs(time, detector, tau=CROSS_DETECTOR_TAU):
    """相邻且**跨探头**、间隔 ≤ τ 的对数。输入必须已按时间排好。"""
    if time.size < 2:
        return 0
    return int(
        ((time[1:] - time[:-1] <= tau) & (detector[1:] != detector[:-1])).sum()
    )


def load_hour(path, override=None):
    """一小时的准入事例，按时间排好：(时刻, 探头, 增益, 死时间秒)。"""
    times, units, gains, deads = [], [], [], []
    with fits.open(path, memmap=True) as hdus:
        min_channel, ladder_length = override or ladder(hdus)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            if data is None or len(data) == 0:
                continue
            pi = np.asarray(data["PI"]).astype(np.int16)
            evt = np.asarray(data["EVT_TYPE"]).astype(np.int8)
            keep = (evt == NORMAL) & (pi >= min_channel) & (pi < ladder_length)
            if not keep.any():
                continue
            t = np.asarray(data["TIME"], float)[keep]
            order = np.argsort(t, kind="stable")
            times.append(t[order])
            units.append(np.full(order.size, int(hdu.name[-2:]), np.int16))
            gains.append(np.asarray(data["GAIN_TYPE"])[keep][order].astype(np.int8))
            deads.append(np.asarray(data["DEAD_TIME"])[keep][order].astype(float) * 1e-6)
    if not times:
        return None
    time = np.concatenate(times)
    order = np.argsort(time, kind="stable")
    return (
        time[order],
        np.concatenate(units)[order],
        np.concatenate(gains)[order],
        np.concatenate(deads)[order],
    )


# 偶然重抽的密度对照：把同一批事例撒在窗长的这些倍数里。1.0 是本体，
# 0.5 把密度翻倍（问"窗内结构更陡会怎样"），10.0 把密度降十倍（灵敏度检验）。
SPREADS = (1.0, 0.5, 10.0)


def main():
    signals = json.load(open(sys.argv[1]))
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    override = None
    if len(sys.argv) > 4:
        override = tuple(int(x) for x in sys.argv[4].split(","))
    rng = np.random.default_rng(20260911)

    by_hour = {}
    for signal in signals:
        by_hour.setdefault(signal["start"][:13], []).append(signal)

    windows = 0
    raw_events = 0            # 窗内通过准入的原始事例（去重前）
    real_merges = 0           # 窗内真配到的对
    real_cross = 0            # 窗内跨探头 ≤ τ 的对
    # 同样窗内、时戳均匀重抽后配到的对（偶然），逐密度档
    null_merges = {s: 0.0 for s in SPREADS}
    null_cross = {s: 0.0 for s in SPREADS}
    window_seconds = 0.0
    reconciled = 0            # 去重后窗内事例数 == 搜索报的 count 的窗数

    for hour_key in sorted(by_hour)[:limit]:
        group = by_hour[hour_key]
        iso = group[0]["start"]
        path = newest(
            f"{ROOT}/{iso[:10].replace('-', '/')}/GRD_EVT/gcg_evt_*_{iso[11:13]}_v*.fits"
        )
        if path is None:
            print(f"  {hour_key} 无文件", flush=True)
            continue
        hour = load_hour(path, override)
        if hour is None:
            print(f"  {hour_key} 无事例", flush=True)
            continue
        time, unit, gain, dead = hour

        hour_windows = hour_real = hour_raw = 0
        hour_null = 0.0
        for signal in group:
            t0 = met(signal["start"]) + signal["delay"]
            t1 = t0 + signal["bin_size_best"]
            lower = np.searchsorted(time, t0, "left")
            upper = np.searchsorted(time, t1, "right")
            if upper - lower < 2:
                continue
            w_time = time[lower:upper]
            w_unit = unit[lower:upper]
            w_gain = gain[lower:upper]
            w_dead = dead[lower:upper]
            n = w_time.size

            windows += 1
            hour_windows += 1
            raw_events += n
            hour_raw += n
            window_seconds += t1 - t0

            real = merges_in_window(w_time, w_unit, w_gain, w_dead)
            real_merges += real
            hour_real += real
            real_cross += cross_detector_pairs(w_time, w_unit)
            # 去重后窗内还剩几条，与搜索报的 count 对账
            if n - real == signal["count"]:
                reconciled += 1

            # 偶然：时戳在同一个窗内均匀重抽，标签跟着事例走
            for spread in SPREADS:
                width = (t1 - t0) * spread
                null_here = cross_here = 0
                for _ in range(draws):
                    drawn = rng.uniform(t0, t0 + width, n)
                    order = np.argsort(drawn, kind="stable")
                    null_here += merges_in_window(
                        drawn[order], w_unit[order], w_gain[order], w_dead[order]
                    )
                    cross_here += cross_detector_pairs(drawn[order], w_unit[order])
                null_merges[spread] += null_here / draws
                null_cross[spread] += cross_here / draws
                if spread == 1.0:
                    hour_null += null_here / draws

        print(
            f"  {hour_key}  窗 {hour_windows:5d}  原始事例 {hour_raw:6d}  "
            f"真并 {hour_real:5d}  偶然 {hour_null:8.1f}",
            flush=True,
        )

    if not windows:
        print("没有可用的窗")
        return

    if override:
        print(f"\n⚠ 准入口径被覆盖成 能阈道 {override[0]} / 梯长 {override[1]}（对照用）")
    print(f"\n候选窗 {windows} 个，合计窗长 {window_seconds * 1e3:.3f} ms")
    print(f"对账：去重后窗内事例数 == 搜索报的 count 的窗 {reconciled}/{windows}"
          f" = {reconciled / windows * 100:.2f}%")
    print(f"窗内密度 {raw_events / window_seconds:.3e} c/s"
          f"（窗内原始事例 {raw_events}，平均每窗 {raw_events / windows:.2f} 条）")

    for title, real, null in (
        ("双增益去重（同探头、死时间窗内、一高一低）", real_merges, null_merges),
        (f"跨探头合并 τ = {CROSS_DETECTOR_TAU * 1e9:.0f} ns", real_cross, null_cross),
    ):
        print(f"\n{title}")
        print(f"  窗内真配到的对   {real}")
        for spread in SPREADS:
            label = {1.0: "本体", 0.5: "半窗（密度 ×2）", 10.0: "十窗（密度 ÷10）"}[spread]
            share = null[spread] / real * 100 if real else float("nan")
            print(f"  偶然 · {label:16s} {null[spread]:9.1f}   占真配对 {share:6.2f}%")

    print(f"\n双增益去重每窗误并 {null_merges[1.0] / windows:.4f} 条（本体档），"
          f"窗内计数因此被压低 "
          f"{null_merges[1.0] / max(raw_events - real_merges, 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
