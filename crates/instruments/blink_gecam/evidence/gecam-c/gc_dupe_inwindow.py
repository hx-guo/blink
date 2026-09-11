"""GECAM-C：双增益去重在**候选窗内**的偶然配对率。

全小时的偶然配对率是 4.16%（平移对照量出来的）。但候选窗是按"事例挤在一起"选出来的，
窗内密度比全小时高几个数量级，**偶然配对率在那里必然更高**——而那才是"去重把多少真
事例误并了"的数。这个数不能从全小时那个数推，要单独量。

办法与全小时那一版同构，只是把统计范围收到候选窗内：

* **真**：按死时间口径在原始流上配对，数有多少对的时戳落在某个候选窗里；
* **偶然**：把低增益支路的时戳整体平移一个远大于死时间的常数，重排后再配一次，
  数有多少对落在**同一批候选窗**里。平移之后配到的对全是偶然的。

平移量取几个不同值，看结果稳不稳——单个平移量可能撞上周期性结构。

用法: gc_dupe_inwindow.py <signals.json> [小时数上限]
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
SHIFTS = (0.11, 0.37, 1.03)

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


def pair_times(time, gain, dead_s):
    """相邻口径配对（与 gc_dedupe.adjacent 同规则），返回每对的时戳（取靠前那条）。"""
    if time.size < 2:
        return np.empty(0)
    ok = (gain[1:] != gain[:-1]) & (time[1:] - time[:-1] <= dead_s[:-1])
    if not ok.any():
        return np.empty(0)
    idx = np.flatnonzero(ok)
    start = np.empty(idx.size, bool)
    start[0] = True
    start[1:] = idx[1:] != idx[:-1] + 1
    seg = np.maximum.accumulate(np.where(start, np.arange(idx.size), -1))
    picked = idx[(np.arange(idx.size) - seg) % 2 == 0]
    return time[picked]


def count_in_windows(times, lows, highs):
    """有多少个 times 落在任一 [low, high] 里。窗互不重叠且已排序。"""
    if times.size == 0:
        return 0
    i = np.searchsorted(highs, times, "left")
    ok = i < lows.size
    hit = np.zeros(times.size, bool)
    hit[ok] = times[ok] >= lows[i[ok]]
    return int(hit.sum())


def main():
    signals = json.load(open(sys.argv[1]))
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    by_hour = {}
    for signal in signals:
        by_hour.setdefault(signal["start"][:13], []).append(signal)

    total_win = 0
    real = 0
    null = {s: 0 for s in SHIFTS}
    total_window_seconds = 0.0
    for hour_key in sorted(by_hour)[:limit]:
        group = by_hour[hour_key]
        iso = group[0]["start"]
        path = newest(f"{ROOT}/{iso[:10].replace('-', '/')}/GRD_EVT/gcg_evt_*_{iso[11:13]}_v*.fits")
        if path is None:
            continue
        lows = np.array([met(s["start"]) + s["delay"] for s in group])
        highs = lows + np.array([s["bin_size_best"] for s in group])
        order = np.argsort(lows)
        lows, highs = lows[order], highs[order]
        # 重叠窗合并，count_in_windows 的前提是互不重叠
        merged_lo, merged_hi = [], []
        for a, b in zip(lows, highs):
            if merged_hi and a <= merged_hi[-1]:
                merged_hi[-1] = max(merged_hi[-1], b)
            else:
                merged_lo.append(a)
                merged_hi.append(b)
        lows = np.array(merged_lo)
        highs = np.array(merged_hi)
        total_win += lows.size
        total_window_seconds += float((highs - lows).sum())

        with fits.open(path, memmap=True) as hdus:
            min_channel, ladder_length = ladder(hdus)
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
                o = np.argsort(t, kind="stable")
                t = t[o]
                gain = np.asarray(data["GAIN_TYPE"])[keep][o].astype(np.int8)
                dead = np.asarray(data["DEAD_TIME"])[keep][o].astype(float) * 1e-6
                real += count_in_windows(pair_times(t, gain, dead), lows, highs)
                for shift in SHIFTS:
                    st = t.copy()
                    st[gain == 1] += shift
                    o2 = np.argsort(st, kind="stable")
                    null[shift] += count_in_windows(
                        pair_times(st[o2], gain[o2], dead[o2]), lows, highs
                    )
        print(f"  {hour_key}  窗 {lows.size}  真 {real}  偶然 "
              + "/".join(str(null[s]) for s in SHIFTS), flush=True)

    print(f"\n候选窗 {total_win} 个，合计窗长 {total_window_seconds*1e3:.3f} ms")
    print(f"窗内真配对 {real}")
    for shift in SHIFTS:
        n = null[shift]
        frac = n / real if real else float("nan")
        # 二项标准误（偶然数本身是泊松，用 sqrt(n)）
        err = np.sqrt(n) / real if real else float("nan")
        print(f"  平移 {shift:5.2f} s：偶然 {n:6d}  →  窗内偶然占比 "
              f"{frac*100:6.2f}% ± {err*100:.2f}%")
    mean_null = np.mean([null[s] for s in SHIFTS])
    print(f"\n**三个平移量的均值：窗内偶然配对占比 {mean_null/real*100:.2f}%**"
          f"（全小时那个数是 4.16%）")


if __name__ == "__main__":
    main()
