"""GECAM 双增益去重的两个实现，以及它们之间差多少。

Rust 侧 `dedupe_gain_pairs` 的语义是**同探头 + 死时间窗内 + 增益一高一低**，从早到晚
贪心，每条最多配一次，保留低增益那条。直译成 Python 是个逐事例的循环——一小时 12 路
两千万个事例要跑十分钟量级，655 个小时就是一星期，不能用。

所以这里有两个实现：

* `greedy`  —— 逐事例直译，与 Rust 同语义，慢，用来当基准；
* `adjacent`—— 只认**相邻**的一对（时间序上挨着），全向量化，快。

`adjacent` 不是近似得没道理：双增益对的两条记录之间几乎不会插进第三个事例（单路
约 460 c/s，死时间窗 4–204 µs 里期望 0.002–0.09 个）。但"几乎"要量，不能假设——
`compare()` 就是量它的，两者的差直接报出来。

相邻口径下"从左到右贪心"有精确解：在一段连续的候选对里（对 (i,i+1)、(i+1,i+2)…
都成立），贪心取的就是起点开始隔一个的那些，不必迭代。

用法: gc_dedupe.py <YYYY-MM-DD> <hour>   —— 逐路比两个实现
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"


def greedy(time, gain, dead_s):
    """与 Rust `dedupe_gain_pairs` 同语义。返回要丢掉的下标（高增益那条）。"""
    n = time.size
    taken = np.zeros(n, dtype=bool)
    drop = []
    for a in range(n):
        if taken[a]:
            continue
        deadline = time[a] + dead_s[a]
        b = a + 1
        while b < n and time[b] <= deadline:
            if not taken[b] and gain[b] != gain[a]:
                taken[a] = taken[b] = True
                drop.append(a if gain[a] < gain[b] else b)
                break
            b += 1
    return np.array(sorted(drop), dtype=np.int64)


def adjacent(time, gain, dead_s):
    """只认相邻的一对，全向量化。返回要丢掉的下标（高增益那条）。"""
    n = time.size
    if n < 2:
        return np.empty(0, dtype=np.int64)
    ok = (gain[1:] != gain[:-1]) & (time[1:] - time[:-1] <= dead_s[:-1])
    if not ok.any():
        return np.empty(0, dtype=np.int64)
    # 候选对的下标 i 表示 (i, i+1)。连续成段的候选对里，从左到右贪心取的是
    # 段起点开始隔一个的那些。
    idx = np.flatnonzero(ok)
    # 段起点：前一个候选对下标不是 idx-1
    start = np.empty(idx.size, dtype=bool)
    start[0] = True
    start[1:] = idx[1:] != idx[:-1] + 1
    # 每个候选对在自己段内的偏移
    seg_start_pos = np.maximum.accumulate(np.where(start, np.arange(idx.size), -1))
    offset = np.arange(idx.size) - seg_start_pos
    picked = idx[offset % 2 == 0]
    # 丢掉一对里增益号小的那条（高增益），保留低增益
    lo_first = gain[picked] < gain[picked + 1]
    return np.where(lo_first, picked, picked + 1).astype(np.int64)


def compare(day, hour):
    path = sorted(glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/*_{hour:02d}_v*.fits"))[-1]
    print("文件", path)
    print("探头       事例数    greedy 丢   adjacent 丢    只 greedy 有   只 adjacent 有")
    tot = [0, 0, 0, 0, 0]
    with fits.open(path, memmap=True) as hdus:
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            if data is None or len(data) == 0:
                continue
            t = np.asarray(data["TIME"], float)
            order = np.argsort(t, kind="stable")
            t = t[order]
            gain = np.asarray(data["GAIN_TYPE"])[order].astype(np.int8)
            dead = np.asarray(data["DEAD_TIME"])[order].astype(float) * 1e-6
            g = greedy(t, gain, dead)
            a = adjacent(t, gain, dead)
            only_g = np.setdiff1d(g, a).size
            only_a = np.setdiff1d(a, g).size
            print(f"{hdu.name}  {t.size:9d}  {g.size:9d}  {a.size:11d}  "
                  f"{only_g:12d}  {only_a:14d}")
            tot[0] += t.size
            tot[1] += g.size
            tot[2] += a.size
            tot[3] += only_g
            tot[4] += only_a
    print(f"\n合计  事例 {tot[0]}  greedy 丢 {tot[1]}  adjacent 丢 {tot[2]}")
    print(f"两者不一致：只 greedy 有 {tot[3]}（{tot[3]/max(tot[1],1)*100:.4f}%），"
          f"只 adjacent 有 {tot[4]}（{tot[4]/max(tot[2],1)*100:.4f}%）")
    print(f"对准入后计数的影响上限 {abs(tot[1]-tot[2])/tot[0]*100:.5f}% 的原始事例")


if __name__ == "__main__":
    compare(sys.argv[1], int(sys.argv[2]))
