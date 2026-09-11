"""GECAM-B：双增益重复计数的实测（OPEN-QUESTIONS 第 17 条只在 C 星量过）。

量四件事，全部逐路 25 路 GRD：

1. **原始重复率**：同探头、死时间内、一高一低的事例对占原始事例的比例；
   以及只按"时戳完全相等"能抓到的那一部分（= 旧判据 e34f560 的覆盖率）。
2. **准入后重复率**：两条都落进 `keep` 窗 [54, 448) 的对——这些才是真被数两遍的。
3. **两支的时戳差分布**。同一个物理光子的两条记录，时戳差就是电子学时戳抖动的
   直接测量，**与任何选样标准构造无关**，可以跨星用。A 星实测有一批差 89 ns 和
   119 ns（3 个和 4 个量化步），正是"时戳相等"判据漏掉的那 15%。
4. **两档的 PI 是不是同一把尺子**：只取高增益条没饱和（PI < 448）的对，看 PI 差。

用法: python3 gb_dupe.py <YYYY-MM-DD> <HH> [更多 天 时 ...]
"""

import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_feat import MIN_CHANNEL, OVERFLOW_CHANNEL, NORMAL_EVT_TYPE, hour_file


def analyse(day, hour):
    path = hour_file(f"{day}T{hour}:00:00", "grd")
    print(f"\n===== {day} {hour}h  {path} =====")
    if path is None:
        print("  没有文件")
        return
    tot = dict(raw=0, pair_dead=0, pair_exact=0, keep=0, keep_pair=0, both_keep=0,
               same_gain_dead=0)
    t_ref = 0.0
    dts, dpi, hi_pi, lo_pi = [], [], [], []
    with fits.open(path, memmap=True) as hdus:
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            t = np.asarray(data["TIME"], float)
            o = np.argsort(t, kind="stable")
            t = t[o]
            pi = np.asarray(data["PI"])[o].astype(int)
            g = np.asarray(data["GAIN_TYPE"])[o].astype(int)
            evt = np.asarray(data["EVT_TYPE"])[o].astype(int)
            dead = np.asarray(data["DEAD_TIME"], float)[o]
            dt_s = np.where(dead > 0, dead * 1e-6, 0.0)
            tot["raw"] += t.size
            if t.size < 2:
                continue
            if t_ref == 0.0:
                t_ref = float(t[t.size // 2])
            # 相邻对：死时间内 + 增益档不同。一路一张表，同表即同探头。
            close = t[1:] <= t[:-1] + dt_s[:-1]
            diff_gain = g[1:] != g[:-1]
            pair = close & diff_gain
            idx = np.flatnonzero(pair)
            tot["pair_dead"] += idx.size
            tot["same_gain_dead"] += int((close & ~diff_gain).sum())
            tot["pair_exact"] += int((t[1:][pair] == t[:-1][pair]).sum())
            dts.append(t[idx + 1] - t[idx])
            a, b = idx, idx + 1
            hi = np.where(g[a] == 0, pi[a], pi[b])
            lo = np.where(g[a] == 0, pi[b], pi[a])
            hi_pi.append(hi)
            lo_pi.append(lo)
            unsat = hi < OVERFLOW_CHANNEL
            dpi.append((hi - lo)[unsat])
            tot["both_keep"] += int((((hi >= MIN_CHANNEL) & (hi < OVERFLOW_CHANNEL))
                                     & ((lo >= MIN_CHANNEL) & (lo < OVERFLOW_CHANNEL))
                                     & (evt[a] == NORMAL_EVT_TYPE) & (evt[b] == NORMAL_EVT_TYPE)).sum())
            k = (evt == NORMAL_EVT_TYPE) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW_CHANNEL)
            tot["keep"] += int(k.sum())

    dts = np.concatenate(dts) if dts else np.array([])
    hi_pi = np.concatenate(hi_pi) if hi_pi else np.array([])
    lo_pi = np.concatenate(lo_pi) if lo_pi else np.array([])
    dpi = np.concatenate(dpi) if dpi else np.array([])
    # q 必须按**本小时的 MET 量级**取，不能写死：TIME 是 float64 秒，格子就是该量级上
    # 的 ulp，跨过 2ⁿ 秒粗一倍（B 星 2021-01-21 之前 7.45 ns、之后 14.9 ns、
    # 2023-04-01 起 29.8 ns）。写死一个 q 会把步号算错，也会把细格纪元的结构抹平。
    q0 = float(np.spacing(t_ref))
    print(f"  原始事例 {tot['raw']:,}；死时间内一高一低的对 {tot['pair_dead']:,} "
          f"({tot['pair_dead'] / tot['raw'] * 100:.3f}%)；同档对 {tot['same_gain_dead']:,} "
          f"({tot['same_gain_dead'] / tot['raw'] * 100:.3f}%)")
    if tot["pair_dead"]:
        print(f"  其中时戳完全相等的 {tot['pair_exact']:,} "
              f"({tot['pair_exact'] / tot['pair_dead'] * 100:.2f}% —— 旧判据 e34f560 的覆盖率)")
    print(f"  准入后事例 {tot['keep']:,}；两条都过准入的对 {tot['both_keep']:,} "
          f"→ 被数两遍的计数占准入后 {tot['both_keep'] / max(tot['keep'], 1) * 100:.3f}%")
    if dts.size:
        print(f"  时戳差（ns）：0 的占 {(dts == 0).mean() * 100:.2f}%，"
              f"中位 {np.median(dts) * 1e9:.2f}，p90 {np.percentile(dts, 90) * 1e9:.2f}，"
              f"p99 {np.percentile(dts, 99) * 1e9:.2f}，max {dts.max() * 1e9:.2f}")
        steps = np.round(dts / q0).astype(int)
        vals, cnt = np.unique(steps, return_counts=True)
        top = np.argsort(-cnt)[:8]
        print("  按量化步 (q = %.2f ns) 分布，前 8：" % (q0 * 1e9)
              + "  ".join(f"{vals[i]}步({vals[i] * q0 * 1e9:.0f}ns):{cnt[i] / steps.size * 100:.2f}%"
                          for i in top))
    if hi_pi.size:
        print(f"  高增益条 PI：中位 {np.median(hi_pi):.0f}，>=448 占 "
              f"{(hi_pi >= OVERFLOW_CHANNEL).mean() * 100:.2f}%；"
              f"低增益条 PI：中位 {np.median(lo_pi):.0f}，>=448 占 "
              f"{(lo_pi >= OVERFLOW_CHANNEL).mean() * 100:.2f}%")
    if dpi.size:
        print(f"  高增益没饱和的对（{dpi.size:,} 对）：PI 相同占 {(dpi == 0).mean() * 100:.2f}%，"
              f"高−低中位 {np.median(dpi):+.0f}，5–95% {np.percentile(dpi, 5):+.0f} .. "
              f"{np.percentile(dpi, 95):+.0f}")


def main():
    args = sys.argv[1:]
    for day, hour in zip(args[::2], args[1::2]):
        analyse(day, hour)


if __name__ == "__main__":
    main()
