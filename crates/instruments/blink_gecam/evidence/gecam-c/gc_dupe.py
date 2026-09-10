"""GECAM-C：同探头同戳事例 = 高/低增益双记录。这一版量它在 keep 过滤之后还剩多少。

keep 是 `EVT_TYPE == 1 且 54 <= PI < 448`。如果一对里高增益那条落在溢出道 448、
被 PI 上限挡掉，那么一个物理事例只留一条计数，搜索不受影响；如果两条都落在 [54,448)，
同一个物理事例就被数了两遍，而且两条时戳完全相同——泊松独立性直接破掉，这正是
候选量爆表该有的样子。
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
day, hour = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/*_{hour:02d}_v*.fits"))
path = files[-1]
print("文件", path)

tot_keep = tot_dup = tot_raw = tot_raw_dup = 0
with fits.open(path, memmap=True) as hdus:
    for hdu in hdus:
        if not hdu.name.startswith("EVENTS"):
            continue
        data = hdu.data
        time = np.asarray(data["TIME"], float)
        order = np.argsort(time, kind="stable")
        time = time[order]
        pi = np.asarray(data["PI"])[order].astype(int)
        gain = np.asarray(data["GAIN_TYPE"])[order].astype(int)
        evt = np.asarray(data["EVT_TYPE"])[order].astype(int)
        flag = np.asarray(data["FLAG"])[order].astype(int)
        pair = np.asarray(data["EVT_PAIR"])[order]

        same = np.flatnonzero(np.diff(time) == 0)
        a, b = same, same + 1
        tot_raw += time.size
        tot_raw_dup += same.size

        keep = (evt == 1) & (pi >= 54) & (pi < 448)
        kt, kg, kpi = time[keep], gain[keep], pi[keep]
        ks = np.flatnonzero(np.diff(kt) == 0)
        tot_keep += kt.size
        tot_dup += ks.size

        if hdu.name in ("EVENTS01", "EVENTS02"):
            print(f"\n== {hdu.name}: 原始 {time.size} 事例，同戳对 {same.size} ({same.size/time.size*100:.3f}%)")
            print("   EVT_PAIR 形状", pair.shape, "dtype", pair.dtype)
            if pair.ndim == 2:
                any_true = pair.any(axis=1)
                print("   EVT_PAIR 任一位为真的事例占 %.3f%%；同戳对中两端都为真的占 %.1f%%"
                      % (any_true.mean() * 100, (any_true[a] & any_true[b]).mean() * 100))
                print("   非同戳事例里 EVT_PAIR 为真的占 %.3f%%"
                      % (any_true[~np.isin(np.arange(time.size), np.concatenate([a, b]))].mean() * 100))
                for k in range(pair.shape[1]):
                    print("     第 %d 位为真 %.3f%%；同戳对两端 (%.1f%%, %.1f%%)"
                          % (k, pair[:, k].mean() * 100, pair[a, k].mean() * 100, pair[b, k].mean() * 100))
            print("   同戳对：(高增益条 PI 中位 %.0f, 低增益条 PI 中位 %.0f)"
                  % (np.median(pi[a][gain[a] == 0]) if (gain[a] == 0).any() else -1,
                     np.median(pi[b][gain[b] == 1]) if (gain[b] == 1).any() else -1))
            hi = np.where(gain[a] == 0, pi[a], pi[b])
            lo = np.where(gain[a] == 0, pi[b], pi[a])
            print("   高增益条 PI: 中位 %.0f, >=448 占 %.1f%%, 在 [54,448) 占 %.1f%%"
                  % (np.median(hi), (hi >= 448).mean() * 100, ((hi >= 54) & (hi < 448)).mean() * 100))
            print("   低增益条 PI: 中位 %.0f, >=448 占 %.1f%%, 在 [54,448) 占 %.1f%%"
                  % (np.median(lo), (lo >= 448).mean() * 100, ((lo >= 54) & (lo < 448)).mean() * 100))
            both = ((hi >= 54) & (hi < 448) & (lo >= 54) & (lo < 448))
            print("   两条都落进 keep 窗口的对占 %.2f%%（这些就是被数两遍的）" % (both.mean() * 100))
            ea, eb = evt[a], evt[b]
            print("   同戳对的 EVT_TYPE 组合:",
                  {f"{int(u[0])},{int(u[1])}": int(c) for u, c in
                   zip(*np.unique(np.stack([ea, eb], 1), axis=0, return_counts=True))})
            print("   同戳对的 FLAG 组合:",
                  {f"{int(u[0])},{int(u[1])}": int(c) for u, c in
                   zip(*np.unique(np.stack([flag[a], flag[b]], 1), axis=0, return_counts=True))})
            if ks.size:
                print("   keep 之后：事例 %d，同戳对 %d (%.4f%%)，其中增益不同的 %.1f%%"
                      % (kt.size, ks.size, ks.size / kt.size * 100, (kg[ks] != kg[ks + 1]).mean() * 100))
                print("     keep 后同戳对的 PI: %.0f / %.0f" % (np.median(kpi[ks]), np.median(kpi[ks + 1])))

print("\n===== 整仪器汇总（12 路 GRD，1 小时）=====")
print("  原始事例 %d，同戳对 %d (%.3f%%)" % (tot_raw, tot_raw_dup, tot_raw_dup / tot_raw * 100))
print("  keep 后事例 %d，同戳对 %d (%.4f%%) → 被数两遍的计数占 keep 事例的 %.4f%%"
      % (tot_keep, tot_dup, tot_dup / tot_keep * 100, 2 * tot_dup / tot_keep * 100))
