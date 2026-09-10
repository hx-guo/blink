"""查 SVOM/GRM 有没有「同一个光子被两个增益支路各写一行」的重复记录。

GECAM-C 实测：同探头同时戳的事例对占原始事例 6.6%，100% 是 GAIN_TYPE 一高一低，
PI 不同（逐行去重抓不到），准入后仍有 3.25% 是重复。SVOM 的事例表同样有
GAIN_TYPE，两档共用一套 PI 刻度，条件相同，必须实测。

统计三档：
  完全同时戳          Δt == 0
  一个量化步以内      0 < Δt <= 2^-20 s（两支路时戳可能差 1 LSB）
  死时间以内          0 < Δt < 3.5 µs（同一路探头物理上不可能，见 evidence/sample/deadtime_check）

用法: python3 svom_gain_dup.py <YYYY/MM/DD> <HH> [更多 天 时 ...]
"""
import glob
import sys

import numpy as np
from astropy.io import fits

D = "/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily"
QUANT = 2.0 ** -20


def latest(pat):
    g = sorted(glob.glob(pat))
    if not g:
        return None
    seen = {}
    for f in g:
        seen[f.rsplit("_v", 1)[0]] = f
    return sorted(seen.values())[-1]


def report(day, hh):
    y, m, dd = day.split("/")
    f = latest("%s/%s/grm_evt/svom_grm_evt_%s%s%s_%s_v*.fits" % (D, day, y[2:], m, dd, hh))
    if f is None:
        print("缺文件", day, hh)
        return
    tot_raw = tot_eq = tot_lsb = tot_dead = 0
    tot_keep = keep_eq = 0
    lines = []
    with fits.open(f) as h:
        for i, det in zip((3, 4, 5), (1, 2, 3)):
            x = h[i].data
            t = np.asarray(x["TIME"], float)
            pi = np.asarray(x["PI"])
            g = np.asarray(x["GAIN_TYPE"])
            et = np.asarray(x["EVT_TYPE"])
            ac = np.asarray(x["ANTI_COIN"])
            o = np.argsort(t, kind="stable")
            t, pi, g, et, ac = t[o], pi[o], g[o], et[o], ac[o]
            d = np.diff(t)
            eq = d == 0
            lsb = (d > 0) & (d <= QUANT * 1.001)
            dead = (d > 0) & (d < 3.5e-6)
            tot_raw += len(t)
            tot_eq += int(eq.sum())
            tot_lsb += int(lsb.sum())
            tot_dead += int(dead.sum())
            gain_mix = int((eq & (g[:-1] != g[1:])).sum())
            pi_same = int((eq & (pi[:-1] == pi[1:])).sum())
            keep = (et == 0) & (ac == 0) & (pi >= 25) & (pi < 256)
            tk = t[keep]
            dk = np.diff(tk)
            tot_keep += int(keep.sum())
            keep_eq += int((dk == 0).sum())
            lines.append("    G%02d 原始 %8d  Δt=0 对 %6d(%.4f%%)  其中增益一高一低 %6d  PI 相同 %5d"
                         "  |  0<Δt≤1LSB %5d  0<Δt<3.5µs %5d  |  准入后 %8d  Δt=0 对 %5d(%.4f%%)"
                         % (det, len(t), int(eq.sum()), 100 * eq.mean(), gain_mix, pi_same,
                            int(lsb.sum()), int(dead.sum()), int(keep.sum()),
                            int((dk == 0).sum()), 100 * (dk == 0).mean() if len(dk) else 0))
        # 三路合并后（旧记录里查的就是这个口径）
        T = []
        for i in (3, 4, 5):
            T.append(np.asarray(h[i].data["TIME"], float))
        T = np.concatenate(T)
        T.sort()
        merged_eq = int((np.diff(T) == 0).sum())
        gt = np.asarray(h[3].data["GAIN_TYPE"])
        pia = np.asarray(h[3].data["PI"])
    print("%s %s:00  原始 %d，Δt=0 对 %d（%.4f%%），三路合并后 Δt=0 对 %d"
          % (day, hh, tot_raw, tot_eq, 100 * tot_eq / tot_raw, merged_eq))
    for ln in lines:
        print(ln)
    print("    GAIN_TYPE=1 占比 %.3f%%，其 PI 范围 [%d, %d]；GAIN_TYPE=0 的 PI 范围 [%d, %d]"
          % (100 * (gt == 1).mean(), pia[gt == 1].min() if (gt == 1).any() else -1,
             pia[gt == 1].max() if (gt == 1).any() else -1,
             pia[gt == 0].min(), pia[gt == 0].max()))
    print("    准入后合计 %d，Δt=0 对 %d（%.4f%%）"
          % (tot_keep, keep_eq, 100 * keep_eq / max(tot_keep, 1)))


if __name__ == "__main__":
    a = sys.argv[1:]
    for i in range(0, len(a), 2):
        report(a[i], a[i + 1])
