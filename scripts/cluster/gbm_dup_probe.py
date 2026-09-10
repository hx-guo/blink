"""GBM 的 TTE 里有没有「同一个物理光子被写成两行」。

GECAM-C 实测过这个错：ADC 的两条增益支路各写一行，两行**时戳完全相同**，
搜索把它们当成两个独立事例，泊松独立性就破了（同探头同时戳的事例对占原始
事例 6.635%，反标定下来 28% 的候选是它造出来的）。

GBM 要不要担心这个，靠这个脚本钉死。量的是**同一路探头内**相邻事例时戳
完全相同的对数——跨探头的同时戳是带电粒子穿整星，那是另一回事，由
`MAX_SIMULTANEOUS_FRACTION` 管，这里分开数。

2014 年只有 2 路 BGO、2019 年 14 路，两年都要查。

用法: python3 gbm_dup_probe.py <YYYY-MM-DD> [hour ...]
"""
import glob
import sys

import numpy as np
from astropy.io import fits

MIRROR = "/hxmtfs/data/Fermi_GBM"
DETS = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb", "b0", "b1"]


def find_tte(day, det, hour):
    y, m, d = day.split("-")
    ymd = y[2:] + m + d
    pats = [
        "%s/%s/%s/%s/current/glg_tte_%s_%s_%02dz_v*.fit*" % (MIRROR, y, m, d, det, ymd, hour),
        "%s/%s/%s/tte/glg_tte_%s_%s_%02dz_v*.fit*" % (MIRROR, y, ymd, det, ymd, hour),
        "%s/BGO/%s/%s/glg_tte_%s_%s_%02dz_v*.fit*" % (MIRROR, y, ymd, det, ymd, hour),
    ]
    for p in pats:
        g = sorted(glob.glob(p))
        if g:
            return g[-1]
    return None


def probe(day, hours):
    print("=== %s ===" % day)
    tot_ev = tot_pair = tot_samepha = 0
    for hour in hours:
        for det in DETS:
            path = find_tte(day, det, hour)
            if not path:
                continue
            with fits.open(path) as h:
                t = np.asarray(h["EVENTS"].data["TIME"], dtype=np.float64)
                pha = np.asarray(h["EVENTS"].data["PHA"])
            if len(t) < 2:
                continue
            o = np.argsort(t, kind="stable")
            t = t[o]; pha = pha[o]
            same = t[1:] == t[:-1]                    # 同一路探头内时戳完全相同的相邻对
            n_pair = int(same.sum())
            n_samepha = int((same & (pha[1:] == pha[:-1])).sum())
            tot_ev += len(t); tot_pair += n_pair; tot_samepha += n_samepha
            print("  %02dz %s  事例 %9d  同时戳对 %7d (%.4f%%)  其中 PHA 也相同 %6d (%.1f%%)"
                  % (hour, det, len(t), n_pair, 100 * n_pair / len(t), n_samepha,
                     100 * n_samepha / n_pair if n_pair else 0.0))
    if tot_ev:
        print("  合计: 事例 %d，同探头同时戳对 %d（%.4f%%），其中 PHA 相同 %d（%.1f%%）"
              % (tot_ev, tot_pair, 100 * tot_pair / tot_ev, tot_samepha,
                 100 * tot_samepha / tot_pair if tot_pair else 0.0))
        print("  对照：GECAM-C 是 6.635%%，且 100%% 是增益一高一低（PHA 相同占 0%%）")


if __name__ == "__main__":
    day = sys.argv[1]
    hours = [int(x) for x in sys.argv[2:]] or [0]
    probe(day, hours)
