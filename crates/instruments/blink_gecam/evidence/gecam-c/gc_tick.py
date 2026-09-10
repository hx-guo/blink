"""GECAM-C 时戳量化步长：在事例流上直接钉死，不从 min_dt 反推。

做法：取一小时一路探头的全部时戳，减去起点，除以候选步长 q，看余数是否恒为 0。
候选 q 取 2^-26 s（前一轮实测的最小正间隔）与 1.5e-8。
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
day, hour = sys.argv[1], int(sys.argv[2])
path = sorted(glob.glob(f"{ROOT}/{day.replace('-', '/')}/GRD_EVT/*_{hour:02d}_v*.fits"))[-1]
print("文件", path)

with fits.open(path, memmap=True) as hdus:
    for hdu in hdus:
        if hdu.name != "EVENTS01":
            continue
        time = np.sort(np.asarray(hdu.data["TIME"], float))
        break

gaps = np.diff(time)
positive = gaps[gaps > 0]
print("事例 %d，正间隔 %d 个" % (time.size, positive.size))
print("最小正间隔 %.9g s" % positive.min())
uniq = np.unique(np.round(positive[positive < 5e-7] / positive.min(), 4))
print("小于 500 ns 的间隔 / 最小正间隔 的取值:", uniq[:15])

for name, q in (("2^-26", 2.0 ** -26), ("1.5e-8", 1.5e-8), ("3.0e-8", 3.0e-8)):
    # 时戳绝对值很大（MET 约 8e7 s），double 只剩约 1e-9 的分辨，
    # 所以对相对时间做检验，起点取整点附近
    rel = time - time[0]
    ticks = rel / q
    err = np.abs(ticks - np.round(ticks))
    print("  q = %-8s (%.9g s): |ticks − round(ticks)| 中位 %.3g，p99 %.3g，最大 %.3g"
          % (name, q, np.median(err), np.percentile(err, 99), err.max()))

print("\n最小正间隔 / 2^-26 = %.6f" % (positive.min() / 2.0 ** -26))
print("MET 量级 %.6g s → double 的 ulp 约 %.3g s（量化步长的 %.4f 倍）"
      % (time[0], np.spacing(time[0]), np.spacing(time[0]) / 2.0 ** -26))
