"""GECAM-C：POSATT 的地心距分布，用来确认坏行判据真的把坏行挡干净了。

**对 C 星，"看 |磁纬| 有没有超过倾角"这条自检不灵敏**——C 星倾角高、覆盖到极冠，
假高纬点混在真高纬点里看不出来。能用的是另一条：**好行的地心距应当是一条窄带**
（近圆轨道，半长轴几乎不变）。带外还剩多少、带内散得多开，直接决定判据够不够。

用法: gc_posatt_radius.py [每天抽几个小时=1]
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
A = 6378137.0
R_MIN, R_MAX = A + 100e3, A + 900e3


def main():
    per_day = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    days = sorted({p.rsplit("/POSATT", 1)[0] for p in glob.glob(f"{ROOT}/*/*/*/POSATT")})
    kept = []
    finite_bad = []
    nonfinite = 0
    total = 0
    for day_dir in days:
        for hour in range(0, 24, max(1, 24 // per_day)):
            paths = sorted(glob.glob(f"{day_dir}/POSATT/*_{hour:02d}_v*.fits"))
            if not paths:
                continue
            try:
                with fits.open(paths[-1], memmap=True) as hdus:
                    t = hdus["Orbit_Attitude"].data
                    r = np.sqrt(
                        np.asarray(t["X_WGS84"], float) ** 2
                        + np.asarray(t["Y_WGS84"], float) ** 2
                        + np.asarray(t["Z_WGS84"], float) ** 2
                    )
            except Exception:
                continue
            total += r.size
            ok = np.isfinite(r)
            nonfinite += int((~ok).sum())
            r = r[ok]
            inside = (r >= R_MIN) & (r <= R_MAX)
            kept.append(r[inside])
            finite_bad.append(r[~inside])
    r = np.concatenate(kept)
    bad = np.concatenate(finite_bad)
    q = np.percentile(r, [0, 0.1, 50, 99.9, 100]) / 1e3
    print(f"POSATT 行 {total}：带内 {r.size}（{r.size / total * 100:.4f}%）、"
          f"带外有限值 {bad.size}、非有限 {nonfinite}")
    print(f"带内地心距 km：min {q[0]:.1f}  p0.1 {q[1]:.1f}  中位 {q[2]:.1f}  "
          f"p99.9 {q[3]:.1f}  max {q[4]:.1f}   带宽 {q[4] - q[0]:.1f} km")
    print(f"  对应高度 km：{q[0] - A / 1e3:.1f} .. {q[4] - A / 1e3:.1f}")
    if bad.size:
        print("带外有限值的地心距（km）分档：")
        edges = [0, 1e3, 3e3, 6e3, 6678, 7278, 1e5, 1e9, np.inf]
        hist = np.histogram(bad / 1e3, bins=edges)[0]
        for lo, hi, n in zip(edges[:-1], edges[1:], hist):
            if n:
                print(f"  {lo:>10.0f} .. {hi:<10.0f} {n:8d}")
    # 带内是不是一条窄带：用中位 ± 50 km 再切一刀看还剩多少
    tight = np.abs(r - np.median(r)) <= 50e3
    print(f"带内再收到中位 ±50 km：{tight.sum()}/{r.size} = "
          f"{tight.sum() / r.size * 100:.4f}%")


if __name__ == "__main__":
    main()
