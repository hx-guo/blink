"""GECAM-C：POSATT 的 `X/Y/Z_WGS84` 从哪天起才有数。

**起因**：2022-08-05 的 POSATT 里 `X/Y/Z_WGS84` **整列全零**，而 `X_J2000` 有数。
后果是那一天 273 个候选的位置全部是 (经度 0, 纬度 180, 高度 −6378137 m)
——**地心原点经地理换算之后的垃圾值，而且它不报错、不被丢弃**：`search` 只在
`interpolate` 返回 `None` 时丢候选，而这里插值成功、返回的就是那个垃圾。

若细梯纪元整段如此，那么第 21 条"多出 29 天曝光、**全部落在 WWLLN 覆盖内**"
这句话就不成立——**没有位置就没有闪电关联、没有磁纬、没有陆海分类**。

逐日点一遍，只读 POSATT 一张表。

用法: gc_posatt_wgs84.py [每天抽几个小时=1]
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"


def main():
    per_day = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    days = sorted({p.rsplit("/POSATT", 1)[0] for p in glob.glob(f"{ROOT}/*/*/*/POSATT")})
    zero_days, good_days, bad_days = [], [], []
    for day_dir in days:
        tag = day_dir[-10:]
        verdict = None
        for hour in range(0, 24, max(1, 24 // per_day)):
            paths = sorted(glob.glob(f"{day_dir}/POSATT/*_{hour:02d}_v*.fits"))
            if not paths:
                continue
            try:
                with fits.open(paths[-1], memmap=True) as hdus:
                    t = hdus["Orbit_Attitude"].data
                    names = t.columns.names
                    if "X_WGS84" not in names:
                        verdict = "无列"
                        break
                    r = np.sqrt(
                        np.asarray(t["X_WGS84"], float) ** 2
                        + np.asarray(t["Y_WGS84"], float) ** 2
                        + np.asarray(t["Z_WGS84"], float) ** 2
                    )
            except Exception:
                continue
            good = np.isfinite(r) & (r > 1e6)
            verdict = "有数" if good.mean() > 0.5 else "全零"
            break
        if verdict == "有数":
            good_days.append(tag)
        elif verdict == "全零":
            zero_days.append(tag)
        elif verdict is not None:
            bad_days.append((tag, verdict))

    print(f"有 POSATT 的天 {len(days)}")
    print(f"  WGS84 有数 {len(good_days)} 天"
          + (f"：{good_days[0]} .. {good_days[-1]}" if good_days else ""))
    print(f"  WGS84 全零 {len(zero_days)} 天"
          + (f"：{zero_days[0]} .. {zero_days[-1]}" if zero_days else ""))
    if bad_days:
        print(f"  其他 {len(bad_days)} 天：{bad_days[:5]}")
    if zero_days and good_days:
        print(f"\n全零段的边界：最后一个全零日 {zero_days[-1]}，"
              f"第一个有数日 {good_days[0]}")
        overlap = sorted(set(zero_days) & set(good_days))
        if overlap:
            print(f"  同日两种都有：{overlap}")
        # 与细梯纪元（2022-08-03 .. 2022-10-15）的关系
        fine = [d for d in zero_days if "2022/08/03" <= d <= "2022/10/15"]
        print(f"  全零日里落在细梯纪元（2022-08-03 .. 10-15）的：{len(fine)} 天")
        print(f"  全零日里落在细梯纪元之外的：{len(zero_days) - len(fine)} 天")


if __name__ == "__main__":
    main()
