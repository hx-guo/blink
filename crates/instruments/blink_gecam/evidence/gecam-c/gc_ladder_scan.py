"""GECAM-C：全任务逐小时点一遍 EBOUNDS，看那把能量梯什么时候换的、换掉多少小时，
并且**按 `EboundsHdu::verify` 的哪一道闸失败分类**。

起因：2022-08-15 的 EBOUNDS 是 **896 行**、ch447 收在 511.0–514.4 keV，而 2022-12
以后是 **470 行**、ch447 收在 9921–10053.5 keV。两把梯子覆盖同一个能量范围
（2.0 keV .. 10053.5 keV），**896 行那把正好细一倍**（ch_896 ≈ 2 × ch_470），所以
同一个道号在两个纪元里是两个能量：`MIN_CHANNEL = 54` 在 470 行梯子上是 39.1–40.2 keV，
在 896 行梯子上是 **14.16–14.54 keV**；`PI < 448` 在前者是 10 MeV 上界、在后者是
**514 keV**。

crate 的 `EboundsHdu::verify` 会整小时报错，这些小时根本没被搜——**闸门正确动作**。
但「被拒」有好几条不同的路，放宽哪一条是两回事，所以这里逐条记：

* `len`     —— EBOUNDS 行数不足 449
* `head`    —— ch0 的 `E_MIN` 不是 2.00 keV
* `tail`    —— ch447 的 `E_MAX` 不是 10053.5 keV
* `gap`     —— 道 0..447 之间有断开（`E_MAX[k-1] != E_MIN[k]`）
* `sentinel`—— ch448 的 `E_MAX` 既不是 20000 也不是 0
* `minch`   —— 40 keV 不落在 ch54

用法: gc_ladder_scan.py <输出 CSV> [年份 ...]
"""

import csv
import glob
import os
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
LADDER = 448
LADDER_MIN_KEV = 2.0
LADDER_MAX_KEV = 10053.5
MIN_ENERGY_KEV = 40.0
MIN_CHANNEL = 54


def latest_per_hour(paths):
    """同一小时多版本取版本号最大的。版本号单独解析出来比大小，不靠字典序。"""
    best = {}
    for path in paths:
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) < 5 or not parts[-1].startswith("v"):
            continue
        try:
            version = int(parts[-1][1:])
        except ValueError:
            continue
        key = (parts[2], parts[3])
        if key not in best or version > best[key][0]:
            best[key] = (version, path)
    return [p for _, p in sorted(best.values(), key=lambda v: v[1])]


def gates(lo, hi):
    """返回失败的闸名列表，空列表 = 全过。与 `EboundsHdu::verify` 同序同判据。"""
    bad = []
    if lo.size <= LADDER:
        return ["len"]
    if abs(lo[0] - LADDER_MIN_KEV) > 0.01:
        bad.append("head")
    if abs(hi[LADDER - 1] - LADDER_MAX_KEV) > 1.0:
        bad.append("tail")
    step = np.abs(lo[1:LADDER] - hi[: LADDER - 1])
    if np.any(step > hi[: LADDER - 1] * 1e-4):
        bad.append("gap")
    overflow = hi[LADDER]
    if abs(overflow - 20000.0) > 1.0 and abs(overflow - 0.0) > 1.0:
        bad.append("sentinel")
    # 40 keV 落在哪一道：第一个 E_MAX >= 40 的道，与 `channel_above` 同义
    channel = int(np.searchsorted(hi[:LADDER], MIN_ENERGY_KEV, "left"))
    if channel != MIN_CHANNEL:
        bad.append("minch")
    return bad


def main():
    out = sys.argv[1]
    years = sys.argv[2:] or ["*"]
    paths = []
    for year in years:
        paths.extend(glob.glob(f"{ROOT}/{year}/*/*/GRD_EVT/*_v*.fits"))
    paths = latest_per_hour(sorted(paths))
    print(f"逐小时文件 {len(paths)} 个（年份 {years}）", flush=True)

    writer = csv.writer(open(out, "w", newline=""), lineterminator="\n")
    writer.writerow(["day", "hour", "nrows", "ch0_min", "ch54_min", "ch54_max",
                     "ch447_max", "kev40_channel", "failed_gates"])
    tally = {}
    handle = open(out, "a")  # 只为逐条 flush，正文仍走 writer
    for i, path in enumerate(paths):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        yymmdd, hh = parts[2], parts[3][:2]
        day = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
        try:
            with fits.open(path, memmap=True) as hdus:
                eb = hdus["EBOUNDS"].data
                lo = np.asarray(eb["E_MIN"], float)
                hi = np.asarray(eb["E_MAX"], float)
        except Exception as exc:
            writer.writerow([day, hh, -1, "", "", "", "", "", f"open:{exc}"])
            tally["读不开"] = tally.get("读不开", 0) + 1
            continue
        bad = gates(lo, hi)
        channel = int(np.searchsorted(hi[:LADDER], MIN_ENERGY_KEV, "left")) if lo.size > LADDER else -1
        writer.writerow([day, hh, lo.size, f"{lo[0]:.3f}",
                         f"{lo[54]:.3f}" if lo.size > 54 else "",
                         f"{hi[54]:.3f}" if lo.size > 54 else "",
                         f"{hi[LADDER-1]:.2f}" if lo.size >= LADDER else "",
                         channel, "|".join(bad)])
        key = (lo.size, "|".join(bad) or "OK")
        tally[key] = tally.get(key, 0) + 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)
    handle.close()

    print("\n（行数, 失败的闸）统计:")
    for key, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {key}  {count} 小时")


if __name__ == "__main__":
    main()
