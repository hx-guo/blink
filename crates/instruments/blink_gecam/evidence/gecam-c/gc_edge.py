"""GECAM-C：逐探头逐增益档从 PI 直方图求满量程道，看堆积包在哪、漂不漂。

OPEN-QUESTIONS 第 19 条在 GECAM-A 上实测到：低增益档撑到 ch≈379 就有一个两个数量级
的悬崖，悬崖之下 15 道是超量程沉积堆出来的包，而准入的 `PI < 448` 整个把它放进来。
机制定为「ADC 满量程经过 c2e 映射到的道号」——A 星 2022–2024 没有在轨 EC，默认标定
把满量程压到 ch379；2025 起有 EC 之后跳到 ch447（梯子顶）。

按这个机制推，B/C 的 c2e 从发射后就有在轨版本，满量程道本来就该在 ch447 附近、
堆积包被现有 448 上界挡住大半。**但那是推断，C 星没测过。** 这个脚本测它。

`edge` 一律从数据本身求（最后一个 ≥ MIN_COUNTS 的道），不查表——CALDB 再更新一次
查表就失效，而直方图相对读事例流的成本可忽略。

用法：gc_edge.py <YYYY-MM-DD> <hour> [<hour> ...]
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
NCH = 512
# 悬崖判定：最后一个计数 ≥ 这个数的道。太小会被单个毛刺带偏，太大会把统计量少的
# 探头的真悬崖往低处拖。A 星用的是 10，这里沿用以便两星可比。
MIN_COUNTS = 10


def hour_files(day, hour):
    directory = f"{ROOT}/{day.replace('-', '/')}/GRD_EVT"
    files = sorted(glob.glob(f"{directory}/*_{hour:02d}_v*.fits"))
    return files[-1] if files else None


def edge_of(hist):
    """最后一个计数 ≥ MIN_COUNTS 的道；全空返回 -1。"""
    nonzero = np.flatnonzero(hist >= MIN_COUNTS)
    return int(nonzero[-1]) if nonzero.size else -1


def bump_ratio(hist, edge):
    """堆积包抬升 = 悬崖下 20 道的峰 / 更低处平台的中位。

    平台取 [edge-100, edge-40]，避开包本身；统计量不够就返回 nan。
    """
    if edge < 120:
        return float("nan"), float("nan"), float("nan")
    plateau = np.median(hist[edge - 100 : edge - 40])
    peak = hist[max(edge - 20, 0) : edge + 1].max()
    if plateau <= 0:
        return float("nan"), float(peak), float(plateau)
    return float(peak / plateau), float(peak), float(plateau)


def main():
    day = sys.argv[1]
    hours = [int(h) for h in sys.argv[2:]]
    emin = emax = None
    for hour in hours:
        path = hour_files(day, hour)
        if path is None:
            print(f"{day} {hour:02d}h  无文件")
            continue
        rows = []
        with fits.open(path, memmap=True) as hdus:
            if emin is None:
                eb = hdus["EBOUNDS"].data
                emin = np.asarray(eb["E_MIN"], float)
                emax = np.asarray(eb["E_MAX"], float)
            for hdu in hdus:
                if not hdu.name.startswith("EVENTS"):
                    continue
                data = hdu.data
                if data is None or len(data) == 0:
                    continue
                pi = np.asarray(data["PI"]).astype(int)
                gain = np.asarray(data["GAIN_TYPE"]).astype(int)
                evt = np.asarray(data["EVT_TYPE"]).astype(int)
                # 准入的是 EVT_TYPE == 1；溢出段（type 2）另算，见下面的 over 列
                good = evt == 1
                for g in (0, 1):
                    sel = good & (gain == g)
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    hist = np.bincount(np.clip(pi[sel], 0, NCH - 1), minlength=NCH)
                    edge = edge_of(hist)
                    ratio, peak, plateau = bump_ratio(hist, edge)
                    over = float((pi[sel] >= 448).mean() * 100)
                    # 落在悬崖下 15 道里的占比——这一段就是被现有准入放进来的堆积包
                    in_bump = float(
                        ((pi[sel] >= edge - 15) & (pi[sel] <= edge)).mean() * 100
                    )
                    rows.append((hdu.name, g, n, edge, ratio, peak, plateau, over, in_bump))
        print(f"\n===== {day} {hour:02d}h  {path.split('/')[-1]} =====")
        print("探头      档  事例数      edge   能量(keV)      包抬升   峰/平台      PI>=448   包内占比")
        for name, g, n, edge, ratio, peak, plateau, over, in_bump in rows:
            e_lo = emin[edge] if 0 <= edge < emin.size else float("nan")
            e_hi = emax[edge] if 0 <= edge < emax.size else float("nan")
            print(
                f"{name}  {'高' if g == 0 else '低'}  {n:9d}  {edge:5d}  "
                f"{e_lo:7.1f}-{e_hi:7.1f}  {ratio:8.2f}  {peak:8.0f}/{plateau:-7.0f}  "
                f"{over:7.3f}%  {in_bump:7.3f}%"
            )
        for g in (0, 1):
            edges = [r[3] for r in rows if r[1] == g and r[3] > 0]
            ratios = [r[4] for r in rows if r[1] == g and np.isfinite(r[4])]
            if edges:
                print(
                    f"  {'高' if g == 0 else '低'}增益汇总: edge 中位 {np.median(edges):.0f} "
                    f"范围 {min(edges)}-{max(edges)} 极差 {max(edges)-min(edges)}  "
                    f"包抬升中位 {np.median(ratios) if ratios else float('nan'):.2f}"
                )


if __name__ == "__main__":
    main()
