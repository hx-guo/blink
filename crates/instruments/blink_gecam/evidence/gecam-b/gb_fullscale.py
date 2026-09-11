"""GECAM-B：逐路满量程道，两条独立路子对一遍。

**路子甲（查表）**：EBOUNDS 溢出段（ch ≥ 448）每一行的 `E_MIN` 就是该 (探头, 增益)
支路的满量程能量，把它反查回能量梯就是该支路的满量程道。B 星 50 行 = 25 路 × 2 档。
好处是逐小时随文件走、自动跟上标定更新，不用 CALDB，也不用统计量。

**路子乙（求直方图）**：从该小时的 PI 直方图逐路逐档求 `edge`（最后一个计数 ≥ 10 的道）。

**两条对上了，这个准入上界才有资格报批。** 对不上就说明溢出段那 50 行的行序/含义
没核实清楚，别拿它当准入依据。

一条已知限定（gecamC 实测）：C 星 896 道纪元里哨兵行**根本不在 EBOUNDS 表里**，
路子甲在那种纪元没有输入。B 星全程 498 行、有 50 行溢出段，不受影响——但这正是
"别把一颗星的表结构当通则"的又一例。

用法: python3 gb_fullscale.py <YYYY-MM-DD> <HH> [更多 天 时 ...]
"""

import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_feat import NORMAL_EVT_TYPE, hour_file

LADDER = 448
THRESHOLD = 10


def analyse(day, hour):
    path = hour_file(f"{day}T{hour}:00:00", "grd")
    print(f"\n===== {day} {hour}h  {path} =====")
    if path is None:
        print("  没有文件")
        return
    with fits.open(path, memmap=True) as hdus:
        eb = None
        for hdu in hdus:
            if hdu.name == "EBOUNDS":
                eb = hdu.data
        if eb is None:
            print("  没有 EBOUNDS")
            return
        e_min = np.asarray(eb["E_MIN"], float)
        e_max = np.asarray(eb["E_MAX"], float)
        n = e_min.size
        print(f"  EBOUNDS {n} 行，能量梯 0..{LADDER - 1}，溢出段 {n - LADDER} 行"
              f"（{'25 路 × 2 档' if n - LADDER == 50 else '行数不是 50，先别用路子甲'}）")
        # 路子甲：溢出段每行的 E_MIN 反查能量梯
        ladder_min, ladder_max = e_min[:LADDER], e_max[:LADDER]
        sentinel = e_min[LADDER:]
        table_edges = []
        for value in sentinel:
            k = int(np.searchsorted(ladder_max, value, "left"))
            table_edges.append(min(k, LADDER - 1))
        table_edges = np.array(table_edges)
        if np.all(sentinel == 0):
            print("  路子甲：溢出段 E_MIN 全是 0.0 —— 这一小时的 pi 库是旧生成器"
                  "（第 15 条 A 星那个 v0.9 印子），**查表法在这种小时没有输入**")
            table_edges = None
        else:
            # 50 行是 25 路 × 2 档，行序与 (探头, 增益) 的对应关系归档里没标注，
            # 但两档的满量程能量差一个数量级，按能量排序对半分就能把两档分开。
            order = np.argsort(sentinel)
            half = sentinel.size // 2
            lo_grp, hi_grp = table_edges[order[:half]], table_edges[order[half:]]
            print(f"  路子甲（溢出段 E_MIN 反查）：E_MIN 范围 "
                  f"{sentinel.min():.1f}–{sentinel.max():.1f} keV，按能量对半分成两档")
            print(f"    低能那 {half} 行（= 高增益支路）：满量程道 中位 {np.median(lo_grp):.0f}，"
                  f"范围 {lo_grp.min()}–{lo_grp.max()}")
            print(f"    高能那 {sentinel.size - half} 行（= 低增益支路）：满量程道 中位 "
                  f"{np.median(hi_grp):.0f}，范围 {hi_grp.min()}–{hi_grp.max()}"
                  f"（{int((sentinel[order[half:]] > ladder_max[-1]).sum())} 行的 E_MIN 已超出梯顶"
                  f" {ladder_max[-1]:.0f} keV，被夹到 ch447）")

        # 路子乙：PI 直方图
        hist_edges = {}
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            d = int(hdu.name[-2:])
            data = hdu.data
            pi = np.asarray(data["PI"]).astype(int)
            g = np.asarray(data["GAIN_TYPE"]).astype(int)
            ok = np.asarray(data["EVT_TYPE"]).astype(int) == NORMAL_EVT_TYPE
            for gain in (0, 1):
                sel = ok & (g == gain)
                if sel.sum() < 1000:
                    continue
                h = np.bincount(np.clip(pi[sel], 0, LADDER + 63), minlength=LADDER + 64)
                nz = np.flatnonzero(h[:LADDER] >= THRESHOLD)
                if nz.size:
                    hist_edges[(d, gain)] = int(nz[-1])
    if not hist_edges:
        print("  路子乙：统计量不足")
        return
    for gain, label in ((0, "高增益"), (1, "低增益")):
        vals = np.array([v for (d, g), v in sorted(hist_edges.items()) if g == gain])
        if vals.size:
            print(f"  路子乙（PI 直方图）{label}：edge 中位 {np.median(vals):.0f}，"
                  f"逐路 {vals.min()}–{vals.max()}（{vals.size} 路）")
    print("  【对比】溢出段那 50 行的行序与 (探头, 增益) 的对应关系归档里没有标注，"
          "所以只能比分布不能逐路比；两个分布的中位和范围对得上才算这条技巧可用。")


def main():
    args = sys.argv[1:]
    for day, hour in zip(args[::2], args[1::2]):
        analyse(day, hour)


if __name__ == "__main__":
    main()
