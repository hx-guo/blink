"""GECAM-B：量程上限 `edge` 与超量程堆积包（OPEN-QUESTIONS 第 19 条）。

第 19 条在 A 星上钉死的机制是：**所谓"上限"不是硬件量程，是 ADC 满量程经过那份
`c2e` 映射到的道号**，所以它随标定版本走——A 星 2022–2024 段没有在轨 EC，低增益
`edge` 被压到 ch≈379（4.0 MeV），堆积包 ch368–383 整个落在 `PI < 448` 的准入里；
2025 年起有 EC 之后 `edge` 回到 ch447。B 星的 `c2e` 从发射后就有在轨版本（67 个
epoch），**按这个机制推 B 星的 edge 本来就该在 447 附近、包被现有上界挡住大半**
——但那是推断，这里实测。

**逐过境求，不查表。** 第 19 条自己就栽在"只看了 2024-01-11 一天"上：跨 epoch
重测之后结论改了两次。所以按 GTI 段（= 过境）分开、逐路逐增益档各求一次，
把逐路极差和逐过境极差都报出来，漂不漂由数说话。

`edge` 的定义与 A 星一致：**最后一个计数 ≥ THRESHOLD 的道**。堆积包抬升 =
`edge` 以下 15 道的平均计数 / 平台（ch200–300）的平均计数。

用法: python3 gb_edge.py <YYYY-MM-DD> <HH> [更多 天 时 ...]
"""

import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_feat import NORMAL_EVT_TYPE, hour_file, read_gti

NBIN = 512
THRESHOLD = 10          # "最后一个计数 >= 10 的道"，与 A 星同口径
PLATEAU = (200, 300)    # 平台参照段
BUMP = 15               # edge 以下这么多道算堆积包


def edge_of(hist):
    nz = np.flatnonzero(hist >= THRESHOLD)
    return int(nz[-1]) if nz.size else -1


def analyse(day, hour):
    path = hour_file(f"{day}T{hour}:00:00", "grd")
    print(f"\n===== {day} {hour}h  {path} =====")
    if path is None:
        print("  没有文件")
        return
    with fits.open(path, memmap=True) as hdus:
        gti = read_gti(hdus)
        segs = list(zip(*gti)) if gti is not None else [(-np.inf, np.inf)]
        print(f"  GTI {len(segs)} 段，合计 {sum(b - a for a, b in segs):.0f} s")
        per_seg = {}
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            d = int(hdu.name[-2:])
            data = hdu.data
            t = np.asarray(data["TIME"], float)
            pi = np.asarray(data["PI"]).astype(int)
            g = np.asarray(data["GAIN_TYPE"]).astype(int)
            evt = np.asarray(data["EVT_TYPE"]).astype(int)
            ok = evt == NORMAL_EVT_TYPE
            for si, (a, b) in enumerate(segs):
                m = ok & (t >= a) & (t <= b)
                for gain in (0, 1):
                    sel = m & (g == gain)
                    if sel.sum() < 1000:
                        continue
                    hist = np.bincount(np.clip(pi[sel], 0, NBIN - 1), minlength=NBIN)
                    per_seg.setdefault((si, gain), {})[d] = hist

    for (si, gain) in sorted(per_seg):
        hists = per_seg[(si, gain)]
        edges, bumps = [], []
        for d, h in sorted(hists.items()):
            e = edge_of(h)
            edges.append(e)
            plateau = h[PLATEAU[0]:PLATEAU[1]].mean()
            bump = h[max(e - BUMP, 0):e + 1].mean() if e > 0 else 0.0
            bumps.append(bump / plateau if plateau > 0 else np.nan)
        edges = np.array(edges)
        bumps = np.array(bumps, float)
        label = "高增益" if gain == 0 else "低增益"
        seg_a, seg_b = segs[si]
        print(f"  段{si} ({seg_b - seg_a:6.0f} s) {label}：edge 中位 {np.median(edges):.0f}，"
              f"逐路 {edges.min()}–{edges.max()}（极差 {edges.max() - edges.min()}），"
              f"堆积包抬升中位 {np.nanmedian(bumps):.2f}×，最大 {np.nanmax(bumps):.2f}×，"
              f"PI>=448 占该档 {sum(h[448:].sum() for h in hists.values()) / sum(h.sum() for h in hists.values()) * 100:.3f}%")

    # 整小时合起来的谱形（低增益），给出 edge 附近逐道计数，看有没有悬崖
    merged = {}
    for (si, gain), hists in per_seg.items():
        for d, h in hists.items():
            merged.setdefault(gain, np.zeros(NBIN, np.int64))
            merged[gain] += h
    for gain in sorted(merged):
        h = merged[gain]
        e = edge_of(h)
        label = "高增益" if gain == 0 else "低增益"
        lo = max(e - 12, 0)
        print(f"  整仪器整小时 {label} edge={e}，ch{lo}..{min(e + 4, NBIN - 1)} 逐道："
              + " ".join(str(int(v)) for v in h[lo:min(e + 5, NBIN)]))
        print(f"    平台(ch{PLATEAU[0]}–{PLATEAU[1]}) 均值 {h[PLATEAU[0]:PLATEAU[1]].mean():.0f}，"
              f"包(ch{max(e - BUMP, 0)}–{e}) 均值 {h[max(e - BUMP, 0):e + 1].mean():.0f}，"
              f"抬升 {h[max(e - BUMP, 0):e + 1].mean() / max(h[PLATEAU[0]:PLATEAU[1]].mean(), 1e-9):.2f}×")


def main():
    args = sys.argv[1:]
    for day, hour in zip(args[::2], args[1::2]):
        analyse(day, hour)


if __name__ == "__main__":
    main()
