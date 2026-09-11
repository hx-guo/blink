"""GECAM-A：三件补充测量。

1. **`edge` 的第二条独立路子**：EBOUNDS 溢出段（PI ≥ 448）每一行的 `E_MIN` 就是
   该 (探头, 增益) 支路的满量程能量；把它落回 0..447 的能量梯上反查道号，与从
   PI 直方图求出的 `edge` 对一遍。两条路子都从同一小时的文件里来，但一条走标定
   表、一条走实测计数，对得上才说明「上限 = ADC 满量程经能标映射到的道」这个机制
   是对的（第 19 条）。
2. **外部真值定的 `min_duration`**：gecamB 用同一套搜索量到 140 个已发表真 TGF 的
   `bin_size_best` 中位 111.4 µs、< 10 µs 只占 6.4%。所以「窗 ≥ 10 µs」不是调出来
   的阈，是拿外部真值量出来的代价已知的准入——代价 6.4% 要带 Clopper–Pearson 区间。
3. **成串尺度扫描**：一次持续的粒子增强会被搜到几十个独立候选。按 C 星那条纪律，
   ±0.5 / 5 / 30 / 60 / 600 s 各扫一次，报整条分离度曲线，不只报一个尺度。

用法:
  python3 ga_extra.py ebounds <hh>            # 第 1 件，hh = 01/02/03
  python3 ga_extra.py chain  <w*.csv ...>     # 第 2、3 件
"""

import csv
import glob
import sys

import numpy as np
from scipy.stats import beta, poisson

YEAR = 3600.0 * 24.0 * 365.25
FPY, MIN_NUMBER = 20.0, 8
EXPOSURE = 7050.0
DAYS = 183
# gecamB 用同一套搜索量的 140 个已发表真 TGF
TGF_N, TGF_SHORT_1US, TGF_SHORT_10US = 140, 3, 9


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def ebounds(hour):
    from astropy.io import fits
    path = sorted(glob.glob(
        f"/gecamfs/Archived-DATA/GSDC/LEVEL1/daily/2024/01/11/GECAM_A/GRD_evt/gag_evt_240111_{hour}*.fits"))[-1]
    print("文件", path)
    with fits.open(path, memmap=True) as hdus:
        data = hdus["EBOUNDS"].data
        cols = data.columns.names
        print("EBOUNDS 列", cols, "行数", len(data))
        det = np.asarray(data["DET_ID"]) if "DET_ID" in cols else None
        gain = np.asarray(data["GAIN_TYPE"]) if "GAIN_TYPE" in cols else None
        chan = np.asarray(data["CHANNEL"]) if "CHANNEL" in cols else None
        emin = np.asarray(data["E_MIN"], float)
        emax = np.asarray(data["E_MAX"], float)
    print("道号范围", chan.min(), chan.max() if chan is not None else None)
    ladder = None
    out = {}
    for d in np.unique(det):
        for g in np.unique(gain):
            sel = (det == d) & (gain == g)
            ch, lo, hi = chan[sel], emin[sel], emax[sel]
            order = np.argsort(ch)
            ch, lo, hi = ch[order], lo[order], hi[order]
            over = ch >= 448
            if not over.any():
                continue
            full = float(lo[over][0])            # 溢出段第一行的 E_MIN = 满量程能量
            normal = ch < 448
            if ladder is None:
                ladder = lo[normal].copy()
            k = int(np.searchsorted(lo[normal], full, "right") - 1)
            out[(int(d), int(g))] = (full, k)
    for key in sorted(out):
        full, k = out[key]
        print(f"  d{key[0]:02d} g{key[1]}  满量程 {full:9.1f} keV → ch{k}")
    return out


def load(paths):
    rows = []
    for p in paths:
        rows.extend(csv.DictReader(open(p)))
    d = {}
    for key in rows[0]:
        try:
            d[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            d[key] = np.array([r[key] for r in rows])
    return d


def chain(paths):
    d = load(paths)
    n = len(d["start"])
    bin_s = d["bin_ns"] * 1e-9
    bin_us = d["bin_ns"] * 1e-3
    long = bin_us >= 10.0
    nc, bk = d["n_cut"], d["b_cut"]
    mean2 = np.where(d["n_bkg"] > 0, d["mean"] * bk / np.maximum(d["n_bkg"], 1), 0.0)
    fa2 = poisson.sf(nc, mean2) * (YEAR / bin_s)
    admit = (nc >= MIN_NUMBER) & (fa2 <= FPY)
    f2c, f3c = d["f2_cut"], d["f3_cut"]

    lo1, hi1 = cp(TGF_SHORT_10US, TGF_N)
    print("=" * 78)
    print("一、外部真值定的 min_duration")
    print("=" * 78)
    print(f"140 个已发表真 TGF（同一套搜索量的 bin_size_best）：< 10 µs 占 "
          f"{TGF_SHORT_10US}/{TGF_N} = {TGF_SHORT_10US / TGF_N * 100:.1f}% "
          f"[{lo1 * 100:.1f}, {hi1 * 100:.1f}]（95% CP）")
    print(f"A 星候选里 < 10 µs 占 {(~long).mean() * 100:.1f}%（{int((~long).sum())} 个）")
    print(f"  → 「窗 ≥ 10 µs」这一刀切掉 {(~long).mean() * 100:.1f}% 的候选、"
          f"代价是 {TGF_SHORT_10US / TGF_N * 100:.1f}% 的真 TGF，杠杆 "
          f"{(~long).mean() / (TGF_SHORT_10US / TGF_N):.1f} : 1")

    print()
    print("=" * 78)
    print("二、完整判据链（每步：切前 → 切后）")
    print("=" * 78)
    steps = [
        ("① 双增益去重后的原始池", np.ones(n, bool)),
        ("② + 准入上界 = 堆积包起点", admit),
        ("③ + 窗 ≥ 10 µs（外部真值定）", admit & long),
        ("④ + f₃ == 0（零参数，≥10 µs 才可用）", admit & long & (f3c == 0)),
        ("⑤ + f₂ ≤ 0.3", admit & long & (f3c == 0) & (f2c <= 0.3)),
        ("⑥ + f₂ == 0（同戳一对都没有）", admit & long & (f3c == 0) & (f2c == 0)),
    ]
    prev = n
    for label, sel in steps:
        k = int(sel.sum())
        print(f"{label:<36}{k:>7}  ({k / prev * 100:5.1f}% of 上一步)  "
              f"{k / EXPOSURE * 86400:9.1f} /天   183 天 {k / EXPOSURE * 86400 * DAYS:10.0f}")
        prev = max(k, 1)
    final = steps[-2][1]
    print()
    print(f"⑤ 的偶然期望：`f₃ > 0` 在 ≥10 µs 档的偶然误否决期望 "
          f"{(1 - np.exp(-d['exp_trip'][admit & long])).sum():.2f} 个／7050 s；"
          f"`f₂ > 0.3` 是调出来的阈、没有零参数解释")
    b_rate = 147 / (274.5 * 86400)
    print(f"验收线：GECAM-B 147 个 TGF / 274.5 天 = {b_rate:.2e} /s（每天 0.54 个）")
    for label, sel in steps[1:]:
        r = sel.sum() / EXPOSURE
        print(f"  {label:<36} {r:9.5f} /s  = B 星的 {r / b_rate:9.0f} 倍 "
              f"({np.log10(r / b_rate):.2f} 个数量级)")

    print()
    print("=" * 78)
    print("三、幸存者成串尺度扫描（邻居数在全量初始候选池里数）")
    print("=" * 78)
    t = np.array([np.datetime64(s) for s in d["start"]]).astype("datetime64[ns]").astype(np.int64) * 1e-9
    order = np.argsort(t)
    ts = t[order]
    hour = d["hour"][order]
    live = {1.0: 2803.0, 2.0: 3599.0, 3.0: 648.0}   # hours.json 的 searched_seconds
    print(f"{'尺度':<10}{'幸存者邻居中位':>16}{'全池邻居中位':>16}{'均匀期望':>12}{'幸存者/期望':>14}")
    for half in (0.5, 5.0, 30.0, 60.0, 600.0):
        lo = np.searchsorted(ts, ts - half, "left")
        hi = np.searchsorted(ts, ts + half, "right")
        neigh = (hi - lo - 1).astype(float)
        sel = final[order]
        pool = {h: float((d["hour"] == h).sum()) / live[h] for h in live}
        exp = np.array([pool[h] for h in hour]) * 2 * half
        print(f"±{half:<9g}{np.median(neigh[sel]):>16.1f}{np.median(neigh):>16.1f}"
              f"{np.median(exp):>12.1f}{np.median(neigh[sel]) / max(np.median(exp), 1e-9):>14.2f}")

    print()
    print(f"⑤ 幸存者 {int(final.sum())} 个的形态：")
    s = final
    if s.sum():
        print(f"  count 中位 {np.median(nc[s]):.0f}（切前 {np.median(d['count'][s]):.0f}）  "
              f"窗长中位 {np.median(bin_us[s]):.1f} µs  fa 中位 {np.median(fa2[s]):.3g}")
        print(f"  |lat| 中位 {np.median(np.abs(d['lat'][s])):.1f}°  "
              f"n_det 中位 {np.median(d['n_det_cut'][s]):.0f}  "
              f"fa ≤ 0.01 的 {int((s & (fa2 <= 0.01)).sum())} 个")
        gap = np.diff(np.sort(d["start"][s].astype("datetime64[ns]").astype(np.int64))) * 1e-9
        print(f"  相邻幸存者时间间隔中位 {np.median(gap):.2f} s，"
              f"间隔 < 1 s 的对数 {(gap < 1).sum()}/{gap.size}")


def tol(paths):
    """同戳容差扫描：`f₂`/`f₃` 按精确相等算会漏掉粒子穿越的跨探头时戳散布。"""
    d = load(paths)
    n = len(d["start"])
    bin_s, bin_us = d["bin_ns"] * 1e-9, d["bin_ns"] * 1e-3
    long = bin_us >= 10.0
    nc, bk = d["n_cut"], d["b_cut"]
    mean2 = np.where(d["n_bkg"] > 0, d["mean"] * bk / np.maximum(d["n_bkg"], 1), 0.0)
    fa2 = poisson.sf(nc, mean2) * (YEAR / bin_s)
    admit = (nc >= MIN_NUMBER) & (fa2 <= FPY)
    base = admit & long
    b_rate = 147 / (274.5 * 86400)
    print("=" * 90)
    print(f"同戳容差扫描（在 ② 准入上界 + ③ 窗 ≥ 10 µs 之后，基数 {int(base.sum())} 个）")
    print("=" * 90)
    print(f"{'τ (ns)':<9}{'f₂中位':>9}{'f₃中位':>9}{'f₃>0 %':>9}{'f₃==0 留':>11}"
          f"{'再 f₂≤0.3':>11}{'再 f₂==0':>10}{'三重偶然期望和':>16}{'/s (f₃==0∩f₂≤.3)':>20}")
    for name in ("t0", "t59", "t150", "t300"):
        f2t, f3t, e3 = d["f2_" + name], d["f3_" + name], d["exp3_" + name]
        k1 = base & (f3t == 0)
        k2 = k1 & (f2t <= 0.3)
        k3 = k1 & (f2t == 0)
        tau = 0 if name == "t0" else int(name[1:]) + 1
        r = k2.sum() / EXPOSURE
        print(f"{tau:<9}{np.median(f2t[base]):>9.3f}{np.median(f3t[base]):>9.3f}"
              f"{(f3t[base] > 0).mean() * 100:>9.1f}{int(k1.sum()):>11}{int(k2.sum()):>11}"
              f"{int(k3.sum()):>10}{np.nansum(1 - np.exp(-e3[base])):>16.2f}"
              f"{r:>13.5f} ({r / b_rate:.0f}×B)")
    print()
    print("同一条扫描，但不先切窗 ≥ 10 µs（基数 = ② 之后的全部）：")
    for name in ("t0", "t59", "t150", "t300"):
        f2t, f3t = d["f2_" + name], d["f3_" + name]
        k2 = admit & (f3t == 0) & (f2t <= 0.3)
        tau = 0 if name == "t0" else int(name[1:]) + 1
        print(f"  τ={tau:<5} f₃==0 ∩ f₂≤0.3 留 {int(k2.sum()):>6} 个"
              f"（其中窗 ≥ 10 µs 的 {int((k2 & long).sum()):>5}）")


if __name__ == "__main__":
    if sys.argv[1] == "ebounds":
        ebounds(sys.argv[2])
    elif sys.argv[1] == "tol":
        tol(sys.argv[2:])
    else:
        chain(sys.argv[2:])
