"""CoV（窗内事例间隔的变异系数）按 A 角 / B 角 / f₃ 标记分组汇总。

判别量来自 HXMT：泊松 CoV = 1、真暴发成团 CoV > 1、缓冲区顺序倾泻或读出定速的假信号
CoV < 1。HXMT 自己把这条从"判定性"下调到 AUC 0.836、不能单独当判据，天格这边同样只
当人群统计用。逐候选的 CoV 由 `diag/lead/grid_cov.py` 算好（窗取 T90 区间，事例准入
与搜索一致）；本脚本只做分组汇总，外加一条新的：**按 f₃ 标记把 A 角拆成干净子集与
粒子污染子集**，看 CoV 能不能独立看出同一批污染。

注意天格的本底本身不是泊松的（本底窗 CoV 中位 1.88），所以一律报"候选窗 ÷ 自己的
本底窗"的归一值，不用 HXMT 那个绝对基准。

用法：
    python3 cov_groups.py --features features_sig_v12.csv --t90 t90_v15.csv \
        --cov cov_v15.csv --f3 f3_overrange.csv [--index burst_events/index.csv]
"""

import argparse
import csv

import numpy as np

# 与 scripts/plot_grid_talk.py 一致的分群轴
T90_CUT_US = 2000.0
MLAT_CUT_DEG = 33.0
POLE_LAT, POLE_LON = np.radians(80.7), np.radians(-72.7)


def dipole_lat(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    s = np.sin(lat) * np.sin(POLE_LAT) + np.cos(lat) * np.cos(POLE_LAT) * np.cos(lon - POLE_LON)
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def tag_to_start(tag):
    _, ymd, hms = tag.split("_")
    return "%s-%s-%sT%s:%s:%s.%s" % (ymd[:4], ymd[4:6], ymd[6:8], hms[:2], hms[2:4], hms[4:6], hms[7:])


def summarise(name, cov, covb, norm):
    ok = np.isfinite(norm)
    if ok.sum() == 0:
        print("%-28s n=0" % name)
        return
    q = np.percentile(norm[ok], [25, 50, 75])
    print("%-28s n=%2d  归一 CoV 中位 %.2f  四分位 %.2f–%.2f  < 1 的 %d/%d  "
          "（窗 CoV 中位 %.2f，本底 %.2f）"
          % (name, ok.sum(), q[1], q[0], q[2], int((norm[ok] < 1).sum()), ok.sum(),
             np.median(cov[ok]), np.median(covb[ok])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--t90", required=True)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--f3", required=True)
    ap.add_argument("--index", help="burst_events/index.csv，用来取闪电认证标记")
    args = ap.parse_args()

    t90 = {(r["sat"], r["start"][:23]): r for r in csv.DictReader(open(args.t90))}
    cov = {(r["sat"], r["start"][:23]): r for r in csv.DictReader(open(args.cov))}
    f3 = {tag_to_start(r["tag"]): float(r["f3"]) for r in csv.DictReader(open(args.f3))}
    lit = set()
    if args.index:
        for r in csv.DictReader(open(args.index)):
            if r["lightning"] == "1":
                lit.add(r["start"][:23])

    rows = list(csv.DictReader(open(args.features)))
    key = [(r["sat"], r["start"][:23]) for r in rows]
    miss = [k for k in key if k not in t90 or k not in cov]
    if miss:
        raise SystemExit("t90/cov 表缺 %d 个候选：%s" % (len(miss), miss[:3]))

    sat = np.array([k[0] for k in key])
    start = np.array([k[1] for k in key])
    mlat = dipole_lat(np.array([_f(r["lat"]) for r in rows]), np.array([_f(r["lon"]) for r in rows]))
    t90_us = np.array([_f(t90[k]["t90_us"]) for k in key])
    c = np.array([_f(cov[k]["cov"]) for k in key])
    cb = np.array([_f(cov[k]["cov_bkg"]) for k in key])
    norm = c / cb
    fv = np.array([f3.get(k[1], np.nan) for k in key])
    is_lit = np.array([k[1] in lit for k in key])

    A = (t90_us < T90_CUT_US) & (np.abs(mlat) < MLAT_CUT_DEG)
    B = (t90_us >= T90_CUT_US) & (np.abs(mlat) >= MLAT_CUT_DEG)
    M = ~A & ~B

    print("候选 %d（A %d / B %d / 中间带 %d），f₃ 表覆盖 %d 个（全是 GRID-03B）"
          % (len(rows), A.sum(), B.sum(), M.sum(), int(np.isfinite(fv).sum())))
    print("本底窗 CoV 中位：全体 %.2f；" % np.median(cb[np.isfinite(cb)])
          + "；".join("%s %.2f" % (s, np.median(cb[(sat == s) & np.isfinite(cb)]))
                      for s in sorted(set(sat))))
    print()
    summarise("A 角（全部）", c[A], cb[A], norm[A])
    summarise("  ├ A 角 f₃ = 0（干净）", c[A & (fv == 0)], cb[A & (fv == 0)], norm[A & (fv == 0)])
    summarise("  └ A 角 f₃ > 0（粒子标记）", c[A & (fv > 0)], cb[A & (fv > 0)], norm[A & (fv > 0)])
    summarise("B 角", c[B], cb[B], norm[B])
    summarise("中间带（全部）", c[M], cb[M], norm[M])
    summarise("  ├ 中间带 f₃ = 0", c[M & (fv == 0)], cb[M & (fv == 0)], norm[M & (fv == 0)])
    summarise("  └ 中间带 f₃ > 0", c[M & (fv > 0)], cb[M & (fv > 0)], norm[M & (fv > 0)])
    summarise("7 个闪电认证", c[is_lit], cb[is_lit], norm[is_lit])
    summarise("f₃ > 0 全体（8 A + 4 中间带）", c[fv > 0], cb[fv > 0], norm[fv > 0])
    print()
    print("秩和检验（Mann–Whitney，双侧）：")
    try:
        from scipy.stats import mannwhitneyu
        for lbl, x, y in (
            ("A 角 f₃=0 vs f₃>0", norm[A & (fv == 0)], norm[A & (fv > 0)]),
            ("A 角 vs B 角", norm[A], norm[B]),
            ("闪电认证 vs f₃>0", norm[is_lit], norm[fv > 0]),
        ):
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            if len(x) > 1 and len(y) > 1:
                print("   %-22s n=%d vs %d  p = %.3f" % (lbl, len(x), len(y),
                                                         mannwhitneyu(x, y).pvalue))
    except ImportError:
        print("   （没有 scipy，跳过）")
    print()
    print("f₃ > 0 的逐个（归一 CoV）：")
    for i in np.flatnonzero(fv > 0):
        print("   %s %-10s f₃ %.2f  T90 %7.0f µs  |mlat| %4.1f°  窗 CoV %5.2f  本底 %5.2f  归一 %5.2f  %s"
              % (start[i], sat[i], fv[i], t90_us[i], abs(mlat[i]), c[i], cb[i], norm[i],
                 "A 角" if A[i] else ("B 角" if B[i] else "中间带")))


if __name__ == "__main__":
    main()
