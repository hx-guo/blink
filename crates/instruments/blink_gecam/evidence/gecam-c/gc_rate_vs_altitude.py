"""GECAM-C：逐小时事例率与轨道高度同不同向。

细梯纪元（2022-08..10）的小时事例数是 2023 那批的 2–3 倍。**这可能根本不是细梯
本身的性质**——C 星的轨道一路衰减（2022Q3 高度 504 km → 2025Q1 325 km），而
**细梯纪元正是全任务里轨道最高的那一段**，高度越高捕获粒子本底越强。

若对上：事例率差有物理来源，不是修法造出来的；顺带也解释了细梯那段为什么有
6.96% 的候选顶到 1 ms 窗长上限（本底高 ⇒ 长窗里也凑够 `min_number`）。
**若对不上，那更值得报**——说明细梯那段有别的东西。

事例率取 `hours.json` 里的 `n_events / searched_seconds`（都是产物里现成的），
高度取同一小时 POSATT 的地心距中位减地球半长轴。**不碰事例流。**

用法: gc_rate_vs_altitude.py <data 根目录> [...]
"""

import collections
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
A = 6378137.0
R_MIN, R_MAX = A + 100e3, A + 900e3


def altitude(day, hour):
    paths = sorted(glob.glob(f"{ROOT}/{day.replace('-', '/')}/POSATT/*_{hour:02d}_v*.fits"))
    if not paths:
        return np.nan
    try:
        with fits.open(paths[-1], memmap=True) as hdus:
            t = hdus["Orbit_Attitude"].data
            r = np.sqrt(
                np.asarray(t["X_WGS84"], float) ** 2
                + np.asarray(t["Y_WGS84"], float) ** 2
                + np.asarray(t["Z_WGS84"], float) ** 2
            )
    except Exception:
        return np.nan
    r = r[np.isfinite(r) & (r >= R_MIN) & (r <= R_MAX)]
    return (np.median(r) - A) / 1e3 if r.size else np.nan


def main():
    rows = []
    for root in sys.argv[1:]:
        for path in sorted(glob.glob(f"{root}/**/*_hours.json", recursive=True)):
            stem = os.path.basename(path)[:8]
            day = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
            raw = json.load(open(path))
            for r in raw if isinstance(raw, list) else raw["hours"]:
                if r["status"] != "searched" or r["searched_seconds"] <= 0:
                    continue
                n = (r.get("metrics") or {}).get("n_events")
                if not n:
                    continue
                rows.append((day, int(r["hour"]), n / r["searched_seconds"],
                             r["n_signals"] / r["searched_seconds"]))
    if not rows:
        print("没有可用的小时")
        return
    # 高度按天算一次就够（一天内变化 < 0.3 km）
    days = sorted({d for d, _, _, _ in rows})
    alt = {}
    for d in days:
        # 归档里一天未必 24 小时都有 POSATT（细梯那段常常只有几个小时），
        # 所以要扫到找着为止，不能只试 0 点和 12 点——否则整季度拿不到高度。
        for hour in range(24):
            value = altitude(d, hour)
            if np.isfinite(value):
                alt[d] = value
                break
        else:
            alt[d] = np.nan

    by_q = collections.defaultdict(list)
    for day, hour, rate, crate in rows:
        q = f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}"
        by_q[q].append((rate, crate, alt[day]))
    print(f"小时 {len(rows)}，天 {len(days)}")
    print("季度      小时数   事例率中位(c/s)   候选率中位(/s)   高度中位(km)")
    xs, ys = [], []
    for q in sorted(by_q):
        v = np.array(by_q[q])
        print(f"{q}   {v.shape[0]:6d}   {np.median(v[:, 0]):14.0f}   "
              f"{np.median(v[:, 1]):14.4f}   {np.nanmedian(v[:, 2]):12.1f}")
        xs.append(np.nanmedian(v[:, 2]))
        ys.append(np.median(v[:, 0]))
    xs, ys = np.array(xs, float), np.array(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() >= 3:
        from scipy import stats
        rho = stats.spearmanr(xs[ok], ys[ok]).statistic
        print(f"\n季度中位：高度 vs 事例率 秩相关 {rho:+.4f}"
              f"（+1 = 高度越高事例率越高，与捕获粒子本底同向）")
    # 逐小时（不按季度聚合）的秩相关，样本大得多
    per = np.array([(alt[d], r) for d, _, r, _ in rows], float)
    ok2 = np.isfinite(per[:, 0]) & np.isfinite(per[:, 1])
    if ok2.sum() >= 50:
        from scipy import stats
        rho2 = stats.spearmanr(per[ok2, 0], per[ok2, 1]).statistic
        print(f"逐小时：高度 vs 事例率 秩相关 {rho2:+.4f}（n = {int(ok2.sum())}）")
        print("  **逐小时那个会被磁纬/南大西洋异常区稀释**，季度中位那个更能看趋势。")


if __name__ == "__main__":
    main()
