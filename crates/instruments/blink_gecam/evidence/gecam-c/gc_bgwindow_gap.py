"""GECAM-C：候选的本底窗跨不跨 GTI 缺口，跨了之后 λ 有没有被压低。

HXMT 发现：**本底窗半边落在数据缺口上 ⇒ λ 低估约 2 倍 ⇒ 几十毫秒内一串普通
涨落同时越线**，一口气解释了它三条分开记的缺陷。**C 星的 GTI 占空比只有 62.7%、
缺口 82.4% 落在极冠，跨缺口的风险比 HXMT 高得多**，所以要查。

**这个检验不用碰事例流**，候选表里已经有两个独立的率估计：

* `mean`（λ）是搜索算的，搜索**把本底窗夹到活时间里再算时长**
  （见 `types/chunk/search.rs` 里 `CPD_BASELINE_SECONDS` 的注释）
  ⇒ `rate_search = mean / bin_size_best`；
* `acd.n_bg` 是审计算的，审计**不夹 GTI**、分母是标称的 2 s
  ⇒ `rate_audit = n_bg / 2.0`，**窗跨了缺口就会被压低**。

于是：

* 两个率在**窗内没有缺口**的候选上应当一致 —— 这是口径对齐的前提，先验它；
* 窗**跨了缺口**时 `rate_search / rate_audit` 应当跟着缺口占比涨 ——
  **涨了说明搜索的夹取在起作用（安全）**，**不涨说明 λ 跟着审计一起被压低（有缺陷）**。

外加一条纯几何的富集检验：候选的本底窗跨缺口的比例，对照是**按活时间均匀撒点**
的同一比例。**富集说明候选偏爱 GTI 边缘，那本身就是可疑的。**

用法: gc_bgwindow_gap.py <data 根目录> [每小时对照点数=2000]
"""

import datetime as dt
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = (2021, 1, 1)
# 搜索的本底窗**全宽** 1 s（候选两侧各 0.5 s），审计的基线窗是两侧各 1 s。
SEARCH_HALF = 0.5
AUDIT_HALF = 1.0


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def newest(pattern):
    best = None
    for path in glob.glob(pattern):
        parts = os.path.basename(path).removesuffix(".fits").split("_")
        if len(parts) >= 5 and parts[-1].startswith("v"):
            try:
                v = int(parts[-1][1:])
            except ValueError:
                continue
            if best is None or v > best[0]:
                best = (v, path)
    return best[1] if best else None


def live_fraction(a, b, lo, hi):
    """[lo, hi] 与 GTI 段 (a, b) 的交集长度 / (hi - lo)。"""
    return float(np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0, None).sum()) / (hi - lo)


def main():
    root = sys.argv[1]
    controls = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    rng = np.random.default_rng(20260911)

    rows = []          # (dead_search, dead_audit, rate_search, rate_audit, n_bg)
    ctrl_dead = []
    for path in sorted(glob.glob(f"{root}/**/*_signals.json", recursive=True)):
        signals = json.load(open(path))
        by_hour = {}
        for s in signals:
            by_hour.setdefault(s["start"][:13], []).append(s)
        for hour_key, group in sorted(by_hour.items()):
            iso = group[0]["start"]
            grd = newest(
                f"{ROOT}/{iso[:10].replace('-', '/')}/GRD_EVT/gcg_evt_*_{iso[11:13]}_v*.fits"
            )
            if grd is None:
                continue
            try:
                with fits.open(grd, memmap=True) as hdus:
                    gti = hdus["GTI"].data
                    a = np.asarray(gti["START"], float)
                    b = np.asarray(gti["STOP"], float)
            except Exception:
                continue
            for s in group:
                t0 = met(s["start"]) + s["delay"]
                t1 = t0 + s["bin_size_best"]
                ds = 1.0 - live_fraction(a, b, t0 - SEARCH_HALF, t1 + SEARCH_HALF)
                da = 1.0 - live_fraction(a, b, t0 - AUDIT_HALF, t1 + AUDIT_HALF)
                acd = s.get("acd") or {}
                n_bg = acd.get("n_bg")
                rate_search = s["mean"] / s["bin_size_best"]
                rate_audit = (n_bg / (2.0 * AUDIT_HALF)) if n_bg else np.nan
                rows.append((ds, da, rate_search, rate_audit))
            # 对照：按活时间均匀撒点
            live = np.maximum(b - a, 0)
            if live.sum() <= 0:
                continue
            pick = rng.choice(live.size, size=controls, p=live / live.sum())
            u = rng.uniform(0, 1, controls)
            t = a[pick] + u * live[pick]
            for centre in t:
                ctrl_dead.append(1.0 - live_fraction(a, b, centre - SEARCH_HALF,
                                                     centre + SEARCH_HALF))

    data = np.array(rows)
    ds, da, rs, ra = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    ctrl = np.array(ctrl_dead)
    print(f"候选 {ds.size}，对照点 {ctrl.size}")

    print("\n=== 几何富集：本底窗跨 GTI 缺口的比例 ===")
    for name, v in (("候选", ds), ("对照（按活时间均匀）", ctrl)):
        print(f"  {name:20s} 有缺口 {(v > 0).mean() * 100:6.3f}%   "
              f"缺口占窗 > 10% 的 {(v > 0.1).mean() * 100:6.3f}%   "
              f"中位缺口占比 {np.median(v):.4f}")
    enrich = (ds > 0).mean() / max((ctrl > 0).mean(), 1e-9)
    print(f"  **富集 {enrich:.2f} 倍**（1.0 = 候选不偏爱 GTI 边缘）")

    print("\n=== λ 有没有被压低：两个率估计之比随缺口占比怎么走 ===")
    print("搜索率 = mean / bin_size_best（搜索夹了 GTI）")
    print("审计率 = n_bg / 2 s       （审计**不夹** GTI，跨缺口必然被压低）")
    ok = np.isfinite(ra) & (ra > 0)
    print("缺口占比档      n     搜索率/审计率 中位   若搜索夹取生效应约为")
    edges = [(-0.001, 1e-9), (1e-9, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)]
    for lo, hi in edges:
        sel = ok & (da > lo) & (da <= hi)
        if sel.sum() < 20:
            continue
        # 审计窗被压低 (1 - da) 倍，搜索窗夹取之后不受影响 ⇒ 比值应约为 1/(1-da)
        expect = 1.0 / max(1.0 - np.median(da[sel]), 1e-6)
        print(f"  {lo:6.3f}–{hi:<6.3f} {int(sel.sum()):6d}   "
              f"{np.median(rs[sel] / ra[sel]):14.3f}   {expect:14.3f}")
    clean = ok & (da <= 1e-9)
    if clean.sum() >= 20:
        r = rs[clean] / ra[clean]
        print(f"\n  **窗内无缺口那一档是口径对齐的前提**：比值中位 "
              f"{np.median(r):.3f}，5–95% {np.percentile(r, 5):.3f}–{np.percentile(r, 95):.3f}"
              f"（应当在 1 附近；偏离说明两个率本来就不是同一个量，"
              f"那下面每一档的读数都要先扣掉这个偏离）")


if __name__ == "__main__":
    main()
