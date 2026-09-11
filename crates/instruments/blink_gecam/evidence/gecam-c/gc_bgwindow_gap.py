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

**⚠️ 缺口必须分三类，混在一起会得出一个构造决定的假富集**（HXMT 在这上面栽过）：

* **正常**——窗内没有任何缺口；
* **小时（chunk）边界**——窗伸到本小时之外。搜索本来就把本底窗夹到 chunk 边界，
  所以**靠近整点的候选必然"有缺口"，这是构造决定的、不是发现**。HXMT 那个
  "距整点 < 1 s 的 9/9 都有大缺口、富集 119 倍"就是这么来的；
* **GTI 内部缺口**——窗落在本小时之内、却落在活时间之外。**只有这一类需要查。**

**C 星在这件事上是全队最可能有真效应的一颗**（占空比 62.7%、缺口 82.4% 在极冠，
那多半是真实的内部缺口而不是文件边界），**也正因如此最不能容忍混进边界效应**。

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


def overlap(a, b, lo, hi):
    """[lo, hi] 与若干段 (a, b) 的交集长度。"""
    return float(np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0, None).sum())


def dead_split(a, b, lo, hi, hour_start, hour_stop):
    """把 [lo, hi] 的死时间拆成「小时边界造成的」与「GTI 内部缺口造成的」两份。

    边界那一份是**构造决定的**（搜索本来就把本底窗夹到 chunk 边界），
    算进富集只会得到一个假结果，所以必须与内部缺口分开。
    """
    width = hi - lo
    inside_hour = overlap(np.array([hour_start]), np.array([hour_stop]), lo, hi)
    boundary = (width - inside_hour) / width
    # 窗落在本小时之内的那一段里，有多少不在活时间上
    live_inside = overlap(np.maximum(a, hour_start), np.minimum(b, hour_stop), lo, hi)
    internal = (inside_hour - live_inside) / width
    return boundary, max(internal, 0.0)


def main():
    root = sys.argv[1]
    controls = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    rng = np.random.default_rng(20260911)

    rows = []          # (boundary, internal, dead_audit_internal, rate_search, rate_audit)
    ctrl_internal = []
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
            # 本小时的边界：整点到整点
            hour_start = met(iso[:13] + ":00:00")
            hour_stop = hour_start + 3600.0
            for s in group:
                t0 = met(s["start"]) + s["delay"]
                t1 = t0 + s["bin_size_best"]
                bd, itn = dead_split(a, b, t0 - SEARCH_HALF, t1 + SEARCH_HALF,
                                     hour_start, hour_stop)
                _, itn_audit = dead_split(a, b, t0 - AUDIT_HALF, t1 + AUDIT_HALF,
                                          hour_start, hour_stop)
                acd = s.get("acd") or {}
                n_bg = acd.get("n_bg")
                rate_search = s["mean"] / s["bin_size_best"]
                rate_audit = (n_bg / (2.0 * AUDIT_HALF)) if n_bg else np.nan
                rows.append((bd, itn, itn_audit, rate_search, rate_audit))
            # 对照：按活时间均匀撒点，只记**内部缺口**那一份
            live = np.maximum(b - a, 0)
            if live.sum() <= 0:
                continue
            pick = rng.choice(live.size, size=controls, p=live / live.sum())
            u = rng.uniform(0, 1, controls)
            t = a[pick] + u * live[pick]
            for centre in t:
                _, itn = dead_split(a, b, centre - SEARCH_HALF, centre + SEARCH_HALF,
                                    hour_start, hour_stop)
                ctrl_internal.append(itn)

    data = np.array(rows)
    bd, itn, itn_audit, rs, ra = (data[:, k] for k in range(5))
    ctrl = np.array(ctrl_internal)
    print(f"候选 {bd.size}，对照点 {ctrl.size}")

    print("\n=== 先分三类（边界那一类是构造决定的，不能算进富集）===")
    normal = (bd <= 0) & (itn <= 0)
    boundary = bd > 0
    internal = (~boundary) & (itn > 0)
    for name, sel in (("正常（窗内无缺口）", normal),
                      ("小时边界（构造决定，排除）", boundary),
                      ("GTI 内部缺口（**只看这一类**）", internal)):
        print(f"  {name:28s} {int(sel.sum()):7d}  {sel.mean() * 100:6.3f}%")

    print("\n=== 几何富集：只用 GTI 内部缺口 ===")
    for name, v in (("候选", itn[~boundary]), ("对照（按活时间均匀）", ctrl)):
        print(f"  {name:20s} 有内部缺口 {(v > 0).mean() * 100:6.3f}%   "
              f"占窗 > 10% 的 {(v > 0.1).mean() * 100:6.3f}%")
    enrich = (itn[~boundary] > 0).mean() / max((ctrl > 0).mean(), 1e-9)
    print(f"  **富集 {enrich:.2f} 倍**（1.0 = 候选不偏爱 GTI 内部缺口）")

    print("\n=== λ 有没有被压低：两个率估计之比随缺口占比怎么走 ===")
    print("搜索率 = mean / bin_size_best（搜索夹了 GTI）")
    print("审计率 = n_bg / 2 s       （审计**不夹** GTI，跨缺口必然被压低）")
    ok = np.isfinite(ra) & (ra > 0) & (~boundary)
    print("（已排除小时边界那一类）")
    print("内部缺口占比档  n     搜索率/审计率 中位   若搜索夹取生效应约为")
    edges = [(-0.001, 1e-9), (1e-9, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)]
    for lo, hi in edges:
        sel = ok & (itn_audit > lo) & (itn_audit <= hi)
        if sel.sum() < 20:
            continue
        # 审计窗被压低 (1 - da) 倍，搜索窗夹取之后不受影响 ⇒ 比值应约为 1/(1-da)
        expect = 1.0 / max(1.0 - np.median(itn_audit[sel]), 1e-6)
        print(f"  {lo:6.3f}–{hi:<6.3f} {int(sel.sum()):6d}   "
              f"{np.median(rs[sel] / ra[sel]):14.3f}   {expect:14.3f}")
    clean = ok & (itn_audit <= 1e-9)
    if clean.sum() >= 20:
        r = rs[clean] / ra[clean]
        print(f"\n  **窗内无缺口那一档是口径对齐的前提**：比值中位 "
              f"{np.median(r):.3f}，5–95% {np.percentile(r, 5):.3f}–{np.percentile(r, 95):.3f}"
              f"（应当在 1 附近；偏离说明两个率本来就不是同一个量，"
              f"那下面每一档的读数都要先扣掉这个偏离）")


if __name__ == "__main__":
    main()
