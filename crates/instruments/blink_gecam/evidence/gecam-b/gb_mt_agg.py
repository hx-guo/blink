"""汇总 `gb_merge_tgf.py`：τ = 150 ns 合并对 141 个召回 TGF 和对整个候选池做了什么。

门槛 (1) 的两半：
* **真 TGF 那半**：逐个报固定窗下的计数、本底、显著性变化。"147 个全活着"不够
  ——净计数系统性掉一大截同样是不能用。
* **功效那半（门槛第 0 条）**：这道"掉一个就不用"的门**有没有触发过**。若合并
  在真 TGF 窗里根本没吃掉几个光子，那"全活着"什么也没测到（天格实测过同一种
  空门：`Σλ₃ = 0.012`，门从未触发，而那条判据其实更差）。

**离阈余量 vs 改动幅度**是逐候选可判的，不用统计：余量 > 改动 ⇒ 必定不掉。

用法: python3 gb_mt_agg.py [目录=mergetgf]
"""

import csv
import datetime as dt
import glob
import os
import sys

import numpy as np

EPOCH = (2019, 1, 1)
CATALOG = os.environ.get("GB_CATALOG", "/scratchfs2/gecam/guohx/gecambrun/gecam_tgf_catalog.csv")
MATCH_LO, MATCH_HI = -1000e-6, 3000e-6
THR = 1e-5


def met(iso):
    head, _, frac = iso.rstrip("Z").partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "mergetgf"
    acc, ev = {}, [0, 0]
    files = sorted(glob.glob(f"{directory}/*.npz"))
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k.startswith("n_event"):
                continue
            acc.setdefault(k, []).append(z[k])
        ev[0] += int(z["n_event_pre"][0])
        ev[1] += int(z["n_event_post"][0])
    D = {k: np.concatenate(v) for k, v in acc.items()}
    n = D["t0"].size
    print(f"小时 {len(files)}，候选 {n}")
    print(f"整小时：合并前 {ev[0]} → 合并后 {ev[1]}，合并掉 "
          f"{(ev[0] - ev[1]) / ev[0] * 100:.3f}%")

    ok = np.isfinite(D["fa_pre"]) & np.isfinite(D["fa_post"]) & (D["n_pre"] > 0)
    drop = (D["n_pre"] - D["n_post"]) / np.maximum(D["n_pre"], 1)
    bg = (D["lam_pre"] - D["lam_post"]) / np.maximum(D["lam_pre"], 1e-30)
    print()
    print("=== 整个候选池（固定窗）===")
    print(f"  窗内计数掉的比例：中位 {np.median(drop[ok]) * 100:.3f}%，"
          f"均值 {np.mean(drop[ok]) * 100:.3f}%，p95 {np.percentile(drop[ok], 95) * 100:.3f}%")
    print(f"  本底 λ 掉的比例：中位 {np.median(bg[ok]) * 100:.3f}%  "
          f"（合并把分子分母一起降，只扣分子会把损失夸大）")
    print(f"  一个都没被合并掉的候选占 {(D['n_pre'][ok] == D['n_post'][ok]).mean() * 100:.2f}%")

    # 真 TGF
    catalog = list(csv.DictReader(open(CATALOG)))
    ut = np.array([met(c["UT"]) for c in catalog])
    sel, names = [], []
    for k, c in enumerate(catalog):
        d = D["t0"] - ut[k]
        m = (d >= MATCH_LO) & (d <= MATCH_HI) & ok
        if not m.any():
            continue
        sel.append(int(np.flatnonzero(m)[np.argmin(D["fa_archive"][m])]))
        names.append(c["UT"])
    sel = np.array(sel)
    print()
    print(f"=== 141 个召回 TGF（匹配窗 [−1000, +3000] µs），实得 {sel.size} 个 ===")
    dt_drop = drop[sel]
    print(f"  净计数掉的比例：中位 {np.median(dt_drop) * 100:.3f}%，"
          f"均值 {np.mean(dt_drop) * 100:.3f}%，"
          f"5–95% {np.percentile(dt_drop, 5) * 100:.3f} .. {np.percentile(dt_drop, 95) * 100:.3f}%，"
          f"max {dt_drop.max() * 100:.3f}%")
    print(f"  计数：合并前中位 {np.median(D['n_pre'][sel]):.0f} → 合并后 "
          f"{np.median(D['n_post'][sel]):.0f}")
    print(f"  本底 λ 掉的比例：中位 {np.median(bg[sel]) * 100:.3f}%")
    nz = int((D["n_pre"][sel] == D["n_post"][sel]).sum())
    print(f"  **一个光子都没被合并掉的 TGF：{nz}/{sel.size} = {nz / sel.size * 100:.1f}%**")
    print(f"  被合并掉 ≥1 条的 {sel.size - nz} 个里，掉的条数中位 "
          f"{np.median((D['n_pre'][sel] - D['n_post'][sel])[D['n_pre'][sel] > D['n_post'][sel]]):.0f}")

    print()
    print("=== 显著性（同一窗、本底同步重算，P(X≥count) 口径）===")
    lo, ln = np.log10(np.maximum(D["fa_pre"][sel], 1e-300)), np.log10(np.maximum(D["fa_post"][sel], 1e-300))
    print(f"  log10(fa) 合并前中位 {np.median(lo):.2f} → 合并后 {np.median(ln):.2f}"
          f"（变化中位 {np.median(ln - lo):+.3f} dex）")
    print(f"  变化 5–95%: {np.percentile(ln - lo, 5):+.3f} .. {np.percentile(ln - lo, 95):+.3f} dex"
          f"；最坏 {np.max(ln - lo):+.3f} dex")
    a = int((D["fa_pre"][sel] <= THR).sum())
    b = int((D["fa_post"][sel] <= THR).sum())
    print(f"  过 fa ≤ {THR:g}：合并前 {a}/{sel.size} → 合并后 {b}/{sel.size}"
          f"   掉出 {int(((D['fa_pre'][sel] <= THR) & (D['fa_post'][sel] > THR)).sum())}")

    print()
    print("=== 门槛第 0 条：这道门有没有功效 ===")
    margin = np.log10(THR) - lo
    change = ln - lo
    print(f"  离阈余量(dex)：中位 {np.median(margin):.2f}，min {margin.min():.2f}")
    print(f"  合并造成的改动(dex)：中位 {np.median(change):+.3f}，max {change.max():+.3f}")
    print(f"  余量 − 改动 的最小值 = {np.min(margin - change):.2f} dex"
          f" ⇒ {'全部为正：这道门在任何一个 TGF 上都不可能触发' if np.min(margin - change) > 0 else '有可能触发'}")
    print()
    print("  逐个 TGF 的 q_i 上界（按固定窗、确定性口径）：")
    print(f"    q_i > 0 的 TGF 个数 = {int((D['fa_post'][sel] > THR).sum())}")
    print(f"    Π(1−q_i) 在确定性口径下 = "
          f"{1.0 if int((D['fa_post'][sel] > THR).sum()) == 0 else 0.0:.1f}"
          f"  ⇒ 门的功效 = 1 − Π(1−q_i) = "
          f"{0.0 if int((D['fa_post'][sel] > THR).sum()) == 0 else 1.0:.1f}")
    print("    （确定性口径只给 0/1。真正的 q_i 要按注入量——本脚本给的是"
          "「合并在真 TGF 窗里到底吃掉了多少」，那是 q_i 的必要条件：吃不到就不可能杀掉。）")

    print()
    print("=== 最受影响的 10 个 TGF（按净计数掉的比例）===")
    order = np.argsort(-dt_drop)[:10]
    print(f"    {'UT':<30}{'n_pre':>7}{'n_post':>7}{'掉%':>8}{'log10 fa 前→后':>22}")
    for i in order:
        j = sel[i]
        print(f"    {names[i]:<30}{D['n_pre'][j]:>7.0f}{D['n_post'][j]:>7.0f}"
              f"{dt_drop[i] * 100:>8.2f}{lo[i]:>11.2f} →{ln[i]:>8.2f}")


if __name__ == "__main__":
    main()
