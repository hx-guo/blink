"""检验 λ₃ 的"窗内均匀"假设：真暴发在窗内并不均匀，会不会把偶然期望算低了。

解析式 λ₃ = m · Σ_{|S|≥3} Π a_d Π(1−a_d) 假定 n 个事例在 T 上均匀独立。真暴发是有
时间结构的（A 角那种几十微秒的针），**非均匀会抬高真实的偶然三重率**——同样的 n 挤在
更短的有效时长里，撞在同一格的机会更大。低估 λ₃ 会让 Poisson 检验的 P 偏小、判据偏向
误杀，**这是不安全的方向**，所以必须量出来。

两个互相独立的做法：

**甲 逐探头随机平移（MC，保结构）。** 把每一路探头的整列事例**整体**平移一个随机整数
tick 量 δ_d ∈ [−s, s]，再取窗内事例数三重簇。同一路内部的时间结构与死时间一个字节没动，
窗内的整体轮廓也基本没动（s ≪ T 时），被破坏的只有**跨探头的同戳对齐**——正是要当成
偶然的那件事。平移量必须取整数 tick，否则两路永远撞不到同一格。

**四路的平移量取互不相同**，否则有一条漏项会把结果污染成假高：两路平移量恰好相等时，
它们原本的同戳对会**原样复原**，于是真的三重簇会漏进"偶然"里。s = ±1 µs（±4 tick，
9 个取值）时一对撞上的概率是 1/9，三重簇整体复原约 1/81，22 个真簇就漏回 0.27 个——
比要测的量还大。取互不相同的平移量把这条漏项精确归零。

**读这张表要注意选择效应。** 带真三重簇的候选，它的"时间结构"本身就是那些粒子簇，
拿它去估"保结构的偶然"等于让被检验的东西参与定义检验——所以要**分开看 k₃ = 0 的子集**
（无簇可漏、无结构可循环），那才是干净的参照，7 个闪电认证也全在里面。

**乙 局部密度解析（保结构到尺度 w）。** 把窗切成宽 w 的小段，在每段内部用同一个逐格
公式（即"段内均匀"），再把各段加起来。w → T 时退回均匀解析式；w 越小越贴合真实密度，
但 w 不能接近 q，否则每段就是数据本身、λ₃ 会被自己的三重簇顶上去。扫 w 看稳不稳。

用法：
    python3 f3_structure.py <burst_events 目录> [-o out.csv] [--trials 400000]
"""

import argparse
import csv
import os
from itertools import combinations

import numpy as np

Q_US = (2.0**-22) * 1e6
TRIPLE = 3
N_DET = 4
SHIFTS_US = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
WIDTHS_US = (5.0, 10.0, 20.0, 50.0, 100.0)


def lambda3_uniform(counts, m):
    if m <= 0:
        return float("nan")
    a = [min(c / m, 1.0) for c in counts]
    dets = range(len(a))
    lam = 0.0
    for size in range(TRIPLE, len(a) + 1):
        for s in combinations(dets, size):
            pr = 1.0
            for d in dets:
                pr *= a[d] if d in s else (1.0 - a[d])
            lam += pr
    return m * lam


def lambda3_local(tk, det, t0, t1, w_ticks):
    """乙：按宽 w 的小段分块，段内用均匀公式，加起来。tk 是整数 tick。"""
    lam = 0.0
    edges = np.arange(t0, t1 + w_ticks, w_ticks)
    edges[-1] = max(edges[-1], t1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = hi - lo
        if m <= 0:
            continue
        sel = (tk >= lo) & (tk < hi)
        if sel.sum() == 0:
            continue
        counts = np.bincount(det[sel], minlength=N_DET)
        lam += lambda3_uniform(counts, m)
    return lam


def shift_mc(tk_all, det_all, t0, t1, s_ticks, trials, rng):
    """甲：逐探头整体平移整数 tick，量窗内三重簇个数的期望。

    四路的平移量互不相同（不放回），真同戳簇因此一个都复原不了。
    """
    per = [np.sort(tk_all[det_all == d]) for d in range(N_DET)]
    grid = np.arange(-s_ticks, s_ticks + 1)
    k3 = np.zeros(trials, dtype=np.int32)
    nn = np.zeros(trials, dtype=np.int32)
    for i in range(trials):
        delta = rng.choice(grid, size=N_DET, replace=False)
        cols = []
        for d in range(N_DET):
            if per[d].size == 0:
                continue
            sh = per[d] + int(delta[d])
            cols.append(sh[(sh >= t0) & (sh <= t1)])
        if not cols:
            continue
        x = np.sort(np.concatenate(cols))
        nn[i] = x.size
        if x.size < TRIPLE:
            continue
        edge = np.flatnonzero(np.diff(x) != 0)
        starts = np.concatenate(([0], edge + 1))
        ends = np.concatenate((edge + 1, [x.size]))
        k3[i] = int(((ends - starts) >= TRIPLE).sum())
    return k3.mean(), float((k3 > 0).mean()), nn.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("burst_dir")
    ap.add_argument("-o", "--out", default="f3_structure.csv")
    ap.add_argument("--trials", type=int, default=200_000)
    args = ap.parse_args()

    rng = np.random.default_rng(20260911)
    index = {r["csv"]: r for r in csv.DictReader(open(os.path.join(args.burst_dir, "index.csv")))}
    rows = []
    for name in sorted(index):
        meta = index[name]
        path = os.path.join(args.burst_dir, name)
        if not os.path.exists(path):
            continue
        recs = list(csv.DictReader(open(path)))
        tk_all = np.array([int(round(float(r["t_us_rel_best_start"]) / Q_US)) for r in recs])
        det_all = np.array([int(r["det"]) for r in recs])
        t_us = float(meta["bin_size_best_us"])
        t0, t1 = 0, int(round(t_us / Q_US))
        inb = (tk_all >= t0) & (tk_all <= t1)
        n = int(inb.sum())
        if n != int(meta["count"]):
            print("对账不过：%s %d 对 %s" % (meta["start"][:23], n, meta["count"]))
            continue
        counts = np.bincount(det_all[inb], minlength=N_DET)
        lam_u = lambda3_uniform(counts, t_us / Q_US)
        tk_in = np.sort(tk_all[inb])
        edge = np.flatnonzero(np.diff(tk_in) != 0)
        st = np.concatenate(([0], edge + 1))
        en = np.concatenate((edge + 1, [tk_in.size]))
        k3_obs = int(((en - st) >= TRIPLE).sum())
        rec = dict(start=meta["start"][:23], lightning=meta["lightning"], n=n,
                   T_us="%.2f" % t_us, k3_obs=k3_obs, lambda3_uniform="%.4e" % lam_u)
        for s in SHIFTS_US:
            st = int(round(s / Q_US))
            mk, p0, mn = shift_mc(tk_all, det_all, t0, t1, st, args.trials, rng)
            rec["shift%g_lam" % s] = "%.4e" % mk
            rec["shift%g_n" % s] = "%.2f" % mn
        for w in WIDTHS_US:
            wt = int(round(w / Q_US))
            rec["local%g_lam" % w] = "%.4e" % lambda3_local(tk_all[inb], det_all[inb], t0, t1, wt)
        rows.append(rec)

    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    A = lambda k: np.array([float(r[k]) for r in rows])
    lu = A("lambda3_uniform")
    lit = np.array([r["lightning"] == "1" for r in rows])
    clean = np.array([int(r["k3_obs"]) == 0 for r in rows])
    print("候选 %d 个（k₃ = 0 的干净子集 %d 个，含全部 7 个闪电认证），均匀解析 λ₃ 合计 %.4f"
          % (len(rows), clean.sum(), lu.sum()))
    print()
    print("甲 逐探头随机平移（四路平移量互不相同，%d 次/候选/档）：" % args.trials)
    print("  %-10s %10s %8s | %10s %8s %8s | %8s" %
          ("平移量 s", "全体 λ₃", "比均匀", "k₃=0 λ₃", "比均匀", "逐候选中位", "窗内 n"))
    for s in SHIFTS_US:
        ls = A("shift%g_lam" % s)
        nn = A("shift%g_n" % s)
        print("  ±%-9.0fµs %10.4f %8.2f | %10.4f %8.2f %8.2f | %8.1f"
              % (s, ls.sum(), ls.sum() / lu.sum(), ls[clean].sum(),
                 ls[clean].sum() / lu[clean].sum(),
                 np.median((ls / lu)[clean]), np.median(nn)))
    print()
    print("乙 局部密度解析（段内均匀）：")
    print("  %-10s %10s %8s | %10s %8s %8s" %
          ("段宽 w", "全体 λ₃", "比均匀", "k₃=0 λ₃", "比均匀", "逐候选中位"))
    for w in WIDTHS_US:
        ll = A("local%g_lam" % w)
        print("  %-10.0fµs %10.4f %8.2f | %10.4f %8.2f %8.2f"
              % (w, ll.sum(), ll.sum() / lu.sum(), ll[clean].sum(),
                 ll[clean].sum() / lu[clean].sum(), np.median((ll / lu)[clean])))
    print()
    print("只看 7 个闪电认证：均匀 λ₃ 合计 %.4f；平移 ±10 µs %.4f（比 %.2f）；局部 w=20 µs %.4f（比 %.2f）"
          % (lu[lit].sum(), A("shift10_lam")[lit].sum(),
             A("shift10_lam")[lit].sum() / lu[lit].sum(),
             A("local20_lam")[lit].sum(), A("local20_lam")[lit].sum() / lu[lit].sum()))
    print()
    print("7 个闪电认证（均匀 λ₃ → 平移 ±10 µs → 局部 w=20 µs）：")
    for r in rows:
        if r["lightning"] == "1":
            print("   %s  %s → %s → %s"
                  % (r["start"][:19], r["lambda3_uniform"], r["shift10_lam"], r["local20_lam"]))


if __name__ == "__main__":
    main()
