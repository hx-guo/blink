"""批量判过境是不是**片上标定脉冲**：逐过境算梳齿占比、周期、开关时刻。

为什么用梳齿占比而不是 `frac_ev_in3`：后者实测与 |偶极磁纬| 和速率**都强相关，而且方向
对 TGF 不利**（峰在 |磁纬| 6–12°，正是 A 角 TGF 所在；见 why-grid03b 第 16 节）。梳齿占比
量的是"四重簇是不是等间隔排列"，**与磁纬、速率都无关**。

要回答两件事（统筹点名）：
  1. `frac_ev_in3` 超阈的那批过境里，有多少是脉冲过境；
  2. **不是脉冲的那些长什么样**——若有一批不是脉冲，那才是真正需要判据的那一类。

所以本脚本**不只跑超阈的**：`--control` 会另外抽同样多的正常过境作对照，好知道梳齿占比
在正常过境上是多少（预期约 0）。**没有对照就分不清"判据有分辨力"和"判据到处都报正"。**

用法（农场 N 分片）：
    python3 pulser_scan.py <trip_*.csv 所在目录> <outdir> --chunk i --nchunk 4
                           [--sat GRID-03B] [--thr 0.05] [--control 1.0]
产物：<outdir>/pulser_NN.csv + done_NN.txt
"""

import argparse
import csv
import glob
import os

import numpy as np

from pulser_comb import TICK, load, quads

ARCH = "/gecamfs/Exchange/GSDC/missions/GRID/{sat}/fits7/{y}/{m}/{d}/{ver}/{f}"
P0 = 4194.5          # 基频初猜（tick）；1 ms × 2²² = 4194.304
TEETH = 29           # 归属到 1..29 倍齿
TOL = 3.0            # 齿宽（tick）


def comb_stats(t4, p0=P0):
    """返回 (梳齿占比, 回归周期 tick, 残差 rms tick, 梳齿上第一个/最后一个簇的时戳)。"""
    if t4.size < 10:
        return float("nan"), float("nan"), float("nan"), None, None
    d4 = np.diff(t4)
    on = np.zeros(d4.size, bool)
    for k in range(1, TEETH + 1):
        on |= np.abs(d4 - p0 * k) <= TOL
    frac = float(on.mean())
    core = np.concatenate(([on[0]], on[:-1] | on[1:], [on[-1]]))[:t4.size]
    if core.sum() < 10:
        return frac, float("nan"), float("nan"), None, None
    idx = np.concatenate(([0], np.cumsum(np.rint(d4 / p0)))).astype(np.int64)
    A = np.vstack([idx[core], np.ones(int(core.sum()))]).T
    P, c0 = np.linalg.lstsq(A, t4[core].astype(float), rcond=None)[0]
    resid = (t4[core] - (c0 + P * idx[core])).std()
    return frac, float(P), float(resid), float(t4[core].min()), float(t4[core].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tripdir")
    ap.add_argument("outdir")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunk", type=int, default=1)
    ap.add_argument("--sat", default="GRID-03B")
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--control", type=float, default=1.0,
                    help="对照样本相对超阈样本的倍数（同速率档抽），0 = 不要对照")
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.tripdir, "trip_*.csv"))):
        rows += [r for r in csv.DictReader(open(f)) if r.get("frac_ev_in3")]
    fr = np.array([float(r["frac_ev_in3"]) for r in rows])
    rate = np.array([float(r["rate_cps"]) for r in rows])
    over = np.flatnonzero(fr > args.thr)

    # 对照：在超阈样本的速率分布上配平地抽正常过境，**不是随便抽**——
    # frac 与速率强相关，不配平的话对照与被检样本比的是速率不是脉冲。
    ctrl = np.zeros(0, int)
    if args.control > 0 and over.size:
        rng = np.random.default_rng(args.seed)
        under = np.flatnonzero(fr <= args.thr)
        edges = np.percentile(rate[over], [0, 20, 40, 60, 80, 100])
        picks = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            need = int(round(args.control * ((rate[over] >= lo) & (rate[over] <= hi)).sum()))
            pool = under[(rate[under] >= lo) & (rate[under] <= hi)]
            if pool.size:
                picks.append(rng.choice(pool, size=min(need, pool.size), replace=False))
        ctrl = np.concatenate(picks) if picks else ctrl

    sel = np.concatenate([over, ctrl])
    tag = np.array(["over"] * over.size + ["control"] * ctrl.size)
    order = np.argsort(sel)
    sel, tag = sel[order], tag[order]
    mine = np.arange(sel.size)[args.chunk::args.nchunk]
    print("超阈 %d，对照 %d，本分片 %d" % (over.size, ctrl.size, mine.size))

    out = open(os.path.join(args.outdir, "pulser_%02d.csv" % args.chunk), "w", newline="")
    w = csv.writer(out)
    w.writerow(["sat", "day", "pass_file", "tag", "frac_ev_in3", "rate_cps", "dur_s",
                "n_quads", "comb_frac", "period_tick", "period_hz", "resid_ns",
                "on_start_s", "on_stop_s", "note"])
    n_done = 0
    for i in mine:
        r = rows[sel[i]]
        y, m, d = r["day"].split("-")
        path = ARCH.format(sat=args.sat, y=y, m=m, d=d, ver=r["version"], f=r["pass_file"])
        try:
            tk, det, pi, emin, gs, ge = load(path)
            t4, e4, _ = quads(tk, pi, emin)
            cf, P, resid, t0, t1 = comb_stats(t4)
        except Exception as ex:
            w.writerow([args.sat, r["day"], r["pass_file"], tag[i], r["frac_ev_in3"],
                        r["rate_cps"], r["dur_s"], "", "", "", "", "", "", "",
                        "ERR " + str(ex)[:50].replace(",", ";")])
            out.flush()
            continue
        w.writerow([args.sat, r["day"], r["pass_file"], tag[i], r["frac_ev_in3"],
                    r["rate_cps"], "%.1f" % (ge - gs), t4.size,
                    "%.4f" % cf if np.isfinite(cf) else "",
                    "%.4f" % P if np.isfinite(P) else "",
                    "%.3f" % (TICK / P) if np.isfinite(P) and P > 0 else "",
                    "%.0f" % (resid / TICK * 1e9) if np.isfinite(resid) else "",
                    "%.3f" % (t0 / TICK - gs) if t0 else "",
                    "%.3f" % (t1 / TICK - gs) if t1 else "", ""])
        out.flush()
        n_done += 1
    out.close()
    with open(os.path.join(args.outdir, "done_%02d.txt" % args.chunk), "w") as fh:
        fh.write("sat,%s\nchunk,%d\nnchunk,%d\nexpected,%d\ndone,%d\nover,%d\ncontrol,%d\n"
                 % (args.sat, args.chunk, args.nchunk, mine.size, n_done, over.size, ctrl.size))
    print("chunk %d done: %d/%d" % (args.chunk, n_done, mine.size))


if __name__ == "__main__":
    main()
