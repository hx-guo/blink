"""B 角候选出现在地磁扰动时段吗？——把阴性证据换成阳性证据的一条腿。

## 为什么值得做

现有支持"B 角不是 TGF"的证据几乎全是阴性的（没有闪电关联、谱像本底、磁纬高），
而阴性证据永远留着"只是没探到"的解释空间。**"这批东西跟地磁活动同步"是阳性的**：
雷暴不跟 Kp 走，磁层沉降电子跟。未决项 14 把这条记成"没验成"。

## 口径（这条判据最容易做错的地方）

**不能拿候选的 ap 分布跟全球 ap 分布比**——要跟**这颗星自己的曝光**在 ap 上的分布比。
曝光表是逐日的（`diag/.../exp*_days.csv`：GTI、可定位、按 |偶极磁纬| 分档），
而 ap 是三小时一格，所以把每天的曝光**均摊到当天 8 格**，并按候选所在的磁纬档
取对应的曝光列（B 角在 |mlat| ≥ 30，就用 `pos_s − mlat_lt30_s`）。

**内部负对照**：A 角是 GRID-03B 的 7 个闪电证实所在的那一群，是真 TGF。
**真 TGF 不该跟 Kp 相关**。若 A 角也给出同样的富集，说明这条判据在量别的东西
（例如曝光模型不对），整条作废。

检验用 Mann–Whitney（候选的 ap 对曝光加权的 ap 抽样），并报中位数与四分位。

用法: python3 kp_test.py <kp.txt> <rows80.json> <曝光表目录>
"""

import csv
import datetime as dt
import os
import json
import sys

import numpy as np


def load_ap(path):
    """返回 {(YYYY,MM,DD): [ap1..ap8]} 与 {(Y,M,D): Ap}。"""
    ap3, apd = {}, {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 28:
                continue
            key = (int(p[0]), int(p[1]), int(p[2]))
            vals = [int(p[i]) for i in range(15, 23)]
            if min(vals) < 0:
                continue
            ap3[key] = vals
            apd[key] = int(p[23])
    return ap3, apd


def exposure_weighted(expfile, ap3, col):
    """按曝光加权的 ap 样本：每天的曝光均摊到 8 个三小时格。返回 (ap 值, 权重秒)。"""
    vals, wts = [], []
    with open(expfile) as f:
        for r in csv.DictReader(f):
            d = r["day"]
            key = (int(d[:4]), int(d[4:6]), int(d[6:8]))
            a = ap3.get(key)
            if a is None:
                continue
            sec = float(r[col[0]]) - (float(r[col[1]]) if col[1] else 0.0)
            if sec <= 0:
                continue
            for v in a:
                vals.append(v)
                wts.append(sec / 8.0)
    return np.array(vals, dtype=float), np.array(wts, dtype=float)


def wquantile(v, w, q):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, v))


def mannwhitney_w(x, v, w, n=200000, seed=7):
    """候选样本 x 对曝光加权样本 (v, w) 的 Mann–Whitney U 与单侧 p（正态近似）。"""
    rng = np.random.default_rng(seed)
    draw = rng.choice(v, size=n, p=w / w.sum())
    gt = sum((draw < xi).sum() + 0.5 * (draw == xi).sum() for xi in x)
    n1, n2 = len(x), n
    u = gt
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u - mu) / sd
    from math import erfc, sqrt
    return float(u / (n1 * n2)), float(z), float(0.5 * erfc(z / sqrt(2)))


def main():
    kp, rowsfile, expdir = sys.argv[1], sys.argv[2], sys.argv[3]
    ap3, apd = load_ap(kp)
    rows = json.load(open(rowsfile))
    files = {"GRID-02": "g02.csv", "GRID-03B": "g03b.csv",
             "GRID-04": "g04.csv", "GRID-07": "g07.csv"}
    for grp, col, tag in (("B", ("pos_s", "mlat_lt30_s"), "|mlat| ≥ 30"),
                          ("A", ("mlat_lt30_s", None), "|mlat| < 30")):
        sub = [r for r in rows if r["grp"] == grp]
        x = []
        for r in sub:
            s = r["start"]
            d = dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
            a = ap3.get((d.year, d.month, d.day))
            if a is None:
                continue
            x.append(a[d.hour // 3])
        x = np.array(x, dtype=float)
        # 曝光基线：把该群涉及的星按各自曝光合起来
        sats = sorted({r["sat"] for r in sub})
        vv, ww = [], []
        for s in sats:
            p = os.path.join(expdir, files[s])
            if not os.path.exists(p):
                continue
            v, w = exposure_weighted(p, ap3, col)
            vv.append(v)
            ww.append(w)
        v = np.concatenate(vv)
        w = np.concatenate(ww)
        auc, z, p = mannwhitney_w(x, v, w)
        print("\n=== %s 角（n=%d，星 %s，曝光取 %s）===" % (grp, len(x), ",".join(sats), tag))
        print("  候选 ap: 中位 %.1f  四分位 %.1f–%.1f  最大 %.0f"
              % (np.median(x), np.percentile(x, 25), np.percentile(x, 75), x.max()))
        print("  曝光 ap: 中位 %.1f  四分位 %.1f–%.1f"
              % (wquantile(v, w, 0.5), wquantile(v, w, 0.25), wquantile(v, w, 0.75)))
        print("  P(候选 ap > 曝光 ap) = %.3f   z = %+.2f   单侧 p = %.2e" % (auc, z, p))
        for thr in (15, 27, 48):
            fc = float((x >= thr).mean())
            fe = float(w[v >= thr].sum() / w.sum())
            print("    ap ≥ %-3d：候选 %5.1f%%  曝光 %5.1f%%  富集 %.2f×"
                  % (thr, 100 * fc, 100 * fe, fc / fe if fe > 0 else float("nan")))


if __name__ == "__main__":
    main()
