"""注入已知 f，看联合估计量能不能无偏地拟回来。

背景：三类共享一个 f 的联合拟合给 0.724，只用陆地那一档给 0.821，差 1.13 倍。
联合的 Δχ² 反而比只用陆地低（23.1 对 23.8）——那两类不是"不贡献信息"，
是**带着轻微的反向拉力**。所以要判的是：把无分辨力的档并进来，估计量还准不准。

办法就是刚才对远洋模板做的那件事，只不过对象换成估计量本身：
**造一批真值已知的样本，用同一套拟合去估，看偏不偏。**

两个变体，因为噪声有两个来源，混在一起就分不清是谁的锅：

- **变体 A（只有样本噪声）**：按真值 f 从模板 T 与本底 B 抽样本，再用**同一个** T、B 拟。
  这只检验估计量本身（边界在 [0,1]、似然曲率）带不带偏。
- **变体 B（加上模板噪声）**：样本照旧从 T 抽，但拟的时候用 T 的一个**多项式重抽**
  （抽样数 = 模板真实的 n），把模板自己的抽样噪声注进去。这才是我们实际的处境。
  回归稀释就在这一项里，它**不随被检验样本的 N 缩小**。

各类的样本量固定为实测值（陆 84 / 近岸 64 / 远洋 87）——类归属是已知协变量，
不该随机。
"""
import csv
import gzip
import math
import os
import sys
from datetime import datetime

import numpy as np

import cartopy.io.shapereader as shpreader
import shapely.ops as ops
from shapely.geometry import Point
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(HERE))
_CANDIDATES = [
    os.path.join(_REPO, "crates/instruments/blink_fermi_gbm/evidence",
                 "gbm_tgf_catalog/gbm_tgf_catalog_offline.csv"),
    os.path.join(HERE, "gbm_tgf_catalog/gbm_tgf_catalog_offline.csv"),
    os.path.join(HERE, "gbm_tgf_catalog_offline.csv"),
]
CAT = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
NBIN = 8
KINDS = ("land", "coast", "ocean")
FGRID = np.linspace(0.0, 1.0, 1001)
FLOOR = 1e-6


def wrap(lon):
    return lon - 360 if lon > 180 else lon


def lst_of(row):
    head = row["start"].rstrip("Z").partition(".")[0]
    t = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    return (t.hour + t.minute / 60 + t.second / 3600 + wrap(float(row["lon"])) / 15) % 24


def hist(xs):
    h = np.zeros(NBIN, dtype=np.int64)
    for x in xs:
        h[min(NBIN - 1, max(0, int((x % 24) / 24 * NBIN)))] += 1
    return h


def normalise(h):
    return np.maximum(np.asarray(h, dtype=float) / np.sum(h), FLOOR)


def logtable(tgf, bkg):
    """LOG[i, k] = ln(f_i * T_k + (1 - f_i) * B_k)，与数据无关，只算一次。"""
    return np.log(np.outer(FGRID, tgf) + np.outer(1 - FGRID, bkg))


def fit_from(tables, counts):
    """counts 是 [(类下标, 8 格计数)]，共享 f 的联合最大似然。"""
    ll = np.zeros_like(FGRID)
    for tab, c in zip(tables, counts):
        ll += tab @ c
    i = int(np.argmax(ll))
    return FGRID[i], 2 * (ll[i] - ll[0])


def draw(rng, n, f, tgf, bkg):
    """真值 f：n 个里二项地分成信号与本底，各自按自己的分布落格。"""
    k = rng.binomial(n, f)
    return rng.multinomial(k, tgf / tgf.sum()) + rng.multinomial(n - k, bkg / bkg.sum())


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(20260911)

    fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    g = ops.unary_union(list(shpreader.Reader(fn).geometries()))
    land, near = prep(g), prep(g.buffer(300.0 / 111.0))

    def klass(lon, lat):
        p = Point(wrap(lon), lat)
        return "land" if land.contains(p) else ("coast" if near.contains(p) else "ocean")

    rows = list(csv.reader(open(CAT)))
    hdr = [h.strip().lstrip("#") for h in rows[0]]
    cat = [dict(zip(hdr, [x.strip() for x in r])) for r in rows[1:]]
    src = [r for r in cat if r["LST"] != "NULL" and r["Date"][:4] != "2014"
           and "2010" <= r["Date"][:4] <= "2016"]
    traw = {k: hist([float(r["LST"]) * 24 for r in src
                     if klass(float(r["Lon"]), float(r["Lat"])) == k]) for k in KINDS}
    tmpl = {k: normalise(traw[k]) for k in KINDS}

    with gzip.open(os.path.join(HERE, "cand_2014b.csv.gz"), "rt") as fh:
        pool = list(csv.DictReader(fh))
    brows = [r for r in pool if float(r["fa"]) > 1.0]
    bkg = {k: normalise(hist([lst_of(r) for r in brows
                              if klass(float(r["lon"]), float(r["lat"])) == k]))
           for k in KINDS}

    # 实测的各类样本量：类归属是已知协变量，固定住
    ex = list(csv.DictReader(open(os.path.join(HERE, "extras_sig_2014b.csv"))))
    nk = {k: sum(1 for r in ex if klass(float(r["lon"]), float(r["lat"])) == k) for k in KINDS}
    print(f"各类样本量（实测，固定）：" + "  ".join(f"{k}={nk[k]}" for k in KINDS))
    print(f"模板 n：" + "  ".join(f"{k}={int(traw[k].sum())}" for k in KINDS))
    print(f"重复次数 {trials}\n")

    base = {k: logtable(tmpl[k], bkg[k]) for k in KINDS}
    diffs = {}

    for variant in ("A 只有样本噪声", "B 加上模板噪声（实际处境）"):
        print(f"=== 变体 {variant} ===")
        print(f"{'真值 f':>7s}  {'联合三类':>26s}   {'只用陆地':>26s}")
        print(f"{'':>7s}  {'估计':>8s}{'偏差':>9s}{'RMSE':>8s}   "
              f"{'估计':>8s}{'偏差':>9s}{'RMSE':>8s}")
        for f_true in (0.0, 0.4, 0.6, 0.8, 1.0):
            joint, only = [], []
            for _ in range(trials):
                counts = [draw(rng, nk[k], f_true, tmpl[k], bkg[k]) for k in KINDS]
                if variant.startswith("A"):
                    tabs = [base[k] for k in KINDS]
                else:
                    # 模板的抽样噪声：按模板真实 n 做多项式重抽，再归一
                    tabs = []
                    for k in KINDS:
                        t = rng.multinomial(int(traw[k].sum()), tmpl[k])
                        tabs.append(logtable(normalise(t), bkg[k]))
                joint.append(fit_from(tabs, counts)[0])
                only.append(fit_from([tabs[0]], [counts[0]])[0])
            j, o = np.array(joint), np.array(only)
            print(f"{f_true:7.2f}  {j.mean():8.3f}{j.mean() - f_true:+9.3f}"
                  f"{np.sqrt(((j - f_true) ** 2).mean()):8.3f}   "
                  f"{o.mean():8.3f}{o.mean() - f_true:+9.3f}"
                  f"{np.sqrt(((o - f_true) ** 2).mean()):8.3f}")
            if abs(f_true - 0.8) < 1e-9:
                diffs[variant] = j - o
        print()

    # 实测那两个数差 0.724 − 0.821 = −0.097，异不异常？
    # 两个估计量共用陆地那批数据，强相关，所以差值的分布只能靠注入给。
    print("=== 观测到的「联合 − 只用陆地」= −0.097 异不异常（注入 f = 0.8）===")
    for variant, d in diffs.items():
        p = (np.abs(d) >= 0.097).mean()
        print(f"  变体 {variant[:1]}：差值均值 {d.mean():+.3f}、标准差 {d.std():.3f}，"
              f"|差| ≥ 0.097 的占 {100 * p:.1f}%")


if __name__ == "__main__":
    main()
