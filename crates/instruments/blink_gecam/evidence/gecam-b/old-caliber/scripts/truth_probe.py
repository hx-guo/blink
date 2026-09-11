"""147 个已发表 TGF 的**构造无关**复测，外加 CPD–GRD 时戳差。

两件事，一次读盘。

**一、构造无关的 mf。** 先前那套量（`multiplet_frac` 等）都是在**我们搜索选出
的最佳格**里量的，而最佳格本身是按显著性挑出来的——用它去判"判据会不会切掉
真 TGF"有循环论证的味道。这里改成只锚在**目录的 UT** 上：在 [UT−3 ms, UT+3 ms]
里用固定的 100 µs 滑窗找计数最高的一格（与我们的搜索无关），在那一格里量
计数、同戳占比、CPD 符合。目录的 `Duration_us` 窗一并量一份。

**二、CPD 判据该用的量。** 「候选窗 ±10 µs 内有没有 CPD 计数」在 140 个真 TGF
上 obs/exp = 7.5——真 TGF 自己就强烈超出（CPD 是塑料闪烁体，对 TGF 的伽马有
康普顿响应，目录自带的 CPDtoGRDcountsRatio 中位 0.040 就是这件事），拿它否决
要误杀 72% 的真 TGF。物理上该分开的是**同一次沉积**（带电粒子穿仪器：CPD 与
多路 GRD 同一个时戳）和**两个不同光子**（TGF：CPD 里那一个是另一个伽马散射
来的，没理由与任何 GRD 计数同戳）。所以量 `min |t_CPD − t_GRD|`，亚微秒分辨，
正样本 = 目录 TGF，对照 = 同一小时随机抽的池候选。

用法: python3 truth_probe.py <shard> <nshard> <out_prefix>
"""
import csv, glob, json, sys, random, datetime as dt
import numpy as np

sys.path.insert(0, "/scratchfs2/gecam/guohx")
from gecam_features import read_events, hour_file, met, span

RUN = "/scratchfs2/gecam/guohx/gecambrun"
SAT = "GECAM-B"
shard, nshard, prefix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SEARCH_HALF = 3e-3      # 锚在 UT 上的搜寻范围
PEAK_W = 1e-4           # 固定 100 µs 滑窗（与已发表判据的分箱同宽）
CPD_HALF = 1e-4         # 取 CPD 的外扩
N_CTRL = 300            # 每小时抽多少池候选当对照


def utc_seconds(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    return (dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S")
            .replace(tzinfo=dt.timezone.utc).timestamp()
            + (float("0." + f) if f else 0.0))


# 目录 UT -> MET：用我们已匹配上的 142 个反推 UTC−MET 的常数（两边都是 TAI 派生，
# 应该是个常数；实测它的离散度就是这套时间口径的一致性检验）
sigs = json.load(open(RUN + "/tgf_signals.json"))
offs = [utc_seconds(s["start"]) - met(s["start"], SAT) for s in sigs]
OFF = float(np.median(offs))
print("UTC−MET 常数 %.6f s，离散度 %.3g s" % (OFF, float(np.std(offs))))

cat = []
for row in csv.DictReader(open(RUN + "/gecam_tgf_catalog.csv")):
    t_utc = utc_seconds(row["UT"])
    cat.append(dict(ut=row["UT"], met=t_utc - OFF,
                    dur=float(row["Duration_us"]) * 1e-6,
                    net=float(row["NetCounts"]),
                    cpd=float(row["CPDtoGRDcountsRatio"]),
                    iso=row["UT"] if row["UT"].endswith("Z") else row["UT"] + "Z"))

hours = sorted({c["ut"][:13] for c in cat})
mine = {h for i, h in enumerate(hours) if i % nshard == shard}
cat = [c for c in cat if c["ut"][:13] in mine]

pool = {}
rng = random.Random(20260911 + shard)
for f in sorted(glob.glob(RUN + "/data/GECAM-B/*/*/*_signals.json")):
    day = f.split("/")[-1][:8]
    day_iso = "%s-%s-%s" % (day[:4], day[4:6], day[6:8])
    if not any(h.startswith(day_iso) for h in mine):
        continue
    for s in json.load(open(f)):
        k = s["start"][:13]
        if k in mine:
            pool.setdefault(k, []).append(s)
for k in pool:
    if len(pool[k]) > N_CTRL:
        pool[k] = rng.sample(pool[k], N_CTRL)
print("这一片 %d 小时, 目录 TGF %d 个, 对照候选 %d 个"
      % (len(mine), len(cat), sum(len(v) for v in pool.values())))


def mf_of(t):
    if len(t) == 0:
        return 0.0, 0
    _, c = np.unique(t, return_counts=True)
    return float(c[c >= 2].sum()) / len(t), int(c.max())


def cpd_dt(grd_t, cpd_t, t0, t1):
    """窗内每个 CPD 计数到最近 GRD 计数的 |Δt|（µs），以及同戳（Δt==0）对数。"""
    clo, chi = span(cpd_t, t0, t1)
    if chi <= clo:
        return [], 0
    glo, ghi = span(grd_t, t0 - 1e-3, t1 + 1e-3)
    if ghi <= glo:
        return [-1.0] * (chi - clo), 0
    g = grd_t[glo:ghi]
    out, ex = [], 0
    for t in cpd_t[clo:chi]:
        i = np.searchsorted(g, t)
        best = np.inf
        for j in (i - 1, i):
            if 0 <= j < len(g):
                best = min(best, abs(g[j] - t))
        out.append(float(best) * 1e6)
        if best == 0.0:
            ex += 1
    return out, ex


rows = []
by_hour = {}
for c in cat:
    by_hour.setdefault(c["ut"][:13], []).append(c)

for k in sorted(set(list(by_hour) + list(pool))):
    iso = k + ":00:00"
    pg, pc = hour_file(SAT, iso, "grd"), hour_file(SAT, iso, "cpd")
    if not pg:
        print("缺 GRD:", k)
        continue
    gt, gch, gdet = read_events(pg, want_pi=True)
    ct = read_events(pc, want_pi=False)[0] if pc else None

    # ---- 目录 TGF：构造无关 ----
    for c in by_hour.get(k, []):
        t = c["met"]
        # 本底
        blo, bhi = span(gt, t - 1.0, t + 1.0)
        hlo, hhi = span(gt, t - 0.01, t + 0.01)
        rate = ((bhi - blo) - (hhi - hlo)) / (2.0 - 0.02)
        # 固定 100 µs 滑窗找峰：候选起点取窗内每个事例的时刻
        slo, shi = span(gt, t - SEARCH_HALF, t + SEARCH_HALF)
        seg = gt[slo:shi]
        best_n, best_t = 0, t
        if len(seg):
            ends = np.searchsorted(seg, seg + PEAK_W, side="left")
            cnts = ends - np.arange(len(seg))
            i = int(np.argmax(cnts))
            best_n, best_t = int(cnts[i]), float(seg[i])
        plo, phi = span(gt, best_t, best_t + PEAK_W)
        mf_pk, mx_pk = mf_of(gt[plo:phi])
        ndet_pk = len(np.unique(gdet[plo:phi]))
        # 目录自己的窗
        clo2, chi2 = span(gt, t, t + c["dur"])
        mf_cat, mx_cat = mf_of(gt[clo2:chi2])
        d_pk, ex_pk = (cpd_dt(gt, ct, best_t - CPD_HALF, best_t + PEAK_W + CPD_HALF)
                       if ct is not None else ([], 0))
        rows.append(dict(tag="tgf", ut=c["ut"], net=c["net"],
                         dur_us=round(c["dur"] * 1e6, 1), cat_cpd=c["cpd"],
                         bg_rate=round(rate, 1),
                         peak_off_us=round((best_t - t) * 1e6, 2),
                         n_peak=best_n, exp_peak=round(rate * PEAK_W, 3),
                         mf_peak=round(mf_pk, 3), max_peak=mx_pk, ndet_peak=ndet_pk,
                         n_cat=chi2 - clo2, mf_cat=round(mf_cat, 3), max_cat=mx_cat,
                         exp_cat=round(rate * c["dur"], 3),
                         cpd_dts=[round(x, 4) for x in d_pk], cpd_exact=ex_pk))

    # ---- 对照：池候选的最佳格 ----
    if ct is not None:
        for s in pool.get(k, []):
            t0 = met(s["start"], SAT) + s["delay"]
            t1 = t0 + s["bin_size_best"]
            lo, hi = span(gt, t0, t1)
            mf, mx = mf_of(gt[lo:hi])
            d, ex = cpd_dt(gt, ct, t0 - CPD_HALF, t1 + CPD_HALF)
            rows.append(dict(tag="ctrl", ut=s["start"],
                             fa=s["false_positive_per_year"],
                             bin_us=round(s["bin_size_best"] * 1e6, 4),
                             n_peak=hi - lo, mf_peak=round(mf, 3), max_peak=mx,
                             ndet_peak=int(len(np.unique(gdet[lo:hi]))),
                             cpd_dts=[round(x, 4) for x in d], cpd_exact=ex))
    del gt, gch, gdet, ct

json.dump(rows, open(prefix + "_%d.json" % shard, "w"))
print("落盘 %d 行 (tgf %d, ctrl %d)"
      % (len(rows), sum(r["tag"] == "tgf" for r in rows),
         sum(r["tag"] == "ctrl" for r in rows)))
