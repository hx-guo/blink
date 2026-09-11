"""GECAM-A：电子学时戳抖动的直接测量，以及按抖动尺度合并跨探头计数后的候选率。

出发点：A 星候选窗 `bin_size_best` 中位 0.15 µs，窗内 8-10 个计数点亮 7-8 路探头。
要判断这是不是"极短的暴"，得先知道仪器能把同一个物理事例的时戳抹开多宽。

**抖动的直接测量是双增益对。** 同一个物理光子被 ADC 两个增益支路各写一行，两条
记录 100% 是同一个事例（OPEN-QUESTIONS 第 17 条实测 `FLAG` 组合全是 (0,10)/(10,0)）。
它们的时戳差就是电子学时戳抖动，**与候选、与搜索完全无关，构造无关**。
对照组用同一路探头、增益档相同的相邻对：那是两个真正独立的事例，时戳差该是泊松的。

量完抖动尺度 τ 之后按 τ 把跨探头的计数并成一个事例（一次粒子穿越在泊松意义上
本来就是一个事件），再看候选还剩多少。合并后本底率同步下降，所以 `fa` 要用
反标定的试验数重算，不能只看窗内计数。

用法: python3 ga_jitter.py <YYYY-MM-DD> <signals.json 或 -> <输出前缀>
"""

import glob
import json
import sys
import datetime as dt

import numpy as np
from astropy.io import fits
from scipy.special import gammainc

ROOT = "/gecamfs/Archived-DATA/GSDC/LEVEL1/daily"
MIN_CHANNEL, OVERFLOW, NORMAL = 54, 448, 1
EPOCH = (2019, 1, 1)
DEADTIME_S = 4e-6
MIN_NUMBER = 8
DAYS_PER_YEAR = 365.25
# 合并窗，秒。0 = 严格同戳；其余按实测抖动尺度铺开
TAUS = (0.0, 3e-8, 6e-8, 1e-7, 1.5e-7, 2e-7, 3e-7, 5e-7, 1e-6)


def met(iso):
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def hour_files(day):
    d = f"{ROOT}/{day.replace('-', '/')}/GECAM_A/GRD_evt"
    stems = {}
    for f in sorted(glob.glob(f"{d}/gag_evt_*_v*.fits")):
        stems.setdefault(f.rsplit("_v", 1)[0], []).append(f)
    return [sorted(v)[-1] for _, v in sorted(stems.items())]


def bump_start(hist):
    """堆积包起点：与 recut.py 同一套求法，逐探头逐过境从数据自己求，不查表。"""
    nz = np.where(hist >= 10)[0]
    if nz.size == 0:
        return -1, -1
    edge = int(nz[-1])
    lo, hi = max(0, edge - 80), max(1, edge - 40)
    cont = float(np.median(hist[lo:hi])) if hi > lo else 0.0
    if cont < 5:
        return edge, -1
    seg = hist[lo:edge + 1]
    idx = np.where(seg >= 3 * cont)[0]
    if idx.size == 0:
        return edge, -1
    gaps = np.where(np.diff(idx) > 3)[0]
    start = idx[gaps[-1] + 1] if gaps.size else idx[0]
    return edge, int(lo + start)


def quantum(t0):
    """归档把时刻存成 float64 秒，量化步就是这个量级上的 ulp。"""
    return float(np.spacing(t0))


def process_hour(path, out):
    times, chans, dets = [], [], []
    jit_diff, ctrl_diff = [], []     # 一高一低 / 同档，同探头相邻对的时戳差
    per_det = {}
    n_raw_total = n_admit_total = 0
    with fits.open(path, memmap=True) as hdus:
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            det = int(hdu.name[-2:])
            d = hdu.data
            if d is None or len(d) == 0:
                continue
            pi = np.asarray(d["PI"], np.int32)
            et = np.asarray(d["EVT_TYPE"], np.int32)
            gt = np.asarray(d["GAIN_TYPE"], np.int32)
            t = np.asarray(d["TIME"], float)
            n_raw_total += t.size
            keep = (et == NORMAL) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW)
            t, pi, gt = t[keep], pi[keep], gt[keep]
            if t.size == 0:
                continue
            n_admit_total += t.size
            o = np.lexsort((-gt, t))
            t, pi, gt = t[o], pi[o], gt[o]

            # (a) 抖动：同探头相邻对，Δt < 死时间。一高一低 = 同一物理事例的两支；
            #     同档 = 两个独立事例，作零假设对照。
            dt_adj = t[1:] - t[:-1]
            near = dt_adj < DEADTIME_S
            cross = near & (gt[1:] != gt[:-1])
            same = near & (gt[1:] == gt[:-1])
            jit_diff.append(dt_adj[cross])
            ctrl_diff.append(dt_adj[same])

            # 双增益去重，与 Rust 侧同规则：保留低增益那条
            drop = np.zeros(t.size, bool)
            drop[1:] = cross
            t, pi, gt = t[~drop], pi[~drop], gt[~drop]

            lowg = gt == 1
            hist = np.bincount(pi[lowg], minlength=OVERFLOW) if lowg.any() else np.zeros(OVERFLOW, int)
            edge, cut = bump_start(hist)
            n_over = int((pi >= cut).sum()) if cut > 0 else 0
            per_det[det] = dict(edge=edge, cut=cut, n=int(t.size), n_low=int(lowg.sum()),
                                n_over=n_over, merged=int(drop.sum()))
            times.append(t)
            chans.append(pi)
            dets.append(np.full(t.size, det, np.int16))

    if not times:
        return None
    time = np.concatenate(times)
    o = np.argsort(time, kind="stable")
    time = time[o]
    chan = np.concatenate(chans)[o]
    det = np.concatenate(dets)[o]
    del times, chans, dets

    jit = np.concatenate(jit_diff)
    ctrl = np.concatenate(ctrl_diff)
    q = quantum(float(time[0]))
    out["quantum_ns"] = q * 1e9
    out["n_raw"] = n_raw_total
    out["n_admit"] = n_admit_total
    out["n_after_dedupe"] = int(time.size)
    out["merged_gain_duplicates"] = int(sum(v["merged"] for v in per_det.values()))
    out["per_det"] = {str(k): v for k, v in per_det.items()}
    n_over = sum(v["n_over"] for v in per_det.values())
    out["overrange_frac_global"] = n_over / max(time.size, 1)

    out["jitter"] = dict(
        n=int(jit.size),
        frac_of_admit=float(jit.size / max(n_admit_total, 1)),
        ns=[float(np.percentile(jit, p) * 1e9) for p in (0, 5, 25, 50, 75, 90, 95, 99, 100)],
        steps=[float(np.percentile(jit, p) / q) for p in (50, 90, 99, 100)],
        zero_frac=float((jit == 0).mean()),
    )
    out["jitter_control"] = dict(
        n=int(ctrl.size),
        ns=[float(np.percentile(ctrl, p) * 1e9) for p in (0, 5, 25, 50, 75, 90, 95, 99, 100)] if ctrl.size else [],
        zero_frac=float((ctrl == 0).mean()) if ctrl.size else None,
    )
    # 抖动的直方（以量化步为单位），看它是不是集中在头几步
    if jit.size:
        steps = np.round(jit / q).astype(np.int64)
        cnt = np.bincount(steps[steps < 64], minlength=64)
        out["jitter_hist_steps"] = [int(x) for x in cnt]

    return time, chan, det, q


def cluster_stats(time, det, tau, sample_cap=400000):
    """按 τ 单链接聚簇。返回簇标签与统计。"""
    if tau <= 0:
        brk = time[1:] != time[:-1]
    else:
        brk = (time[1:] - time[:-1]) > tau
    labels = np.empty(time.size, np.int64)
    labels[0] = 0
    labels[1:] = np.cumsum(brk)
    n_cluster = int(labels[-1]) + 1
    size = np.bincount(labels, minlength=n_cluster)
    multi = np.where(size >= 2)[0]
    stat = dict(tau_ns=tau * 1e9, n_cluster=n_cluster,
                merged_frac=float(1 - n_cluster / time.size),
                n_multi=int(multi.size),
                size_hist=[int(x) for x in np.bincount(size, minlength=12)[:12]])
    # 簇内不同探头数：只对多成员簇算，必要时抽样
    if multi.size:
        pick = multi if multi.size <= sample_cap else multi[
            np.linspace(0, multi.size - 1, sample_cap).astype(np.int64)]
        start = np.searchsorted(labels, pick, "left")
        stop = np.searchsorted(labels, pick, "right")
        nd = np.array([np.unique(det[a:b]).size for a, b in zip(start, stop)])
        span = np.array([time[b - 1] - time[a] for a, b in zip(start, stop)])
        stat["multi_ndet"] = dict(
            median=float(np.median(nd)), mean=float(nd.mean()),
            frac_ndet_ge2=float((nd >= 2).mean()), frac_ndet_ge3=float((nd >= 3).mean()),
            hist=[int(x) for x in np.bincount(nd, minlength=12)[:12]])
        stat["multi_span_ns"] = [float(np.percentile(span, p) * 1e9) for p in (50, 90, 99, 100)]
    return labels, stat


def main():
    day, sig_path, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    report = {"day": day, "hours": {}}
    signals = []
    if sig_path != "-":
        signals = sorted(json.load(open(sig_path)), key=lambda s: s["start"])
        print(f"候选 {len(signals)}", flush=True)

    by_hour = {}
    for s in signals:
        by_hour.setdefault(s["start"][11:13], []).append(s)

    rows = []
    for path in hour_files(day):
        hh = path.rsplit("_", 2)[-2]
        out = {}
        res = process_hour(path, out)
        if res is None:
            continue
        time, chan, det, q = res
        j = out["jitter"]
        print(f"[{hh}] 事例 {out['n_admit']:,} -> 去重后 {out['n_after_dedupe']:,} "
              f"(双增益并 {out['merged_gain_duplicates']:,} = {out['merged_gain_duplicates']/max(out['n_admit'],1)*100:.2f}%)"
              f" q={out['quantum_ns']:.2f}ns 超量程全局占比 {out['overrange_frac_global']*100:.2f}%",
              flush=True)
        print(f"     抖动 n={j['n']:,} 中位 {j['ns'][3]:.1f}ns p90 {j['ns'][5]:.1f}ns "
              f"p99 {j['ns'][7]:.1f}ns 严格同戳占 {j['zero_frac']*100:.1f}%", flush=True)
        c = out["jitter_control"]
        if c["ns"]:
            print(f"     对照(同档) n={c['n']:,} 中位 {c['ns'][3]:.1f}ns p5 {c['ns'][1]:.1f}ns", flush=True)

        # 候选窗的索引区间先定下来，逐 τ 算完就把标签释放（一份标签 350 MB）
        hour_signals = by_hour.get(hh, [])
        base = []
        for s in hour_signals:
            t0 = met(s["start"]) + s["delay"]
            t1 = t0 + s["bin_size_best"]
            i0 = int(np.searchsorted(time, t0, "left"))
            i1 = int(np.searchsorted(time, t1, "right"))
            if i1 > i0:
                base.append([s["start"][:23], hh, s["false_positive_per_year"], s["count"],
                             s["mean"], s["bin_size_best"] * 1e6, i1 - i0, i0, i1])

        out["clusters"] = []
        for tau in TAUS:
            labels, stat = cluster_stats(time, det, tau)
            out["clusters"].append(stat)
            md = stat.get("multi_ndet", {})
            print(f"     τ={tau*1e9:6.1f}ns 簇 {stat['n_cluster']:,} 并掉 {stat['merged_frac']*100:5.2f}% "
                  f"多成员簇 {stat['n_multi']:,} 簇内探头中位 {md.get('median', 0):.0f} "
                  f"≥2 路占 {md.get('frac_ndet_ge2', 0)*100:5.1f}%", flush=True)
            for r in base:
                r.append(int(labels[r[8] - 1] - labels[r[7]] + 1))
            del labels
        rows.extend([r[:7] + r[9:] for r in base])
        del time, chan, det
        report["hours"][hh] = out

    json.dump(report, open(prefix + "_report.json", "w"))

    if rows:
        import csv
        hdr = ["start", "hour", "fa", "count", "mean", "bin_us", "n_core"] + [f"n_t{int(t*1e9)}" for t in TAUS]
        with open(prefix + "_cand.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            for r in rows:
                w.writerow(r)
        arr = np.array([[r[3], r[6]] + r[7:] for r in rows], float)
        hit = int((arr[:, 0] == arr[:, 1]).sum())
        print(f"计数对账 n_core == count: {hit}/{len(rows)} = {hit/len(rows)*100:.2f}%", flush=True)

        # 合并后重算 fa。`fa = sf * 一年秒数 / bin_size_best` 是精确关系
        # （480,241 个候选实测最大偏离 2.2e-16），不需要反标定任何常数；
        # 先前这里用 `gammaincc(count, mean)` 当 sf 反解出一个 "T"，那是把
        # 上尾当成了下尾，量出来的 T 跨 17 个数量级、四列 fa 全部塌成同一个数。
        #
        # 另外：搜索报的 sf 是 P(X > count)（statrs 的 DiscreteCDF::sf），而
        # 观测到 count 个计数时的 p 值应当是 P(X >= count) = gammainc(count, mean)。
        # 这里按正确口径算，所以重算出来的 fa 与 signals.json 里的不可直接比，
        # 差一个逐候选的 (count+1)/lambda（实测中位 1.16e4）。见 OPEN-QUESTIONS 第 36 条。
        fa = np.array([r[2] for r in rows])
        cnt = np.array([r[3] for r in rows], float)
        mean = np.array([r[4] for r in rows], float)
        bin_s = np.array([r[5] for r in rows], float) * 1e-6
        year = 3600.0 * 24.0 * DAYS_PER_YEAR
        print("\n合并后候选存活（近似重算，原窗口不动）")
        print(f"{'τ(ns)':>7} {'并掉全局':>9} {'n>=8':>8} {'fa<=20':>8} {'fa<=1':>8} {'fa<=0.01':>9}")
        for k, tau in enumerate(TAUS):
            shrink = np.mean([1 - h["clusters"][k]["merged_frac"] for h in report["hours"].values()])
            n2 = arr[:, 2 + k]
            m2 = mean * shrink
            sf2 = gammainc(np.maximum(n2, 1), m2)
            fa2 = sf2 * year / bin_s
            ok = n2 >= MIN_NUMBER
            print(f"{tau*1e9:7.1f} {(1-shrink)*100:8.2f}% {int(ok.sum()):8d} "
                  f"{int((ok&(fa2<=20)).sum()):8d} {int((ok&(fa2<=1)).sum()):8d} {int((ok&(fa2<=0.01)).sum()):9d}")


if __name__ == "__main__":
    main()
