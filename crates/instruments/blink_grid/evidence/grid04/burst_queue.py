"""B 角候选是不是"被读出队列摊开的短亮暴"：数暴内背靠背帧的游程。

## 假说

共帧星的读出成本实测是 `τ(m) = 120 + 37(m−1)` tick（`frame_cost.py`）。若一次暴发
的入射率超过这个吞吐，事例会排队，读出就以**背靠背帧**的形式把它们等间距吐出来，
表观时长 ≈ 计数 × 帧成本。B 角候选净超出中位 47 个计数、T90 约 3 ms，而
47 × (120…231) tick = 1.3…2.6 ms——**数量级对得上**。

## 判据

对每一帧 i，它到下一帧的间隔 Δᵢ 有一个**硬下限** `120 + 37(mᵢ−1)` tick。
把 Δᵢ 落在下限 +2 tick 以内的帧叫"背靠背"。

- 队列饱和 ⇒ 背靠背帧连成**长游程**（游程长度 ≈ 队列深度）；
- 纯丢弃 / 普通本底 ⇒ 背靠背是偶发的，游程长度近似几何分布、均值 < 1.3。

对每个候选分别在**暴窗**（T90 区间，退而求其次用最佳格）与**本底窗**（±0.5 s 里
扣掉暴窗）上量，**本底窗就是每个候选自己的负对照**。

用法: python3 burst_queue.py <SAT> <tgfs.json> <out.csv> [fa 上限]
"""

import datetime as dt
import glob
import json
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
TICK = 2.0**22
EPOCH = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
BASE_COST, PER_EVENT = 120, 37
PAD = 0.5


def met_of(s):
    """ISO 时刻 → MET 秒。整秒与小数秒分开加，不让 1e8 量级的数吃掉纳秒。"""
    s = s.rstrip("Z")
    head, _, frac = s.partition(".")
    whole = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (whole - EPOCH).total_seconds() + (float("0." + frac) if frac else 0.0)


def load_window(sat, day, t0, t1):
    """取 [t0, t1] 内的事例（合并四路、已排序）。day 形如 'YYYY/MM/DD'。"""
    out = []
    for dpath in (day, prev_day(day)):
        vers = sorted(glob.glob("%s/%s/fits7/%s/evt_v*" % (BASE, sat, dpath)))
        if not vers:
            continue
        for path in sorted(glob.glob(vers[-1] + "/*.fits")):
            try:
                with fits.open(path, memmap=False) as hd:
                    g = hd["GTI"].data
                    gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
                    if ge < t0 or gs > t1:
                        continue
                    emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
                    nch = emin.size
                    for i in range(4):
                        d = hd["EVENTS%d" % i].data
                        t = np.asarray(d["TIME"], dtype=np.float64)
                        pi = np.asarray(d["PI"], dtype=np.int32)
                        ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
                        ok = (pi >= 1) & (pi < nch)
                        e = np.zeros(t.size)
                        e[ok] = emin[pi[ok] - 1]
                        sel = (ty == 1) & ok & (e >= ETH) & (t >= t0) & (t <= t1)
                        out.append(t[sel])
            except Exception:
                continue
    return np.sort(np.concatenate(out)) if out else np.zeros(0)


def prev_day(day):
    y, m, d = (int(x) for x in day.split("/"))
    p = dt.date(y, m, d) - dt.timedelta(days=1)
    return "%04d/%02d/%02d" % (p.year, p.month, p.day)


def runs(times):
    """返回 (帧数, 背靠背帧占比, 最长游程, 游程长度均值, 游程长度列表)。"""
    if times.size < 3:
        return 0, float("nan"), 0, float("nan"), []
    tk = np.rint(times * TICK).astype(np.int64)
    ed = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], ed + 1))
    sz = np.concatenate((ed + 1, [tk.size])) - st
    ft = tk[st]
    if ft.size < 3:
        return int(ft.size), float("nan"), 0, float("nan"), []
    d = np.diff(ft)
    floor = BASE_COST + PER_EVENT * (sz[:-1] - 1)
    bb = d <= floor + 2
    rl, cur = [], 0
    for v in bb:
        if v:
            cur += 1
        elif cur:
            rl.append(cur)
            cur = 0
    if cur:
        rl.append(cur)
    return (int(ft.size), float(bb.mean()), (max(rl) if rl else 0),
            (float(np.mean(rl)) if rl else 0.0), rl)


def main():
    sat, jpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fa_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    cands = json.load(open(jpath))
    fh = open(out, "w")
    fh.write("sat,start,fa,count,dur_us,lat,lon,n_burst,nf_burst,bb_burst,maxrun_burst,"
             "meanrun_burst,n_bkg,nf_bkg,bb_bkg,maxrun_bkg,meanrun_bkg,span_us\n")
    for c in cands:
        s = c["signal"]
        fa = s.get("false_positive_per_year", 1e9)
        if fa > fa_max:
            continue
        t0 = met_of(s["start"]) + s.get("delay", 0.0)
        t1 = t0 + s["bin_size_best"]
        day = s["start"][:10].replace("-", "/")
        ev = load_window(sat, day, t0 - PAD, t1 + PAD)
        if ev.size < 200:
            continue
        inb = (ev >= t0) & (ev <= t1)
        # 暴窗放宽到 ±3 ms，因为假说本身说的就是"被摊开"，只看最佳格会自我实现
        wide = (ev >= t0 - 0.003) & (ev <= t1 + 0.003)
        bkg = ~wide
        nf_b, bb_b, mr_b, ar_b, rl_b = runs(ev[wide])
        nf_k, bb_k, mr_k, ar_k, _ = runs(ev[bkg])
        span = (ev[wide].max() - ev[wide].min()) * 1e6 if wide.sum() > 1 else 0.0
        pos = s.get("position") or {}
        fh.write("%s,%s,%.3e,%d,%.1f,%.2f,%.2f,%d,%d,%.4f,%d,%.3f,%d,%d,%.4f,%d,%.3f,%.0f\n"
                 % (sat, s["start"], fa, s["count"], s["bin_size_best"] * 1e6,
                    pos.get("latitude", float("nan")), pos.get("longitude", float("nan")),
                    int(inb.sum()), nf_b, bb_b, mr_b, ar_b,
                    int(bkg.sum()), nf_k, bb_k, mr_k, ar_k, span))
        fh.flush()
        print("%s fa=%.1e count=%d  暴窗 bb=%.3f 最长游程 %d | 本底 bb=%.4f 最长 %d"
              % (s["start"], fa, s["count"], bb_b, mr_b, bb_k, mr_k))
    fh.close()


if __name__ == "__main__":
    main()
