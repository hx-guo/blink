"""未决项 18 那个"B 角窗内 CoV 只有自己本底窗的 0.40"是不是**窗长**造成的。

## 怀疑点

原来的比较是"候选窗（T90 区间，毫秒量级）÷ 本底窗（±0.5 s）"。**两个窗差两个半
数量级**，而共帧星的本底流在毫秒尺度上有读出空洞（已知现象：几毫秒密集帧后跟
几毫秒空洞），空洞会把长窗的间隔分布拉出一条长尾、**把长窗的 CoV 抬上去**
（GRID-04 本底窗 CoV 实测 2.17，纯泊松是 1）。短窗里装不下空洞，CoV 自然低。

**若如此，0.40 这个比值量的是窗长，不是暴发的性质。**

## 做法

对每个候选算三个 CoV，**第三个是关键**：

1. `cov_burst`：暴窗（最佳格 ±3 ms）内相邻事例间隔的 CoV；
2. `cov_bkg_long`：±0.5 s 本底窗（原口径）；
3. `cov_bkg_short`：**从本底区里抽同样长的短窗**（与暴窗等长），取中位。

`cov_burst / cov_bkg_short` 才是扣掉窗长之后的量。若它回到 1 附近，
未决项 18 的那条判别量就只是窗长效应。

用法: python3 cov_window.py <SAT> <tgfs.json> <out.csv> [fa 上限]
"""

import datetime as dt
import glob
import json
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
EPOCH = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
PAD = 0.5
HALF = 0.003
RNG = np.random.default_rng(11)


def met_of(s):
    s = s.rstrip("Z")
    head, _, frac = s.partition(".")
    whole = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (whole - EPOCH).total_seconds() + (float("0." + frac) if frac else 0.0)


def prev_day(day):
    y, m, d = (int(x) for x in day.split("/"))
    p = dt.date(y, m, d) - dt.timedelta(days=1)
    return "%04d/%02d/%02d" % (p.year, p.month, p.day)


def load_window(sat, day, t0, t1):
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
                        out.append(t[(ty == 1) & ok & (e >= ETH) & (t >= t0) & (t <= t1)])
            except Exception:
                continue
    return np.sort(np.concatenate(out)) if out else np.zeros(0)


def cov(t):
    if t.size < 8:
        return float("nan")
    d = np.diff(t)
    d = d[d > 0]
    if d.size < 6 or d.mean() <= 0:
        return float("nan")
    return float(d.std() / d.mean())


def main():
    sat, jpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fa_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    fh = open(out, "w")
    fh.write("sat,start,fa,count,n_burst,cov_burst,cov_bkg_long,cov_bkg_short,"
             "ratio_long,ratio_short\n")
    for c in json.load(open(jpath)):
        s = c["signal"]
        if s.get("false_positive_per_year", 1e9) > fa_max:
            continue
        t0 = met_of(s["start"]) + s.get("delay", 0.0)
        t1 = t0 + s["bin_size_best"]
        ev = load_window(sat, s["start"][:10].replace("-", "/"), t0 - PAD, t1 + PAD)
        if ev.size < 500:
            continue
        lo, hi = t0 - HALF, t1 + HALF
        burst = ev[(ev >= lo) & (ev <= hi)]
        bkg = ev[(ev < lo) | (ev > hi)]
        span = hi - lo
        # 从本底区抽 200 个与暴窗等长的短窗
        shorts = []
        for _ in range(200):
            a = ev[0] + RNG.random() * max(ev[-1] - ev[0] - span, 1e-6)
            if a < hi and a + span > lo:
                continue
            seg = ev[(ev >= a) & (ev <= a + span)]
            v = cov(seg)
            if np.isfinite(v):
                shorts.append(v)
        cs = float(np.median(shorts)) if shorts else float("nan")
        cb, cl = cov(burst), cov(bkg)
        fh.write("%s,%s,%.3e,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n"
                 % (sat, s["start"], s["false_positive_per_year"], s["count"],
                    burst.size, cb, cl, cs, cb / cl if cl else float("nan"),
                    cb / cs if cs else float("nan")))
        print("%s  n=%3d  暴窗 %.3f  本底长窗 %.3f  本底等长短窗 %.3f  ->  比值 长 %.3f / 短 %.3f"
              % (s["start"][:19], burst.size, cb, cl, cs,
                 cb / cl if cl else float("nan"), cb / cs if cs else float("nan")))
    fh.close()


if __name__ == "__main__":
    main()
