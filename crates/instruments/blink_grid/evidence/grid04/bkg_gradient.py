"""本底梯度对照：候选会不会只是坐在本底上升沿上。

高磁纬处本底在 ±0.5 s 的本底窗内可以变很快。搜索用的是**窗内平均**本底，若候选正好
坐在上升沿上，平均值就低估了候选处的真实本底，**凭空造出"超出"**——这正是当初造出
成帧伪信号的机制，也是守恒检验里那 3 个例外（落在本底陡变处）的解释。

对每个候选给三个本底估计，并用它们各自重算超出与泊松显著性：

- `wide`：±0.5 s 窗内（挖掉暴窗）的平均率——**搜索用的就是这个**；
- `local`：±50 ms 内（挖掉暴窗）的平均率；
- `trend`：对 ±0.5 s 内 10 ms 分格的计数做一次线性拟合，在候选时刻取值。

若 `local` / `trend` 比 `wide` 高很多，该候选就可疑。**判据是比值，不是绝对值。**

用法: python3 bkg_gradient.py <SAT> <tgfs.json> <out.csv> [fa 上限]
"""

import datetime as dt
import glob
import json
import math
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
EPOCH = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
PAD = 0.5
LOCAL = 0.05


def met_of(s):
    s = s.rstrip("Z")
    head, _, frac = s.partition(".")
    whole = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (whole - EPOCH).total_seconds() + (float("0." + frac) if frac else 0.0)


def prev_day(day):
    y, m, d = (int(x) for x in day.split("/"))
    p = dt.date(y, m, d) - dt.timedelta(days=1)
    return "%04d/%02d/%02d" % (p.year, p.month, p.day)


def load(sat, day, t0, t1):
    out, seg = [], []
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
                    seg.append((max(gs, t0), min(ge, t1)))
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
    return (np.sort(np.concatenate(out)) if out else np.zeros(0)), seg


def live_in(seg, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in seg)


def poisson_sf(k, mu):
    if mu <= 0:
        return 0.0 if k > 0 else 1.0
    s, term = 0.0, math.exp(-mu)
    for i in range(0, int(k)):
        s += term
        term *= mu / (i + 1)
    return max(0.0, 1.0 - s)


def main():
    sat, jpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fa_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    fh = open(out, "w")
    fh.write("sat,start,count,dur_us,r_wide,r_local,r_trend,ratio_local,ratio_trend,"
             "excess_wide,excess_trend,p_wide,p_trend\n")
    for c in json.load(open(jpath)):
        s = c["signal"]
        if s.get("false_positive_per_year", 1e9) > fa_max:
            continue
        t0 = met_of(s["start"]) + s.get("delay", 0.0)
        t1 = t0 + s["bin_size_best"]
        ev, seg = load(sat, s["start"][:10].replace("-", "/"), t0 - PAD, t1 + PAD)
        if ev.size < 500 or not seg:
            continue
        n_in = int(((ev >= t0) & (ev <= t1)).sum())
        w = t1 - t0
        hole = 0.003
        lo, hi = t0 - hole, t1 + hole
        # wide
        nw = int(((ev < lo) | (ev > hi)).sum())
        lw = live_in(seg, t0 - PAD, t1 + PAD) - live_in(seg, lo, hi)
        r_wide = nw / lw if lw > 0 else float("nan")
        # local
        nl = int((((ev >= t0 - LOCAL) & (ev < lo)) | ((ev > hi) & (ev <= t1 + LOCAL))).sum())
        ll = live_in(seg, t0 - LOCAL, t1 + LOCAL) - live_in(seg, lo, hi)
        r_local = nl / ll if ll > 0 else float("nan")
        # trend：±0.5 s 内 10 ms 分格线性拟合，挖掉暴窗附近的格
        edges = np.arange(t0 - PAD, t1 + PAD, 0.010)
        cnt = np.histogram(ev, bins=edges)[0]
        mid = 0.5 * (edges[:-1] + edges[1:])
        keep = np.abs(mid - t0) > 0.02
        if keep.sum() > 10:
            a, b = np.polyfit(mid[keep] - t0, cnt[keep] / 0.010, 1)
            r_trend = float(b)
        else:
            r_trend = float("nan")
        mu_w, mu_t = r_wide * w, r_trend * w
        fh.write("%s,%s,%d,%.1f,%.1f,%.1f,%.1f,%.3f,%.3f,%.1f,%.1f,%.3e,%.3e\n"
                 % (sat, s["start"], n_in, w * 1e6, r_wide, r_local, r_trend,
                    r_local / r_wide if r_wide else float("nan"),
                    r_trend / r_wide if r_wide else float("nan"),
                    n_in - mu_w, n_in - mu_t,
                    poisson_sf(n_in, mu_w), poisson_sf(n_in, mu_t)))
        print("%s n=%3d 窗 %6.0f µs  本底 wide %6.1f local %6.1f trend %6.1f "
              "(local/wide %.2f trend/wide %.2f)  p: %.1e -> %.1e"
              % (s["start"][:19], n_in, w * 1e6, r_wide, r_local, r_trend,
                 r_local / r_wide if r_wide else float("nan"),
                 r_trend / r_wide if r_wide else float("nan"),
                 poisson_sf(n_in, mu_w), poisson_sf(n_in, mu_t)))
    fh.close()


if __name__ == "__main__":
    main()
