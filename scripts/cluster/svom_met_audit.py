"""量化 met() 把 ISO 小数秒截到微秒对窗内计数的影响。

搜索侧 `Candidate::new(data[cursor].time(), data[cursor+step].time(), ...)`：
候选的 start/stop **就是窗内第一个和最后一个事例的时刻**，count 是这两端之间
（含两端）的事例数。合并过的候选，count 对应的是最显著子窗
[start+delay, start+delay+bin_size_best]。

所以逐候选重算窗内计数，就能直接跟搜索报的 count 对账。三种解析口径：
  exact  —— 不截断，float("0."+frac)
  trunc  —— 现行写法 (frac + "000000")[:6]
  tol    —— exact 且两端各放 0.5 µs（时间戳量化步长 2^-20 s ≈ 0.954 µs 的一半），
            这一档才是"真值"：MET 存成 f64 秒时 ulp 已经有 60 ns。

用法: python3 svom_met_audit.py <tgfs.json> <out.csv> [WORKERS] [IDX] [LIMIT]
"""
import csv
import datetime as dt
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from astropy.io import fits

D = "/gecamfs/SVOM/Archived-DATA/GRM-DATA/L1B/daily"
REF = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
TOL = 0.5 * 2 ** -20          # 半个时间戳量化步


def met_trunc(iso):
    """现行写法：小数秒截到 6 位。"""
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    body = head + "." + (frac + "000000")[:6] if frac else head + ".000000"
    t = dt.datetime.strptime(body, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=dt.timezone.utc)
    return (t - REF).total_seconds()


def met_exact(iso):
    """整秒走 strptime，小数秒单独加，不丢位。"""
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (stamp - REF).total_seconds() + (float("0." + frac) if frac else 0.0)


def load_hour(day, hh):
    pat = "%s/%s/grm_evt/svom_grm_evt_%s_%s_v*.fits" % (
        D, day, dt.datetime.strptime(day, "%Y/%m/%d").strftime("%y%m%d"), hh)
    g = sorted(glob.glob(pat))
    if not g:
        return None
    T, P = [], []
    with fits.open(g[-1]) as h:
        gti = [(float(a), float(b)) for a, b in zip(h["GTI"].data["START"], h["GTI"].data["STOP"])]
        for i in (3, 4, 5):
            d = h[i].data
            t = np.asarray(d["TIME"], float)
            pi = np.asarray(d["PI"])
            et = np.asarray(d["EVT_TYPE"])
            ac = np.asarray(d["ANTI_COIN"])
            k = (et == 0) & (ac == 0) & (pi >= 25) & (pi < 256)   # v8 的 keep()
            T.append(t[k])
            P.append(pi[k])
    T = np.concatenate(T)
    P = np.concatenate(P)
    o = np.argsort(T, kind="stable")
    T, P = T[o], P[o]
    ingti = np.zeros(len(T), bool)
    for a, b in gti:
        ingti |= (T >= a) & (T <= b)
    return T[ingti], P[ingti]


def main(inp, out, workers=1, idx=0, limit=0):
    recs = json.load(open(inp))
    picked = []
    for r in recs:
        s = r["signal"]
        if s["false_positive_per_year"] > 1e-5:
            continue
        li = r.get("lightning") or {}
        picked.append((s, bool(li.get("associated"))))
    if limit:
        rng = np.random.default_rng(0)
        conf = [x for x in picked if x[1]]
        rest = [x for x in picked if not x[1]]
        keep = min(limit - len(conf), len(rest))
        sel = rng.choice(len(rest), size=max(keep, 0), replace=False)
        picked = conf + [rest[i] for i in sel]

    by_hour = defaultdict(list)
    for s, assoc in picked:
        t0 = REF + dt.timedelta(seconds=met_exact(s["start"]))
        by_hour[(t0.strftime("%Y/%m/%d"), t0.strftime("%H"))].append((s, assoc))

    w = csv.writer(open(out, "w", newline=""))
    w.writerow(["start", "fa", "assoc", "count_report", "n_best_exact", "n_best_trunc",
                "n_best_tol", "n_core_exact", "n_core_trunc", "n_core_tol",
                "pi_sum_core_tol", "pi_sum_core_trunc", "err_start_ns", "err_stop_ns",
                "best_ms", "span_ms"])
    n_hour = 0
    for (day, hh), items in sorted(by_hour.items()):
        n_hour += 1
        if (n_hour - 1) % workers != idx:
            continue
        hourly = load_hour(day, hh)
        if hourly is None:
            print("缺文件", day, hh, flush=True)
            continue
        T, P = hourly
        for s, assoc in items:
            e0, e1 = met_exact(s["start"]), met_exact(s["stop"])
            t0, t1 = met_trunc(s["start"]), met_trunc(s["stop"])
            b0e = e0 + s["delay"]
            b1e = b0e + s["bin_size_best"]
            b0t = t0 + s["delay"]
            b1t = b0t + s["bin_size_best"]

            def n(a, b, tol=0.0):
                return int(((T >= a - tol) & (T <= b + tol)).sum())

            def pisum(a, b, tol=0.0):
                m = (T >= a - tol) & (T <= b + tol)
                return int(P[m].sum())

            w.writerow([s["start"], "%.6g" % s["false_positive_per_year"], int(assoc),
                        s["count"],
                        n(b0e, b1e), n(b0t, b1t), n(b0e, b1e, TOL),
                        n(e0, e1), n(t0, t1), n(e0, e1, TOL),
                        pisum(e0, e1, TOL), pisum(t0, t1),
                        "%.1f" % ((e0 - t0) * 1e9), "%.1f" % ((e1 - t1) * 1e9),
                        "%.4f" % (s["bin_size_best"] * 1e3), "%.4f" % ((e1 - e0) * 1e3)])
        print("done", day, hh, len(items), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         int(sys.argv[4]) if len(sys.argv) > 4 else 0,
         int(sys.argv[5]) if len(sys.argv) > 5 else 0)
