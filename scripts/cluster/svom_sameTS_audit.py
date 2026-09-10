"""跨探头同时戳事例在候选窗里是否富集。

三路合并后每小时有 4.4 万–6.1 万对 Δt==0（占事例 0.31–0.35%）。同探头内是 0，
所以不是重复记录，而是跨探头同时戳：0.954 µs 的量化步造成的偶然同格 + 真实的
跨探头符合（宇宙线穿星、康普顿散射）。前者与信号无关、后者是带电粒子的特征，
两者都可能把本来不显著的窗推过线。

逐候选量：最佳格 [start+delay, +bin_size_best] 内属于同时戳对的事例占比，
与邻域本底（±1 s 挖 ±10 ms）的同一个量对比。

用法: python3 svom_sameTS_audit.py <tgfs.json> <out.csv> [WORKERS] [IDX]
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


def met(iso):
    b = iso.rstrip("Z")
    h, _, f = b.partition(".")
    s = dt.datetime.strptime(h, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (s - REF).total_seconds() + (float("0." + f) if f else 0.0)


def load_hour(day, hh):
    pat = "%s/%s/grm_evt/svom_grm_evt_%s_%s_v*.fits" % (
        D, day, dt.datetime.strptime(day, "%Y/%m/%d").strftime("%y%m%d"), hh)
    g = sorted(glob.glob(pat))
    if not g:
        return None
    T, DET = [], []
    with fits.open(g[-1]) as h:
        gti = [(float(a), float(b)) for a, b in zip(h["GTI"].data["START"], h["GTI"].data["STOP"])]
        for i, det in zip((3, 4, 5), (1, 2, 3)):
            d = h[i].data
            t = np.asarray(d["TIME"], float)
            pi = np.asarray(d["PI"])
            et = np.asarray(d["EVT_TYPE"])
            ac = np.asarray(d["ANTI_COIN"])
            k = (et == 0) & (ac == 0) & (pi >= 25) & (pi < 256)
            T.append(t[k])
            DET.append(np.full(int(k.sum()), det, np.int8))
    T = np.concatenate(T)
    DET = np.concatenate(DET)
    o = np.argsort(T, kind="stable")
    T, DET = T[o], DET[o]
    ok = np.zeros(len(T), bool)
    for a, b in gti:
        ok |= (T >= a) & (T <= b)
    T, DET = T[ok], DET[ok]
    # 同时戳标记：与前一个或后一个事例时戳完全相同
    same = np.zeros(len(T), bool)
    eq = T[1:] == T[:-1]
    same[:-1] |= eq
    same[1:] |= eq
    # 同时戳且不同探头（同探头内实测为 0，这里只是把口径说死）
    cross = np.zeros(len(T), bool)
    ce = eq & (DET[1:] != DET[:-1])
    cross[:-1] |= ce
    cross[1:] |= ce
    return T, same, cross


def main(inp, out, workers=1, idx=0):
    recs = [r for r in json.load(open(inp)) if r["signal"]["false_positive_per_year"] <= 1e-5]
    by_hour = defaultdict(list)
    for r in recs:
        t0 = REF + dt.timedelta(seconds=met(r["signal"]["start"]))
        by_hour[(t0.strftime("%Y/%m/%d"), t0.strftime("%H"))].append(r)

    w = csv.writer(open(out, "w", newline=""))
    w.writerow(["start", "fa", "assoc", "n_best", "n_same_best", "n_cross_best",
                "n_bkg", "n_same_bkg", "n_cross_bkg", "hour_frac_same"])
    for n, ((day, hh), items) in enumerate(sorted(by_hour.items())):
        if n % workers != idx:
            continue
        got = load_hour(day, hh)
        if got is None:
            continue
        T, same, cross = got
        hour_frac = float(same.mean())
        for r in items:
            s = r["signal"]
            b0 = met(s["start"]) + s["delay"]
            b1 = b0 + s["bin_size_best"]
            core = (T >= b0) & (T <= b1)
            near = ((T >= b0 - 1.0) & (T <= b1 + 1.0)) & ~((T >= b0 - 0.01) & (T <= b1 + 0.01))
            w.writerow([s["start"], "%.6g" % s["false_positive_per_year"],
                        int(bool((r.get("lightning") or {}).get("associated"))),
                        int(core.sum()), int(same[core].sum()), int(cross[core].sum()),
                        int(near.sum()), int(same[near].sum()), int(cross[near].sum()),
                        "%.6f" % hour_frac])
        print("done", day, hh, len(items), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         int(sys.argv[4]) if len(sys.argv) > 4 else 0)
