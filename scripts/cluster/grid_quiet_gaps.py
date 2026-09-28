"""安静时段里的长空当：逐个找出来并记录特征。

"安静"指空当前后各 1 s 的计数率都 < 1500 c/s（辐射带里的成帧丢数不在这里）。
空当 = 相邻两个事例（全部事例、四路合计，不做能量准入）的间隔 L，满足 r·L > 20
（r 取空当前后各 1 s 的平均计数率，e^{-20} ≈ 2e-9，泊松本底造不出来）。

每个空当记：起止 MET、长度；前后各 1 s 的计数率；四路各自在空当前最后一个事例、空当后
第一个事例的时刻（看是否四路同时停、同时恢复）；起止时刻在 UTC 整秒内的相位（MET 与 UTC
相差整数秒，所以 MET 小数部分就是 UTC 秒内相位）；离 GTI 起止多远；该过境的 GTI 与文件名。

用法: python3 grid_quiet_gaps.py <卫星目录名> <worker> <nworkers> <抽几天> <输出 CSV>
"""
import csv, glob, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID"
QUIET = 1500.0
K = 20.0


def load(f):
    with fits.open(f, memmap=False) as h:
        g = h["GTI"].data
        gs, ge = float(g["START"][0]), float(g["STOP"][0])
        T, D = [], []
        for k in range(4):
            t = np.asarray(h[f"EVENTS{k}"].data["TIME"], float)
            T.append(t); D.append(np.full(t.size, k, np.int8))
    t = np.concatenate(T); o = np.argsort(t, kind="stable")
    return gs, ge, t[o], np.concatenate(D)[o]


sat, w, nw, ndays, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
days = sorted(glob.glob(f"{G}/{sat}/fits7/20*/*/*"))
pick = [days[int(round(k * (len(days) - 1) / max(ndays - 1, 1)))] for k in range(ndays)]
rows, live = [], []
for d in pick[w::nw]:
    vs = sorted(os.listdir(d))
    if not vs:
        continue
    for f in sorted(glob.glob(f"{d}/{vs[-1]}/*.fits")):
        try:
            gs, ge, t, det = load(f)
        except Exception:
            continue
        m = (t >= gs) & (t <= ge)
        t, det = t[m], det[m]
        if t.size < 1000:
            continue
        # 该过境安静时段的总时长（逐秒计数率 < 1500 的秒数），作为发生率的分母
        c = np.histogram(t, np.arange(np.floor(gs), np.ceil(ge) + 1))[0]
        live.append(dict(file=os.path.basename(f), gti_s=ge - gs, quiet_s=int((c < QUIET).sum())))
        dt = np.diff(t)
        cand = np.flatnonzero(dt > 0.02)
        for i in cand:
            a, b = t[i], t[i + 1]
            n_before = i + 1 - np.searchsorted(t, a - 1.0)
            n_after = np.searchsorted(t, b + 1.0, side="right") - (i + 1)
            rb = n_before / min(1.0, a - gs) if a - gs > 0.05 else np.nan
            ra = n_after / min(1.0, ge - b) if ge - b > 0.05 else np.nan
            r = np.nanmean([rb, ra])
            if not np.isfinite(r) or r * (b - a) <= K or max(np.nan_to_num(rb), np.nan_to_num(ra)) >= QUIET:
                continue
            j0 = np.searchsorted(t, a - 2.0); j1 = np.searchsorted(t, b + 2.0, side="right")
            tb, db = t[j0:i + 1], det[j0:i + 1]
            ta, da = t[i + 1:j1], det[i + 1:j1]
            last = [tb[db == k][-1] if (db == k).any() else np.nan for k in range(4)]
            first = [ta[da == k][0] if (da == k).any() else np.nan for k in range(4)]
            rows.append(dict(sat=sat, file=os.path.basename(f), gti_start="%.6f" % gs, gti_stop="%.6f" % ge,
                             start="%.7f" % a, stop="%.7f" % b, L_ms="%.3f" % ((b - a) * 1e3),
                             rate_before="%.1f" % rb, rate_after="%.1f" % ra,
                             last_spread_ms="%.3f" % ((np.nanmax(last) - np.nanmin(last)) * 1e3),
                             first_spread_ms="%.3f" % ((np.nanmax(first) - np.nanmin(first)) * 1e3),
                             last_gap_ms=";".join("%.2f" % ((a - x) * 1e3) for x in last),
                             first_gap_ms=";".join("%.2f" % ((x - b) * 1e3) for x in first),
                             phase_start="%.6f" % (a % 1.0), phase_stop="%.6f" % (b % 1.0),
                             from_gti_start_s="%.3f" % (a - gs), to_gti_stop_s="%.3f" % (ge - b)))
with open(out, "w", newline="") as fh:
    if rows:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
with open(out.replace(".csv", "_live.csv"), "w", newline="") as fh:
    if live:
        wr = csv.DictWriter(fh, fieldnames=list(live[0].keys())); wr.writeheader(); wr.writerows(live)
print("worker", w, "gaps", len(rows), "passes", len(live))
