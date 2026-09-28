"""GRID-03B 单路探头的沉默段：每路按自己的计数率找泊松造不出的空当。

四路合计的长空当是各路沉默恰好重叠的结果；单路沉默才是基本现象。安静 = 该路空当前后各 1 s
的四路合计计数率 < 1500 c/s。沉默 = 该路相邻事例间隔 L，满足 r_k·L > 20（r_k 取该路空当前后各
1 s 的平均率）。全部事例（不做能量准入）。

每段记：探头、起止 MET、长度、该路前后计数率、起止时刻的 UTC 秒内相位、离 GTI 起点、
同一时段其余三路各自的计数（看是不是只有这一路停了）、该路沉默前最后一个事例的 PI 与 EVT_TYPE。
另记每次过境每路的安静时长，作发生率分母。

用法: python3 grid_channel_silences.py <卫星目录名> <worker> <nworkers> <抽几天> <输出 CSV>
"""
import csv, glob, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID"
QUIET, K = 1500.0, 20.0

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
            with fits.open(f, memmap=False) as h:
                gs, ge = float(h["GTI"].data["START"][0]), float(h["GTI"].data["STOP"][0])
                ev = []
                for k in range(4):
                    e = h[f"EVENTS{k}"].data
                    t = np.asarray(e["TIME"], float); m = (t >= gs) & (t <= ge)
                    ev.append((t[m], np.asarray(e["PI"], int)[m], np.asarray(e["EVT_TYPE"], int)[m]))
        except Exception:
            continue
        tall = np.sort(np.concatenate([x[0] for x in ev]))
        if tall.size < 1000:
            continue
        c = np.histogram(tall, np.arange(np.floor(gs), np.ceil(ge) + 1))[0]
        live.append(dict(file=os.path.basename(f), quiet_s=int((c < QUIET).sum()), gti_s=ge - gs))
        for k, (t, pi, ty) in enumerate(ev):
            if t.size < 100:
                continue
            dt = np.diff(t)
            for i in np.flatnonzero(dt > 0.05):
                a, b = t[i], t[i + 1]
                nb = i + 1 - np.searchsorted(t, a - 1.0); na = np.searchsorted(t, b + 1.0, side="right") - (i + 1)
                if a - gs < 1.0 or ge - b < 1.0:
                    continue
                rk = (nb + na) / 2.0
                tot_b = np.searchsorted(tall, a) - np.searchsorted(tall, a - 1.0)
                tot_a = np.searchsorted(tall, b + 1.0) - np.searchsorted(tall, b)
                if rk * (b - a) <= K or max(tot_b, tot_a) >= QUIET:
                    continue
                others = []
                for j, (tj, _, _) in enumerate(ev):
                    if j != k:
                        others.append(int(np.searchsorted(tj, b) - np.searchsorted(tj, a)))
                rows.append(dict(file=os.path.basename(f), det=k, start="%.7f" % a, stop="%.7f" % b,
                                 L_ms="%.2f" % ((b - a) * 1e3), rate_det="%.1f" % rk, rate_all="%.0f" % ((tot_b + tot_a) / 2),
                                 others=";".join(map(str, others)), phase_start="%.5f" % (a % 1), phase_stop="%.5f" % (b % 1),
                                 from_gti_start_s="%.2f" % (a - gs), last_pi=int(pi[i]), last_type=int(ty[i]),
                                 next_pi=int(pi[i + 1]), next_type=int(ty[i + 1])))
with open(out, "w", newline="") as fh:
    if rows:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
with open(out.replace(".csv", "_live.csv"), "w", newline="") as fh:
    if live:
        wr = csv.DictWriter(fh, fieldnames=list(live[0].keys())); wr.writeheader(); wr.writerows(live)
print("worker", w, "silences", len(rows), "passes", len(live))
