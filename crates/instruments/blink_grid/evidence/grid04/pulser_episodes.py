"""全任务的标定脉冲 episode 清单：起止时刻、时长、频率、幅度。

判据（从数据导出，不是写死名单）：把四重同戳簇按时间排好，**到下一个四重簇的间隔
落在 8388–8389 tick** 的标成"周期对"（2.000093 ms = 500.00 Hz，02/04/07 实测同一个值）。
把相邻的周期对按间隔 < 2 s 并成一个 episode，只留 ≥ 100 个周期对的。

输出每个 episode：星、日期、过境、起止 MET、时长、幅度道号与能量、以及该 episode
期间四路合计的计数率（= 曝光/本底修正的输入）。

**两个计数列的含义不同，别混**：`n_pair` 是**间隔恰为 8388/8389 tick 的相邻对**，
只有当连续两个脉冲都完整记成四重簇时才计到——**它是脉冲数的下限**（实测
`pair_per_s` 约 300，而周期本身是 500 Hz，说明约四成的连续对被打断）。
`n_quad` 是 episode 内全部四重簇，更接近真实脉冲数。**真实频率以间隔的众数为准
（8388/8389 tick = 500.00 Hz），不要用 `pair_per_s`。**

用法: python3 pulser_episodes.py <SAT> <out.csv> <trip_pass_*.csv> [更多]
"""

import csv
import glob
import sys

import numpy as np
from astropy.io import fits

ARCH = "/gecamfs/Exchange/GSDC/missions/GRID/%s"
ETH = 30.0
TICK = 2.0**22
PER_LO, PER_HI = 8388, 8389
GAP_S = 2.0
MIN_PULSE = 100


def load(sat, day, key):
    v = sorted(glob.glob(ARCH % sat + "/fits7/%s/evt_v*" % day.replace("-", "/")))
    if not v:
        return None
    f = sorted(glob.glob(v[-1] + "/*%s*" % key))
    if not f:
        return None
    with fits.open(f[0], memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        eb = hd["EBOUNDS"].data
        emin = np.asarray(eb["E_MIN"], dtype=np.float64)
        emax = np.asarray(eb["E_MAX"], dtype=np.float64)
        nch = emin.size
        T, P = [], []
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            sel = (ty == 1) & ok & (e >= ETH)
            T.append(t[sel])
            P.append(pi[sel])
    t = np.concatenate(T)
    o = np.argsort(t, kind="stable")
    return gs, ge, t[o], np.concatenate(P)[o], np.sqrt(emin * emax)


def episodes(sat, day, key):
    got = load(sat, day, key)
    if got is None:
        return []
    gs, ge, t, pi, egeo = got
    if t.size < 1000:
        return []
    tk = np.rint(t * TICK).astype(np.int64)
    ed = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], ed + 1))
    sz = np.concatenate((ed + 1, [tk.size])) - st
    big = np.flatnonzero(sz >= 4)
    if big.size < MIN_PULSE:
        return []
    ft = tk[st[big]]
    d = np.diff(ft)
    per = np.flatnonzero((d >= PER_LO) & (d <= PER_HI))
    if per.size < MIN_PULSE:
        return []
    # 周期对的时刻（取对的左端）
    pt = t[st[big[per]]]
    cut = np.flatnonzero(np.diff(pt) > GAP_S)
    starts = np.concatenate(([0], cut + 1))
    ends = np.concatenate((cut, [pt.size - 1]))
    out = []
    for a, b in zip(starts, ends):
        n = b - a + 1
        if n < MIN_PULSE:
            continue
        t0, t1 = pt[a], pt[b]
        dur = t1 - t0
        if dur <= 0:
            continue
        # 该 episode 内四重簇事例的道号
        sel = big[per[a:b + 1]]
        idx = np.concatenate([np.arange(st[k], st[k] + sz[k]) for k in sel])
        md = int(np.median(pi[idx]))
        inwin = ((t >= t0) & (t <= t1)).sum()
        # episode 内全部四重簇（不只周期对）——脉冲总数的更好估计
        inep = big[(t[st[big]] >= t0) & (t[st[big]] <= t1)]
        out.append(dict(sat=sat, day=day, key=key, t0=t0, t1=t1, dur=dur,
                        n=int(n), n4=int(inep.size), f=n / dur, pi=md,
                        e=float(egeo[md - 1]), rate=inwin / dur, gti=ge - gs))
    return out


def main():
    sat, out = sys.argv[1], sys.argv[2]
    rows = []
    for pat in sys.argv[3:]:
        for p in sorted(glob.glob(pat)):
            with open(p) as f:
                rows.extend(r for r in csv.DictReader(f)
                            if r.get("sat") == sat and r.get("k4"))
    cand = [r for r in rows
            if float(r["k4"]) >= 200 and float(r["k4"]) / max(float(r["k3"]), 1) > 0.8]
    fh = open(out, "w")
    fh.write("sat,day,pass_file,met_start,met_stop,dur_s,n_pair,n_quad,pair_per_s,"
             "quad_per_s,pi_med,e_kev,rate_in_episode_cps,gti_s\n")
    tot, ndur = 0, 0.0
    for r in cand:
        key = "_".join(r["pass_file"].split("_")[2:4])
        for e in episodes(sat, r["day"], key):
            fh.write("%s,%s,%s,%.6f,%.6f,%.2f,%d,%d,%.1f,%.1f,%d,%.1f,%.1f,%.0f\n"
                     % (e["sat"], e["day"], r["pass_file"], e["t0"], e["t1"], e["dur"],
                        e["n"], e["n4"], e["f"], e["n4"] / e["dur"], e["pi"], e["e"],
                        e["rate"], e["gti"]))
            tot += 1
            ndur += e["dur"]
        fh.flush()
    fh.close()
    print("%s: 疑似脉冲过境 %d，找到 episode %d 个，合计 %.1f s（%.3f h）"
          % (sat, len(cand), tot, ndur, ndur / 3600.0))


if __name__ == "__main__":
    main()
