"""读出空洞门：现行判据与新判据在同一批窗上的误触发率与检出率。

现行（search.rs 的 has_dead_gap）：r = max(窗内平均率, 整次过境平均率)，r·L > 9.2 即否决。
新判据：r = ln2 / 窗内相邻间隔中位数（对空洞不敏感：丢数时大多数间隔在帧内）；
        窗里 n 个间隔（含两端到窗边的距离）中最长的一个 ≥ L 的概率
            P = 1 − (1 − e^{−r·L})^n
        P < α 即否决（α 取 1e-3 与 1e-4 两档）。
L 都是窗内最长空段，含窗两端。事例准入同 `Event::keep`。

两类窗：
  quiet  —— 过境内均匀随机取中心、窗内平均率 < 1500 c/s（安静本底，量误触发）；
  frame  —— 中心落在"丢数模式"的秒里（量检出）。丢数模式的独立标签来自读出缓冲的帧大小：
            该秒按 > 0.5 ms 空洞切帧，至少 2 帧的事例数是满帧（03B：1260 的整数倍；
            02/04/07：320–352）。标签用全部事例（缓冲区装的是全部事例），判据用准入后的事例。

用法: python3 grid_deadgap_check2.py <卫星目录名> <worker> <nworkers> <抽几天> <输出 CSV>
"""
import csv, glob, math, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
N_QUIET = 150
N_FRAME = 40
rng = np.random.default_rng(20260927)


def is_full(n, sat):
    if sat == "GRID-03B":
        return n >= 1260 and n % 1260 == 0
    return 320 <= n <= 352


def load(f):
    with fits.open(f, memmap=False) as h:
        g = h["GTI"].data
        gs, ge = float(g["START"][0]), float(g["STOP"][0])
        emin = np.asarray(h["EBOUNDS"].data["E_MIN"], float)
        K, A = [], []
        for k in range(4):
            e = h[f"EVENTS{k}"].data
            t = np.asarray(e["TIME"], float); pi = np.asarray(e["PI"], int); ty = np.asarray(e["EVT_TYPE"], int)
            ok = (pi >= 1) & (pi < emin.size)
            en = np.zeros(t.size); en[ok] = emin[pi[ok] - 1]
            K.append(t[(ty == 1) & ok & (en >= ETH)]); A.append(t)
    keep = np.sort(np.concatenate(K)); allv = np.sort(np.concatenate(A))
    return gs, ge, keep[(keep >= gs) & (keep <= ge)], allv


def frame_seconds(allv, sat):
    """丢数模式的秒（标签）。"""
    if allv.size < 1000:
        return []
    edges = np.arange(np.floor(allv[0]), np.ceil(allv[-1]) + 1)
    c = np.histogram(allv, edges)[0]
    out = []
    for i in np.flatnonzero(c >= 2000):
        w = allv[(allv >= edges[i]) & (allv < edges[i] + 1)]
        fr = np.split(w, np.flatnonzero(np.diff(w) > 0.0005) + 1)
        if sum(is_full(x.size, sat) for x in fr) >= 2:
            out.append(edges[i])
    return out


def judge(t, gs, ge, c, pass_rate):
    lo, hi = max(c - 0.5, gs), min(c + 0.5, ge)
    a, b = np.searchsorted(t, lo), np.searchsorted(t, hi, side="right")
    win = t[a:b]
    span = hi - lo
    if win.size < 3:
        return None
    d = np.diff(win)
    L = max(win[0] - lo, hi - win[-1], d.max())
    rw = win.size / span
    old = max(rw, pass_rate) * L > 9.2
    med = np.median(d)
    r_med = math.log(2) / med if med > 0 else rw
    n = d.size + 2
    x = r_med * L
    # log P(最长 ≥ L) = log(1 − (1 − e^{−x})^n)，x 大时用 n·e^{−x} 近似防下溢
    q = -math.expm1(n * math.log1p(-math.exp(-x))) if x < 700 else n * math.exp(-x)
    return dict(span="%.3f" % span, n=win.size, rate_win="%.1f" % rw, rate_pass="%.1f" % pass_rate,
                rate_med="%.1f" % r_med, L_ms="%.3f" % (L * 1e3), P_long="%.3g" % q,
                old=int(old), new3=int(q < 1e-3), new4=int(q < 1e-4))


sat, w, nw, ndays, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
days = sorted(glob.glob(f"{G}/{sat}/fits7/20*/*/*"))
pick = [days[int(round(k * (len(days) - 1) / max(ndays - 1, 1)))] for k in range(ndays)]
rows = []
for d in pick[w::nw]:
    vs = sorted(os.listdir(d))
    if not vs:
        continue
    for f in sorted(glob.glob(f"{d}/{vs[-1]}/*.fits")):
        try:
            gs, ge, t, allv = load(f)
        except Exception:
            continue
        if t.size < 1000 or ge - gs < 10:
            continue
        pr = t.size / (ge - gs)
        for c in rng.uniform(gs, ge, N_QUIET):
            j = judge(t, gs, ge, c, pr)
            if j and float(j["rate_win"]) < 1500:
                rows.append(dict(sat=sat, kind="quiet", file=os.path.basename(f), centre="%.6f" % c, **j))
        fs = frame_seconds(allv, sat)
        if fs:
            for s0 in rng.choice(fs, size=min(N_FRAME, len(fs)), replace=False):
                j = judge(t, gs, ge, s0 + rng.uniform(0, 1), pr)
                if j:
                    rows.append(dict(sat=sat, kind="frame", file=os.path.basename(f), centre="%.6f" % s0, **j))
with open(out, "w", newline="") as fh:
    if rows:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print("worker", w, "rows", len(rows))
