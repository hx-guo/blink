"""去掉计数率上限之前：高计数率、又没有读出空白的本底窗，数据还是不是泊松的？

对每个窗（中心 ±0.5 s、裁到 GTI）：
  * 新空洞判据（v15）：r = ln2/间隔中位数，1 − (1 − e^{−rL})^n < 1e-3 即判为有空白；
  * 离散度检验：1 ms 分格计数 x_i，D = Σ(x_i − x̄)²/x̄，纯泊松时 D ~ χ²(k−1)；
    报右尾 p（方差比泊松大）与 D/(k−1)（Fano 因子）。死时间让方差变小（Fano < 1），不会造假暴发。
窗分三类：quiet（窗内 < 1500 c/s）、high（≥ 1500 c/s）、frame（中心在帧大小标签判为丢数的秒里）。
事例准入同 `Event::keep`（判据看的是准入后的流）；帧标签用全部事例。

用法: python3 grid_dispersion_check.py <卫星目录名> <worker> <nworkers> <抽几天> <输出 CSV>
"""
import csv, glob, math, os, sys
import numpy as np
from astropy.io import fits
from scipy.stats import chi2

G = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
N_RAND = 150
N_FRAME = 30
rng = np.random.default_rng(20260927)


def is_full(n, sat):
    return (n >= 1260 and n % 1260 == 0) if sat == "GRID-03B" else (320 <= n <= 352)


def load(f):
    with fits.open(f, memmap=False) as h:
        gs, ge = float(h["GTI"].data["START"][0]), float(h["GTI"].data["STOP"][0])
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


def judge(t, gs, ge, c):
    lo, hi = max(c - 0.5, gs), min(c + 0.5, ge)
    if hi - lo < 0.5:
        return None
    a, b = np.searchsorted(t, lo), np.searchsorted(t, hi, side="right")
    win = t[a:b]
    if win.size < 3:
        return None
    d = np.diff(win)
    L = max(win[0] - lo, hi - win[-1], d.max())
    med = np.median(d)
    r = math.log(2) / med if med > 0 else win.size / (hi - lo)
    x = r * L
    n = d.size + 2
    p_gap = -math.expm1(n * math.log1p(-math.exp(-x))) if x < 700 else n * math.exp(-x)
    k = int((hi - lo) / 1e-3)
    cnt = np.histogram(win, lo + np.arange(k + 1) * 1e-3)[0]
    m = cnt.mean()
    D = ((cnt - m) ** 2).sum() / m if m > 0 else 0.0
    # 去掉慢起伏：每格的期望取前后各 H 格的平均（不含自己）。03B 单路丢包让四路合计率
    # 少 1/4、持续约 0.2 s，这种慢起伏在整窗平均下也表现为方差超出，但不会造假暴发；
    # 毫秒尺度的扎堆才会。E[(x−e)²] = λ(1 + 1/(2H))，据此归一。
    H = 12
    ker = np.ones(2 * H + 1); ker[H] = 0
    s_ = np.convolve(cnt, ker, mode="same"); w_ = np.convolve(np.ones(k), ker, mode="same")
    e = s_ / w_
    sel = slice(H, k - H)
    ok = e[sel] > 0
    Dl = (((cnt[sel] - e[sel]) ** 2)[ok] / e[sel][ok]).sum() / (1 + 1 / (2 * H))
    kl = int(ok.sum())
    return dict(rate="%.1f" % (win.size / (hi - lo)), p_gap="%.3g" % p_gap, gap=int(p_gap < 1e-3),
                fano="%.4f" % (D / (k - 1)), p_disp="%.3g" % chi2.sf(D, k - 1),
                fano_local="%.4f" % (Dl / max(kl, 1)), p_local="%.3g" % chi2.sf(Dl, max(kl, 1)))


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
        for c in rng.uniform(gs, ge, N_RAND):
            j = judge(t, gs, ge, c)
            if j:
                rows.append(dict(sat=sat, kind="quiet" if float(j["rate"]) < 1500 else "high", file=os.path.basename(f), **j))
        # 高计数率段再加抽：计数率 ≥ 1500 的秒里取中心（随机取窗落在那里的太少）
        edges = np.arange(np.floor(gs), np.ceil(ge))
        cs = np.histogram(t, edges)[0]
        hs = edges[:-1][cs >= 1500]
        for s0 in (rng.choice(hs, size=min(N_RAND, hs.size), replace=False) if hs.size else []):
            j = judge(t, gs, ge, s0 + rng.uniform(0, 1))
            if j:
                rows.append(dict(sat=sat, kind="high", file=os.path.basename(f), **j))
        fs = frame_seconds(allv, sat)
        for s0 in (rng.choice(fs, size=min(N_FRAME, len(fs)), replace=False) if fs else []):
            j = judge(t, gs, ge, s0 + rng.uniform(0, 1))
            if j:
                rows.append(dict(sat=sat, kind="frame", file=os.path.basename(f), **j))
with open(out, "w", newline="") as fh:
    if rows:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print("worker", w, "rows", len(rows))
