"""读出空洞的光变曲线：一次过境的逐秒计数率与空洞占比，高/低计数率处的 1 ms 分探头计数。

GRID-03B 与 GRID-04 装在同一颗卫星上（NORAD 51830），同一时刻的计数率可以互相对照：
若 03B 的空洞是数据丢了，空洞期间 04 仍在计数；若只是时戳挤在一起、数没丢，03B 的
逐秒计数率应与 04 同步变化、不该在空洞多的秒里掉下来。

事例准入与 `Event::keep` 一致：EVT_TYPE==1、PI ∈ [1, n_ch)、该道 E_MIN ≥ 30 keV。

用法: python3 grid_gap_lightcurve.py <YYYY-MM-DD> <输出目录> [03B 文件名，缺省取当天计数率最高的过境]
"""
import glob, json, os, sys
import numpy as np
from astropy.io import fits

G = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
GAP = 0.003          # 四路合计连续 3 ms 无事例算一个空洞（与 grid_gaps.py 同口径）
HALF_MS = 1000       # 1 ms 分格画 ±1 s


def pass_files(sat, day):
    y, m, d = day.split("-")
    base = f"{G}/{sat}/fits7/{y}/{m}/{d}"
    if not os.path.isdir(base):
        return []
    v = sorted(os.listdir(base))[-1]
    return sorted(glob.glob(f"{base}/{v}/*.fits"))


def load(path):
    with fits.open(path, memmap=False) as h:
        g = h["GTI"].data
        gs, ge = float(g["START"][0]), float(g["STOP"][0])
        emin = np.asarray(h["EBOUNDS"].data["E_MIN"], float)
        T, D = [], []
        for k in range(4):
            e = h[f"EVENTS{k}"].data
            t = np.asarray(e["TIME"], float)
            pi = np.asarray(e["PI"], int)
            ty = np.asarray(e["EVT_TYPE"], int)
            ok = (pi >= 1) & (pi < emin.size)
            en = np.zeros(t.size)
            en[ok] = emin[pi[ok] - 1]
            keep = (ty == 1) & ok & (en >= ETH)
            T.append(t[keep])
            D.append(np.full(int(keep.sum()), k, np.int8))
    t = np.concatenate(T)
    o = np.argsort(t, kind="stable")
    return gs, ge, t[o], np.concatenate(D)[o]


def per_second(gs, ge, t):
    """逐秒：计数、空洞累计时长（F(x) = 截至 x 的空洞累计长度，秒内空洞 = F(s+1) − F(s)）。"""
    s0, s1 = int(np.floor(gs)), int(np.ceil(ge))
    edges = np.arange(s0, s1 + 1, dtype=float)
    cnt = np.histogram(t, edges)[0]
    d = np.diff(t)
    big = d > GAP
    a, b = t[:-1][big], t[1:][big]
    # 每个空洞按秒切开累加
    gap = np.zeros(len(edges) - 1)
    for x, y in zip(a, b):
        i0 = int(np.floor(x)) - s0
        i1 = int(np.floor(y)) - s0
        for i in range(max(i0, 0), min(i1, len(gap) - 1) + 1):
            lo, hi = max(x, edges[i]), min(y, edges[i + 1])
            if hi > lo:
                gap[i] += hi - lo
    live = np.clip(np.minimum(edges[1:], ge) - np.maximum(edges[:-1], gs), 0, 1)
    return edges[:-1], cnt, gap, live


def ms_bins(t, det, centre):
    edges = centre + np.arange(-HALF_MS, HALF_MS + 1) * 1e-3
    return [np.histogram(t[det == k], edges)[0].tolist() for k in range(4)]


def main():
    day, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    # 03B：挑逐秒计数率最高的一次过境
    best = None
    files = pass_files("GRID-03B", day)
    if len(sys.argv) > 3:
        files = [x for x in files if os.path.basename(x) == sys.argv[3]]
    for f in files:
        gs, ge, t, det = load(f)
        if t.size < 1000:
            continue
        s, c, g, live = per_second(gs, ge, t)
        ok = live > 0.5
        peak = (c[ok] / live[ok]).max()
        if best is None or peak > best[0]:
            best = (peak, f)
    peak, f = best
    gs, ge, t, det = load(f)
    s, c, g, live = per_second(gs, ge, t)
    ok = live > 0.5
    rate = np.where(ok, c / np.maximum(live, 1e-9), np.nan)
    gfrac = np.where(ok, g / np.maximum(live, 1e-9), np.nan)
    i_hi = int(np.nanargmax(rate))
    # 低率对照：率在 500–1500 c/s 的秒里取离最高点最远的一个
    cand = np.flatnonzero(ok & (rate > 500) & (rate < 1500))
    i_lo = int(cand[np.argmax(np.abs(cand - i_hi))]) if cand.size else None

    # 放大哪里：空洞占比超出泊松预期最多的那一秒（率 > 2 kc/s），以该秒里最长的空洞为中心。
    # 取最高计数率那一格会落在连续段上、看不到空洞。
    x = np.where(ok, rate, 0) * GAP
    excess = np.where(ok & (rate > 2000), gfrac - np.exp(-x) * (1 + x), -np.inf)
    i_hi = int(np.argmax(excess))
    sec_ev = (t >= s[i_hi]) & (t < s[i_hi] + 1)
    ts = t[sec_ev]
    j = int(np.argmax(np.diff(ts)))
    c_hi = 0.5 * (ts[j] + ts[j + 1])
    c_lo = s[i_lo] + 0.5 if i_lo is not None else None

    # 帧内速率：高率秒里去掉空洞后的计数率
    frame_rate = c[i_hi] / max(live[i_hi] - g[i_hi], 1e-9)

    res = dict(sat="GRID-03B", file=os.path.basename(f), gti=[gs, ge],
               sec=s.tolist(), rate=np.nan_to_num(rate, nan=-1).tolist(),
               gap_frac=np.nan_to_num(gfrac, nan=-1).tolist(),
               hi=dict(sec=float(s[i_hi]), rate=float(rate[i_hi]), gap_frac=float(gfrac[i_hi]),
                       frame_rate=float(frame_rate), centre=float(c_hi), ms=ms_bins(t, det, c_hi)),
               lo=None if i_lo is None else dict(sec=float(s[i_lo]), rate=float(rate[i_lo]),
                                                 gap_frac=float(gfrac[i_lo]), centre=float(c_lo),
                                                 ms=ms_bins(t, det, c_lo)))

    # 同一颗卫星上的 GRID-04：取与这次过境时间重叠的文件
    comp = []
    for f4 in pass_files("GRID-04", day):
        gs4, ge4, t4, det4 = load(f4)
        if ge4 < gs or gs4 > ge or t4.size < 100:
            continue
        s4, c4, g4, l4 = per_second(gs4, ge4, t4)
        ok4 = l4 > 0.5
        r4 = np.where(ok4, c4 / np.maximum(l4, 1e-9), np.nan)
        gf4 = np.where(ok4, g4 / np.maximum(l4, 1e-9), np.nan)
        comp.append(dict(file=os.path.basename(f4), gti=[gs4, ge4], sec=s4.tolist(),
                         rate=np.nan_to_num(r4, nan=-1).tolist(),
                         gap_frac=np.nan_to_num(gf4, nan=-1).tolist(),
                         ms_hi=ms_bins(t4, det4, c_hi) if gs4 <= c_hi <= ge4 else None))
    res["grid04"] = comp
    json.dump(res, open(os.path.join(out, "gap_lc_%s.json" % os.path.basename(f)[8:18]), "w"))
    print("03B", res["file"], "peak %.0f c/s gap %.2f frame_rate %.0f c/s" % (res["hi"]["rate"], res["hi"]["gap_frac"], frame_rate))
    print("low", res["lo"] and (res["lo"]["rate"], res["lo"]["gap_frac"]))
    print("GRID-04 overlapping passes:", [c["file"] for c in comp])


if __name__ == "__main__":
    main()
