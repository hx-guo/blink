"""共帧星的读出参数直接测量：入射率、死时间占比、**有效符合窗 w**。四星通用。

## 为什么需要它

共帧星（GRID-02/04/07）的读出被当成"触发后 τ = 120 tick 内每路最多贡献 1 个事例、
全帧共用触发时戳"。注入检验的折扣因子、进而"只有 GRID-03B 探到 TGF"这条结论，
全部压在这个模型上。GRID-04 实测发现模型高估帧内多路符合 **6.9 倍**：真实读出只在
触发后约 4 µs 内 latch 其余各路，(w, τ) 段的击中并不共用触发时戳。
**w 必须逐星实测**，三颗共帧星能不能共用一个折扣因子取决于此，而 02 与 07 没人量过。

那些不共戳的击中去哪了，有两种后果完全不同的可能——**丢掉**（计数真的少）与
**排队**（计数保住、只是被摊到随后的帧上、每帧一个新时戳）。判别量见下面的 β 与
守恒比，灵敏度由 `sim_readout.py` 在已知真值的合成数据上标定。

## 估计量（不经速率反解，避免一阶近似）

同戳簇就是帧（相邻帧至少隔 τ，不会并簇）。把帧当更新过程：一帧占 τ，其后等下一个
入射击中，等待时间 ~ Exp(Λ)。于是

    E[周期] = τ + 1/Λ   ⟹   **Λ = 1 / (T/n_frames − τ)**       ← 入射总率，直接可测
    死时间占比 = τ / (τ + 1/Λ)
    P(探头 d 出现在某帧里) = k_d / n_frames = λ_d/Λ + (1 − λ_d/Λ)·q_d
    q_d = 1 − e^{−λ_d w}   ⟹   **w = −ln(1 − q_d) / λ_d**

λ_d 按 k_d 的比例分配 Λ（二阶偏差，帧内占用低时可忽略）。这套只用"帧数"和"每帧
有哪几路"，不用任何小 q 近似，也不用把观测率当入射率。

## 自检：更新模型自己成不成立

帧间隔 Δ 若真是"τ + Exp(Λ)"（丢弃），则 Δ 的分布是平移指数，**间隔恰为 τ 的帧不存在**。
排队会插进一批背靠背帧（Δ = τ），把均值拉短而不动尾部斜率。所以：

    β = 1 − Λ_tail/Λ_mean        丢弃 ⇒ β ≈ 0；排队 ⇒ β ≈ 1 − e^{−Λτ}
    bb_exc                       Δ 落在 τ ±1.5 tick 的占比减去局部连续本底
    R_obs/Λ_tail                 丢弃 ⇒ 随速率下降；排队 ⇒ 恒为 1

`bb_exc` 是三者里最稳的：它只看 τ 处的一个尖峰，**不受毫秒尺度读出空洞的影响**，
而 `Λ_tail` 会被空洞拉低。`lam_trunc` 是把尾部截到 5 个平均等待时间以内重算的版本，
它与 `lam_tail` 之差就是空洞污染的大小。

用法: python3 readout_window.py <SAT> <tau_tick> <out.csv> <YYYY/MM/DD> [更多日期...]
"""

import glob
import math
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID"
ETH = 30.0
TICK = 2.0**22
SLICE_S = 10.0
POLE_LAT, POLE_LON = math.radians(80.7), math.radians(-72.7)


def dipole_lat(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    s = (np.sin(la) * math.sin(POLE_LAT)
         + np.cos(la) * math.cos(POLE_LAT) * np.cos(lo - POLE_LON))
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def load_pass(path):
    """返回 (gs, ge, 合并排序后的时刻, 对应探头号)。准入与 `Event::keep` 一致。"""
    with fits.open(path, memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
        nch = emin.size
        ts, ds = [], []
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            sel = (ty == 1) & ok & (e >= ETH)
            ts.append(t[sel])
            ds.append(np.full(int(sel.sum()), i, dtype=np.int8))
    t = np.concatenate(ts)
    d = np.concatenate(ds)
    o = np.argsort(t, kind="stable")
    return gs, ge, t[o], d[o]


def posatt_for(path, sat, daypath):
    key = "_".join(path.split("/")[-1].split("_")[2:4])
    for v in reversed(sorted(glob.glob("%s/%s/fits8/%s/posatt_v*" % (BASE, sat, daypath)))):
        hits = glob.glob("%s/*%s*" % (v, key))
        if not hits:
            continue
        try:
            with fits.open(hits[0], memmap=False) as hd:
                a = hd["ORBIT_ATTITUDE"].data
                t = np.asarray(a["TIME"], dtype=np.float64)
                la = np.asarray(a["Latitude"], dtype=np.float64)
                lo = np.asarray(a["Longitude"], dtype=np.float64)
                pt = np.asarray(a["POS_TYPE"], dtype=np.int32)
                x = np.asarray(a["X_WGS84"], dtype=np.float64)
                y = np.asarray(a["Y_WGS84"], dtype=np.float64)
                z = np.asarray(a["Z_WGS84"], dtype=np.float64)
        except Exception:
            continue
        ok = (pt != 0) & np.isfinite(la) & np.isfinite(lo) & (np.sqrt(x * x + y * y + z * z) > 1.0)
        if ok.sum() >= 2:
            return t[ok], dipole_lat(la[ok], lo[ok]), la[ok], lo[ok]
    return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)


def measure(times, dets, dur, tau_s):
    """一段数据上的读出参数。返回 dict 或 None。"""
    n = times.size
    if n < 500:
        return None
    tk = np.rint(times * TICK).astype(np.int64)
    ed = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], ed + 1))
    en = np.concatenate((ed + 1, [n]))
    nf = st.size
    if nf < 200:
        return None
    ftime = times[st]
    # 入射率：均值法与尾部法
    dt = np.diff(ftime)
    dt = dt[dt > 0]
    if dt.size < 100:
        return None
    mean_cycle = dur / nf
    lam_mean = 1.0 / (mean_cycle - tau_s) if mean_cycle > tau_s else float("nan")
    # 尾部法：Δ ≥ τ 的部分若是平移指数，1/(E[Δ|Δ≥τ0] − τ0) 就是 Λ；取 τ0 = τ + 1/Λ_mean
    tail0 = tau_s + (1.0 / lam_mean if np.isfinite(lam_mean) and lam_mean > 0 else 0.0)
    sel = dt >= tail0
    lam_tail = 1.0 / (dt[sel].mean() - tail0) if sel.sum() > 50 else float("nan")
    # 截断版：高速率下毫秒尺度的读出空洞会混进尾部、把 Λ_tail 压低，只取 5 个平均
    # 等待时间以内的部分重算一次，两者之差就是空洞污染的大小。
    hi = tail0 + 5.0 / lam_mean if np.isfinite(lam_mean) and lam_mean > 0 else np.inf
    sel2 = sel & (dt <= hi)
    if sel2.sum() > 50:
        m2 = dt[sel2].mean() - tail0
        span = hi - tail0
        # 截断指数的均值反解：E[X|X≤s] = 1/Λ − s·e^{−Λs}/(1−e^{−Λs})，数值解
        lo_, hi_ = 1e-6, 10.0 / max(m2, 1e-9)
        for _ in range(60):
            mid = 0.5 * (lo_ + hi_)
            x = mid * span
            ex = math.exp(-x) if x < 700 else 0.0
            val = 1.0 / mid - span * ex / (1 - ex) if ex < 1 else m2
            if val > m2:
                lo_ = mid
            else:
                hi_ = mid
        lam_trunc = 0.5 * (lo_ + hi_)
    else:
        lam_trunc = float("nan")
    # 背靠背帧：间隔落在 τ 的 ±1.5 tick 内。排队会在这里堆一个尖峰，丢弃不会
    # （丢弃下该处只有连续本底，密度 ≈ Λ·3/2²²，量级 1e-4）。这个量不受长空洞影响。
    one = 1.0 / TICK
    bb = float(((dt >= tau_s - 1.5 * one) & (dt <= tau_s + 1.5 * one)).mean())
    cont = float(((dt >= tau_s + 3 * one) & (dt <= tau_s + 23 * one)).mean()) * (3.0 / 20.0)
    bb_exc = bb - cont
    # 每路出现在多少帧里
    kd = np.zeros(4)
    for d in range(4):
        m = dets == d
        if not m.any():
            continue
        kd[d] = np.unique(np.searchsorted(st, np.flatnonzero(m), side="right")).size
    P = kd / nf
    lam_d = lam_mean * kd / max(kd.sum(), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = lam_d / lam_mean
        q = (P - frac) / (1.0 - frac)
        w = np.where((q > 0) & (q < 1) & (lam_d > 0), -np.log(1.0 - np.clip(q, 1e-12, 1 - 1e-12)) / lam_d, np.nan)
    sz = en - st
    return dict(n=n, nf=nf, rate=n / dur, em=float(sz.mean()),
                lam=lam_mean, lam_tail=lam_tail, lam_trunc=lam_trunc,
                bb=bb, bb_exc=bb_exc,
                dead=tau_s / (tau_s + 1.0 / lam_mean) if np.isfinite(lam_mean) else float("nan"),
                q_med=float(np.nanmedian(q)), w_med=float(np.nanmedian(w)),
                f2=float(sz[sz == 2].sum()) / n, f3=float(sz[sz >= 3].sum()) / n)


def main():
    sat, tau_tick, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    tau_s = tau_tick / TICK
    fh = open(out, "w")
    fh.write("sat,day,pass_file,islice,dur_s,n,rate_cps,nframe,em,lam_inc,lam_tail,"
             "lam_trunc,tail_over_mean,bb,bb_exc,dead_frac,q_med,w_us,f2,f3,mlat,lat,lon\n")
    for daypath in sys.argv[4:]:
        vers = sorted(glob.glob("%s/%s/fits7/%s/evt_v*" % (BASE, sat, daypath)))
        if not vers:
            continue
        day = daypath.replace("/", "-")
        for path in sorted(glob.glob(vers[-1] + "/*.fits")):
            try:
                gs, ge, t, d = load_pass(path)
            except Exception:
                continue
            if t.size < 2000:
                continue
            pt, pml, pla, plo = posatt_for(path, sat, daypath)
            nsl = int(math.ceil((ge - gs) / SLICE_S))
            edges = gs + SLICE_S * np.arange(nsl + 1)
            b = np.searchsorted(t, edges)
            for i in range(nsl):
                a, z = b[i], b[i + 1]
                d_i = min(SLICE_S, ge - edges[i])
                m = measure(t[a:z], d[a:z], d_i, tau_s)
                if m is None:
                    continue
                tm = edges[i] + d_i / 2
                if pt.size >= 2:
                    mv = abs(float(np.interp(tm, pt, pml, left=np.nan, right=np.nan)))
                    la = float(np.interp(tm, pt, pla, left=np.nan, right=np.nan))
                    lo = float(np.interp(tm, pt, plo, left=np.nan, right=np.nan))
                else:
                    mv = la = lo = float("nan")
                fh.write("%s,%s,%s,%d,%.1f,%d,%.1f,%d,%.5f,%.1f,%.1f,%.1f,%.4f,"
                         "%.6f,%.6f,%.5f,%.6f,%.4f,%.6f,%.6f,%.2f,%.2f,%.2f\n"
                         % (sat, day, path.split("/")[-1], i, d_i, m["n"], m["rate"],
                            m["nf"], m["em"], m["lam"], m["lam_tail"], m["lam_trunc"],
                            m["lam_tail"] / m["lam"] if m["lam"] else float("nan"),
                            m["bb"], m["bb_exc"],
                            m["dead"], m["q_med"], m["w_med"] * 1e6, m["f2"], m["f3"],
                            mv, la, lo))
            fh.flush()
    fh.close()
    print("done ->", out)


if __name__ == "__main__":
    main()
