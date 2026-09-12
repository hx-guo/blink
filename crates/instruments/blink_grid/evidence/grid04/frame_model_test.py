"""帧模型的直接检验：同戳簇的**重数分布**对速率，逐 10 s 片。

## 为什么用重数而不是三重

`frac_ev_in3` 是二阶量（∝ q²），在低速率下被真实贯穿粒子主导，量不到帧参数。
**二重占比是一阶量**（∝ q），统计量大三个数量级，且直接就是"帧内另一路也被收进来"
的概率。所以检验帧模型要用整条重数分布，不能只看三重。

## 两种帧模型，重数分布能分开

- **模型 A（收满整帧）**：触发后 τ = 120 tick 内每路最多贡献 1 个事例，帧内所有
  事例共用触发时戳。另一路被收进来的概率 q = 1 − e^{−λτ}。
- **模型 B（短符合窗 + 长读出死时间）**：触发后只在 w ≪ τ 内latch 各路，其余
  (w, τ) 段的击中**整个丢掉**，读出占满 τ。另一路被收进来的概率 q = 1 − e^{−λw}。

两者的逐探头死时间硬边沿都是 τ（数据实测 120 tick），**但重数分布差一个 τ/w 倍**。
模型 B 丢的计数远多于模型 A ⇒ 折扣因子更低 ⇒ "只有 03B 看到 TGF"的反常更小。
所以这个量不是诊断细节，它直接定折扣因子。

还有一条不依赖模型的硬约束：单链读出的**输出率上限**是 E[m]/τ。若实测速率越过 1/τ
= 34.95 kc/s，则帧内必然多路（E[m] > 1），模型 B 的 w ≪ τ 被直接证伪。

## 输出

逐 10 s 片一行：速率、逐重数占比、E[m]、模型 A 的预期、磁纬。
用法: python3 frame_model_test.py <SAT> <out.csv> <YYYY/MM/DD> [更多日期...]
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


def poisson_binomial(ps):
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for i, p in enumerate(ps):
        dist[1:i + 2] = dist[1:i + 2] * (1 - p) + dist[0:i + 1] * p
        dist[0] *= (1 - p)
    return dist


def model_a(rates_obs, tau_s):
    """模型 A 的预期重数分布（占事例的比例）与 E[m]。输入逐路观测率。"""
    r = np.asarray(rates_obs, dtype=np.float64)
    if r.sum() <= 0:
        return np.full(5, np.nan), float("nan")
    lam = r.copy()
    for _ in range(300):
        Lam = lam.sum()
        if Lam <= 0:
            break
        q = 1.0 - np.exp(-lam * tau_s)
        cyc = tau_s + 1.0 / Lam
        pred = (lam / Lam + (1.0 - lam / Lam) * q) / cyc
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = lam * np.clip(np.where(pred > 0, r / pred, 1.0), 0.5, 2.0)
    Lam = lam.sum()
    q = 1.0 - np.exp(-lam * tau_s)
    pm = np.zeros(5)
    for d0 in range(4):
        w = lam[d0] / Lam
        dist = poisson_binomial([q[d] for d in range(4) if d != d0])
        for k, pk in enumerate(dist):
            pm[1 + k] += w * pk
    em = float(sum(m * pm[m] for m in range(5)))
    frac = np.array([m * pm[m] / em for m in range(5)]) if em > 0 else np.full(5, np.nan)
    return frac, em


def load_pass(path):
    with fits.open(path, memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
        nch = emin.size
        T = []
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            T.append(np.sort(t[(ty == 1) & ok & (e >= ETH)]))
    return gs, ge, T


def posatt_for(path, sat, daypath):
    key = "_".join(path.split("/")[-1].split("_")[2:4])
    for v in reversed(sorted(glob.glob("%s/%s/fits8/%s/posatt_v*" % (BASE, sat, daypath)))):
        hits = glob.glob("%s/*%s*" % (v, key))
        if not hits:
            continue
        try:
            with fits.open(hits[0], memmap=False) as hd:
                d = hd["ORBIT_ATTITUDE"].data
                t = np.asarray(d["TIME"], dtype=np.float64)
                la = np.asarray(d["Latitude"], dtype=np.float64)
                lo = np.asarray(d["Longitude"], dtype=np.float64)
                pt = np.asarray(d["POS_TYPE"], dtype=np.int32)
                x = np.asarray(d["X_WGS84"], dtype=np.float64)
                y = np.asarray(d["Y_WGS84"], dtype=np.float64)
                z = np.asarray(d["Z_WGS84"], dtype=np.float64)
        except Exception:
            continue
        ok = (pt != 0) & np.isfinite(la) & np.isfinite(lo) & (np.sqrt(x * x + y * y + z * z) > 1.0)
        if ok.sum() >= 2:
            return t[ok], dipole_lat(la[ok], lo[ok])
    return np.zeros(0), np.zeros(0)


def main():
    sat, out = sys.argv[1], sys.argv[2]
    tau_s = {"GRID-02": 119, "GRID-04": 120, "GRID-07": 119}.get(sat, 0) / TICK
    fh = open(out, "w")
    fh.write("sat,day,pass_file,islice,dur_s,n,rate_cps,f1,f2,f3,f4,em,"
             "f1_a,f2_a,f3_a,f4_a,em_a,mlat\n")
    for daypath in sys.argv[3:]:
        vers = sorted(glob.glob("%s/%s/fits7/%s/evt_v*" % (BASE, sat, daypath)))
        if not vers:
            continue
        day = daypath.replace("/", "-")
        for path in sorted(glob.glob(vers[-1] + "/*.fits")):
            try:
                gs, ge, T = load_pass(path)
            except Exception:
                continue
            allt = np.sort(np.concatenate(T))
            if allt.size < 1000:
                continue
            pt, pml = posatt_for(path, sat, daypath)
            nsl = int(math.ceil((ge - gs) / SLICE_S))
            edges = gs + SLICE_S * np.arange(nsl + 1)
            b = np.searchsorted(allt, edges)
            pd = [np.searchsorted(t, edges) for t in T]
            for i in range(nsl):
                a, z = b[i], b[i + 1]
                n = z - a
                if n < 200:
                    continue
                d_i = min(SLICE_S, ge - edges[i])
                seg = allt[a:z]
                tk = np.rint(seg * TICK).astype(np.int64)
                ed = np.flatnonzero(np.diff(tk) != 0)
                sz = np.concatenate((ed + 1, [tk.size])) - np.concatenate(([0], ed + 1))
                f = [float(sz[sz == k].sum()) / n for k in (1, 2, 3, 4)]
                em = float(sz.mean())
                r_i = [(pd[d][i + 1] - pd[d][i]) / d_i for d in range(4)]
                fa, ema = model_a(r_i, tau_s)
                mv = (float(np.interp(edges[i] + d_i / 2, pt, pml, left=np.nan, right=np.nan))
                      if pt.size >= 2 else float("nan"))
                fh.write("%s,%s,%s,%d,%.1f,%d,%.1f,%.5f,%.5f,%.5f,%.5f,%.4f,"
                         "%.5f,%.5f,%.5f,%.5f,%.4f,%.2f\n"
                         % (sat, day, path.split("/")[-1], i, d_i, n, n / d_i,
                            f[0], f[1], f[2], f[3], em,
                            fa[1], fa[2], fa[3], fa[4], ema, abs(mv)))
    fh.close()
    print("done ->", out)


if __name__ == "__main__":
    main()
