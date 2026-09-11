"""方位扫描：在 φ + Δφ 上重算三路有效面积，看逐 GRD 份额的似然有没有峰。

83 个闪电证实 TGF 的份额与 CALDB 在真实入射方向上的预测对不上（ε ≈ 0 而不是 1，
第 29 条）。若是方向链里有一个固定的方位误差，扫 Δφ 应当在某个角度上冒出峰，
而且落在 90°/120°/180° 这种整数上就是轴序或符号错了；整条平则说明是响应本身的
探头间相对归一，或者源根本不是点源。

方向链另有一条独立核验（`diag_svom_orb_frames.py`，0.02° 精度），所以本条的
作用主要是确认曲线是平的。

输入是 `scripts/cluster/svom_resp_scan.py` 的 npz（83 × 13 × 3 的有效面积，
13 = 12 个方位偏移 + θ→180−θ）。
用法: python3 diag_svom_resp_azimuth.py "<rscan_*.npz>" "<respw_*.npz>"
"""
import glob
import sys

import numpy as np
from scipy.optimize import minimize_scalar


def load_scan(pattern):
    files = sorted(glob.glob(pattern))
    z0 = np.load(files[0])
    area = np.zeros_like(z0["area"])
    done = np.zeros(z0["done"].shape, bool)
    for f in files:
        z = np.load(f)
        d = z["done"]
        area[d] = z["area"][d]
        done |= d
    return area, done, z0["dphi"], z0["starts"]


def load_counts(pattern):
    files = sorted(glob.glob(pattern))
    z0 = np.load(files[0])
    core = np.zeros_like(z0["core"])
    meta = np.zeros_like(z0["meta"])
    ok = np.zeros(core.shape[0], bool)
    for f in files:
        z = np.load(f)
        d = z["done"]
        core[d] = z["core"][d]
        meta[d] = z["meta"][d]
        ok |= d
    return core[:, :, 25:230].sum(2).astype(float), meta, ok


def selfconsistent(A, n, tw, tau):
    """死时间由模型计数给，避免用实测计数算的修正反噬预测（第 29 条）。"""
    p = np.zeros_like(A)
    for i in range(len(A)):
        phi = n[i] / max(A[i].sum(), 1e-30)
        m = A[i] * phi
        for _ in range(60):
            m = A[i] * phi / (1 + A[i] * phi * tau / tw[i])
            phi *= n[i] / max(m.sum(), 1e-30)
        p[i] = m / max(m.sum(), 1e-30)
    return p


def fit_eps(c, p):
    def nll(e):
        return -float((c * np.log(np.clip((1 - e) / 3.0 + e * p, 1e-6, 1))).sum())

    r = minimize_scalar(nll, bounds=(-3, 3), method="bounded", options=dict(xatol=1e-4))
    e0, f0 = float(r.x), float(r.fun)
    step = 0.01
    while step < 5 and nll(e0 + step) - f0 <= 0.5:
        step *= 1.1
    return e0, step, -f0


def main(scan_pat, resp_pat):
    area, done, dphi, starts = load_scan(scan_pat)
    c_all, meta, ok = load_counts(resp_pat)
    good = ok & done.all(1)
    print("方位扫描：%d 个 TGF 全部 %d 个变体都算出来了（共 %d）"
          % (good.sum(), area.shape[1], len(good)))
    c = c_all[good]
    n = c.sum(1)
    tw = meta[good, 0]
    dead = meta[good, 3:6]
    tau = float(np.nansum(dead * tw[:, None]) / np.nansum(c))

    print("\n%-16s %8s %8s %10s" % ("变体", "ε", "±", "lnL(最佳 ε)"))
    best = None
    for v in range(area.shape[1]):
        tag = ("Δφ = %3d°" % dphi[v]) if v < len(dphi) else "θ → 180−θ"
        p = selfconsistent(area[good, v, :], n, tw, tau)
        e, s, ll = fit_eps(c, p)
        print("%-16s %+8.2f %8.2f %10.1f" % (tag, e, s, ll))
        if best is None or ll > best[2]:
            best = (tag, e, ll)
    print("\n似然最高的是 %s（ε = %+.2f）。判据：若某个 Δφ 能把 ε 拉到接近 1，"
          "那就是方向链的方位错了；整条曲线都在 ε ≈ 0 附近，说明不是方位问题。"
          % (best[0], best[1]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
