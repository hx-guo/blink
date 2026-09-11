"""83 个闪电证实 TGF 的逐 GRD 计数份额 vs CALDB 在各自真实入射方向上的预测。

拟合 q_d = (1−ε)/3 + ε·p_d：ε = 1 表示 CALDB 的点源预测成立，ε = 0 表示三路
等照射、份额不含方向信息。

**死时间必须小心。** 预测里若直接乘 live_d = 1 − f_d，而 f_d = Σ DEAD_TIME / 窗长
是从该探头自己的事例算出来的，那么某一路泊松涨高时它的 f_d 也高、预测反而被压低，
凭空造出反相关。核心窗只有约 190 µs、每事例死时间 4.2 µs、最忙一路 f 中位 0.24，
这个人为项与真实各向异性同量级。所以三种处理并排给：

  (a) 用实测 f_d —— 有上述人为反相关，仅作对照
  (b) 完全不修死时间
  (c) 自洽：死时间由模型自己的计数给，m_d = A_d Φ / (1 + A_d Φ τ / T)

用法: python3 diag_svom_share_vs_response.py "<respw_*.npz>" <incidence.csv>
"""
import csv
import glob
import itertools
import sys

import numpy as np
from scipy.optimize import minimize_scalar

CH_LO, CH_HI = 25, 230
RE = 6378.137


def load(pattern):
    files = sorted(glob.glob(pattern))
    z0 = np.load(files[0])
    core = np.zeros_like(z0["core"])
    meta = np.zeros_like(z0["meta"])
    R = np.zeros(z0["resp"].shape, np.float32)
    done = np.zeros(core.shape[0], bool)
    for f in files:
        z = np.load(f)
        d = z["done"]
        core[d] = z["core"][d]
        meta[d] = z["meta"][d]
        R[d] = z["resp"][d]
        done |= d
    return core, meta, R, done, z0


def fit_eps(c, p, mask=None):
    cc = c if mask is None else c[mask]
    pp = p if mask is None else p[mask]

    def nll(eps):
        return -float((cc * np.log(np.clip((1 - eps) / 3.0 + eps * pp, 1e-6, 1))).sum())

    r = minimize_scalar(nll, bounds=(-3, 3), method="bounded", options=dict(xatol=1e-4))
    e0, f0 = float(r.x), float(r.fun)
    step = 0.01
    while step < 5 and nll(e0 + step) - f0 <= 0.5:
        step *= 1.1
    return e0, step, nll(1.0) - f0, nll(0.0) - f0


def main(pattern, inc):
    core, meta, R, done, z0 = load(pattern)
    starts = z0["starts"][done]
    rows = {r["start"]: r for r in csv.DictReader(open(inc))}
    dist = np.array([float(rows[s]["dist_km"]) for s in starts])
    sep = np.array([float(rows[s]["sep_nadir_deg"]) for s in starts])
    zen = sep + np.degrees(dist / RE)
    ec = np.sqrt(z0["elo"] * z0["ehi"])
    de = z0["ehi"] - z0["elo"]
    c = core[done][:, :, CH_LO:CH_HI].sum(2).astype(float)
    n = c.sum(1)
    th = meta[done, 8]
    dead = meta[done, 3:6]
    tw = meta[done, 0]
    w = (ec / 100.0) ** (-1.0) * de
    A = np.einsum("idec,e->id", R[done][:, :, :, CH_LO:CH_HI].astype(np.float64), w)

    print("样本 %d 个，核心窗净计数中位 %.0f，窗长中位 %.0f µs"
          % (len(n), np.median(n), 1e6 * np.median(tw)))
    amp = (A / A.sum(1)[:, None])
    print("CALDB 预测的份额峰谷差：中位 %.3f，p90 %.3f，最大 %.3f"
          % (np.median(amp.max(1) - amp.min(1)),
             np.percentile(amp.max(1) - amp.min(1), 90),
             (amp.max(1) - amp.min(1)).max()))

    pa = A * (1 - np.clip(dead, 0, 0.95))
    pa /= pa.sum(1)[:, None]
    pb = A / A.sum(1)[:, None]
    tau = float(np.nansum(dead * tw[:, None]) / np.nansum(c))
    pc = np.zeros_like(A)
    for i in range(len(A)):
        phi = n[i] / max(A[i].sum(), 1e-30)
        m = A[i] * phi
        for _ in range(60):
            m = A[i] * phi / (1 + A[i] * phi * tau / tw[i])
            phi *= n[i] / max(m.sum(), 1e-30)
        pc[i] = m / m.sum()
    print("反解的等效每事例死时间 τ = %.2f µs（表头 EVT_DEAD = 4.0，实测众数 4.167，"
          "反解偏大是因为暴发窗里高道事例多、死时间长）" % (tau * 1e6))

    print("\n%-24s %8s %8s %12s %12s" % ("死时间的处理", "ε", "±", "ΔlnL(ε=1)", "ΔlnL(ε=0)"))
    for tag, p in (("(a) 用实测 f_d", pa), ("(b) 不修死时间", pb), ("(c) 自洽死时间", pc)):
        e, s, d1, d0 = fit_eps(c, p)
        print("%-24s %+8.2f %8.2f %12.1f %12.1f" % (tag, e, s, d1, d0))

    print("\n== 用 (c) 分档 ==")
    for lo, hi in ((0, 60), (60, 90), (90, 120), (120, 180)):
        m = (th >= lo) & (th < hi)
        e, s, _, _ = fit_eps(c, pc, m)
        print("  θ ∈ [%3d, %3d)  N=%2d  ε = %+.2f ± %.2f" % (lo, hi, m.sum(), e, s))
    # 源展宽假设：ζ 小（斜程短、末次散射区紧致）时 ε 该更接近 1
    q = np.percentile(zen, [33, 67])
    for lo, hi, tag in ((0, q[0], "ζ 最小三分之一"), (q[0], q[1], "中间"),
                        (q[1], 1e9, "ζ 最大三分之一")):
        m = (zen >= lo) & (zen < hi)
        e, s, _, _ = fit_eps(c, pc, m)
        print("  %-14s（ζ %.0f–%.0f°）N=%2d  ε = %+.2f ± %.2f"
              % (tag, zen[m].min(), zen[m].max(), m.sum(), e, s))
    m = n >= 20
    e, s, _, _ = fit_eps(c, pc, m)
    print("  计数 ≥ 20 的 %d 个：ε = %+.2f ± %.2f" % (m.sum(), e, s))

    print("\n== 探头编号错位的对照（六种排列，用 (c)）==")
    for perm in itertools.permutations((0, 1, 2)):
        e, s, d1, _ = fit_eps(c, pc[:, list(perm)])
        print("  %-10s ε = %+.2f ± %.2f，ΔlnL(ε=1) = %.1f"
              % ("".join("G%d" % (k + 1) for k in perm), e, s, d1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
