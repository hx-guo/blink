"""SVOM/GRM 叠加光子谱的「系统项」拆解 + 逐 TGF 注量。

`plot_svom_photon_spectrum.py` 给出 83 个闪电证实 TGF 的叠加光子谱
α = 0.878 ± 0.032 (stat)，但按入射角二分给 0.689/1.094、按落点距离二分给
0.765/0.981。这里把这两支和堆积那一支分开量——它们的物理含义完全不同：

  θ    载荷系入射角。SVOM 惯性指向，天底在载荷系里扫遍全域，所以 θ 与大气
       几何无关；α 随 θ 变只能出在响应（CALDB 的质量模型）上，是仪器项。
  ζ    TGF 产生点（取 15 km）看卫星的天顶角，sec ζ 是大气斜程的空气质量因子。
       它与仪器无关；α 随它变只能出在大气里的康普顿散射，是物理项。
  f    核心窗内最忙一路的死时间占比。堆积把两个光子并成一个高道事例，
       让谱看起来更硬，是仪器项。

做法是一次全局拟合：逐 TGF 的谱指数写成 α_i = α0 + Σ β_k (x_ik − x̄_k)，
注量归一 A_i 仍逐 TGF 剖面掉。这样三支互相控制、83 个样本全用上，
不像切格子那样把样本稀释成每格 20 个。

注量由同一个 A_i 给出：A_i·s(E) 就是入射在卫星上的光子谱
（响应单位是 cm²），fluence = ∫ E·A_i·s(E) dE。

输入是 `scripts/cluster/svom_tgf_response.py` 在集群上导出的 npz
（入射方向取 WWLLN 的真实落点）与 `svom_wwlln_match.py` 的 incidence 表。
用法: python3 plot_svom_photon_systematics.py "<respw_*.npz>" <incidence.csv> \
          -o <png> --fluence <csv> [--seam "<seamtgf_*.csv>"]
"""
import argparse
import csv
import glob

import numpy as np
from scipy.optimize import minimize, minimize_scalar

CH_LO, CH_HI = 25, 230        # 可用道段，见 OPEN-QUESTIONS 第 25 条
RE = 6378.137
H_TGF = 15.0
KEV_TO_ERG = 1.602176634e-9
CORE_FRAC = 0.9545            # 核心窗 t0 ± 2σ 对高斯脉冲只收 95.45%
BAND = (42.0, 5488.0)         # 与拟合道段同一个能带
WIDE = (20.0, 1e4)            # 文献常用的 20 keV – 10 MeV，靠幂律外推


def load(pattern):
    files = sorted(glob.glob(pattern))
    z0 = np.load(files[0])
    out = dict(core=np.zeros_like(z0["core"]), bkg=np.zeros_like(z0["bkg"]),
               resp=np.zeros(z0["resp"].shape, np.float32), meta=np.zeros_like(z0["meta"]),
               done=np.zeros(z0["core"].shape[0], bool))
    for f in files:
        z = np.load(f)
        d = z["done"]
        for k in ("core", "bkg", "resp", "meta"):
            out[k][d] = z[k][d]
        out["done"] |= d
    for k in ("elo", "ehi", "clo", "chi", "starts"):
        out[k] = z0[k]
    return out


def build(z, keep=None, mask=None):
    """取出（沉积谱, 折算到核心窗的本底, 三路加权响应, ΔE, 道中心, 用哪些道）。"""
    d = z["done"].copy()
    if keep is not None:
        d &= keep
    core = z["core"][d][:, :, CH_LO:CH_HI].sum(1)
    bkg = z["bkg"][d][:, :, CH_LO:CH_HI].sum(1)
    meta = z["meta"][d]
    bkg = bkg * (meta[:, 0] / meta[:, 1])[:, None]
    live = 1.0 - np.clip(meta[:, 3:6], 0, 0.95)
    R = np.einsum("id,idec->iec", live,
                  z["resp"][d][:, :, :, CH_LO:CH_HI].astype(np.float64))
    m = np.ones(core.shape, bool) if mask is None else mask[d]
    return core, bkg, R, z["ehi"] - z["elo"], np.sqrt(z["elo"] * z["ehi"]), m


def cash(D, M):
    M = np.maximum(M, 1e-12)
    return 2.0 * np.sum(M - D * np.log(M))


def expect(alpha, R, de, ec):
    """逐 TGF 的 α（标量或数组）→ 单位归一下的期望计数。"""
    al = np.broadcast_to(np.atleast_1d(alpha), (R.shape[0],))
    return np.einsum("iec,ie->ic", R, (ec[None, :] / 100.0) ** (-al[:, None]) * de[None, :])


def profile(D, B, m0, m):
    A = np.empty(len(D))
    for i in range(len(D)):
        k = m[i]
        tot = max(D[i][k].sum() - B[i][k].sum(), 1.0)
        a0 = np.log(tot / max(m0[i][k].sum(), 1e-30))
        A[i] = np.exp(minimize_scalar(
            lambda lna: cash(D[i][k], np.exp(lna) * m0[i][k] + B[i][k]),
            bracket=(a0 - 2, a0, a0 + 2)).x)
    return A


def stat(alpha, D, B, R, de, ec, m):
    m0 = expect(alpha, R, de, ec)
    A = profile(D, B, m0, m)
    M = A[:, None] * m0 + B
    return cash(D[m], M[m]), A, M


def bracket_one(f, c, lo, hi):
    """f(step) − c 在 [lo, hi] 上跨过 1，二分到 2% 精度。"""
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        if f(mid) - c > 1.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fit_const(args):
    """所有 TGF 共用一个 α。误差取 ΔCash = 1（先几何扩张再二分，精度约 2%）。"""
    r = minimize_scalar(lambda a: stat(a, *args)[0], bounds=(0.2, 2.2),
                        method="bounded", options=dict(xatol=1e-4))
    a, c = float(r.x), float(r.fun)
    g = lambda s: stat(a + s, *args)[0]
    step, prev = 0.005, 0.0
    while step < 3 and g(step) - c <= 1.0:
        prev, step = step, step * 1.4
    return a, bracket_one(g, c, prev, min(step, 3.0)), c, stat(a, *args)


def fit_cov(X, args, p0):
    """α_i = p[0] + X[i] · p[1:]。误差逐参数剖面（其余重新极小化）。"""
    def f(p):
        return stat(p[0] + X @ np.asarray(p[1:]), *args)[0]

    best = None
    for jit in (0.0, 0.15, -0.15):
        r = minimize(f, np.array(p0) + jit, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-5, fatol=1e-5))
        if best is None or r.fun < best.fun:
            best = r
    p, c = best.x, float(best.fun)
    errs = []
    for k in range(len(p)):
        free = [j for j in range(len(p)) if j != k]

        def prof(step, sgn):
            """把第 k 个参数钉在 p[k] + sgn·step 上，其余重新极小化。"""
            q = p.copy()
            q[k] += sgn * step

            def g(y):
                v = q.copy()
                for m, j in enumerate(free):
                    v[j] = y[m]
                return f(v)

            return float(minimize(g, p[free], method="Nelder-Mead",
                                  options=dict(maxiter=2000, xatol=1e-5,
                                               fatol=1e-5)).fun)

        e = []
        for sgn in (-1, 1):
            step, prev = 0.01, 0.0
            while step < 4 and prof(step, sgn) - c <= 1.0:
                prev, step = step, step * 1.5
            e.append(bracket_one(lambda s: prof(s, sgn), c, prev, min(step, 4.0)))
        errs.append(0.5 * (e[0] + e[1]))
    return p, np.array(errs), c


def submask(z, m):
    k = np.zeros(len(z["done"]), bool)
    k[np.where(z["done"])[0][m]] = True
    return k


def grouped(D, M, clo, edges):
    idx = np.digitize(clo[CH_LO:CH_HI], edges) - 1
    rows = []
    for j in range(len(edges) - 1):
        k = idx == j
        if k.sum() == 0:
            continue
        o, e = D[:, k].sum(), M[:, k].sum()
        rows.append((edges[j], edges[j + 1], o, e, o / e, np.sqrt(max(o, 1)) / e))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resp")
    ap.add_argument("incidence")
    ap.add_argument("-o", "--out", default="svom_photon_systematics.png")
    ap.add_argument("--fluence", default=None, help="逐 TGF 注量表（CSV）")
    ap.add_argument("--seam", default=None,
                    help="逐 TGF 的增益交接道表（svom_seam_per_tgf.py 的输出通配）")
    a = ap.parse_args()

    z = load(a.resp)
    starts = z["starts"][z["done"]]
    rows = {r["start"]: r for r in csv.DictReader(open(a.incidence))}
    dist = np.array([float(rows[s]["dist_km"]) for s in starts])
    sep = np.array([float(rows[s]["sep_nadir_deg"]) for s in starts])
    th = z["meta"][z["done"], 8]
    ph = z["meta"][z["done"], 9]
    dead = z["meta"][z["done"], 3:6].max(1)
    # TGF 产生点看卫星的天顶角 = 星下点到落点的地心角 + 星上看到的源–天底夹角
    zen = sep + np.degrees(dist / RE)
    airmass = 1.0 / np.cos(np.radians(zen))
    n = len(th)

    print("== 样本（%d 个闪电证实 TGF，%s .. %s）==" % (n, starts[0][:10], starts[-1][:10]))
    print("  θ 中位 %.1f°（%.1f–%.1f）；落点距离中位 %.0f km（%.0f–%.0f）"
          % (np.median(th), th.min(), th.max(), np.median(dist), dist.min(), dist.max()))
    print("  源天顶角 ζ 中位 %.1f°（%.1f–%.1f），sec ζ 中位 %.2f（%.2f–%.2f）"
          % (np.median(zen), zen.min(), zen.max(), np.median(airmass),
             airmass.min(), airmass.max()))
    print("  最忙一路死时间 f 中位 %.3f，p90 %.3f" % (np.median(dead), np.percentile(dead, 90)))
    print("  三个协变量两两相关：θ–secζ %+.3f，θ–f %+.3f，secζ–f %+.3f"
          % (np.corrcoef(th, airmass)[0, 1], np.corrcoef(th, dead)[0, 1],
             np.corrcoef(airmass, dead)[0, 1]))

    args = build(z)
    a0, e0, c0, (_, _, M0) = fit_const(args)
    print("\n== 基准（单一 α）：α = %.3f ± %.3f，Cash = %.1f ==" % (a0, e0, c0))

    u = (th - np.median(th)) / 90.0
    v = airmass - np.median(airmass)
    w = dead - np.median(dead)
    names = ["θ 每 90°（仪器）", "sec ζ 每 +1（大气）", "死时间 f 每 +1（堆积）"]
    X = np.column_stack([u, v, w])

    print("\n== 单协变量（一次只放一个，会互相混进来）==")
    for nm, x in zip(names, (u, v, w)):
        p, e, c = fit_cov(x[:, None], args, [a0, 0.0])
        print("  %-20s β = %+.3f ± %.3f   ΔCash = %+.1f" % (nm, p[1], e[1], c - c0))

    print("\n== 三个一起（互相控制，这是结论用的那组）==")
    p3, e3, c3 = fit_cov(X, args, [a0, 0.4, 0.3, -1.0])
    print("  α0（三个量都取中位数处）= %.3f ± %.3f，ΔCash = %+.1f（3 自由度）"
          % (p3[0], e3[0], c3 - c0))
    for nm, x, b, e in zip(names, (u, v, w), p3[1:], e3[1:]):
        span = np.percentile(x, 90) - np.percentile(x, 10)
        print("  %-20s β = %+.3f ± %.3f (%.1fσ)，样本 10–90%% 跨度上 Δα = %+.3f ± %.3f"
              % (nm, b, e, abs(b) / e, b * span, e * span))

    print("\n== θ × sec ζ 二维分格（各按中位数二分）==")
    tc, vc = np.median(th), np.median(airmass)
    cell = {}
    for tn, tm in (("θ<%.0f°" % tc, th < tc), ("θ≥%.0f°" % tc, th >= tc)):
        for dn, dm in (("secζ<%.2f" % vc, airmass < vc), ("secζ≥%.2f" % vc, airmass >= vc)):
            m = tm & dm
            aa, ee, _, _ = fit_const(build(z, submask(z, m)))
            cell[(tn, dn)] = (aa, ee, int(m.sum()))
            print("  %-24s N=%2d  α = %.3f ± %.3f" % ("%s, %s" % (tn, dn), m.sum(), aa, ee))
    k = list(cell)
    print("  沿 θ 的边际 Δα = %+.3f；沿 sec ζ 的边际 Δα = %+.3f"
          % (0.5 * ((cell[k[2]][0] - cell[k[0]][0]) + (cell[k[3]][0] - cell[k[1]][0])),
             0.5 * ((cell[k[1]][0] - cell[k[0]][0]) + (cell[k[3]][0] - cell[k[2]][0]))))

    print("\n== θ 分四档：是 90° 处的台阶还是平滑漂移 ==")
    qs = np.percentile(th, [25, 50, 75])
    edges_t = [th.min() - 1, qs[0], qs[1], qs[2], th.max() + 1]
    th_bins = []
    for i in range(4):
        m = (th > edges_t[i]) & (th <= edges_t[i + 1])
        aa, ee, _, _ = fit_const(build(z, submask(z, m)))
        th_bins.append((np.median(th[m]), aa, ee))
        print("  θ ∈ (%.0f°, %.0f°]  N=%2d  α = %.3f ± %.3f"
              % (edges_t[i], edges_t[i + 1], m.sum(), aa, ee))

    print("\n== 排除「其实是某一路 GRD 标定偏了」：那会表现为 φ / 主导探头依赖 ==")
    dom = z["core"][z["done"]][:, :, CH_LO:CH_HI].sum(2).argmax(1)
    for d in (0, 1, 2):
        m = dom == d
        aa, ee, _, _ = fit_const(build(z, submask(z, m)))
        print("  主导 G%02d  N=%2d  α = %.3f ± %.3f" % (d + 1, m.sum(), aa, ee))
    for lo_, hi_ in ((0, 90), (90, 180), (180, 270), (270, 360)):
        m = (ph >= lo_) & (ph < hi_)
        aa, ee, _, _ = fit_const(build(z, submask(z, m)))
        print("  φ ∈ [%3d°, %3d°)  N=%2d  α = %.3f ± %.3f" % (lo_, hi_, m.sum(), aa, ee))

    print("\n== 逐能段的实测/模型（单一 α 的折叠模型，找响应缺陷的位置）==")
    ee_edges = np.array([42, 80, 120, 180, 250, 300, 400, 500, 640, 800, 1100,
                         1600, 2600, 5488.])
    tab = grouped(args[0], M0, z["clo"], ee_edges)
    for lo_, hi_, o, m_, r, er in tab:
        flag = "  <<<" if r + 2 * er < 1 else ("  >>>" if r - 2 * er > 1 else "")
        print("  %6.0f–%-6.0f keV  实测 %5.0f  模型 %6.1f  比 %.3f ± %.3f%s"
              % (lo_, hi_, o, m_, r, er, flag))
    o3 = args[0][:, (z["clo"][CH_LO:CH_HI] >= 300) & (z["chi"][CH_LO:CH_HI] <= 640)].sum()
    m3 = M0[:, (z["clo"][CH_LO:CH_HI] >= 300) & (z["chi"][CH_LO:CH_HI] <= 640)].sum()
    print("  300–640 keV 整段：实测 %.0f，模型 %.1f，比 %.3f ± %.3f（%.1fσ）"
          % (o3, m3, o3 / m3, np.sqrt(o3) / m3, abs(m3 - o3) / np.sqrt(o3)))

    if a.seam:
        seam = {}
        for f in sorted(glob.glob(a.seam)):
            for r in csv.DictReader(open(f)):
                seam[r["start"]] = int(r["seam_lo"])
        print("\n== 挡掉两档增益的交接道（逐 TGF，%d 个，ch%d–%d）=="
              % (len(seam), min(seam.values()), max(seam.values())))
        chs = np.arange(CH_LO, CH_HI)
        for pad in (2, 3):
            msk = np.ones((len(z["done"]), CH_HI - CH_LO), bool)
            for i, s in enumerate(z["starts"]):
                if s in seam:
                    msk[i, np.abs(chs - seam[s]) <= pad] = False
            aa, ee, _, _ = fit_const(build(z, mask=msk))
            print("  挡交接道 ±%d 道：α = %.3f ± %.3f（不挡 %.3f）" % (pad, aa, ee, a0))

    print("\n== 注量 ==")
    low = dead < 0.3
    al, el, _, _ = fit_const(build(z, submask(z, low)))
    ec, de = args[4], args[3]
    print("  谱形取低堆积子样本的 α = %.3f（堆积是仪器效应，扣掉后才是真谱形）；"
          "归一 A_i 逐 TGF 自由；核心窗只收 %.1f%% 的脉冲，已补回" % (al, 100 * CORE_FRAC))
    _, A, _ = stat(al, *args)
    s = (ec / 100.0) ** (-al)
    kb = (ec >= BAND[0]) & (ec <= BAND[1])
    kw = (ec >= WIDE[0]) & (ec <= WIDE[1])
    fe = A * float((ec[kb] * s[kb] * de[kb]).sum()) * KEV_TO_ERG / CORE_FRAC
    fp = A * float((s[kb] * de[kb]).sum()) / CORE_FRAC
    fw = A * float((ec[kw] * s[kw] * de[kw]).sum()) * KEV_TO_ERG / CORE_FRAC
    print("  %.0f keV–%.1f MeV：能量注量中位 %.2e erg/cm²（16–84%% %.2e–%.2e，"
          "范围 %.2e–%.2e）" % (BAND[0], BAND[1] / 1e3, np.median(fe),
                              np.percentile(fe, 16), np.percentile(fe, 84), fe.min(), fe.max()))
    print("  光子注量中位 %.3f ph/cm²（范围 %.3f–%.3f）；同一幂律外推到 20 keV–10 MeV "
          "的能量注量中位 %.2e erg/cm²" % (np.median(fp), fp.min(), fp.max(), np.median(fw)))
    print("  log10 注量的散布只有 %.2f dex——样本是流强受限的，"
          "不能当成源的注量分布（阈值还随 θ 变：注量与 θ 相关 %+.2f）"
          % (np.std(np.log10(fe)), np.corrcoef(fe, th)[0, 1]))
    for tag, alpha in (("α 固定 0.88", 0.88), ("α 固定 1.10", 1.10)):
        _, A2, _ = stat(alpha, *args)
        s2 = (ec / 100.0) ** (-alpha)
        f2 = A2 * float((ec[kb] * s2[kb] * de[kb]).sum()) * KEV_TO_ERG / CORE_FRAC
        print("  谱形改成 %s：注量中位 %.2e erg/cm²（对基准 ×%.2f）"
              % (tag, np.median(f2), np.median(f2) / np.median(fe)))
    sat = np.where(dead >= 0.9)[0]
    if len(sat):
        print("  单列：%s 有一路死时间 f = %.2f 已饱和，注量是下限"
              % (starts[sat[0]][:23], dead[sat[0]]))

    if a.fluence:
        with open(a.fluence, "w", newline="") as fh:
            wtr = csv.writer(fh)
            wtr.writerow(["start", "theta", "phi", "dist_km", "zenith_src_deg", "airmass",
                          "dead_frac_max", "n_core", "sigma_us", "fluence_erg_cm2",
                          "fluence_ph_cm2"])
            for i, s_ in enumerate(starts):
                wtr.writerow([s_, "%.2f" % th[i], "%.2f" % ph[i], "%.1f" % dist[i],
                              "%.2f" % zen[i], "%.3f" % airmass[i], "%.3f" % dead[i],
                              int(args[0][i].sum()), "%.1f" % (z["meta"][z["done"], 6][i] * 1e6),
                              "%.3e" % fe[i], "%.4f" % fp[i]])
        print("  → %s" % a.fluence)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["PingFang SC", "Arial Unicode MS"],
                         "font.family": "sans-serif", "axes.unicode_minus": False})
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

    x = [b[0] for b in th_bins]
    y = [b[1] for b in th_bins]
    er = [b[2] for b in th_bins]
    ax[0].errorbar(x, y, yerr=er, fmt="o-", color="C3", ms=5)
    ax[0].axvline(90, color="0.7", ls="--", lw=1)
    ax[0].set_xlabel("载荷系入射角 θ (°)")
    ax[0].set_ylabel(r"光子谱指数 $\alpha$")
    ax[0].set_title("沿 θ：仪器项（响应）\nβ = %+.2f ± %.2f 每 90°（分档未控其余两项）"
                    % (p3[1], e3[1]))

    qv = np.percentile(airmass, [12.5, 37.5, 62.5, 87.5])
    ev = np.percentile(airmass, [0, 25, 50, 75, 100])
    xv, yv, errv = [], [], []
    for i in range(4):
        m = (airmass >= ev[i]) & (airmass <= ev[i + 1])
        aa, ee, _, _ = fit_const(build(z, submask(z, m)))
        xv.append(np.median(airmass[m]))
        yv.append(aa)
        errv.append(ee)
    ax[1].errorbar(xv, yv, yerr=errv, fmt="o-", color="C0", ms=5)
    ax[1].set_xlabel(r"大气斜程因子 sec $\zeta$")
    ax[1].set_title("沿 sec ζ：物理项（大气散射）\nβ = %+.2f ± %.2f 每 +1（已控 θ 与堆积；"
                    "分档本身没控，所以不单调）" % (p3[2], e3[2]))
    ax[1].set_ylabel(r"光子谱指数 $\alpha$")

    ax[2].hist(np.log10(fe), bins=14, color="0.6", edgecolor="k", lw=0.5)
    ax[2].axvline(np.log10(np.median(fe)), color="C3", lw=1.6,
                  label="中位 %.1e erg/cm²" % np.median(fe))
    ax[2].set_xlabel(r"$\log_{10}$ 注量 (erg/cm²，42 keV–5.5 MeV)")
    ax[2].set_ylabel("个数")
    ax[2].set_title("83 个闪电证实 TGF 的注量\n（流强受限样本，非源分布）")
    ax[2].legend(fontsize=8)
    fig.suptitle("SVOM/GRM：叠加光子谱的系统项拆解与注量")
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("→", a.out)


if __name__ == "__main__":
    main()
