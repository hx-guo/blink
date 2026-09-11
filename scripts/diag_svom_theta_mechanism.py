"""θ 那一支的机理定位：它住在哪个能段，需要多大的响应误差才能造出来。

第 27 条量到 α 随载荷系入射角 θ 变，β_θ = +0.461 ± 0.062（7.4σ）。θ 与大气几何
无关（SVOM 惯性指向，天底在载荷系里扫遍全域），所以只能出在响应上。第 27 条给了
一个**机理候选但没验证**：实测 CALDB 背/正有效面积比从 42–80 keV 的 0.124 单调升
到 5–8 MeV 的 0.870，背面低能响应只有正面的 12%，是个小数，质量模型里不大的
**绝对**误差在那一段就是很大的**相对**误差；而那一段正是驱动 α 的杠杆端。

第 31 条另外留下一条待查：**42–80 keV 的软超出 1.260 ± 0.090**（2.9σ），是整个
42 keV–5.5 MeV 里唯一超 2σ 的一段，紧贴 ch25 能阈。

这两条如果是同一件事，那就只有一个机理：**CALDB 低估了背面入射的低能有效面积**。
它同时预言两件可测的事：

* **预言一**：θ 项住在低能端。把低能道从拟合里去掉，β_θ 应当塌下去；
  从高能端切同样多的道作对照，β_θ 不该有同样的变化。
* **预言二**：42–80 keV 的超出应当集中在大 θ 一侧。若两侧一样大，
  那超出与 θ 项就是两件事，这个机理只能解释其中一件。

再加一个定量的可证伪点：**把背面 42–80 keV 的响应乘上 (1+ε) 重拟，求使 β_θ = 0
的 ε**，换算成绝对面积，看它是不是一个「对正面来说不大」的数。若需要的 ε 大到
离谱（例如面积要翻几倍），这个机理就不成立。

用法: python3 diag_svom_theta_mechanism.py "<respw_*.npz>" <incidence.csv>
"""
import argparse
import csv
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot_svom_photon_systematics import CH_LO, CH_HI, RE, load, cash

def build_masked(z, ch_lo, ch_hi):
    """与 plot_svom_photon_systematics.build 相同，但道段可调。"""
    d = z["done"]
    core = z["core"][d][:, :, ch_lo:ch_hi].sum(1)
    bkg = z["bkg"][d][:, :, ch_lo:ch_hi].sum(1)
    meta = z["meta"][d]
    bkg = bkg * (meta[:, 0] / meta[:, 1])[:, None]
    live = 1.0 - np.clip(meta[:, 3:6], 0, 0.95)
    R = np.einsum("id,idec->iec", live,
                  z["resp"][d][:, :, :, ch_lo:ch_hi].astype(np.float64))
    return core, bkg, R


def expect(alpha, R, de, ec):
    al = np.broadcast_to(np.atleast_1d(alpha), (R.shape[0],))
    return np.einsum("iec,ie->ic", R, (ec[None, :] / 100.0) ** (-al[:, None]) * de[None, :])


def profile_robust(D, B, m0):
    """逐 TGF 剖面掉注量归一 A_i。

    与 plot_svom_photon_systematics.profile 等价，但把 Brent 的 bracket 换成有界
    极小化——切掉低能道之后某些 TGF 的净计数掉到个位数，Brent 会 bracket 失败。
    """
    A = np.empty(len(D))
    for i in range(len(D)):
        tot = max(D[i].sum() - B[i].sum(), 1.0)
        a0 = np.log(tot / max(m0[i].sum(), 1e-30))
        r = minimize_scalar(lambda lna: cash(D[i], np.exp(lna) * m0[i] + B[i]),
                            bounds=(a0 - 8.0, a0 + 8.0), method="bounded",
                            options=dict(xatol=1e-6))
        A[i] = np.exp(float(r.x))
    return A


def stat(alpha, D, B, R, de, ec):
    m0 = expect(alpha, R, de, ec)
    A = profile_robust(D, B, m0)
    M = A[:, None] * m0 + B
    return cash(D, M), A, M


def fit_cov(X, D, B, R, de, ec, p0):
    """α_i = p[0] + X[i] · p[1:]，只要最佳值不要剖面误差（ε 扫描用）。"""
    def f(p):
        return stat(p[0] + X @ np.asarray(p[1:]), D, B, R, de, ec)[0]

    best = None
    for jit in (0.0, 0.12):
        r = minimize(f, np.array(p0) + jit, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-5, fatol=1e-5))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(best.fun)


def covariates(z, incidence):
    starts = z["starts"][z["done"]]
    rows = {r["start"]: r for r in csv.DictReader(open(incidence))}
    dist = np.array([float(rows[s]["dist_km"]) for s in starts])
    sep = np.array([float(rows[s]["sep_nadir_deg"]) for s in starts])
    meta = z["meta"][z["done"]]
    th = meta[:, 8]
    # ζ：TGF 产生点看卫星的天顶角 = 星下点到落点的地心角 + 星上看到的源–天底夹角
    # （与 plot_svom_photon_systematics.py 逐字一致，协变量口径必须同一套）
    zen = sep + np.degrees(dist / RE)
    sec = 1.0 / np.cos(np.radians(zen))
    f = meta[:, 3:6].max(1)
    X = np.column_stack([(th - np.median(th)) / 90.0, sec - np.median(sec),
                         f - np.median(f)])
    return X, th, sec, f


def band_rows(D, M, clo, ch_lo, ch_hi, edges):
    idx = np.digitize(clo[ch_lo:ch_hi], edges) - 1
    out = []
    for j in range(len(edges) - 1):
        k = idx == j
        if k.sum() == 0:
            continue
        o, e = D[:, k].sum(), M[:, k].sum()
        out.append((edges[j], edges[j + 1], o, e, o / e, np.sqrt(max(o, 1)) / e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resp")
    ap.add_argument("incidence")
    a = ap.parse_args()
    z = load(a.resp)
    clo, chi = z["clo"], z["chi"]
    elo, ehi = z["elo"], z["ehi"]
    de, ec = ehi - elo, np.sqrt(elo * ehi)
    X, th, sec, f = covariates(z, a.incidence)
    n = len(th)
    print("83 个闪电证实里 done = %d 个。θ ≥ 90° 的 %d 个，θ < 90° 的 %d 个。"
          % (n, int((th >= 90).sum()), int((th < 90).sum())))
    print("可用道段 ch%d–%d，对应沉积能 %.1f–%.1f keV。"
          % (CH_LO, CH_HI - 1, clo[CH_LO], chi[CH_HI - 1]))

    D, B, R = build_masked(z, CH_LO, CH_HI)
    p, c0 = fit_cov(X, D, B, R, de, ec, [0.9, 0.4, 0.3, -1.0])
    print("\n基准三协变量拟合：α0 = %.3f，β_θ(每 90°) = %+.3f，β_secζ = %+.3f，β_f = %+.3f"
          % (p[0], p[1], p[2], p[3]))

    # ---- 预言一：θ 项住在哪个能段 -------------------------------------------
    print("\n" + "=" * 84)
    print("预言一：把低能道切掉，β_θ 应当塌；从高能端切同样多的道作对照")
    print("=" * 84)
    print("%-28s %6s %8s %10s %10s" % ("拟合用的道段", "道数", "净计数", "β_θ", "相对基准"))

    def report(tag, lo, hi):
        Dx, Bx, Rx = build_masked(z, lo, hi)
        px, _ = fit_cov(X, Dx, Bx, Rx, de, ec, list(p))
        net = float(Dx.sum() - Bx.sum())
        print("%-28s %6d %8.0f %10.3f %10.2f"
              % (tag, hi - lo, net, px[1], px[1] / p[1]))
        return px[1]

    b_base = report("ch%d–%d（基准，%.0f keV 起）" % (CH_LO, CH_HI - 1, clo[CH_LO]),
                    CH_LO, CH_HI)
    lows = []
    for e_cut in (80.0, 150.0, 300.0, 600.0):
        lo = int(np.searchsorted(clo, e_cut))
        if lo >= CH_HI - 20:
            continue
        lows.append((e_cut, lo, report("切掉 < %.0f keV（ch%d 起）" % (e_cut, lo), lo, CH_HI)))
    # 高能端对照：切掉与低能端同样多的道
    for e_cut, lo, _ in lows:
        ncut = lo - CH_LO
        hi = CH_HI - ncut
        if hi <= CH_LO + 20:
            continue
        report("对照：高端切掉同样 %d 道（到 %.0f keV）" % (ncut, chi[hi - 1]),
               CH_LO, hi)

    # ---- 预言二：42–80 keV 的超出是不是集中在大 θ 一侧 -----------------------
    print("\n" + "=" * 84)
    print("预言二：42–80 keV 的超出按 θ 分半。两侧一样大 ⇒ 超出与 θ 项是两件事")
    print("=" * 84)
    edges = np.array([42.0, 80.0, 150.0, 300.0, 640.0, 1200.0, 2500.0, 5488.0])
    for tag, k in (("全样本", np.ones(n, bool)), ("θ < 90°（正面）", th < 90),
                   ("θ ≥ 90°（背面）", th >= 90)):
        Dk, Bk, Rk = D[k], B[k], R[k]
        # 每个子样本自己拟一个常数 α，这样比值反映的是形状不匹配而不是 α 差异
        r = minimize_scalar(lambda al: stat(al, Dk, Bk, Rk, de, ec)[0],
                            bounds=(0.2, 2.2), method="bounded",
                            options=dict(xatol=1e-4))
        al = float(r.x)
        _, _, M = stat(al, Dk, Bk, Rk, de, ec)
        rows = band_rows(Dk, M, clo, CH_LO, CH_HI, edges)
        print("\n  %-14s N=%2d  自拟 α = %.3f" % (tag, int(k.sum()), al))
        print("    %-18s %8s %10s %16s" % ("沉积能段", "实测", "模型", "实测/模型"))
        for e0, e1, o, e, ratio, err in rows:
            flag = "  <<" if abs(ratio - 1) > 2 * err else ""
            print("    %6.0f–%-11.0f %8.0f %10.1f %10.3f ± %.3f%s"
                  % (e0, e1, o, e, ratio, err, flag))

    # ---- 定量：要多大的背面低能响应误差才能把 β_θ 抹平 ------------------------
    print("\n" + "=" * 84)
    print("定量：把背面（θ ≥ 90°）42–80 keV 的入射响应乘 (1+ε)，求 β_θ = 0 的 ε")
    print("=" * 84)
    eband = (ec >= 42.0) & (ec < 80.0)
    back = th >= 90
    print("  被扰动的入射能格 %d 个（%.0f–%.0f keV），被扰动的 TGF %d 个。"
          % (int(eband.sum()), elo[eband][0], ehi[eband][-1], int(back.sum())))
    print("  %8s %10s %12s" % ("ε", "β_θ", "ΔCash"))
    grid, betas = [], []
    for eps in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        Rp = R.copy()
        Rp[np.ix_(back, eband)] *= (1.0 + eps)
        px, cx = fit_cov(X, D, B, Rp, de, ec, list(p))
        grid.append(eps)
        betas.append(px[1])
        print("  %8.2f %10.3f %12.1f" % (eps, px[1], cx - c0))
    grid, betas = np.array(grid), np.array(betas)
    if betas.min() < 0 < betas.max() or betas[-1] < betas[0]:
        j = np.argsort(betas)
        eps0 = float(np.interp(0.0, betas[j], grid[j]))
        print("\n  线性内插给 β_θ = 0 的 ε ≈ %.2f" % eps0)
        a_back = 30.7   # evidence/systematics/effective_area.txt，42–80 keV 背面
        a_front = 247.5
        print("  换算：背面 42–80 keV 有效面积 %.1f → %.1f cm²（+%.1f cm²），"
              % (a_back, a_back * (1 + eps0), a_back * eps0)
              + "相当于同能段正面面积 %.1f cm² 的 %.1f%%。"
              % (a_front, 100.0 * a_back * eps0 / a_front))
    else:
        print("\n  β_θ 在扫描范围内没有过零，这个机理单独解释不了 θ 项。")


if __name__ == "__main__":
    main()
