"""SVOM/GRM 83 个闪电证实 TGF 的叠加**光子**谱：正向折叠拟合与作图。

响应矩阵不可逆，所以不做解卷积。做法是假设一个光子谱形 s(E)，用 CALDB 的
方向相关响应折成各道的期望计数，再与实测沉积谱比：

    m[i, ch] = A_i · Σ_d (1 − f[i,d]) · Σ_E R[i,d,E,ch] · s(E) · ΔE

  A_i    第 i 个 TGF 的注量归一（自由参数，只吸收亮度差别，不影响谱形）
  f[i,d] 该核心窗内该探头的死时间占比（Σ DEAD_TIME / 窗长）
  R      官方 RSP_Generator 在该 TGF 的**真实入射方向**（WWLLN 落点，不是天底）
         与该 MET 下给出的响应（cm²，已含能量分辨与增益）

统计量用 Cash；本底当已知量加进期望（核心窗只有 0.15 ms，本底占 3%）。

输入是 `scripts/cluster/svom_tgf_response.py` 在集群上导出的 npz。
用法: python3 plot_svom_photon_spectrum.py "<respw_*.npz>" <incidence.csv> -o <png>
"""
import argparse
import csv
import glob

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2 as chi2dist

# 可用道段。上限来自实测：本底谱在 ch230 之后隆起（ch240–244 是 ch200–204 的
# 8 倍）、到 ch250 断崖归零，是可用 ADC 量程顶端的假象，不是物理。
CH_LO, CH_HI = 25, 230

PL = lambda e, p: (e / 100.0) ** (-p[0])
CPL = lambda e, p: (e / 100.0) ** (-p[0]) * np.exp(-e / np.clip(np.exp(p[1]), 1, 1e7))
RREA = lambda e, p: (e / 100.0) ** (-1.0) * np.exp(-e / np.clip(np.exp(p[0]), 1, 1e7))


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


def build(z, keep=None, ch_lo=CH_LO, ch_hi=CH_HI):
    d = z["done"].copy()
    if keep is not None:
        d &= keep
    core = z["core"][d][:, :, ch_lo:ch_hi]
    bkg = z["bkg"][d][:, :, ch_lo:ch_hi]
    resp = z["resp"][d][:, :, :, ch_lo:ch_hi].astype(np.float64)
    meta = z["meta"][d]
    live = 1.0 - np.clip(meta[:, 3:6], 0, 0.95)
    R = np.einsum("id,idec->iec", live, resp)
    return (core.sum(1), bkg.sum(1) * (meta[:, 0] / meta[:, 1])[:, None], R,
            z["ehi"] - z["elo"], np.sqrt(z["elo"] * z["ehi"]))


def cash(D, M):
    M = np.maximum(M, 1e-12)
    return 2.0 * np.sum(M - D * np.log(M))


def stat(shape, p, D, B, R, de, ec):
    m0 = np.einsum("iec,e->ic", R, shape(ec, p) * de)
    A = np.empty(len(D))
    for i in range(len(D)):
        tot = max(D[i].sum() - B[i].sum(), 1.0)
        a0 = np.log(tot / max(m0[i].sum(), 1e-30))
        A[i] = np.exp(minimize_scalar(
            lambda lna: cash(D[i], np.exp(lna) * m0[i] + B[i]),
            bracket=(a0 - 2, a0, a0 + 2)).x)
    return cash(D, A[:, None] * m0 + B), A, A[:, None] * m0 + B


def fit(shape, p0, args):
    f = lambda p: stat(shape, p, *args)[0]
    r = minimize(f, p0, method="Nelder-Mead",
                 options=dict(maxiter=500, xatol=1e-4, fatol=1e-4))
    c = f(r.x)
    errs = []
    for k in range(len(r.x)):
        e = []
        for sgn in (-1, 1):
            step = 0.005
            while step < 6:
                q = r.x.copy()
                q[k] += sgn * step
                if f(q) - c > 1.0:
                    break
                step *= 1.1
            e.append(min(step, 6.0))
        errs.append(e)
    return r.x, errs, c, stat(shape, r.x, *args)


def submask(z, m):
    k = np.zeros(len(z["done"]), bool)
    k[np.where(z["done"])[0][m]] = True
    return k


def group(D, M, clo, chi, nb=12):
    ed = clo[CH_LO:CH_HI]
    edges = np.geomspace(ed[0], chi[CH_HI - 1], nb + 1)
    idx = np.digitize(ed, edges) - 1
    rows = []
    for j in range(nb):
        m = idx == j
        if m.sum() == 0:
            continue
        rows.append((edges[j], edges[j + 1], D[:, m].sum(), M[:, m].sum()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resp", help="respw_*.npz 的通配（引号括起来）")
    ap.add_argument("incidence", help="tgf_incidence_wwlln.csv")
    ap.add_argument("-o", "--out", default="svom_photon_spectrum.png")
    a = ap.parse_args()

    z = load(a.resp)
    starts = z["starts"][z["done"]]
    rows = {r["start"]: r for r in csv.DictReader(open(a.incidence))}
    dist = np.array([float(rows[s]["dist_km"]) for s in starts])
    th = z["meta"][z["done"], 8]
    dead = z["meta"][z["done"], 3:6].max(1)

    full = build(z)
    p, e, c, (_, A, M) = fit(PL, [0.9], full)
    print("全样本 N=%d，叠加计数 %.0f（本底 %.1f）：α = %.3f −%.3f/+%.3f"
          % (len(full[0]), full[0].sum(), full[1].sum(), p[0], e[0][0], e[0][1]))
    rows_g = group(full[0], M, z["clo"], z["chi"])
    chi2 = sum((o - m) ** 2 / m for _, _, o, m in rows_g)
    dof = len(rows_g) - 2
    print("  分组 χ² = %.1f / %d = %.2f（p = %.3f）"
          % (chi2, dof, chi2 / dof, 1 - chi2dist.cdf(chi2, dof)))

    low = submask(z, dead < 0.3)
    args_l = build(z, low)
    pl, el, cl, (_, Al, Ml) = fit(PL, [1.0], args_l)
    print("低堆积（最忙一路 f<0.3）N=%d，计数 %.0f：α = %.3f −%.3f/+%.3f"
          % (len(args_l[0]), args_l[0].sum(), pl[0], el[0][0], el[0][1]))
    pc, ec_, cc, _ = fit(CPL, [0.9, np.log(1e4)], args_l)
    print("  加指数截断：α = %.3f，Ec = %.0f keV，ΔCash = %+.1f（不显著则说明波段内无截断）"
          % (pc[0], min(np.exp(pc[1]), 9.9e6), cc - cl))

    # 正面/背面
    for tag, m in (("θ<90°", th < 90), ("θ≥90°", th >= 90)):
        pp, ee, _, _ = fit(PL, [0.9], build(z, submask(z, m)))
        print("  %s（N=%d）：α = %.3f ± %.3f" % (tag, m.sum(), pp[0], ee[0][0]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["PingFang SC", "Arial Unicode MS"],
                         "font.family": "sans-serif", "axes.unicode_minus": False})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                             gridspec_kw=dict(height_ratios=[2.2, 1]))
    for col, (tag, args, pp, ee, MM) in enumerate((
            ("全部 83 个", full, p, e, M),
            ("低堆积 55 个（f<0.3）", args_l, pl, el, Ml))):
        ax, axr = axes[0, col], axes[1, col]
        g = group(args[0], MM, z["clo"], z["chi"])
        x = np.array([np.sqrt(lo * hi) for lo, hi, _, _ in g])
        w = np.array([hi - lo for lo, hi, _, _ in g])
        o = np.array([oo for _, _, oo, _ in g])
        mm = np.array([m for _, _, _, m in g])
        ax.errorbar(x, o / w, yerr=np.sqrt(o) / w, fmt="o", ms=4, color="k",
                    label="实测沉积谱（本底已计入模型）")
        ax.step(x, mm / w, where="mid", color="C3", lw=1.6,
                label="正向折叠：dN/dE ∝ E$^{-%.2f}$" % pp[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("计数 / keV")
        ax.set_title("%s，α = %.2f ± %.2f" % (tag, pp[0], ee[0][0]))
        ax.legend(fontsize=8)
        axr.axhline(1.0, color="0.6", lw=1)
        axr.errorbar(x, o / mm, yerr=np.sqrt(o) / mm, fmt="o", ms=4, color="k")
        axr.set_xscale("log")
        axr.set_xlabel("沉积能量 (keV)")
        axr.set_ylabel("实测 / 模型")
        axr.set_ylim(0.5, 1.6)
    fig.suptitle("SVOM/GRM 闪电证实 TGF 的叠加光子谱（CALDB 方向相关响应正向折叠）")
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("→", a.out)


if __name__ == "__main__":
    main()
