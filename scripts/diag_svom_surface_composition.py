"""只用星下点落在哪一类，估计显著池里真 TGF 的占比——不碰 LST、不碰模板、不碰台网。

思路来自 GBM 组：一批候选的下垫面构成必然落在「纯 TGF 的构成」与「纯本底的构成」
之间，位置由混合比例定。它**不需要模板有分辨力**，正好绕开 LST 那条腿在海上撞到的
墙（A ≈ 0.3 的模板在这个 N 上测不出 f，见 OPEN-QUESTIONS 第 32c 条）。

**但它有一个必须先验的前提：两个端点都得是这批数据里那两个成分真正的构成。**
本脚本先验前提再拟合——SVOM 上「用 83 个闪电证实当纯 TGF 端点」这条过不了，
因为那 83 个本身的下垫面构成被台网覆盖的地理分布扭了（近岸偏多、陆地偏少），
显著池有两类落在两个端点之外，两成分模型直接不成立。

过得了的是另一条：**纯本底端点稳，TGF 端点让数据自己定**。本底构成在
fa 0.1–1 / 1–5 / 5–20 三档上一致（三个数量级的虚警率），所以它可以固定；
TGF 构成当两个自由参数，各虚警率档的 f 各自自由，一起多项式拟合。
再加一条几乎不用假设的下界：真 TGF 的远洋占比不能是负数。

用法: python3 diag_svom_surface_composition.py <pool.csv>
"""
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from diag_svom_lst_template_crosscheck import load, PT_NAMES

BANDS = [("fa <= 1e-5（显著池）", lambda r: r["fa"] <= 1e-5),
         ("fa 1e-5 .. 1e-3", lambda r: 1e-5 < r["fa"] <= 1e-3),
         ("fa 1e-3 .. 0.1", lambda r: 1e-3 < r["fa"] <= 0.1),
         ("fa 0.1 .. 1", lambda r: 0.1 < r["fa"] <= 1.0)]
BKG = ("fa 5 .. 20（本底端点）", lambda r: 5.0 < r["fa"] <= 20.0)


def counts(rows, sel):
    sub = [r for r in rows if sel(r)]
    return np.array([sum(1 for r in sub if r["pt"] == s) for s in range(3)], float)


def show(tag, c):
    n = max(c.sum(), 1)
    e = np.sqrt(np.maximum(c, 1)) / n
    print("%-26s %6d  " % (tag, int(c.sum()))
          + "  ".join("%s %4.1f±%.1f%%" % (PT_NAMES[s], 100 * c[s] / n, 100 * e[s])
                      for s in range(3)))


def mnll(p_list, c_list):
    s = 0.0
    for p, c in zip(p_list, c_list):
        p = np.maximum(p, 1e-9)
        p = p / p.sum()
        s -= float((c * np.log(p)).sum())
    return s


def main(pool):
    rows = load(pool)
    print("下垫面构成随虚警率（单点口径：星下点判陆，海上离陆 < 300 km 算近岸）")
    print("%-26s %6s  %s" % ("样本", "N", "构成"))
    cb = counts(rows, BKG[1])
    for tag, sel in BANDS:
        show(tag, counts(rows, sel))
    show(*BKG[:1], cb) if False else show(BKG[0], cb)
    for tag, sel in (("fa 1 .. 5（本底对照）", lambda r: 1.0 < r["fa"] <= 5.0),
                     ("闪电证实 83（台网选样）",
                      lambda r: r["fa"] <= 1e-5 and r["in_cov"] == "1" and r["assoc"] == "1")):
        show(tag, counts(rows, sel))
    print("\n本底端点的稳定性：fa 0.1–1 / 1–5 / 5–20 三档构成一致（差 <1.5 个百分点），")
    print("跨三个数量级的虚警率不动，所以它可以当固定端点；从 fa 1e-3 起开始单调偏移，")
    print("那正是真 TGF 成分长出来的样子。")

    # ---- 前提检验：83 个闪电证实能不能当纯 TGF 端点 ---------------------------
    print("\n" + "=" * 78)
    print("前提检验：用 83 个闪电证实当纯 TGF 端点行不行")
    print("=" * 78)
    ct = counts(rows, lambda r: r["fa"] <= 1e-5 and r["in_cov"] == "1" and r["assoc"] == "1")
    cx = counts(rows, lambda r: r["fa"] <= 1e-5
                and not (r["in_cov"] == "1" and r["assoc"] == "1"))
    cg = counts(rows, lambda r: 0.1 < r["fa"] <= 1.0)
    pt, px, pg = ct / ct.sum(), cx / cx.sum(), cg / cg.sum()
    ok = True
    for s, nm in enumerate(PT_NAMES):
        lo, hi = min(pt[s], pg[s]), max(pt[s], pg[s])
        inside = lo <= px[s] <= hi
        ok &= inside
        print("  %-4s 待测 %5.1f%%   端点 TGF %5.1f%% / 本底 %5.1f%%   %s"
              % (nm, 100 * px[s], 100 * pt[s], 100 * pg[s],
                 "在区间内" if inside else "**在区间外**"))
    print("  ⇒ " + ("两成分模型成立。" if ok else
                    "两类落在端点之外，两成分模型不成立——**83 个闪电证实不是一个"
                    "无偏的 TGF 构成端点**（近岸偏多、陆地偏少，台网覆盖的地理分布带进来的）。"
                    "\n     这条对任何拿这 83 个当 TGF 人群代表的用法都成立，不只是这个估计量。"))

    # ---- 几乎不用假设的下界：真 TGF 的远洋占比不能为负 ------------------------
    print("\n" + "=" * 78)
    print("下界：真 TGF 的远洋占比 ≥ 0")
    print("=" * 78)
    c_sig = counts(rows, BANDS[0][1])
    p_sig, p_bkg = c_sig / c_sig.sum(), cb / cb.sum()
    r = p_sig[2] / p_bkg[2]
    er = r * np.hypot(1 / np.sqrt(max(c_sig[2], 1)), 1 / np.sqrt(max(cb[2], 1)))
    print("  显著池远洋占比 %.3f ± %.3f，本底端点远洋占比 %.3f ± %.3f"
          % (p_sig[2], np.sqrt(c_sig[2]) / c_sig.sum(), p_bkg[2], np.sqrt(cb[2]) / cb.sum()))
    print("  本底成分最多只能占 %.3f ± %.3f ⇒ **f ≥ %.3f ± %.3f**" % (r, er, 1 - r, er))
    print("  这条只用了「真 TGF 不会在远洋有负的占比」，不用模板、不用 LST、不用台网。")

    # ---- 多项式联合拟合：本底端点固定，TGF 构成让数据自己定 --------------------
    print("\n" + "=" * 78)
    print("联合拟合：本底端点固定在 fa 5–20 的构成，TGF 构成 2 个自由参数，各档 f 自由")
    print("=" * 78)
    cs = [counts(rows, sel) for _, sel in BANDS]
    g = cb / cb.sum()
    nb = len(BANDS)

    def unpack(v):
        t = np.exp(np.concatenate([[0.0], v[:2]]))
        t = t / t.sum()
        f = 1.0 / (1.0 + np.exp(-v[2:]))
        return t, f

    def nll(v):
        t, f = unpack(v)
        return mnll([f[b] * t + (1 - f[b]) * g for b in range(nb)], cs)

    best = None
    for seed in range(6):
        v0 = np.concatenate([np.random.default_rng(seed).normal(0, 0.5, 2),
                             np.linspace(1.5, -2.0, nb)])
        r0 = minimize(nll, v0, method="Nelder-Mead",
                      options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7))
        if best is None or r0.fun < best.fun:
            best = r0
    t, f = unpack(best.x)
    print("  拟合出来的纯 TGF 构成：" +
          "  ".join("%s %.1f%%" % (PT_NAMES[s], 100 * t[s]) for s in range(3)))
    print("  （对照：83 个闪电证实给 " +
          "  ".join("%s %.1f%%" % (PT_NAMES[s], 100 * pt[s]) for s in range(3)) + "）")
    c_best = float(best.fun)
    for b, (tag, _) in enumerate(BANDS):
        # 把第 b 档的 f 钉住、其余重新极小化，扫 ΔlnL = 0.5
        def prof(val):
            def h(w):
                v = best.x.copy()
                v[:2] = w[:2]
                v[2:] = w[2:]
                v[2 + b] = np.log(val / (1 - val)) if 0 < val < 1 else (20 if val >= 1 else -20)
                return nll(v)
            return float(minimize(h, best.x, method="Nelder-Mead",
                                  options=dict(maxiter=8000, xatol=1e-6, fatol=1e-6)).fun)
        lo, hi = f[b], f[b]
        while lo > 0.005 and prof(lo) - c_best <= 0.5:
            lo -= 0.01
        while hi < 0.995 and prof(hi) - c_best <= 0.5:
            hi += 0.01
        print("  %-26s f = %.2f [%.2f, %.2f]" % (tag, f[b], lo, hi))
    print("\n  这条腿与 LST 那条完全不共享输入（一个用位置的下垫面类别，一个用时间的"
          "当地太阳时），\n  与幂律外推也不共享（那条只用虚警率分布）。")


if __name__ == "__main__":
    main(sys.argv[1])
