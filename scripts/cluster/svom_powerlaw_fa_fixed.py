"""第 15 条的幂律外推，用改正后的 fa 重做一遍（画法与 plot_svom_powerlaw.py 一致）。"""
import json
import math
import sys

import numpy as np
from scipy.optimize import curve_fit


def pmf(c, lam):
    if lam <= 0:
        return 1.0 if c == 0 else 0.0
    return math.exp(-lam + c * math.log(lam) - math.lgamma(c + 1.0))


def power_law(x, a, b):
    return a * x ** b


data = [r for r in json.load(open(sys.argv[1]))
        if r["signal"]["start"] < "2025-01-01" and r["lightning"].get("in_coverage", True)]
fa0 = np.array([r["signal"]["false_positive_per_year"] for r in data])
corr = np.array([1.0 + pmf(int(r["signal"]["count"]), float(r["signal"]["mean"]))
                 / float(r["signal"]["sf"]) if float(r["signal"]["sf"]) > 0 else 1.0
                 for r in data])
assoc = np.array([bool(r["lightning"].get("associated")) for r in data])
print("覆盖内候选 %d 个，关联 %d 个。改正因子中位 %.2f" % (len(data), assoc.sum(), np.median(corr)))

for tag, fa in (("原 fa", fa0), ("改正后 fa", fa0 * corr)):
    mn, mx, nb = min(fa.min(), 1e-30) / 10.0, fa.max(), 100
    edges = np.logspace(np.log10(mn), np.log10(mx), nb + 1)
    cen = np.sqrt(edges[:-1] * edges[1:])
    n_all, _ = np.histogram(fa, bins=edges)
    n_as, _ = np.histogram(fa[assoc], bins=edges)
    print("\n== %s ==" % tag)
    fits = {}
    for name, cond, y in (("all_bkg", cen > 1e-3, n_all), ("all_tgf", cen < 1e-8, n_all),
                          ("assoc", cen < 1e-2, n_as)):
        m = cond & (y > 0)
        if m.sum() < 3:
            continue
        p, _ = curve_fit(power_law, cen[m], y[m], p0=(1.0, 0.1), maxfev=40000)
        fits[name] = p
        print("  fit %-8s a=%.4g b=%.4f" % (name, p[0], p[1]))
    sig = fa <= 1e-5
    acc = (fa < 1e-5) | ((fa < 1.0) & assoc)
    print("  显著 fa<=1e-5: %d，判选接受 %d（直接 %d + 仅关联 %d）"
          % (sig.sum(), acc.sum(), (fa < 1e-5).sum(),
             ((fa >= 1e-5) & (fa < 1) & assoc).sum()))
    if "all_bkg" in fits:
        a, b = fits["all_bkg"]
        # 把本底幂律积到显著段：sum over bins of a*cen^b
        k = cen <= 1e-5
        bkg_pred = float((a * cen[k] ** b).sum())
        obs = float(n_all[k].sum())
        print("  本底幂律外推到 fa<=1e-5：期望本底 %.0f，实测 %.0f ⇒ 真 TGF %.0f，纯度 %.3f"
              % (bkg_pred, obs, obs - bkg_pred, (obs - bkg_pred) / max(obs, 1)))
