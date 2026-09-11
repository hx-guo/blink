"""用当地太阳时（LST）估「这批候选里有多少是真 TGF」，不依赖闪电台网。

雷暴有强日变化（陆地午后峰），本底没有。把一批候选的 LST 直方图拆成
「真 TGF 模板 + 平坦本底」两成分拟合，拟合出的 f_TGF 就是这批里真的占几成。
这是与 WWLLN 完全独立的第二条腿——WWLLN 只到 2024-12-31、台网效率不均、
暗 TGF 的母闪电常探不到，所以关联数一直只是下限。

四条不循环的硬要求：
  1. 真 TGF 模板来自**外部真值样本**（闪电证实的那 83 个），不来自被检验的候选；
  2. 本底模板来自**留出的、与被检验样本不重叠**的 fa 区间；
  3. 先验证本底的 LST 分布确实是平的——曝光的轨道进动、SAA 穿越相位都可能
     造出假的 LST 结构；
  4. 判别量与选样标准在构造上无关：LST 由位置与 UTC 算，选样只看显著性。

LST = UTC + 经度/15° + 时差方程（后者幅度 ±0.29 h，本脚本算进去）。

用法: python3 diag_svom_local_time.py <pool.csv>
      pool.csv 列：start, fa, lon, lat, assoc, in_cov, is_train
"""
import csv
import datetime as dt
import sys

import numpy as np
from scipy.optimize import minimize_scalar

NB = 8                       # 3 小时一格


def eot_hours(doy):
    """时差方程（小时），标准近似，幅度 ±0.29 h。"""
    b = 2 * np.pi * (doy - 81) / 364.0
    return (9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)) / 60.0


def lst(rows):
    out = []
    for r in rows:
        t = dt.datetime.strptime(r["start"][:19], "%Y-%m-%dT%H:%M:%S")
        utc = t.hour + t.minute / 60.0 + t.second / 3600.0
        h = utc + float(r["lon"]) / 15.0 + eot_hours(t.timetuple().tm_yday)
        out.append(h % 24.0)
    return np.array(out)


def hist(v):
    h, _ = np.histogram(v, bins=NB, range=(0, 24))
    return h.astype(float)


def fit_frac(obs, tmpl, bkg):
    """obs = N·[f·tmpl + (1−f)·bkg]，泊松似然对 f 求极值，误差取 ΔlnL = 0.5。"""
    n = obs.sum()
    t = tmpl / tmpl.sum()
    b = bkg / bkg.sum()

    def nll(f):
        m = np.maximum(n * (f * t + (1 - f) * b), 1e-9)
        return float((m - obs * np.log(m)).sum())

    r = minimize_scalar(nll, bounds=(-0.5, 2.0), method="bounded", options=dict(xatol=1e-4))
    f0, c = float(r.x), float(r.fun)
    lo, hi = f0, f0
    while lo > -0.5 and nll(lo) - c <= 0.5:
        lo -= 0.005
    while hi < 2.0 and nll(hi) - c <= 0.5:
        hi += 0.005
    return f0, max(lo, -0.5), min(hi, 2.0), nll(0.0) - c


def main(path, ext_tmpl=None):
    rows = [r for r in csv.DictReader(open(path)) if abs(float(r["lon"])) <= 180]
    for r in rows:
        r["fa"] = float(r["fa"])
    pool = [r for r in rows if r["is_train"] == "0"]
    conf = [r for r in pool if r["fa"] <= 1e-5 and r["in_cov"] == "1" and r["assoc"] == "1"]
    print("池 %d（去列车后 %d），闪电证实 %d" % (len(rows), len(pool), len(conf)))

    t_conf = lst(conf)
    h_conf = hist(t_conf)
    print("\n== 模板（83 个闪电证实 TGF 的 LST）==")
    print("  " + "  ".join("%02d-%02d %2d(%.0f%%)" % (3 * i, 3 * i + 3, h_conf[i],
                                                      100 * h_conf[i] / h_conf.sum())
                           for i in range(NB)))
    # 模板本身有多少「日变化」：与均匀比的 χ²
    e = h_conf.sum() / NB
    chi = float(((h_conf - e) ** 2 / e).sum())
    print("  与均匀比 χ² = %.1f / %d 自由度" % (chi, NB - 1))
    r_ = np.abs(np.exp(2j * np.pi * t_conf / 24).mean())
    print("  瑞利检验 R = %.3f，p = %.2f" % (r_, np.exp(-len(t_conf) * r_ ** 2)))

    t_bkg = lst([r for r in pool if r["fa"] > 1])
    h_bkg = hist(t_bkg)
    print("\n== 本底模板（留出：fa > 1 的 %d 个）==" % len(t_bkg))
    print("  " + "  ".join("%02d-%02d %.1f%%" % (3 * i, 3 * i + 3,
                                                 100 * h_bkg[i] / h_bkg.sum())
                           for i in range(NB)))
    e = h_bkg.sum() / NB
    chi_b = float(((h_bkg - e) ** 2 / e).sum())
    print("  与均匀比 χ² = %.1f / %d —— 平不平决定这套方法能不能用" % (chi_b, NB - 1))

    print("\n== 两成分拟合：f_TGF ==")
    print("%-34s %6s %8s %16s %10s" % ("被检验的样本", "N", "f_TGF", "68% 区间", "ΔlnL(f=0)"))
    tests = [("显著 fa ≤ 1e-5（全部）", lambda r: r["fa"] <= 1e-5),
             ("显著、覆盖内、未关联", lambda r: (r["fa"] <= 1e-5 and r["in_cov"] == "1"
                                         and r["assoc"] == "0")),
             ("显著、覆盖外（无法验证的 696）",
              lambda r: r["fa"] <= 1e-5 and r["in_cov"] == "0"),
             ("目录里非证实的那部分（fa≤1e-5 去掉 83 个）",
              lambda r: r["fa"] <= 1e-5 and not (r["in_cov"] == "1" and r["assoc"] == "1")),
             ("fa 1e-5 .. 1e-3", lambda r: 1e-5 < r["fa"] <= 1e-3),
             ("fa 1e-3 .. 0.1", lambda r: 1e-3 < r["fa"] <= 0.1),
             ("fa 0.1 .. 1", lambda r: 0.1 < r["fa"] <= 1.0)]
    for tag, sel in tests:
        sub = [r for r in pool if sel(r)]
        if len(sub) < 20:
            continue
        h = hist(lst(sub))
        f, lo, hi, d0 = fit_frac(h, h_conf, h_bkg)
        print("%-34s %6d %8.2f   [%.2f, %.2f] %10.1f"
              % (tag, len(sub), f, lo, hi, d0))

    print("\n注：模板与本底都来自与被检验样本不重叠的集合；「显著、覆盖内、未关联」"
          "与模板不重叠，「目录里非证实的那部分」也已剔掉那 83 个。")

    if ext_tmpl is None:
        return
    # 外部模板：同样日期、同样纬度带的 WWLLN 落点自己的 LST 分布。
    # 两个自带偏差：全部闪电 ≠ 产生 TGF 的闪电；WWLLN 探测效率自己有日变化
    # （VLF 夜间传播好）。所以它主要用来回答「模板该有多大对比度、N 要多少」，
    # 拿它去拟合 f_TGF 只作参考。
    z = np.load(ext_tmpl)
    h_ext = (z["weighted"] if hasattr(z, "files") and "weighted" in z.files
             else np.asarray(z)).astype(float)
    files = getattr(z, "files", [])
    p_ext = h_ext / h_ext.sum()
    a = 2 * np.abs((p_ext * np.exp(2j * np.pi * (np.arange(NB) + 0.5) * 3 / 24)).sum())
    print("\n== 外部模板（WWLLN 落点，|lat| ≤ 30°）==")
    print("  " + "  ".join("%02d-%02d %.1f%%" % (3 * i, 3 * i + 3, 100 * p_ext[i])
                           for i in range(NB)))
    print("  峰/谷 = %.2f，等效正弦幅度 A ≈ %.2f" % (p_ext.max() / p_ext.min(), a))
    print("  若真 TGF 按这个分布，瑞利检验的期望 p：" + "，".join(
        "N=%d %.3f" % (n_, np.exp(-n_ * (a / 2) ** 2)) for n_ in (83, 200, 611, 2547)))
    # 似然比检验比瑞利更有力：它用上了模板的形状
    ll_t = float((h_conf * np.log(p_ext)).sum())
    ll_u = float((h_conf * np.log(np.ones(NB) / NB)).sum())
    print("  83 个证实 TGF：lnL(雷暴模板) − lnL(均匀) = %+.2f"
          "（正说明更像雷暴模板；|Δ| ≳ 2 才算有话说）" % (ll_t - ll_u))
    print("\n== 用外部模板重做两成分拟合（参考值，带上述偏差）==")
    for tag, sel in tests:
        sub = [r for r in pool if sel(r)]
        if len(sub) < 20:
            continue
        h = hist(lst(sub))
        f, lo, hi, d0 = fit_frac(h, h_ext, h_bkg)
        print("%-34s %6d %8.2f   [%.2f, %.2f] %10.1f" % (tag, len(sub), f, lo, hi, d0))

    if "land" not in files:
        return
    # 陆海分开：陆地雷暴峰在午后、海洋峰在凌晨，混在一起互相抵消。
    # 分开后陆地模板的幅度是混合模板的 2.5 倍，杠杆大得多。
    from global_land_mask import globe
    t_land = z["land"].astype(float)
    t_ocn = z["ocean"].astype(float)
    isl = {}
    for r in pool:
        isl[id(r)] = bool(globe.is_land(float(r["lat"]), float(r["lon"])))
    print("\n== 陆海分开（候选按星下点判；模板用 WWLLN 各自那一半）==")
    for nm, tm in (("陆地", t_land), ("海洋", t_ocn)):
        b = hist(lst([r for r in pool if float(r["fa"]) > 1 and isl[id(r)] == (nm == "陆地")]))
        e = b.sum() / NB
        print("  %s本底（留出 fa>1，%d 个）：" % (nm, b.sum())
              + " ".join("%.1f%%" % (100 * v / b.sum()) for v in b)
              + "  与均匀比 χ² = %.1f/%d" % (float(((b - e) ** 2 / e).sum()), NB - 1))
    print("%-34s %6s %8s %16s %10s" % ("被检验的样本", "N", "f_TGF", "68% 区间", "ΔlnL(f=0)"))
    for tag, sel in tests:
        sub = [r for r in pool if sel(r)]
        if len(sub) < 20:
            continue
        parts = []
        for nm, tm in (("陆地", t_land), ("海洋", t_ocn)):
            k = [r for r in sub if isl[id(r)] == (nm == "陆地")]
            b = hist(lst([r for r in pool if float(r["fa"]) > 1
                          and isl[id(r)] == (nm == "陆地")]))
            parts.append((hist(lst(k)), tm, b, len(k)))
        f, lo, hi, d0 = fit_joint(parts)
        print("%-34s %6d %8.2f   [%.2f, %.2f] %10.1f"
              % (tag + "（陆%d/海%d）" % (parts[0][3], parts[1][3]), len(sub),
                 f, lo, hi, d0))


def fit_joint(parts):
    """陆、海两块共用一个 f_TGF，各自用自己的模板与本底。"""
    def nll(f):
        s = 0.0
        for obs, tmpl, bkg, _ in parts:
            n = obs.sum()
            if n == 0:
                continue
            m = np.maximum(n * (f * tmpl / tmpl.sum() + (1 - f) * bkg / bkg.sum()), 1e-9)
            s += float((m - obs * np.log(m)).sum())
        return s

    from scipy.optimize import minimize_scalar as _ms
    r = _ms(nll, bounds=(-0.5, 2.0), method="bounded", options=dict(xatol=1e-4))
    f0, c = float(r.x), float(r.fun)
    lo, hi = f0, f0
    while lo > -0.5 and nll(lo) - c <= 0.5:
        lo -= 0.005
    while hi < 2.0 and nll(hi) - c <= 0.5:
        hi += 0.005
    return f0, max(lo, -0.5), min(hi, 2.0), nll(0.0) - c


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
