"""按下垫面（陆地／近岸／远洋）分类拟合 f_TGF，模板用 GBM 的零偏差模板。

为什么要分类而不是套一个混合模板：陆地雷暴峰在午后、海洋峰在凌晨，相位差接近
12 h，混在一起互相抵消，混合模板的对比度只有陆地的三分之一。而**下垫面是每个
候选自己的星下点决定的，是已知协变量，不是待拟参数**——把它当自由参数拟会把
对比度的增益又吃回去（GBM 实测：两成分自由拟的误差带 0.425 比单模板的 0.374 还宽）。
所以逐候选分类、每类配同类模板、每类只留一个自由参数。

**注意：本脚本原先写着「三类的 f 应当相等，是一个免费的自洽检验」，那是错的，
已撤回**（2026-09-11，见 OPEN-QUESTIONS 第 32c 条）。纯度 = 该类里 TGF 率 /
（TGF 率 + 本底率），TGF 的产生密度随下垫面变化而本底不跟着同一张地图变，
所以 f 本来就应当在海上更低。下面那个「三类共用一个 f」的联合拟合因此是一个
**加了约束的模型**，不是自洽检验，**也不是池级纯度**（它按统计权重加权、被陆地类
主导）。池级纯度要按计数加权算，见 `diag_svom_lst_template_crosscheck.py`。

模板取 GBM 的：它来自盲搜 gamma 目录的认证 TGF，**选样与 WWLLN 无关**，
没有 VLF 夜间效率偏差，而且测的正是 TGF 人群自己的日变化。分类口径也整套用
GBM 的（星下点在陆上算陆地；在海上离陆 < 300 km 算近岸，否则远洋）。

还有一条不用拟合的检验：**陆地那一类候选自己的 LST 峰位**应当落在当地下午，
这是形状预言不是比例预言。

用法: python3 diag_svom_lst_by_surface.py <pool.csv> [wwlln_lst_landsea.npz]
"""
import csv
import datetime as dt
import sys

import numpy as np
from global_land_mask import globe
from scipy.optimize import minimize_scalar

NB = 8
RE = 6378.137
COAST_KM = 300.0

# GBM 的模板（8 个 3 h 格，格心 1.5, 4.5, … 22.5 h）。零 WWLLN 选样。
GBM = {
    "陆地": np.array([0.1089, 0.1018, 0.0406, 0.0330, 0.1067, 0.2801, 0.2031, 0.1257]),
    "近岸": np.array([0.1516, 0.1638, 0.1125, 0.0648, 0.0990, 0.1510, 0.1375, 0.1198]),
    "远洋": np.array([0.1556, 0.1787, 0.1525, 0.0817, 0.0847, 0.1125, 0.1248, 0.1094]),
    "全体": np.array([0.1331, 0.1385, 0.0867, 0.0533, 0.1002, 0.2026, 0.1649, 0.1208]),
}


def amp(p):
    """统一刻度的幅度：(max − min) / 2 / 均值，8 格归一化时等于 (max − min) × 4。"""
    p = p / p.sum()
    return (p.max() - p.min()) * NB / 2.0


def eot_hours(doy):
    b = 2 * np.pi * (doy - 81) / 364.0
    return (9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)) / 60.0


def surface(lat, lon):
    """0 = 陆地，1 = 近岸（海上离陆 < 300 km），2 = 远洋。"""
    if globe.is_land(lat, lon):
        return 0
    for d in (100.0, 200.0, 300.0):
        ang = d / RE
        for az in np.arange(0, 360, 22.5):
            a = np.radians(az)
            p, l = np.radians(lat), np.radians(lon)
            p2 = np.arcsin(np.sin(p) * np.cos(ang) + np.cos(p) * np.sin(ang) * np.cos(a))
            l2 = l + np.arctan2(np.sin(a) * np.sin(ang) * np.cos(p),
                                np.cos(ang) - np.sin(p) * np.sin(p2))
            if globe.is_land(np.degrees(p2), (np.degrees(l2) + 540) % 360 - 180):
                return 1
    return 2


def rayleigh(t):
    z = np.exp(2j * np.pi * np.asarray(t) / 24.0).mean()
    return abs(z), (np.angle(z) / (2 * np.pi) * 24.0) % 24.0, np.exp(-len(t) * abs(z) ** 2)


def fit(obs, tmpl, bkg):
    n = obs.sum()
    t, b = tmpl / tmpl.sum(), bkg / bkg.sum()

    def nll(f):
        m = np.maximum(n * (f * t + (1 - f) * b), 1e-9)
        return float((m - obs * np.log(m)).sum())

    r = minimize_scalar(nll, bounds=(-0.5, 2.5), method="bounded", options=dict(xatol=1e-4))
    f0, c = float(r.x), float(r.fun)
    lo, hi = f0, f0
    while lo > -0.5 and nll(lo) - c <= 0.5:
        lo -= 0.005
    while hi < 2.5 and nll(hi) - c <= 0.5:
        hi += 0.005
    return f0, lo, hi, nll(0.0) - c


def main(pool, wwlln_npz=None):
    rows = [r for r in csv.DictReader(open(pool))
            if abs(float(r["lon"])) <= 180 and r["is_train"] == "0"]
    cache = {}
    for r in rows:
        r["fa"] = float(r["fa"])
        lat, lon = float(r["lat"]), float(r["lon"])
        key = (round(lat, 1), round(lon, 1))
        if key not in cache:
            cache[key] = surface(lat, lon)
        r["surf"] = cache[key]
        t = dt.datetime.strptime(r["start"][:19], "%Y-%m-%dT%H:%M:%S")
        r["lst"] = ((t.hour + t.minute / 60.0 + t.second / 3600.0 + lon / 15.0
                     + eot_hours(t.timetuple().tm_yday)) % 24.0)

    def hist(sub):
        h, _ = np.histogram([r["lst"] for r in sub], bins=NB, range=(0, 24))
        return h.astype(float)

    names = ["陆地", "近岸", "远洋"]
    print("GBM 模板的统一刻度幅度 A = (max−min)/2/均值："
          + "，".join("%s %.3f" % (k, amp(v)) for k, v in GBM.items()))
    bkg = [r for r in rows if r["fa"] > 1]
    print("留出本底 fa>1 共 %d：" % len(bkg)
          + "，".join("%s %d" % (names[s], sum(1 for r in bkg if r["surf"] == s))
                      for s in (0, 1, 2)))
    for s in (0, 1, 2):
        b = hist([r for r in bkg if r["surf"] == s])
        e = b.sum() / NB
        print("  %s本底 " % names[s] + " ".join("%.1f%%" % (100 * v / b.sum()) for v in b)
              + "  与均匀比 χ² = %.1f/%d" % (float(((b - e) ** 2 / e).sum()), NB - 1))

    tests = [("显著 fa ≤ 1e-5", lambda r: r["fa"] <= 1e-5),
             ("目录里非证实的那部分", lambda r: r["fa"] <= 1e-5
              and not (r["in_cov"] == "1" and r["assoc"] == "1")),
             ("显著、覆盖外", lambda r: r["fa"] <= 1e-5 and r["in_cov"] == "0"),
             ("fa 1e-3 .. 0.1（对照）", lambda r: 1e-3 < r["fa"] <= 0.1),
             ("fa 0.1 .. 1（对照）", lambda r: 0.1 < r["fa"] <= 1.0)]
    for tag, sel in tests:
        sub = [r for r in rows if sel(r)]
        if len(sub) < 30:
            continue
        print("\n== %s（N = %d）==" % (tag, len(sub)))
        for s in (0, 1, 2):
            k = [r for r in sub if r["surf"] == s]
            b = hist([r for r in bkg if r["surf"] == s])
            if len(k) < 15 or b.sum() == 0:
                print("  %s N=%d 太少" % (names[s], len(k)))
                continue
            f, lo, hi, d0 = fit(hist(k), GBM[names[s]], b)
            print("  %-4s N=%4d  f_TGF = %.2f [%.2f, %.2f]  ΔlnL(f=0) = %.1f"
                  % (names[s], len(k), f, lo, hi, d0))
        # 一类一类拟完，再给一个共用 f 的联合拟合（三类同一个纯度）
        parts = []
        for s in (0, 1, 2):
            k = [r for r in sub if r["surf"] == s]
            b = hist([r for r in bkg if r["surf"] == s])
            if len(k) >= 15 and b.sum() > 0:
                parts.append((hist(k), GBM[names[s]], b))

        def nll(f):
            t = 0.0
            for o, tm, b in parts:
                n = o.sum()
                m = np.maximum(n * (f * tm / tm.sum() + (1 - f) * b / b.sum()), 1e-9)
                t += float((m - o * np.log(m)).sum())
            return t

        r0 = minimize_scalar(nll, bounds=(-0.5, 2.5), method="bounded",
                             options=dict(xatol=1e-4))
        f0, c = float(r0.x), float(r0.fun)
        lo, hi = f0, f0
        while lo > -0.5 and nll(lo) - c <= 0.5:
            lo -= 0.005
        while hi < 2.5 and nll(hi) - c <= 0.5:
            hi += 0.005
        print("  三类共用一个 f：%.2f [%.2f, %.2f]，ΔlnL(f=0) = %.1f"
              % (f0, lo, hi, nll(0.0) - c))

    print("\n== 不用拟合的形状检验：各类候选自己的 LST 峰位 ==")
    print("  （GBM 模板的峰：陆地 18.9 h、近岸 23.5 h、远洋 3.2 h）")
    for tag, sel in tests[:3]:
        sub = [r for r in rows if sel(r)]
        line = []
        for s in (0, 1, 2):
            k = [r["lst"] for r in sub if r["surf"] == s]
            if len(k) < 15:
                continue
            R, ph, p = rayleigh(k)
            line.append("%s N=%d 峰 %.1f h R=%.3f p=%.1e" % (names[s], len(k), ph, R, p))
        print("  %-22s " % tag + " | ".join(line))

    if wwlln_npz is None:
        return
    # WWLLN 落点模板 ÷ GBM 模板 = 台网昼夜效率的**上限**：差值里也可能含
    # 「产生 TGF 的闪电 ≠ 全部闪电」的成分，两者分不开。
    z = np.load(wwlln_npz)
    print("\n== WWLLN 落点模板 ÷ GBM 模板（台网昼夜效率的上限估计）==")
    for nm, key in (("陆地", "land"),):
        w = z[key].astype(float)
        w = w / w.sum()
        g = GBM[nm] / GBM[nm].sum()
        r = w / g
        print("  %s：" % nm + " ".join("%.2f" % v for v in r / r.mean())
              + "  （夜/昼 = %.2f）" % (r[:2].mean() / r[5:7].mean()))
        print("  WWLLN %s A = %.3f，GBM %s A = %.3f" % (nm, amp(w), nm, amp(g)))
    print("  限制：这个比值里混着「全部闪电 ≠ 产生 TGF 的闪电」，两者分不开，"
          "所以只能当效率日变化的上限。")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
