"""近岸/远洋两类到底是模板问题、本底问题，还是灵敏度不够。

第 32b 条留下的未决项：SVOM 的陆地类 f = 0.84 [0.75, 0.93] 站得住，近岸/远洋
两类不可用——纯本底档（fa 0.1–1）用 GBM 远洋模板拟出 f = −0.23 [−0.34, −0.11]，
一个纯本底样本拟出显著负值。当时记成「模板搬不过来」，但那不是唯一的解释：

* **模板问题**：GBM 25.6° 倾角、SVOM 30°，海上日变化弱且地域性强，形状搬不过来。
* **本底问题**（HXMT 组提的）：LST = UT + 经度/15，本底在「它自己的经度分布下」
  平不等于在「被检验样本的经度分布下」也平。这一条换模板换不掉，两套模板会给出
  同样的负号，会被误判成「两套都搬不过来」。
* **灵敏度问题**：海上模板的幅度 A 只有陆地的四分之一，f 的误差按 1/A 放大，
  同样的本底噪声在海上类翻译成大得多的 f。

三者要分开，靠的是三组只变一个量的对照：

1. **只变模板来源**：GBM 单点口径 + GBM 模板 vs 同口径 + HXMT 单点模板
   （HXMT 组按 GBM 口径重分给出）。同一批候选、同一份本底、同一个分类规则。
2. **只变分类口径**：HXMT 单点模板 vs HXMT 九点模板，同一批 HXMT 候选的两种切法。
3. **只变本底模型**：本底原样 vs 按被检验样本的经度分区重加权。

再加一条零假设分布：从留出本底里抽**同样大小**的子样本去拟剩下的本底（真值恒为 0），
给出「本底自己能造出多大的 f」。观测值落在这个分布里，就不是任何一种「问题」，
只是这一类的灵敏度不够。

用法: python3 diag_svom_lst_template_crosscheck.py <pool.csv>
"""
import csv
import datetime as dt
import sys

import numpy as np
from global_land_mask import globe
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as chi2_dist

NB = 8
RE = 6378.137
# 经度三分区，用来做本底的经度配平（LST 与经度直接耦合）
SECTORS = [("美洲", -150.0, -30.0), ("非洲欧洲", -30.0, 60.0), ("亚洲海洋大陆", 60.0, 210.0)]

# ---- 模板 -------------------------------------------------------------------
# GBM：盲搜 gamma 目录认证的 TGF，零 WWLLN 选样。星下点**单点**判陆；
# 海上离陆 < 300 km 算近岸、> 300 km 算远洋。
GBM = {
    "陆地": np.array([0.1089, 0.1018, 0.0406, 0.0330, 0.1067, 0.2801, 0.2031, 0.1257]),
    "近岸": np.array([0.1516, 0.1638, 0.1125, 0.0648, 0.0990, 0.1510, 0.1375, 0.1198]),
    "远洋": np.array([0.1556, 0.1787, 0.1525, 0.0817, 0.0847, 0.1125, 0.1248, 0.1094]),
}
GBM_N = {"陆地": 1846, "近岸": 1636, "远洋": 649}

# HXMT 显著未关联（3467 个，不经台网），**按 GBM 单点口径重分**——与 GBM 那三条
# 口径完全一致，所以「GBM 模板 vs HXMT 单点模板」只差模板来源这一个量。
# 原始计数由 HXMT 组现算给出（scripts/cluster/hxmt_lst_gbm_compare.py）。
HXMT_PT = {
    "陆地": np.array([172.0, 135, 92, 58, 179, 383, 335, 207]),
    "近岸": np.array([144.0, 156, 84, 66, 104, 168, 135, 165]),
    "远洋": np.array([142.0, 138, 99, 74, 94, 95, 125, 117]),
}
HXMT_PT_N = {"陆地": 1561, "近岸": 1022, "远洋": 884}
# 同口径本底（fa > 1，1/12 抽样，误差按实际计数 118103 算，不按 ×12）
HXMT_PT_BKG = {
    "陆地": np.array([3516.0, 3475, 3652, 3723, 3771, 3701, 3641, 3591]),
    "近岸": np.array([1759.0, 1825, 1802, 1843, 1969, 1763, 1846, 1776]),
    "远洋": np.array([9135.0, 9253, 9334, 9665, 9454, 9413, 9094, 9102]),
}

# HXMT 同一批 3467 个候选的另一种切法：星下点 + 八方位 600 km **九点**，
# 全陆算陆核、全海算海核，混的算近岸。与上面那套不是独立样本。
HXMT_9P = {
    "陆核": np.array([0.0706, 0.0573, 0.0534, 0.0324, 0.1679, 0.2939, 0.2252, 0.0992]),
    "近岸": np.array([0.1395, 0.1310, 0.0778, 0.0496, 0.0975, 0.1873, 0.1667, 0.1506]),
    "海核": np.array([0.1544, 0.1501, 0.1034, 0.0992, 0.1006, 0.1034, 0.1473, 0.1416]),
}
HXMT_9P_N = {"陆核": 524, "近岸": 2237, "海核": 706}
HXMT_9P_BKG_A = {"陆核": 0.056, "近岸": 0.035, "海核": 0.034}

PT_NAMES = ["陆地", "近岸", "远洋"]
NP_NAMES = ["陆核", "近岸", "海核"]


def amp(p):
    """统一刻度的幅度：(max − min) / 2 / 均值，8 格归一化时等于 (max − min) × 4。"""
    p = np.asarray(p, float)
    if p.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    return (p.max() - p.min()) * NB / 2.0


def tmpl_peak(p):
    """模板自己的瑞利相位（按格心加权），比取 argmax 的格心稳。"""
    p = np.asarray(p, float)
    c = np.arange(NB) * 24.0 / NB + 12.0 / NB
    z = (p / p.sum() * np.exp(2j * np.pi * c / 24.0)).sum()
    return (np.angle(z) / (2 * np.pi) * 24.0) % 24.0


def eot_hours(doy):
    b = 2 * np.pi * (doy - 81) / 364.0
    return (9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)) / 60.0


def ring(lat, lon, d_km, n_az):
    """以 (lat, lon) 为心、半径 d_km 的大圆环上取 n_az 个方位点。"""
    ang = d_km / RE
    az = np.radians(np.arange(0.0, 360.0, 360.0 / n_az))
    p, l = np.radians(lat), np.radians(lon)
    p2 = np.arcsin(np.sin(p) * np.cos(ang) + np.cos(p) * np.sin(ang) * np.cos(az))
    l2 = l + np.arctan2(np.sin(az) * np.sin(ang) * np.cos(p),
                        np.cos(ang) - np.sin(p) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def surf_point(lat, lon):
    """GBM 口径：0 = 陆地（星下点单点），1 = 近岸（海上离陆 < 300 km），2 = 远洋。"""
    if globe.is_land(lat, lon):
        return 0
    for d in (100.0, 200.0, 300.0):
        la, lo = ring(lat, lon, d, 16)
        if bool(np.any(globe.is_land(la, lo))):
            return 1
    return 2


def surf_nine(lat, lon):
    """HXMT 口径：星下点 + 八方位 600 km 九点。全陆 = 0，全海 = 2，混 = 1。"""
    la, lo = ring(lat, lon, 600.0, 8)
    pts = np.concatenate(([bool(globe.is_land(lat, lon))],
                          np.asarray(globe.is_land(la, lo), bool)))
    if pts.all():
        return 0
    if not pts.any():
        return 2
    return 1


def sector(lon):
    x = (lon + 360.0) % 360.0
    for k, (_, lo, hi) in enumerate(SECTORS):
        a, b = (lo + 360.0) % 360.0, (hi + 360.0) % 360.0
        if (a < b and a <= x < b) or (a > b and (x >= a or x < b)):
            return k
    return 2


def rayleigh(t):
    t = np.asarray(t, float)
    z = np.exp(2j * np.pi * t / 24.0).mean()
    return abs(z), (np.angle(z) / (2 * np.pi) * 24.0) % 24.0, float(np.exp(-len(t) * abs(z) ** 2))


def hist_of(sub):
    h, _ = np.histogram([r["lst"] for r in sub], bins=NB, range=(0, 24))
    return h.astype(float)


def matched_bkg(bkg, sub):
    """把本底按被检验样本的经度分区构成重新加权（0c 对照）。"""
    w = np.zeros(len(SECTORS))
    for r in sub:
        w[r["sec"]] += 1
    out = np.zeros(NB)
    for c in range(len(SECTORS)):
        if w[c] == 0:
            continue
        h = hist_of([r for r in bkg if r["sec"] == c])
        if h.sum() == 0:
            continue
        out += h / h.sum() * w[c]
    return out if out.sum() > 0 else hist_of(bkg)


def fit(obs, tmpl, bkg, band=False):
    """两成分泊松似然：obs ~ n · [f·模板 + (1−f)·本底]，f 是唯一的自由参数。"""
    n = obs.sum()
    t, b = tmpl / tmpl.sum(), bkg / bkg.sum()

    def nll(f):
        m = np.maximum(n * (f * t + (1 - f) * b), 1e-9)
        return float((m - obs * np.log(m)).sum())

    r = minimize_scalar(nll, bounds=(-1.5, 2.5), method="bounded", options=dict(xatol=1e-4))
    f0 = float(r.x)
    if not band:
        return f0
    c = nll(f0)
    lo, hi = f0, f0
    while lo > -1.5 and nll(lo) - c <= 0.5:
        lo -= 0.005
    while hi < 2.5 and nll(hi) - c <= 0.5:
        hi += 0.005
    return f0, lo, hi, nll(0.0) - c


def chi2_shape(a, b):
    """两个直方图的形状是否一致（把 b 归一到 a 的总数后比）。"""
    if a.sum() == 0 or b.sum() == 0:
        return float("nan"), NB - 1, float("nan")
    e = b / b.sum() * a.sum()
    c = float(((a - e) ** 2 / np.maximum(e, 1e-9)).sum())
    return c, NB - 1, float(chi2_dist.sf(c, NB - 1))


def load(pool):
    rows = [r for r in csv.DictReader(open(pool))
            if abs(float(r["lon"])) <= 180 and r["is_train"] == "0"]
    cache = {}
    for r in rows:
        r["fa"] = float(r["fa"])
        lat, lon = float(r["lat"]), float(r["lon"])
        key = (round(lat, 1), round(lon, 1))
        if key not in cache:
            cache[key] = (surf_point(lat, lon), surf_nine(lat, lon))
        r["pt"], r["np"] = cache[key]
        r["sec"] = sector(lon)
        r["day"] = r["start"][:10]
        t = dt.datetime.strptime(r["start"][:19], "%Y-%m-%dT%H:%M:%S")
        r["lst"] = ((t.hour + t.minute / 60.0 + t.second / 3600.0 + lon / 15.0
                     + eot_hours(t.timetuple().tm_yday)) % 24.0)
    return rows


TESTS = [("显著 fa ≤ 1e-5", lambda r: r["fa"] <= 1e-5),
         ("显著、覆盖外", lambda r: r["fa"] <= 1e-5 and r["in_cov"] == "0"),
         ("fa 1e-3 .. 0.1", lambda r: 1e-3 < r["fa"] <= 0.1),
         ("fa 0.1 .. 1（纯本底）", lambda r: 0.1 < r["fa"] <= 1.0)]


def section_templates():
    print("=" * 92)
    print("三套模板并排：A 与瑞利峰位")
    print("=" * 92)
    print("%-22s %6s %8s %8s   %s" % ("模板", "N", "A", "峰 (h)", "八格归一化"))
    for tag, tm, tn in (("GBM 单点", GBM, GBM_N), ("HXMT 单点", HXMT_PT, HXMT_PT_N),
                        ("HXMT 九点", HXMT_9P, HXMT_9P_N)):
        for nm, v in tm.items():
            print("%-22s %6d %8.3f %8.1f   %s"
                  % ("%s %s" % (tag, nm), tn[nm], amp(v), tmpl_peak(v),
                     " ".join("%.3f" % x for x in v / v.sum())))
    print("\n只变模板来源（口径都是单点）：")
    for nm in PT_NAMES:
        g, h = GBM[nm] / GBM[nm].sum(), HXMT_PT[nm] / HXMT_PT[nm].sum()
        c, dof, p = chi2_shape(HXMT_PT[nm], GBM[nm])
        print("  %-4s  A: GBM %.3f vs HXMT %.3f（比 %.2f）  峰: %.1f vs %.1f h（差 %.1f h）"
              "  形状 χ² = %.1f/%d (p = %.3f)"
              % (nm, amp(g), amp(h), amp(h) / amp(g), tmpl_peak(g), tmpl_peak(h),
                 ((tmpl_peak(h) - tmpl_peak(g) + 36) % 24) - 12, c, dof, p))
    print("  注：HXMT 那三条来自显著未关联样本、本身掺本底，A 是**下限**（HXMT 组自报）。")
    print("      所以两套模板不一致时，不能直接判成「GBM 搬不过来」——模板纯度不是受控变量。")


def section_background(rows):
    print("\n" + "=" * 92)
    print("本底诊断：类内平坦性、经度配平、本底对本底")
    print("=" * 92)
    bkg = [r for r in rows if r["fa"] > 1]
    ctl = [r for r in rows if 0.1 < r["fa"] <= 1.0]
    for tag, key, names, ref in (("单点", "pt", PT_NAMES, None),
                                 ("九点", "np", NP_NAMES, HXMT_9P_BKG_A)):
        print("\n-- %s口径 --" % tag)
        for s, nm in enumerate(names):
            b = [r for r in bkg if r[key] == s]
            c = [r for r in ctl if r[key] == s]
            hb, hc = hist_of(b), hist_of(c)
            hm = matched_bkg(b, c)
            e = hb.sum() / NB
            chi_u = float(((hb - e) ** 2 / max(e, 1e-9)).sum())
            x1 = chi2_shape(hc, hb)
            x2 = chi2_shape(hc, hm)
            extra = ("  HXMT 同类本底 A = %.3f" % ref[nm]) if ref else ""
            print("  %-4s 本底 N=%6d  A = %.3f  与均匀 χ² = %5.1f/%d (p=%.2f)%s"
                  % (nm, len(b), amp(hb), chi_u, NB - 1,
                     chi2_dist.sf(chi_u, NB - 1), extra))
            print("       本底对本底（fa 0.1–1 N=%4d）：原样 χ² = %.1f/%d (p=%.2f)   "
                  "经度配平后 χ² = %.1f/%d (p=%.2f)"
                  % (len(c), x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]))


def section_fits(rows):
    print("\n" + "=" * 92)
    print("主拟合：每套模板配它自己的口径；本底一律按被检验样本的经度分区配平")
    print("=" * 92)
    bkg = [r for r in rows if r["fa"] > 1]
    chains = [("GBM 单点", "pt", PT_NAMES, GBM), ("HXMT 单点", "pt", PT_NAMES, HXMT_PT),
              ("HXMT 九点", "np", NP_NAMES, HXMT_9P)]
    for label, sel in TESTS:
        sub_all = [r for r in rows if sel(r)]
        if len(sub_all) < 30:
            continue
        print("\n== %s（N = %d）==" % (label, len(sub_all)))
        print("  %-14s %-6s %6s %22s %8s  %s"
              % ("模板", "类", "N", "f [68%]", "ΔlnL", "自身峰 / 模板峰"))
        for tag, key, names, tmpl in chains:
            for s, nm in enumerate(names):
                k = [r for r in sub_all if r[key] == s]
                b = [r for r in bkg if r[key] == s]
                if len(k) < 15 or not b:
                    continue
                hb = matched_bkg(b, k)
                f, lo, hi, d0 = fit(hist_of(k), tmpl[nm], hb, band=True)
                R, ph, p = rayleigh([r["lst"] for r in k])
                print("  %-14s %-6s %6d   %+.2f [%+.2f, %+.2f] %8.1f  %.1f h (p=%.0e) / %.1f h"
                      % (tag, nm, len(k), f, lo, hi, d0, ph, p, tmpl_peak(tmpl[nm])))


def section_null(rows, n_draw=1500, seed=20260911):
    """零假设分布：从留出本底里按天抽同样大小的子样本，拟剩下的本底。真值恒为 0。

    观测到的 f 若落在这个分布里，那它既不是模板问题也不是样本问题——
    只是这一类的灵敏度不够，本底自己就能造出那么大的 f。
    """
    print("\n" + "=" * 92)
    print("零假设分布：本底抽同样大小的子样本拟剩下的本底（真值 0），%d 次" % n_draw)
    print("=" * 92)
    rng = np.random.default_rng(seed)
    bkg = [r for r in rows if r["fa"] > 1]
    chains = [("GBM 单点", "pt", PT_NAMES, GBM), ("HXMT 单点", "pt", PT_NAMES, HXMT_PT),
              ("HXMT 九点", "np", NP_NAMES, HXMT_9P)]
    for label, sel, want in (("fa 0.1–1（纯本底）", lambda r: 0.1 < r["fa"] <= 1.0, True),
                             ("显著 fa ≤ 1e-5", lambda r: r["fa"] <= 1e-5, False)):
        print("\n-- 被检验样本：%s --" % label)
        print("  %-14s %-6s %6s %10s %10s %10s %10s"
              % ("模板", "类", "N", "观测 f", "零分布 σ", "|f|/σ", "双侧 p"))
        for tag, key, names, tmpl in chains:
            for s, nm in enumerate(names):
                cls = [r for r in bkg if r[key] == s]
                sub = [r for r in rows if sel(r) and r[key] == s]
                if len(sub) < 15 or len(cls) < 400:
                    continue
                f_obs = fit(hist_of(sub), tmpl[nm], matched_bkg(cls, sub))
                by_day = {}
                for r in cls:
                    by_day.setdefault(r["day"], []).append(r)
                keys = list(by_day)
                frac = len(sub) / float(len(cls))
                vals = []
                for _ in range(n_draw):
                    m = rng.random(len(keys)) < frac
                    a = [r for k, take in zip(keys, m) if take for r in by_day[k]]
                    rest = [r for k, take in zip(keys, m) if not take for r in by_day[k]]
                    if len(a) < 15 or len(rest) < 100:
                        continue
                    vals.append(fit(hist_of(a), tmpl[nm], matched_bkg(rest, a)))
                v = np.asarray(vals)
                sd = float(v.std())
                p = float(np.mean(np.abs(v - v.mean()) >= abs(f_obs - v.mean())))
                print("  %-14s %-6s %6d %10.2f %10.3f %10.1f %10.3f"
                      % (tag, nm, len(sub), f_obs, sd, abs(f_obs - v.mean()) / max(sd, 1e-9), p))
        if want:
            print("  （纯本底档观测 f 落在零分布里 ⇒ 没有任何「问题」，只是灵敏度不够。）")


def section_intersect(rows):
    """只变模板来源、只变口径的两组对照，都在同一批候选上做。"""
    print("\n" + "=" * 92)
    print("只变一个量的对照")
    print("=" * 92)
    bkg = [r for r in rows if r["fa"] > 1]
    print("\n[只变模板来源] 单点口径，同一批候选、同一份经度配平本底，只换模板：")
    for label, sel in TESTS:
        sub_all = [r for r in rows if sel(r)]
        line = []
        for s, nm in enumerate(PT_NAMES):
            k = [r for r in sub_all if r["pt"] == s]
            b = [r for r in bkg if r["pt"] == s]
            if len(k) < 15 or not b:
                continue
            hb, o = matched_bkg(b, k), hist_of(k)
            fg = fit(o, GBM[nm], hb, band=True)
            fh = fit(o, HXMT_PT[nm], hb, band=True)
            line.append("%s N=%d GBM %+.2f[%+.2f,%+.2f] / HXMT %+.2f[%+.2f,%+.2f]"
                        % (nm, len(k), fg[0], fg[1], fg[2], fh[0], fh[1], fh[2]))
        print("  %-22s %s" % (label, "\n                         ".join(line)))

    print("\n[只变口径] HXMT 的同一批候选切两次，各配自己那套 SVOM 分类：")
    for label, sel in TESTS:
        sub_all = [r for r in rows if sel(r)]
        line = []
        for (key, names, tmpl, tag) in (("pt", PT_NAMES, HXMT_PT, "单点"),
                                        ("np", NP_NAMES, HXMT_9P, "九点")):
            for s, nm in enumerate(names):
                if s != 2:
                    continue
                k = [r for r in sub_all if r[key] == s]
                b = [r for r in bkg if r[key] == s]
                if len(k) < 15 or not b:
                    continue
                f = fit(hist_of(k), tmpl[nm], matched_bkg(b, k), band=True)
                line.append("%s %s N=%d f = %+.2f [%+.2f, %+.2f]"
                            % (tag, nm, len(k), f[0], f[1], f[2]))
        print("  %-22s %s" % (label, "  |  ".join(line)))


def main(pool):
    rows = load(pool)
    n_sig = sum(1 for r in rows if r["fa"] <= 1e-5)
    print("候选池 %d 个（已去掉 is_train），显著 fa ≤ 1e-5 共 %d 个。" % (len(rows), n_sig))
    print("两套口径分出来的类别构成（显著池）：")
    for tag, key, names in (("单点", "pt", PT_NAMES), ("九点 600 km", "np", NP_NAMES)):
        c = [sum(1 for r in rows if r["fa"] <= 1e-5 and r[key] == s) for s in range(3)]
        print("  %-12s " % tag + "  ".join("%s %d (%.0f%%)" % (names[s], c[s], 100.0 * c[s] / n_sig)
                                           for s in range(3)))
    print("混淆表（显著池；行 = 单点，列 = 九点）：      陆核     近岸     海核")
    for s in range(3):
        row = [sum(1 for r in rows if r["fa"] <= 1e-5 and r["pt"] == s and r["np"] == t)
               for t in range(3)]
        print("                             %-8s %-8d %-8d %-8d" % (PT_NAMES[s], *row))
    section_templates()
    section_background(rows)
    section_fits(rows)
    section_intersect(rows)
    section_null(rows)


if __name__ == "__main__":
    main(sys.argv[1])
