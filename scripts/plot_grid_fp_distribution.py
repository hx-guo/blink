"""天格候选的 fp 分布（每年误报次数的直方图），照慧眼 fp-distribution 的做法画。

慧眼上的读法：纯本底候选堆在弱显著一侧；真 TGF 在显著一侧是一条独立的幂律，
闪电关联曲线与全候选显著侧同斜率（b ≈ 0.036 / 0.039）即说明显著侧已是纯 TGF。
这里对天格做同样的事，并把两类慧眼上没有的东西拆开：
  - 候选窗本底 > 5000 计数/秒（mean / 窗长，搜索自己报的口径）
  - WWLLN 一步标出的成串候选（train.is_train，阈 34 是慧眼上定的）

分箱与拟合区间照 plot_fp_distribution_v4_paperfmt.py：200 个对数箱，从全体最小值到 20；
弱显著侧 1e-4..20，显著侧 1e-50..1e-8（这里样本少，另报每侧参与拟合的箱数）。

用法:
    python3 scripts/plot_grid_fp_distribution.py <tgfs_grid*.json ...> -o <png> [--slide <png>]
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cjk_font import FAMILIES as CJK_FAMILIES
from scipy.optimize import curve_fit

plt.rcParams.update({"font.sans-serif": CJK_FAMILIES, "font.family": "sans-serif",
                     "axes.unicode_minus": False, "font.size": 13})
HIGH_RATE = 5000.0


def power_law(x, a, b):
    return a * x ** b


def load(paths):
    rows = []
    for p in paths:
        for r in json.load(open(p)):
            s = r["signal"]
            rows.append(dict(sat=s["instrument"], start=s["start"], fpy=s["false_positive_per_year"],
                             rate=s["mean"] / s["bin_size_best"],
                             assoc=bool(r["lightning"].get("associated")),
                             coinc=float(r["lightning"].get("coincidence_probability") or 0.0),
                             train=bool(r.get("train", {}).get("is_train")),
                             day=s["start"][:10]))
    return rows


def fit(centers, n, lo, hi, p0=None):
    sel = (centers > lo) & (centers < hi) & (n > 0)
    if sel.sum() < 3:
        return None, int(sel.sum())
    try:
        p, _ = curve_fit(power_law, centers[sel], n[sel], p0=p0 or (1.0, 0.0), maxfev=20000)
    except RuntimeError:
        return None, int(sel.sum())
    return p, int(sel.sum())


def panel(ax, rows, bins, title):
    centers = (bins[:-1] + bins[1:]) / 2
    fpy = np.array([r["fpy"] for r in rows])
    hi = np.array([r["rate"] > HIGH_RATE for r in rows])
    tr = np.array([r["train"] for r in rows])
    asc = np.array([r["assoc"] for r in rows])
    coinc = np.array([r["coinc"] for r in rows])
    clip = lambda x: np.clip(x, bins[0], None)
    n_all, _, _ = ax.hist(clip(fpy), bins=bins, histtype="step", color="0.6", lw=1.0, label="全部候选 %d" % len(fpy))
    clean = ~hi & ~tr
    n_clean, _, _ = ax.hist(clip(fpy[clean]), bins=bins, histtype="step", color="C0", lw=1.3,
                            label="去掉高本底与成串 %d" % clean.sum())
    ax.hist(clip(fpy[hi]), bins=bins, histtype="step", color="C3", lw=1.0, label="本底 > 5000/秒 %d" % hi.sum())
    ax.hist(clip(fpy[tr & ~hi]), bins=bins, histtype="step", color="C4", lw=1.0, label="成串（本底不高）%d" % (tr & ~hi).sum())
    n_asc, _, _ = ax.hist(clip(fpy[asc]), bins=bins, histtype="step", color="C2", lw=1.3, label="有闪电对应 %d" % asc.sum())
    mis = np.zeros(len(centers))
    for f, c in zip(fpy, coinc):
        i = np.digitize(max(f, bins[0]), bins) - 1
        if 0 <= i < len(mis):
            mis[i] += c
    ax.stairs(mis, bins, color="C1", lw=1.0, label="期望误关联 %.1f" % coinc.sum())
    out = {}
    for name, n, lo, hi_ in (("clean_weak", n_clean, 1e-4, 20), ("clean_sig", n_clean, 1e-50, 1e-8),
                             ("assoc", n_asc, 1e-50, 1e-2)):
        p, k = fit(centers, n, lo, hi_)
        out[name] = (p, k)
        if p is not None:
            x = np.logspace(np.log10(max(lo, bins[0])), np.log10(hi_), 50)
            ax.plot(x, power_law(x, *p), ls="--", color="0.4", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(20, bins[0]); ax.set_ylim(0.5, None)
    ax.axvline(1e-5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("泊松假设下的每年误报次数"); ax.set_ylabel("每箱候选数")
    ax.set_title(title)
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tgfs", nargs="+"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--slide", help="另出一张讲稿用的简洁图")
    ap.add_argument("--classes", nargs=3, metavar=("FEATURES", "T90", "PNG"), help="按 A/B/中间/辐射带拆开的判据图")
    ap.add_argument("--hxmt", help="慧眼候选表 data/tgfs_v6.csv（day,fpy,assoc,...,is_train），给讲稿图加一个对照面板")
    a = ap.parse_args()
    rows = load(a.tgfs)
    if a.classes:
        fig_criterion(rows, a.classes[0], a.classes[1], a.classes[2])
    if a.slide:
        hx = None
        if a.hxmt:
            import pandas as pd
            d = pd.read_csv(a.hxmt)
            d = d[(d.day < 20250101) & (d.is_train == 0)]
            hx = {"fpy": d.fpy.values, "assoc": d.assoc.values.astype(bool)}
        slide_figure(rows, a.slide, hx)
    fmin = min(r["fpy"] for r in rows if r["fpy"] > 0)
    bins = np.logspace(np.log10(fmin), np.log10(20), 201)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), sharey=True)
    groups = [("GRID-03B（FPGA）", [r for r in rows if r["sat"] == "GRID-03B"]),
              ("GRID-02 / 04 / 07（MCU）", [r for r in rows if r["sat"] != "GRID-03B"])]
    for ax, (t, rs) in zip(axes, groups):
        out = panel(ax, rs, bins, t)
        print(t, len(rs))
        for k, (p, nb) in out.items():
            print("  %-10s bins=%3d  %s" % (k, nb, "a=%.3g b=%.4f" % tuple(p) if p is not None else "no fit"))
        fpy = np.array([r["fpy"] for r in rs]); hi = np.array([r["rate"] > HIGH_RATE for r in rs])
        tr = np.array([r["train"] for r in rs]); asc = np.array([r["assoc"] for r in rs])
        sig = fpy <= 1e-5
        print("  significant: all %d | high-rate %d | train(not hi) %d | clean %d | clean&assoc %d"
              % (sig.sum(), (sig & hi).sum(), (sig & tr & ~hi).sum(), (sig & ~hi & ~tr).sum(), (sig & ~hi & ~tr & asc).sum()))
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight"); print("wrote", a.out)




SIG_HI, SIG_LO = 1e-8, 1e-50  # 显著一侧的拟合区间，与慧眼 fp-distribution 相同


def mle_slope(x):
    """显著一侧幂律斜率的最大似然估计：每单位 ln x 的候选数 ∝ x^b，截断在 [SIG_LO, SIG_HI]。

    直方图每箱只有一两个计数时，按箱最小二乘拟合不稳定，所以改用逐候选的似然。
    """
    from scipy.optimize import brentq
    u = np.log(SIG_HI / np.clip(x, SIG_LO, None)); U = np.log(SIG_HI / SIG_LO)
    m = u.mean()
    return brentq(lambda b: 1 / b - U / np.expm1(b * U) - m, 1e-6, 5)


def _fp_panel(ax, fpy, hi, asc, title, bins, fmin, show_hi=True, xmin=None):
    """一个面板：直方线 + 噪声一侧（细箱拟合）+ 显著一侧（最大似然）两条幂律。"""
    n_beyond = int((fpy < bins[0]).sum()) if xmin is None else int((fpy < xmin).sum())
    fpy = np.clip(fpy, bins[0], None)
    dln = np.log(bins[1] / bins[0])
    ax.axvspan(1e-5, bins[0], color="#F3F5F8", zorder=0)
    lab = "本底不高" if show_hi else "全部候选"
    ax.hist(fpy[~hi], bins=bins, histtype="step", color="#1B3454", lw=2.2, label=lab)
    if show_hi and hi.sum() > 20:
        ax.hist(fpy[hi], bins=bins, histtype="step", color="#D62728", lw=1.8, label="本底 > 5000/秒")
    if asc.any():
        ax.hist(fpy[asc], bins=bins, histtype="stepfilled", color="#2CA02C", alpha=0.6, label="有闪电对应")
    # 噪声一侧照慧眼：200 个细箱（最小值到 20）上拟合，再把高度换算到这里的箱宽
    fine = np.logspace(np.log10(fmin), np.log10(20), 201)
    fc = (fine[:-1] + fine[1:]) / 2
    n_low, _ = np.histogram(fpy[~hi], bins=fine)
    pw, _ = fit(fc, n_low.astype(float), 1e-4, 20)
    if pw is not None:
        xw = np.logspace(np.log10(20), -6, 50)
        ax.plot(xw, dln / np.log(fine[1] / fine[0]) * power_law(xw, *pw), color="#E8833A", lw=2.2, ls="--",
                zorder=5, label="幂律 b = %.2f" % pw[1])
    xs = fpy[~hi]; xs = xs[(xs > SIG_LO) & (xs < SIG_HI)]
    if len(xs) >= 5:
        bb = mle_slope(xs); U = np.log(SIG_HI / SIG_LO)
        xr = np.logspace(-5, np.log10(xmin or bins[0]), 50)
        ax.plot(xr, len(xs) * bb * (xr / SIG_HI) ** bb / -np.expm1(-bb * U) * dln, color="#7B3FA0", lw=2.4,
                ls="--", zorder=5, label="幂律 b = %.3f" % bb)
    ax.axvline(1e-5, color="0.3", lw=1.2, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(20, xmin or bins[0]); ax.set_ylim(0.3, None)
    if xmin is not None and n_beyond:
        ax.text(0.98, 0.52, "另有 %d 个更显著，\n最远到 10$^{%d}$ →" % (n_beyond, int(np.floor(np.log10(fpy.min())))),
                transform=ax.transAxes, ha="right", va="top", fontsize=20, color="0.35")
    ax.set_xlabel("每年误报次数（越往右越显著）")
    ax.set_title(title, fontsize=30)
    ax.legend(frameon=False, loc="upper right", fontsize=21, handlelength=1.4, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def slide_figure(rows, out, hxmt=None):
    """讲稿用：慧眼（可选）、03B、MCU 三个面板，各自纵轴；每个数量级一箱。

    噪声一侧：200 个细箱上按箱最小二乘，区间 1e-4..20（与慧眼论文相同），画时换算到每数量级一箱。
    显著一侧：[1e-50, 1e-8] 内逐候选最大似然（每十年只有约一个候选，按箱拟合不稳）。
    慧眼：v6 目录 2025 年以前、去掉成串候选（is_train）后的全部候选。
    """
    plt.rcParams.update({"font.size": 26})
    # 细箱起点照慧眼论文取各自数据的最小值（天格两组共用天格的最小值）
    fmin_grid = min(r["fpy"] for r in rows if r["fpy"] > 0)
    fmin_all = fmin_grid if hxmt is None else min(fmin_grid, hxmt["fpy"][hxmt["fpy"] > 0].min())
    # 每箱一个数量级：显著一侧每十年只有约一个候选，箱再窄就全是 0 和 1
    bins = 10.0 ** np.arange(np.floor(np.log10(fmin_all)), 2.0)
    # 三栏共用横轴，截到天格的最显著处；慧眼更显著的部分只在图上注明
    xmin = 10.0 ** np.floor(np.log10(fmin_grid))
    panels = []
    if hxmt is not None:
        panels.append(("慧眼 HE（2017–2024）", hxmt["fpy"], np.zeros(len(hxmt["fpy"]), bool), hxmt["assoc"], False,
                       hxmt["fpy"][hxmt["fpy"] > 0].min()))
    for t, rs in (("天格 03B（FPGA）", [r for r in rows if r["sat"] == "GRID-03B"]),
                  ("天格 02 / 04 / 07（MCU）", [r for r in rows if r["sat"] != "GRID-03B"])):
        panels.append((t, np.array([r["fpy"] for r in rs]), np.array([r["rate"] > HIGH_RATE for r in rs]),
                       np.array([r["assoc"] for r in rs]), True, fmin_grid))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.6 * len(panels), 7.2))
    for ax, (t, f, h, a_, sh, fm) in zip(axes, panels):
        _fp_panel(ax, f, h, a_, t, bins, fm, sh, xmin)
    axes[0].set_ylabel("候选数")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)


def fig_classes(rows, features, t90, out):
    """讲稿用判据图：全部候选的每年误报分布，显著候选按 A / B / 中间 / 辐射带拆开，
    各画一条线并给 A、B 各拟一条显著侧幂律，看两群落在哪、怎么分开。

    类别口径与 plot_grid_slides.py 相同：本底 > 5000/秒（mean / 窗长）为辐射带；
    其余 T90 < 2 ms 且 |偶极磁纬| < 33° 为 A，T90 ≥ 2 ms 且 |磁纬| ≥ 33° 为 B，其余为中间。
    """
    import csv
    from plot_grid_talk import dipole_lat, T90_CUT_US, MLAT_CUT_DEG
    plt.rcParams.update({"font.size": 22})
    tt = {(r["sat"], r["start"][:23]): r for r in csv.DictReader(open(t90))}
    cls = {}
    for r in csv.DictReader(open(features)):
        k = (r["sat"], r["start"][:23])
        if float(r["mean"]) / (float(r["dur_us"]) * 1e-6) > HIGH_RATE:
            cls[k] = "hi"; continue
        t = float(tt[k]["t90_us"]); m = abs(dipole_lat(float(r["lat"]), float(r["lon"])))
        cls[k] = "A" if (t < T90_CUT_US and m < MLAT_CUT_DEG) else ("B" if (t >= T90_CUT_US and m >= MLAT_CUT_DEG) else "M")
    fpy = np.array([r["fpy"] for r in rows])
    key = [(r["sat"], r["start"][:23]) for r in rows]
    c = np.array([cls.get(k, "") for k in key])
    fmin = fpy[fpy > 0].min()
    bins = 10.0 ** np.arange(np.floor(np.log10(fmin)), 2.0)
    dln = np.log(10.0)
    fig, ax = plt.subplots(figsize=(17, 7.6))
    ax.axvspan(1e-5, bins[0], color="#F3F5F8", zorder=0)
    x = np.clip(fpy, bins[0], None)
    ax.hist(x, bins=bins, histtype="step", color="0.35", lw=2.0, label="全部候选 %d" % len(x))
    for code, col, lab in (("hi", "#E8833A", "辐射带（本底 > 5000/秒）"), ("B", "#2F6497", "B 群：约 3 毫秒、高磁纬"),
                           ("M", "0.6", "中间"), ("A", "#D62728", "A 群：短、低磁纬")):
        sel = c == code
        if sel.any():
            ax.hist(x[sel], bins=bins, histtype="stepfilled" if code in ("A", "B") else "step", alpha=0.55 if code in ("A", "B") else 1,
                    color=col, lw=2.2, label="%s %d" % (lab, sel.sum()))
    # 噪声一侧：全部候选，细箱拟合（与慧眼论文同口径）
    fine = np.logspace(np.log10(fmin), np.log10(20), 201); fc = (fine[:-1] + fine[1:]) / 2
    n_low, _ = np.histogram(fpy, bins=fine)
    pw, _ = fit(fc, n_low.astype(float), 1e-4, 20)
    if pw is not None:
        xw = np.logspace(np.log10(20), -7, 50)
        ax.plot(xw, dln / np.log(fine[1] / fine[0]) * power_law(xw, *pw), color="0.25", lw=2, ls=":", label="本底涨落 b = %.2f" % pw[1])
    slopes = {}
    for code, col in (("A", "#D62728"), ("B", "#2F6497")):
        xs = fpy[(c == code) & (fpy > SIG_LO) & (fpy < SIG_HI)]
        if len(xs) >= 5:
            bb = mle_slope(xs); slopes[code] = (bb, len(xs)); U = np.log(SIG_HI / SIG_LO)
            lo = max(fpy[c == code].min(), bins[0])
            xr = np.logspace(-5, np.log10(lo), 40)
            ax.plot(xr, len(xs) * bb * (xr / SIG_HI) ** bb / -np.expm1(-bb * U) * dln, color=col, lw=2.6, ls="--",
                    label="%s 群幂律 b = %.3f" % (code, bb))
    ax.axvline(1e-5, color="0.3", lw=1.3, ls="--")
    ax.text(1e-5, ax.get_ylim()[1] if False else 3e4, "  显著\n  10$^{-5}$", va="top", fontsize=19, color="0.3")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(20, bins[0]); ax.set_ylim(0.5, 5e4)
    ax.set_xlabel("每年误报次数（越往右越显著）"); ax.set_ylabel("候选数")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=17, ncol=2)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    ext = {k: float(fpy[c == k].min()) for k in ("A", "B", "M", "hi") if (c == k).any()}
    print("wrote", out, "noise b", None if pw is None else round(pw[1], 3), "slopes", slopes, "most significant", ext)


def fig_criterion(rows, features, t90, out):
    """讲稿用判据图，两栏：
    左：全部候选，辐射带（本底 > 5000/秒）单独画；其余候选的噪声一侧幂律与 10⁻⁵ 拐折。
    右：只看显著一侧，A 群与 B 群的分布和各自的幂律（逐候选最大似然）。"""
    import csv
    from plot_grid_talk import dipole_lat, T90_CUT_US, MLAT_CUT_DEG
    plt.rcParams.update({"font.size": 21})
    tt = {(r["sat"], r["start"][:23]): r for r in csv.DictReader(open(t90))}
    rate = {(r["sat"], r["start"][:23]): float(r["mean"]) / (float(r["dur_us"]) * 1e-6) for r in csv.DictReader(open(features))}
    cls = {}
    for r in csv.DictReader(open(features)):
        k = (r["sat"], r["start"][:23])
        if rate[k] > HIGH_RATE: cls[k] = "hi"; continue
        t = float(tt[k]["t90_us"]); m = abs(dipole_lat(float(r["lat"]), float(r["lon"])))
        cls[k] = "A" if (t < T90_CUT_US and m < MLAT_CUT_DEG) else ("B" if (t >= T90_CUT_US and m >= MLAT_CUT_DEG) else "M")
    fpy = np.array([r["fpy"] for r in rows])
    hi_all = np.array([r["rate"] > HIGH_RATE for r in rows])       # 写出的全部候选都能按本底率分
    c = np.array([cls.get((r["sat"], r["start"][:23]), "") for r in rows])
    fmin = fpy[fpy > 0].min()
    bins = 10.0 ** np.arange(np.floor(np.log10(fmin)), 2.0)
    dln = np.log(10.0)
    x = np.clip(fpy, bins[0], None)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(21, 7.6), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.14))
    # 左
    a1.axvspan(1e-5, bins[0], color="#F3F5F8", zorder=0)
    a1.hist(x[hi_all], bins=bins, histtype="step", color="#E8833A", lw=2.2, label="辐射带（本底 > 5000/秒）")
    a1.hist(x[~hi_all], bins=bins, histtype="step", color="#1B3454", lw=2.6, label="其余候选")
    fine = np.logspace(np.log10(fmin), np.log10(20), 201); fc = (fine[:-1] + fine[1:]) / 2
    n_low, _ = np.histogram(fpy[~hi_all], bins=fine)
    pw, _ = fit(fc, n_low.astype(float), 1e-4, 20)
    if pw is not None:
        xw = np.logspace(np.log10(20), -8, 50)
        a1.plot(xw, dln / np.log(fine[1] / fine[0]) * power_law(xw, *pw), color="0.35", lw=2, ls=":", label="本底涨落 b = %.2f" % pw[1])
    a1.axvline(1e-5, color="0.3", lw=1.3, ls="--")
    a1.text(1e-5, 2.5e4, " 显著 10$^{-5}$", va="top", fontsize=19, color="0.3")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(20, bins[0]); a1.set_ylim(0.5, 5e4)
    a1.set_xlabel("每年误报次数（越往右越显著）"); a1.set_ylabel("候选数")
    a1.set_title("全部候选", loc="left", fontsize=23)
    a1.legend(frameon=False, loc="upper right", fontsize=18)
    # 右
    rb = 10.0 ** np.arange(np.floor(np.log10(fmin)), -4.0)
    a2.axvspan(1e-5, rb[0], color="#F3F5F8", zorder=0)
    slopes = {}
    for code, col, lab in (("B", "#2F6497", "B 群：约 3 毫秒、高磁纬"), ("A", "#D62728", "A 群：短、低磁纬")):
        sel = c == code
        a2.hist(np.clip(fpy[sel], rb[0], None), bins=rb, histtype="stepfilled", alpha=0.5, color=col, label="%s（%d）" % (lab, sel.sum()))
        xs = fpy[sel & (fpy > SIG_LO) & (fpy < SIG_HI)]
        bb = mle_slope(xs); slopes[code] = bb; U = np.log(SIG_HI / SIG_LO)
        xr = np.logspace(-5, np.log10(max(fpy[sel].min(), rb[0])), 40)
        a2.plot(xr, len(xs) * bb * (xr / SIG_HI) ** bb / -np.expm1(-bb * U) * dln, color=col, lw=3, ls="--", label="幂律 b = %.3f" % bb)
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(1e-5, 1e-45); a2.set_ylim(0.5, 60)
    a2.set_xlabel("每年误报次数（只看显著一侧）")
    a2.set_title("显著候选：A 群和 B 群", loc="left", fontsize=23)
    a2.legend(frameon=False, loc="upper right", fontsize=18, ncol=1)
    for a in (a1, a2): a.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "noise b %.3f" % pw[1], "A b %.4f B b %.4f" % (slopes["A"], slopes["B"]))


if __name__ == "__main__":
    main()
