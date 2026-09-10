"""天格 GRID TGF 搜索的讲图：给报告用，一张图讲一件事，字号按投影调大。

图 1：显著候选在 T90 × 磁纬 × 谱硬度里的结构，以及谱硬度的噪声底。
图 2：地理分布。
图 3：两类各举一例的光变与逐事例沉积能量。

分类依据是**测出来的** T90（扣本底后累积计数的 5%–95%，见 scripts/cluster/grid_t90.py）
和偶极磁纬，不是搜索窗长——窗长顶在 1 ms 的搜索上限上，是搜索参数不是时长测量。

用法:
    python3 scripts/plot_grid_talk.py <features_sig.csv> <t90.csv> <tgfs_*.json ...> \
        -o <目录> [--lightcurves <dir>]
"""
import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS"], "font.family": "sans-serif",
    "axes.unicode_minus": False, "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "lines.linewidth": 2,
})

# 两个"干净角"的边界。T90 2 ms 是四星里最能把 GRID-03B 与另外三星分开的单一判据
# （80 个候选里只有 4 个跨界）；|磁纬| 33° 是另外三星候选的下边界。两条线都是描述
# 数据里已有的空隙，不是从物理导出的阈值。
T90_CUT_US = 2000.0
MLAT_CUT_DEG = 33.0
# 谱硬度的名义分界，画出来是为了说明它在低计数下切不动，不用它分类。
HARDNESS_NOMINAL = 1.6

RED, BLUE, GREY = "#c53030", "#2b6cb0", "0.55"
SAT_COLORS = {"GRID-02": "#d97706", "GRID-03B": "#c53030", "GRID-04": "#2b6cb0", "GRID-07": "#2f855a"}
# 实测的读出时间尺度（逐探头相邻事例 dt 直方的硬边沿，单位 µs）。四星时戳栅格相同
# （都是 2^-22 s = 0.2384 µs），差别在这里：GRID-03B 是逐探头独立的死时间 4.77 µs，
# 另外三颗是四路共用的**帧长** 28.4–28.6 µs——帧内每路最多 1 个事例、全帧共用触发那
# 一击的时戳。见 blink_grid/OPEN-QUESTIONS.md 未决项 6 与 16。
DEAD_TIME_US = {"GRID-02": 28.37, "GRID-03B": 4.77, "GRID-04": 28.61, "GRID-07": 28.37}
DEAD_TIME_NAME = {"GRID-03B": "探头死时间", "GRID-02": "读出帧长",
                  "GRID-04": "读出帧长", "GRID-07": "读出帧长"}
ENERGY_THRESHOLD_KEV = 30.0

POLE_LAT, POLE_LON = np.radians(80.7), np.radians(-72.7)


def dipole_lat(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    s = np.sin(lat) * np.sin(POLE_LAT) + np.cos(lat) * np.cos(POLE_LAT) * np.cos(lon - POLE_LON)
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_particles(path):
    """f₃ 表：落在 ≥3 重同戳簇里的计数占比，带电粒子的签名。

    只有 GRID-03B 有——另外三星是共帧读出，时戳本来就塌在帧上，这个量在那里含义不同。
    键是 `GRID03B_YYYYMMDD_HHMMSSpmmm`，还原成 `start[:23]` 好和特征表对上。
    """
    out = {}
    for r in csv.DictReader(open(path)):
        _, ymd, hms = r["tag"].split("_")
        key = "%s-%s-%sT%s:%s:%s.%s" % (ymd[:4], ymd[4:6], ymd[6:8],
                                        hms[:2], hms[2:4], hms[4:6], hms[7:])
        out[key] = float(r["f3"])
    return out


def load(features, t90, tgfs, particles=None):
    assoc = {}
    for path in tgfs:
        for rec in json.load(open(path)):
            s = rec["signal"]
            assoc[(s["instrument"], s["start"][:23])] = bool(rec["lightning"].get("associated"))
    # 两张表都按 (卫星, start[:23]) 对键：start 列现在写全精度，旧表截到毫秒，取
    # 前 23 个字符两边都能对上。
    t = {(r["sat"], r["start"][:23]): r for r in csv.DictReader(open(t90))}
    rows = [r for r in csv.DictReader(open(features))]
    key = [(r["sat"], r["start"][:23]) for r in rows]
    missing = [k for k in key if k not in t]
    if missing:
        raise SystemExit("t90 表缺 %d 个候选，先补齐再画图：%s" % (len(missing), missing[:3]))
    d = dict(
        sat=np.array([r["sat"] for r in rows]),
        start=np.array([k[1] for k in key]),
        fa=np.array([_f(r["fa"]) for r in rows]),
        dur=np.array([_f(r["dur_us"]) for r in rows]),
        lon=np.array([_f(r["lon"]) for r in rows]),
        lat=np.array([_f(r["lat"]) for r in rows]),
        t90=np.array([_f(t[k]["t90_us"]) for k in key]),
        n=np.array([_f(t[k]["n_t90"]) for k in key]),
        rate=np.array([_f(t[k]["rate_bkg"]) for k in key]),
        hard=np.array([_f(t[k]["hardness_e"]) for k in key]),
        hlo=np.array([_f(t[k]["hard_e_lo"]) for k in key]),
        hhi=np.array([_f(t[k]["hard_e_hi"]) for k in key]),
        h95=np.array([_f(t[k].get("hard_e_95", "")) for k in key]),
        null95=np.array([_f(t[k]["null_p95"]) for k in key]),
        nullp=np.array([_f(t[k]["null_p"]) for k in key]),
    )
    d["assoc"] = np.array([assoc.get(k, False) for k in key])
    f3 = load_particles(particles) if particles else {}
    d["f3"] = np.array([f3.get(k[1], np.nan) for k in key])
    d["mlat"] = dipole_lat(d["lat"], d["lon"])
    # 两个干净角 + 中间带
    d["corner_a"] = (d["t90"] < T90_CUT_US) & (np.abs(d["mlat"]) < MLAT_CUT_DEG)
    d["corner_b"] = (d["t90"] >= T90_CUT_US) & (np.abs(d["mlat"]) >= MLAT_CUT_DEG)
    d["middle"] = ~d["corner_a"] & ~d["corner_b"]
    return d


def _sat_legend(ax, d, **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=c, mec="k", mew=0.5,
                      label="%s (%d)" % (sat, int((d["sat"] == sat).sum())))
               for sat, c in SAT_COLORS.items() if (d["sat"] == sat).any()]
    return ax.legend(handles=handles, **kw)


def fig_two_populations(d, out):
    """图 1：真实时长、磁纬、谱硬度三个量里的结构，外加谱硬度的噪声底。

    (a) 是主图：两个干净角 + 中间带。(c) 是"别把噪声当结构"的那张——零假设下
    谱硬度能涨到多少，随计数急剧变化。
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    a, b, m = d["corner_a"], d["corner_b"], d["middle"]
    asc = d["assoc"]
    is03b = d["sat"] == "GRID-03B"
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.9))

    # ---- (a) T90 × |偶极磁纬| ----
    ax = axes[0]
    ax.add_patch(Rectangle((20, 0), T90_CUT_US - 20, MLAT_CUT_DEG, facecolor=RED, alpha=0.07, zorder=0))
    ax.add_patch(Rectangle((T90_CUT_US, MLAT_CUT_DEG), 12000, 90 - MLAT_CUT_DEG,
                           facecolor=BLUE, alpha=0.07, zorder=0))
    for sat, c in SAT_COLORS.items():
        sel = (d["sat"] == sat) & ~asc
        ax.scatter(d["t90"][sel], np.abs(d["mlat"][sel]), s=56, c=c, lw=0.5, edgecolor="k",
                   alpha=0.9, zorder=4)
    ax.scatter(d["t90"][asc], np.abs(d["mlat"][asc]), s=230, marker="*", c="#f6e05e",
               lw=0.8, edgecolor="k", zorder=6)
    # 带电粒子签名：f₃ > 0（≥3 重同戳簇里的计数占比）。只有 GRID-03B 有这个量。
    part = d["f3"] > 0
    if part.any():
        ax.scatter(d["t90"][part], np.abs(d["mlat"][part]), s=110, marker="x", c="k",
                   lw=1.6, zorder=7)
    ax.axvline(T90_CUT_US, color="0.4", ls="--", lw=1.4)
    ax.axhline(MLAT_CUT_DEG, color="0.4", ls="--", lw=1.4)
    ax.set_xscale("log"); ax.set_xlim(20, 12000); ax.set_ylim(0, 80)
    ax.set_xlabel("T90 真实时长 (µs)"); ax.set_ylabel("|偶极磁纬| (°)")
    # 两个角的说明放进标题，图里只留最短的标签——B 角那一带点密，长文本压不下去
    ax.text(24, 31.5, "A 角 短·低磁纬 %d 个\n全是 GRID-03B" % int(a.sum()),
            fontsize=12.5, color=RED, va="top", ha="left", linespacing=1.5)
    ax.text(11500, 34.5, "B 角 毫秒·高磁纬 %d 个\n%d 个来自另外三星"
            % (int(b.sum()), int((b & ~is03b).sum())),
            fontsize=12.5, color=BLUE, va="bottom", ha="right", linespacing=1.5)
    ax.set_title("(a) 真实时长 × 磁纬：两个干净角 + %d 个中间带" % int(m.sum()), pad=10)
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=c, mec="k", mew=0.5,
                      label="%s (%d)" % (sat, int((d["sat"] == sat).sum())))
               for sat, c in SAT_COLORS.items() if (d["sat"] == sat).any()]
    handles.append(Line2D([], [], marker="*", ls="", ms=14, mfc="#f6e05e", mec="k", mew=0.6,
                          label="闪电证实 (%d)" % int(asc.sum())))
    if part.any():
        handles.append(Line2D([], [], marker="x", ls="", ms=9, mec="k", mew=1.6,
                              label="带电粒子签名 f₃>0 (%d)" % int(part.sum())))
    ax.legend(handles=handles, loc="upper left", fontsize=10.5, framealpha=0.9, ncol=1)

    # ---- (b) T90 × 谱硬度 ----
    ax = axes[1]
    for sat, c in SAT_COLORS.items():
        sel = (d["sat"] == sat) & ~asc
        ax.scatter(d["t90"][sel], d["hard"][sel], s=56, c=c, lw=0.5, edgecolor="k", alpha=0.9, zorder=4)
    ax.scatter(d["t90"][asc], d["hard"][asc], s=230, marker="*", c="#f6e05e", lw=0.8,
               edgecolor="k", zorder=6)
    ax.errorbar(d["t90"][asc], d["hard"][asc], yerr=[d["hard"][asc] - d["hlo"][asc],
                                                     d["hhi"][asc] - d["hard"][asc]],
                fmt="none", ecolor="0.35", elinewidth=1.4, capsize=3, zorder=5)
    ax.axhline(1.0, color="0.35", ls=":", lw=1.6)
    ax.axvline(T90_CUT_US, color="0.4", ls="--", lw=1.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(20, 12000); ax.set_ylim(0.22, 40)
    ax.set_xlabel("T90 真实时长 (µs)"); ax.set_ylabel("谱硬度：暴内能量中位 ÷ 本底")
    ok95 = np.isfinite(d["h95"])
    b_pow = np.median(d["h95"][b & ok95]) if (b & ok95).any() else np.nan
    ax.text(0.02, 0.03,
            "误差棒只画在 %d 个闪电证实上（暴内事例自举 16%%–84%%）\n"
            "B 角硬度中位 %.2f、自举 95%% 上限中位 %.2f：是测出来与本底同谱，不是没功效"
            % (int(asc.sum()), np.median(d["hard"][b]), b_pow),
            transform=ax.transAxes, fontsize=11, va="bottom", color="0.3", linespacing=1.5)
    ax.set_title("(b) 时长越短谱越硬", pad=10)

    # ---- (c) 谱硬度的噪声底 ----
    ax = axes[2]
    # 逐候选的零假设分位依赖各自的本底谱，直接连线是锯齿；按计数分对数格取中位数。
    bins = np.logspace(np.log10(6), np.log10(220), 11)
    idx = np.digitize(d["n"], bins) - 1
    xs, ys = [], []
    for i in range(len(bins) - 1):
        sel = idx == i
        if sel.sum():
            xs.append(np.sqrt(bins[i] * bins[i + 1])); ys.append(np.median(d["null95"][sel]))
    ax.plot(xs, ys, color="k", lw=2.4, marker="o", ms=5, zorder=5,
            label="零假设 95% 分位\n（谱与本底相同时硬度能涨到多少）")
    for sat, c in SAT_COLORS.items():
        sel = (d["sat"] == sat) & ~asc
        ax.scatter(d["n"][sel], d["hard"][sel], s=52, c=c, lw=0.5, edgecolor="k", alpha=0.85, zorder=4)
    ax.scatter(d["n"][asc], d["hard"][asc], s=230, marker="*", c="#f6e05e", lw=0.8,
               edgecolor="k", zorder=6, label="闪电证实的 TGF")
    ax.axhline(HARDNESS_NOMINAL, color=RED, ls="--", lw=1.6)
    ax.text(190, HARDNESS_NOMINAL * 1.08, "硬度 %.1f 这一刀" % HARDNESS_NOMINAL,
            color=RED, fontsize=12, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(6, 220); ax.set_ylim(0.22, 40)
    ax.set_xlabel("T90 区间内的事例数"); ax.set_ylabel("谱硬度")
    n_bad = int(((d["hard"] >= HARDNESS_NOMINAL) & (d["nullp"] >= 0.05)).sum())
    n_lost = int(((d["hard"] < HARDNESS_NOMINAL) & (d["nullp"] < 0.05)).sum())
    n_tgf_ns = int((d["nullp"][asc] >= 0.05).sum())
    ax.set_title("(c) 低计数下这一刀切不动", pad=10)
    ax.text(0.03, 0.04,
            "8 个计数时，谱与本底相同也能到 2.7\n噪声冒充硬 %d 个，真硬被切掉 %d 个\n%d 个证实的 TGF 里 %d 个硬度不显著"
            % (n_bad, n_lost, int(asc.sum()), n_tgf_ns),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5, color="0.25",
            linespacing=1.55)
    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.92)

    fig.suptitle("天格 %d 个显著候选：只有 GRID-03B 探测到 TGF——A 角 %d 个全是它，"
                 "B 角 %d/%d 来自另外三星（四路共帧读出，帧长是 03B 死时间的 6 倍）"
                 % (len(d["fa"]), int(a.sum()), int((b & ~is03b).sum()), int(b.sum())), fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=160); print("wrote", out)
    plt.close(fig)


def fig_map(d, out):
    """图 2：候选的地理分布，颜色分卫星、形状分类别。

    足点检验（B 角是不是 TGF 的电子束）暂不进讲图，分析还没定论；做法与数字留在
    `crates/instruments/blink_grid/OPEN-QUESTIONS.md` 和 `scripts/grid_footpoints.py`。
    """
    from matplotlib.lines import Line2D

    a, b, m, asc = d["corner_a"], d["corner_b"], d["middle"], d["assoc"]
    fig = plt.figure(figsize=(14, 6.8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    span = min(90, np.ceil(np.abs(d["lat"]).max()) + 4)
    ax.set_extent([-180, 180, -span, span], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.94")
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="0.45")
    ax.gridlines(draw_labels=False, lw=0.3, color="0.9")

    # 形状分类别、颜色分卫星：A 角只出在 GRID-03B，这一点在图上要一眼看得出来
    kinds = [(b, "s", 54, "B 角：毫秒·高磁纬"),
             (m, "^", 60, "中间带"),
             (a & ~asc, "o", 54, "A 角：短·低磁纬"),
             (a & asc, "*", 210, "A 角，闪电证实")]
    for sat in SAT_COLORS:
        for sel_kind, marker, size, _ in kinds:
            sel = sel_kind & (d["sat"] == sat)
            if not sel.any():
                continue
            ax.scatter(d["lon"][sel], d["lat"][sel], s=size, marker=marker, c=SAT_COLORS[sat],
                       lw=0.5, edgecolor="k", alpha=0.9, transform=ccrs.PlateCarree(),
                       zorder=6 if marker == "*" else 4)

    sat_handles = [Line2D([], [], marker="o", ls="", ms=10, mfc=c, mec="k", mew=0.5,
                          label="%s (%d)" % (sat, int((d["sat"] == sat).sum())))
                   for sat, c in SAT_COLORS.items() if (d["sat"] == sat).any()]
    kind_handles = [Line2D([], [], marker=marker, ls="", ms=13 if marker == "*" else 9,
                           mfc="0.75", mec="k", mew=0.5, label="%s (%d)" % (name, int(sel_kind.sum())))
                    for sel_kind, marker, _, name in kinds]
    leg = ax.legend(handles=sat_handles, loc="upper left", bbox_to_anchor=(0.0, -0.04),
                    ncol=2, frameon=False, fontsize=13, title="卫星", alignment="left")
    leg.get_title().set_fontsize(13)
    ax.add_artist(leg)
    leg2 = ax.legend(handles=kind_handles, loc="upper right", bbox_to_anchor=(1.0, -0.04),
                     ncol=1, frameon=False, fontsize=13, title="类别", alignment="left")
    leg2.get_title().set_fontsize(13)

    ax.set_title("天格候选的地理分布：A 角在低纬雷暴区、B 角在高磁纬海域\n"
                 "A 角 %d 个与 %d 个闪电证实的 TGF 全部来自 GRID-03B；"
                 "B 角经 GECAM 的带电粒子探测器判为磁层沉降电子"
                 % (int(a.sum()), int(asc.sum())),
                 fontsize=15.5, pad=12, linespacing=1.5)
    fig.savefig(out, dpi=160, bbox_inches="tight"); print("wrote", out)
    plt.close(fig)


# 图 3 的两个例子：A 角取闪电证实里最显著的一个，B 角取谱硬度正好等于 1.00（与本底
# 完全同谱）的那个。两颗星的探头死时间差 6 倍，正是 A 角只出在 03B 的原因。
EXAMPLES = [
    dict(tag="GRID-03B_20221004T000956543", sat="GRID-03B", color=RED,
         title="A 角：GRID-03B 2022-10-04 00:09:56.5 UTC",
         note="152.9°E 11.96°S｜偶极磁纬 −18°｜同时刻有闪电，巧合概率 3×10⁻⁴"),
    dict(tag="GRID-02_20210124T195431291", sat="GRID-02", color=BLUE,
         title="B 角：GRID-02 2021-01-24 19:54:31.3 UTC",
         note="149.3°W 47.2°S｜偶极磁纬 −45°｜WWLLN 有覆盖但无闪电｜同型事件经 GECAM 的带电粒子探测器判为粒子沉降"),
]


def _load_example(lc_dir, ex):
    """读逐事例 CSV 与 EBOUNDS，返回相对暴中心的时间、沉积能量、本底率、T90。

    事例准入在这里再过一道，和搜索一致（非溢出道 + 道下限能量 >= 30 keV）：早期导出的
    逐事例 CSV 只筛了 EVT_TYPE，直接画会多出一层 30 keV 以下的假本底。
    """
    from scipy.stats import poisson

    eb = np.genfromtxt(os.path.join(lc_dir, "ebounds_" + ex["sat"] + ".csv"), delimiter=",", names=True)
    ch = np.atleast_1d(eb["ch"]).astype(int)
    e_min = np.atleast_1d(eb["e_min"])
    e_max = np.atleast_1d(eb["e_max"])
    n_ch = len(ch)
    keep_ch = {int(c): np.sqrt(lo * hi) for c, lo, hi in zip(ch, e_min, e_max)
               if 1 <= int(c) < n_ch and lo >= ENERGY_THRESHOLD_KEV}

    d = np.genfromtxt(os.path.join(lc_dir, ex["tag"] + ".csv"), delimiter=",", names=True)
    t = np.atleast_1d(d["dt_ms"]).astype(float)
    pi = np.atleast_1d(d["pi"]).astype(int)
    ok = np.array([int(c) in keep_ch for c in pi])
    t, pi = t[ok], pi[ok]
    energy = np.array([keep_ch[int(c)] for c in pi])

    # 本底谱取导出文件里的 ±1 s 样本（已扣掉暴附近 ±5 ms），不要用 ±50 ms 窗里那
    # 一两百个事例：样本太小，中位数会跳，跟 grid_t90.py 的表也对不上。
    b = np.genfromtxt(os.path.join(lc_dir, ex["tag"] + "_bkg.csv"), delimiter=",", names=True)
    live = float(np.atleast_1d(b["live_s"])[0])
    bpi = np.atleast_1d(b["pi"])
    bpi = bpi[np.isfinite(bpi)].astype(int)
    bpi = np.array([int(c) for c in bpi if int(c) in keep_ch])
    bkg_energy = np.array([keep_ch[int(c)] for c in bpi])
    rate = float(len(bpi)) / live

    # 先用 1 ms 格粗框出暴的范围：含最大格、逐格泊松 p < 1e-4 的连续段，两边各放宽一格。
    # （更细的格里单个低计数格会把连续段打断，不能直接拿来量时长。）
    edges = np.arange(-15.0, 15.01, 1.0)
    cnt, _ = np.histogram(t, bins=edges)
    sig = poisson.sf(cnt - 1, rate / 1000.0) < 1e-4
    peak = int(np.argmax(cnt))
    lo = hi = peak
    while lo - 1 >= 0 and sig[lo - 1]:
        lo -= 1
    while hi + 1 < len(sig) and sig[hi + 1]:
        hi += 1
    b0, b1 = edges[max(lo - 1, 0)], edges[min(hi + 2, len(edges) - 1)]

    # 框内扣掉本底的累积计数，取 5%–95% 作为 T90；暴中心取 T90 区间中点。
    ts = np.sort(t[(t >= b0) & (t < b1)])
    cum = np.arange(1, len(ts) + 1) - rate / 1000.0 * (ts - b0)
    total = cum[-1]
    t05 = float(np.interp(0.05 * total, cum, ts))
    t95 = float(np.interp(0.95 * total, cum, ts))
    centre = 0.5 * (t05 + t95)
    inside = (t >= t05) & (t <= t95)
    excess = float(inside.sum() - rate / 1000.0 * (t95 - t05))
    return t - centre, energy, rate, inside, (t95 - t05), excess, bkg_energy


def fig_lightcurves(lc_dir, out):
    """图 3：两类各举一例，上排光变、下排逐事例沉积能量。"""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.4),
                             gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.22))
    half, w = 8.0, 0.2   # 上排统一 ±8 ms、200 µs 格，两类的时长直接可比

    for col, ex in enumerate(EXAMPLES):
        t, energy, rate, inside, span, excess, bkg_energy = _load_example(lc_dir, ex)
        mu = rate * w / 1000.0

        ax = axes[0, col]
        edges = np.arange(-half, half + w / 2, w)
        ax.hist(t, bins=edges, color=ex["color"], alpha=0.85, edgecolor=ex["color"])
        ax.axhline(mu, color="0.35", ls="--", lw=1.5)
        ax.set_xlim(-half, half)
        ax.set_ylim(0, max(np.histogram(t, bins=edges)[0].max() * 1.28, 3))
        ax.set_xlabel("相对暴中心时间 (ms)")
        ax.set_ylabel(f"计数 / {w * 1000:.0f} µs")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(ex["title"], pad=10)
        ax.text(0.02, 0.95, f"({'ab'[col]})", transform=ax.transAxes, va="top", fontsize=15, weight="bold")
        ax.text(0.98, 0.95,
                f"超出本底 {excess:.0f} 计数    T90 = {span * 1000:.0f} µs\n本底 {rate / 1000:.2f} 计数/ms（虚线）\n"
                f"{DEAD_TIME_NAME[ex['sat']]} {DEAD_TIME_US[ex['sat']]:.2f} µs",
                transform=ax.transAxes, ha="right", va="top", fontsize=13, color=ex["color"],
                linespacing=1.5)

        # A 角那例在 200 µs 格里只剩一根针，插一个 20 µs 格的放大图
        if span < 0.5:
            ins = ax.inset_axes([0.53, 0.24, 0.43, 0.44])
            fine = 0.02
            ins.hist(t, bins=np.arange(-0.3, 0.3 + fine / 2, fine), color=ex["color"], edgecolor=ex["color"])
            ins.set_xlim(-0.3, 0.3)
            ins.set_title("放大：20 µs 格", fontsize=12, pad=3)
            ins.tick_params(labelsize=11)
            ins.set_xlabel("ms", fontsize=11, labelpad=1)

        ax = axes[1, col]
        far = np.abs(t) > 1.5 * max(span, 0.3)
        ax.scatter(t[far], energy[far], s=22, color=GREY, alpha=0.55, label="本底事例", zorder=2)
        ax.scatter(t[inside], energy[inside], s=52, color=ex["color"], edgecolor="white",
                   linewidth=0.6, label="暴内事例", zorder=3)
        med_b, med_i = np.nanmedian(bkg_energy), np.nanmedian(energy[inside])
        # 两条线重合时（B 角正是这种情况）标签要分开放，否则叠成一团
        same = abs(np.log10(med_i / med_b)) < 0.06
        ax.axhline(med_b, color="0.35", ls="--", lw=1.5)
        ax.text(-half * 0.98, med_b, f" 本底中位 {med_b:.0f} keV（±1 s，{len(bkg_energy)} 个事例）",
                ha="left", va="top" if same else "bottom", fontsize=12, color="0.35")
        ax.axhline(med_i, color=ex["color"], ls="-", lw=1.5, alpha=0.7)
        ax.text(half * 0.98, med_i, f"暴内中位 {med_i:.0f} keV ", ha="right",
                va="bottom" if (med_i > med_b or same) else "top", fontsize=13, color=ex["color"])
        ax.set_yscale("log")
        ax.set_ylim(20, 2500)
        ax.set_xlim(-half, half)
        ax.set_xlabel("相对暴中心时间 (ms)")
        ax.set_ylabel("沉积能量 (keV)")
        ax.text(0.02, 0.95, f"({'cd'[col]})", transform=ax.transAxes, va="top", fontsize=15, weight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.text(0.5, -0.30, ex["note"], transform=ax.transAxes, ha="center", fontsize=13, color="0.3")

    fig.suptitle("两类各一例：A 角是几十微秒的硬脉冲，B 角是 3 ms 的平顶、能谱与本底相同；"
                 "GRID-03B 是 4.77 µs 的逐探头死时间，GRID-02 是 28.4 µs 的四路共帧读出", y=0.985)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features"); ap.add_argument("t90"); ap.add_argument("tgfs", nargs="+")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--lightcurves", help="逐事例导出目录，见 scripts/cluster/grid_lightcurve.py")
    ap.add_argument("--particles", help="f₃ 表（带电粒子签名），只有 GRID-03B 有")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d = load(args.features, args.t90, args.tgfs, args.particles)
    fig_two_populations(d, os.path.join(args.outdir, "grid_talk_1_populations.png"))
    fig_map(d, os.path.join(args.outdir, "grid_talk_2_map.png"))
    if args.lightcurves:
        fig_lightcurves(args.lightcurves, os.path.join(args.outdir, "grid_talk_3_lightcurves.png"))


if __name__ == "__main__":
    main()
