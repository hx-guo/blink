"""GECAM-B：跨探头 τ 合并的门槛 (3)——合并后的事例流到底是不是泊松的。

**为什么必须带模拟对照。** 合并本身会吃掉 τ 内的事例，那是一种死时间效应，
**它单独就会把方差压到均值以下**。所以"合并后 F 更接近 1"这句话没有分母：
要拿同速率的齐次泊松流走同一条合并管线，得到 `F_sim`，再问实测的 `F_post`
偏离 `F_sim` 多少。**只有 `F_post − F_sim` 才是真正剩下的过离散。**

同理，合并前也要报，否则"改善了多少"没有分母（合并前的对照就是 `F_pre`）。

**安静段的定义不能把被检验的量本身切掉。** 候选率实测 7.4 个/s，任何 10 s 的
窗里都有候选——候选本身就是过离散的表现，"挑没有候选的段"等于把要测的东西
先删掉（循环论证）。所以这里的安静段只切三样**与事例流无关**的东西：
GTI 之外、SAA、以及（作为第二档对照）显著候选 `fa ≤ 1e-5` 的 ±20 ms。
两档都报：只切 GTI+SAA 的是正题，再切显著候选的用来说明结论不是几个亮事件带的。

段要跨不同磁纬与轨道相位，不能只挑一段：本脚本把一小时内所有合规的 10 s 段
全部取出，逐段报 (地磁纬、升/降交、速率)，汇总时按段给分布而不是只给一个数。

Fano 因子在小 λ 下卡方近似不好用，所以**显著性一律由模拟给**（同段、同总数、
同探头份额，重抽 `N_SIM` 次取散布），不用解析式。

用法: python3 gb_poisson.py <小时清单> <输出目录> [worker] [workers]
"""

import json
import os
import sys

import numpy as np
from astropy.io import fits

import gb_feat as gf

TAU = 150e-9            # 跨探头合并容差
SEG_SECONDS = 10.0      # 安静段长度
GTI_MARGIN = 2.0        # GTI 两端各削掉的秒数
SIG_FA = 1e-5           # 第二档对照里当作"显著候选"的阈
SIG_GUARD = 0.020       # 显著候选两侧各挖掉的秒数
N_SIM = int(os.environ.get("GB_NSIM", 20))          # 每段每个档的模拟重抽次数
MAX_SEGMENTS = int(os.environ.get("GB_MAXSEG", 12))  # 每小时段数上限，在合规段里均匀抽
BINS_US = (1.0, 10.0, 100.0, 1000.0, 10000.0)
# SAA 的地理框（宽松取，宁可多切）：经度 −100..+45，纬度 −55..+5
SAA_LON, SAA_LAT = (-100.0, 45.0), (-55.0, 5.0)
# IGRF 偶极北极（约 2020 纪元）
POLE_LAT, POLE_LON = np.radians(80.65), np.radians(-72.68)
WGS84_A, WGS84_F = 6378137.0, 1.0 / 298.257223563


# 轨道半径的合规带：B 星约 600 km 圆轨，实测好行 |r| 全落在 6960–6980 km
R_MIN, R_MAX = 6.6e6, 7.2e6


def read_posatt(iso_hour):
    """这一小时的轨道位置。**坏行必须按物理半径带滤，"X=Y=Z=0" 这个口径不够。**

    实测 2022-06-15 一天 75,971 行里 1,565 行（2.06%）是坏的，而**没有一行是精确
    的零**：绝大多数是 `4.28e−38` 这种反常规格化垃圾（|r| ≈ 0），但还有 36 行的
    |r| 落在 1362–9060 km、以及一行 |r| = 2.5e22 m。**`|r| > 1e6` 那道闸放行了
    后面这批**，于是这一天算出 |地理纬| 最大 90.00°、|磁纬| 87.13°——一颗倾角
    29° 的星不可能到那儿。改用合规带 [6600, 7200] km 一次全挡掉。
    """
    day = iso_hour[:10]
    directory = f"{gf.ARCHIVE}/{day[:4]}/{day[5:7]}/{day[8:10]}/GECAM_B/posatt"
    if not os.path.isdir(directory):
        return None
    best = {}
    for name in os.listdir(directory):
        if not name.endswith(".fits"):
            continue
        stem, _, ver = name[:-len(".fits")].rpartition("_v")
        try:
            version = int(ver)
        except ValueError:
            continue
        if version > best.get(stem, (-1, None))[0]:
            best[stem] = (version, os.path.join(directory, name))
    rows = []
    for _, path in sorted(best.values(), key=lambda x: x[1]):
        with fits.open(path, memmap=False) as hdus:
            data = hdus[1].data
            t = np.asarray(data["TIME"], float)
            xyz = np.stack([np.asarray(data[f"{a}_WGS84"], float) for a in "XYZ"], 1)
            vz = np.asarray(data["VZ_WGS84"], float)
            rows.append((t, xyz, vz))
    if not rows:
        return None
    t = np.concatenate([r[0] for r in rows])
    xyz = np.concatenate([r[1] for r in rows])
    vz = np.concatenate([r[2] for r in rows])
    radius = np.linalg.norm(xyz, axis=1)
    good = np.isfinite(radius) & (radius > R_MIN) & (radius < R_MAX)
    t, xyz, vz, radius = t[good], xyz[good], vz[good], radius[good]
    order = np.argsort(t)
    t, xyz, vz, radius = t[order], xyz[order], vz[order], radius[order]
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lon = (lon + 180.0) % 360.0 - 180.0
    # 地理纬度：WGS84 椭球，Bowring 一次迭代足够（0.001° 量级）
    p = np.hypot(xyz[:, 0], xyz[:, 1])
    e2 = WGS84_F * (2 - WGS84_F)
    lat = np.arctan2(xyz[:, 2], p * (1 - e2))
    for _ in range(3):
        n = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(xyz[:, 2] + e2 * n * np.sin(lat), p)
    lat = np.degrees(lat)
    mlat = np.degrees(np.arcsin(
        np.sin(np.radians(lat)) * np.sin(POLE_LAT)
        + np.cos(np.radians(lat)) * np.cos(POLE_LAT) * np.cos(np.radians(lon) - POLE_LON)))
    return {"t": t, "lon": lon, "lat": lat, "mlat": mlat, "vz": vz}


def at(orbit, when):
    if orbit is None or orbit["t"].size == 0:
        return None
    i = int(np.clip(np.searchsorted(orbit["t"], when), 0, orbit["t"].size - 1))
    return {k: float(orbit[k][i]) for k in ("lon", "lat", "mlat", "vz")}


def merge_cross_det(t, det, tau):
    """跨探头单链接合并：相邻两条 Δt ≤ τ 且探头不同则连边，连通分量并成一条
    （保留分量里最早的那条）。返回分量首事例的下标。"""
    if t.size == 0:
        return np.zeros(0, np.int64)
    link = (np.diff(t) <= tau) & (det[1:] != det[:-1])
    return np.concatenate(([0], np.flatnonzero(~link) + 1))


def fano_curve(t, t0, t1, bins_us):
    """逐 bin 宽算 Fano = var/mean。返回 {bin_us: (F, lambda, n_bin)}。

    **不开 nb 长的直方数组**——1 µs 档一段就是 1e7 格，开一次几十 ms、乘上模拟
    重抽次数就变成分钟级。空格对 Σc 与 Σc² 都贡献 0，所以只数非空格的占据数：
    `Σc² = Σ 非空格的 c²`，`var = (Σc² − (Σc)²/nb) / (nb − 1)`，与开满数组等价。
    """
    out = {}
    for bw_us in bins_us:
        bw = bw_us * 1e-6
        nb = int((t1 - t0) / bw)
        if nb < 200 or t.size == 0:
            out[bw_us] = (np.nan, np.nan, nb)
            continue
        idx = ((t - t0) / bw).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < nb)]
        _, c = np.unique(idx, return_counts=True)
        c = c.astype(np.float64)
        s1_, s2_ = c.sum(), (c * c).sum()
        mean = s1_ / nb
        var = (s2_ - s1_ * s1_ / nb) / (nb - 1)
        out[bw_us] = ((var / mean if mean > 0 else np.nan), mean, nb)
    return out


def simulate(n, t0, t1, det_share, rng, tau, bins_us):
    """同段、同总数、同探头份额的齐次泊松流，走同一条合并管线。"""
    ts = np.sort(rng.uniform(t0, t1, n))
    det = rng.choice(det_share.size, size=n, p=det_share).astype(np.int8)
    pre = fano_curve(ts, t0, t1, bins_us)
    head = merge_cross_det(ts, det, tau)
    post = fano_curve(ts[head], t0, t1, bins_us)
    return pre, post, head.size


def ks_exponential(gaps):
    """到达间隔对指数分布的 KS 统计量（速率由样本均值给）。"""
    g = np.sort(gaps[np.isfinite(gaps) & (gaps >= 0)])
    if g.size < 100:
        return np.nan, g.size
    cdf = 1.0 - np.exp(-g / g.mean())
    n = g.size
    d = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))
    return float(d), n


def main():
    hours_file, outdir = sys.argv[1], sys.argv[2]
    worker = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.makedirs(outdir, exist_ok=True)
    hours = [x.strip() for x in open(hours_file) if x.strip()]
    rng = np.random.default_rng(20260911 + worker)

    for idx, iso in enumerate(hours):
        if idx % workers != worker:
            continue
        path = gf.hour_file(iso, "grd")
        if path is None:
            print(f"{iso}: 没有 GRD 文件，跳过", flush=True)
            continue
        ev = gf.read_grd(path)
        if ev is None:
            print(f"{iso}: 事例为空，跳过", flush=True)
            continue
        keep = ev["keep"]
        time, det = ev["time"][keep], ev["det"][keep]
        gti = ev["gti"]
        orbit = read_posatt(iso)

        # 显著候选的时刻（第二档对照用）
        day = iso[:10]
        sig_path = (f"{gf.SIGNAL_ROOT}/{day[:4]}/{day[5:7]}/"
                    f"{day.replace('-', '')}_signals.json")
        sig_t = []
        if os.path.exists(sig_path):
            for s in json.load(open(sig_path)):
                if s["start"][:13] != iso:
                    continue
                if s.get("false_positive_per_year", 1e9) <= SIG_FA:
                    sig_t.append(gf.met(s["start"]) + s.get("delay", 0.0))
        sig_t = np.sort(np.array(sig_t, float))

        # 合规的 10 s 段：GTI 内（两端削 GTI_MARGIN）、非 SAA
        segments = []
        if gti is not None:
            for a, b in zip(*gti):
                a, b = a + GTI_MARGIN, b - GTI_MARGIN
                k = 0
                while a + (k + 1) * SEG_SECONDS <= b:
                    s0 = a + k * SEG_SECONDS
                    k += 1
                    pos = at(orbit, s0 + SEG_SECONDS / 2)
                    if pos is None:
                        continue
                    if (SAA_LON[0] <= pos["lon"] <= SAA_LON[1]
                            and SAA_LAT[0] <= pos["lat"] <= SAA_LAT[1]):
                        continue
                    segments.append((s0, s0 + SEG_SECONDS, pos))

        if len(segments) > MAX_SEGMENTS:
            pick = np.linspace(0, len(segments) - 1, MAX_SEGMENTS).round().astype(int)
            segments = [segments[i] for i in dict.fromkeys(pick.tolist())]

        rows = []
        for s0, s1, pos in segments:
            lo = int(np.searchsorted(time, s0, "left"))
            hi = int(np.searchsorted(time, s1, "right"))
            ts, ds = time[lo:hi], det[lo:hi]
            if ts.size < 5000:
                continue
            share = np.bincount(ds.astype(int), minlength=26).astype(float)
            share /= share.sum()

            pre = fano_curve(ts, s0, s1, BINS_US)
            head = merge_cross_det(ts, ds, TAU)
            post = fano_curve(ts[head], s0, s1, BINS_US)
            ks_pre, n_pre = ks_exponential(np.diff(ts))
            ks_post, _ = ks_exponential(np.diff(ts[head]))

            # 簇大小分布：过离散的量与"多少事例成簇"是同一件事的两面，
            # 而复合泊松的尾巴（= fa 被抬多少）只由这张分布决定，所以逐段存下来。
            csz = np.diff(np.concatenate((head, [ts.size])))
            csz_hist = np.bincount(np.clip(csz, 0, 6), minlength=7)[1:].tolist()

            sim_pre = {b: [] for b in BINS_US}
            sim_post = {b: [] for b in BINS_US}
            sim_kept = []
            for _ in range(N_SIM):
                sp, sq, nk = simulate(ts.size, s0, s1, share, rng, TAU, BINS_US)
                for b in BINS_US:
                    sim_pre[b].append(sp[b][0])
                    sim_post[b].append(sq[b][0])
                sim_kept.append(nk)

            row = {
                "hour": iso, "t0": s0, "n": int(ts.size), "rate": ts.size / SEG_SECONDS,
                "n_merged": int(ts.size - head.size),
                "merged_frac": (ts.size - head.size) / ts.size,
                "sim_merged_frac": float(np.mean([ts.size - k for k in sim_kept])) / ts.size,
                "mlat": pos["mlat"], "lat": pos["lat"], "lon": pos["lon"],
                "asc": int(pos["vz"] > 0),
                "ks_pre": ks_pre, "ks_post": ks_post, "n_gap": n_pre,
                "near_sig": int(sig_t.size and np.any((sig_t > s0 - SIG_GUARD)
                                                      & (sig_t < s1 + SIG_GUARD))),
                "csz_hist": csz_hist,           # 簇大小 1,2,3,4,5,>=6 的簇数
                "csz_mean2": float((csz * csz).mean()),   # E[m²]，复合泊松尾巴要用
            }
            for b in BINS_US:
                row[f"F_pre_{b:g}"] = pre[b][0]
                row[f"F_post_{b:g}"] = post[b][0]
                row[f"lam_{b:g}"] = pre[b][1]
                row[f"Fsim_pre_{b:g}"] = float(np.mean(sim_pre[b]))
                row[f"Fsim_pre_sd_{b:g}"] = float(np.std(sim_pre[b], ddof=1))
                row[f"Fsim_post_{b:g}"] = float(np.mean(sim_post[b]))
                row[f"Fsim_post_sd_{b:g}"] = float(np.std(sim_post[b], ddof=1))
            rows.append(row)

        if not rows:
            print(f"{iso}: 没有合规段", flush=True)
            continue
        out = f"{outdir}/poisson_{iso.replace('-', '').replace('T', '_')}.json"
        json.dump(rows, open(out, "w"))
        summarize(iso, rows)


def summarize(iso, rows):
    import statistics as st
    n = len(rows)
    rate = st.median(r["rate"] for r in rows)
    print(f"\n{iso}: {n} 段 × {SEG_SECONDS:g} s，速率中位 {rate:.0f} c/s，"
          f"磁纬 {min(r['mlat'] for r in rows):.1f}..{max(r['mlat'] for r in rows):.1f}°，"
          f"升交 {sum(r['asc'] for r in rows)}/{n}", flush=True)
    print(f"  合并掉的比例：实测中位 {st.median(r['merged_frac'] for r in rows) * 100:.4f}%，"
          f"同速率泊松模拟 {st.median(r['sim_merged_frac'] for r in rows) * 100:.4f}%", flush=True)
    head = f"  {'bin':>8} {'lam':>8} {'F_pre':>9} {'Fsim_pre':>9} {'F_post':>9} {'Fsim_post':>9} {'(F_post-Fsim)/sd':>18}"
    print(head, flush=True)
    for b in BINS_US:
        fp = st.median(r[f"F_pre_{b:g}"] for r in rows)
        sp = st.median(r[f"Fsim_pre_{b:g}"] for r in rows)
        fq = st.median(r[f"F_post_{b:g}"] for r in rows)
        sq = st.median(r[f"Fsim_post_{b:g}"] for r in rows)
        sd = st.median(r[f"Fsim_post_sd_{b:g}"] for r in rows)
        lam = st.median(r[f"lam_{b:g}"] for r in rows)
        z = (fq - sq) / sd if sd > 0 else float("nan")
        print(f"  {b:8g} {lam:8.3f} {fp:9.4f} {sp:9.4f} {fq:9.4f} {sq:9.4f} {z:18.1f}", flush=True)


if __name__ == "__main__":
    main()
