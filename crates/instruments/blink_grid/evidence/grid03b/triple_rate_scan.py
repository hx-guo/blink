"""逐过境扫 `frac_ev_in3`：找 2022-03-19 那种「大半事例落在同戳簇里」的过境。

2022-03-19T06:14:28–06:19:12 那次过境实测：keep 后 3371 c/s、≥3 重同戳簇率 **599 个/s**、
四重簇 463 个/s、**64% 的事例落在 ≥3 重同戳簇里**。对照 38 次正常过境（速率 141–6318
c/s）：r₃ 平在 1.0–2.0 个/s、事例占比 8×10⁻⁴–1.7×10⁻²。**不是速率效应**——速率 5722
与 6318 c/s 的两次过境 r₃ 只有 1.97 和 1.69。这次过境产出了一个 v11 的新显著候选
（`2022-03-19T06:16:38`，count = 8、窗 8.3 µs、四路同戳、f₃ 正好 0.500），而速率门
`RATE_CEILING = 5000 c/s` 差 338 c/s 没拦住（局部速率 4662 c/s）。

本脚本要回答三件事，缺一不可：

  1. 全任务有多少次过境 `frac_ev_in3` 高，占比多少、落在哪些日期；
  2. 它们是成群（同一天、同一轨道段）还是散的；
  3. **`frac_ev_in3` 与 |偶极磁纬| 的关系。** 若它就是辐射带的影子，这道门与磁纬强相关，
     那必须说清它挡的是**环境**而不是**位置**——否则等于偷偷加了一道磁纬门。

所以除了逐过境一行，还累一张 **|磁纬| 分箱**的账：把过境切成 `--seg` 秒的小段，逐段取
位置算偶极磁纬，把事例数、落在 ≥3 重簇里的事例数、簇数、秒数按磁纬分箱累加。这张账直接
给出 frac_ev_in3 对 |磁纬| 的曲线，且不受"一次过境跨很多磁纬"的干扰。

**位置口径与 Rust 管线一致**（`io/posatt.rs` + `io/orbit_fit.rs`），三条都不能省：
  - 只用有限的经纬度行（2024-02 起 `POS_TYPE = 0`，整段是 NaN）；
  - **插值不跨 30 s 以上的采样空档**（`MAX_SAMPLE_GAP_SECONDS`）。位姿有成段缺失
    （03B 2022-10-01 有 6 段各 74 s），`np.interp` 会把缺口两头连成直线，比丢掉更糟；
  - **位置黑名单**（`$GRID_ORBIT_FIT_DIR/posatt_blacklist.txt`）里的日子位置是错的
    （GRID-07 2024-02-08/09 把 8 kc/s 的辐射带峰放在磁纬 35°，错约 380 s），抹掉改用
    拟合轨道表。没有位姿位置时同样退回拟合轨道表 `<sat>/<YYYYMMDD>.csv`。
位置来源逐过境记在 `pos_src` 列（posatt / orbit_fit / none），**不许静默合并**。

用法（农场 N 分片）：
    python3 triple_rate_scan.py <chunk> <nchunk> <outdir> [--sat GRID-03B] [--seg 10]
产物：
    <outdir>/trip_NN.csv    逐过境一行
    <outdir>/mlat_NN.csv    |磁纬| 分箱的累计账
"""

import argparse
import glob
import os

import numpy as np
from astropy.io import fits

ARCH = "/gecamfs/Exchange/GSDC/missions/GRID/{sat}"
ETH = 30.0
TICK = 2.0**22
TRIPLE = 3
# 与 cov_groups.py 同一组极点。全队别处用 (80.65, -72.68)，两者差 < 0.05°，
# 对 |磁纬| 的影响远小于本图 2° 的分箱；口径统一列在 OPEN-QUESTIONS。
POLE_LAT, POLE_LON = np.radians(80.7), np.radians(-72.7)
MLAT_EDGES = np.arange(0, 92, 2.0)
# io/posatt.rs::MAX_SAMPLE_GAP_SECONDS
MAX_GAP_S = 30.0
# 位置坏行的物理半径带（见 load_posatt）。03B/04/07 是 SSO 约 500 km、02 约 55° 倾角低轨；
# 硬带取整个 LEO，窄带再按本文件高度中位数收。
ALT_BAND_KM = (150.0, 1500.0)
ALT_SPREAD_KM = 100.0
# 分块转时戳格的块大小（见下），只为省临时数组，不影响结果
TICK_CHUNK = 8_000_000


def dipole_lat(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    s = np.sin(lat) * np.sin(POLE_LAT) + np.cos(lat) * np.cos(POLE_LAT) * np.cos(lon - POLE_LON)
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def cluster_arrays(tk):
    """返回 (簇起点索引, 簇长度)。tk 已升序。"""
    if tk.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    edge = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], edge + 1))
    en = np.concatenate((edge + 1, [tk.size]))
    return st, en - st


def blacklisted(sat, day):
    """`$GRID_ORBIT_FIT_DIR/posatt_blacklist.txt` 里的 `<星目录名> <YYYY-MM-DD>`。"""
    d = os.environ.get("GRID_ORBIT_FIT_DIR")
    if not d:
        return False
    try:
        with open(os.path.join(d, "posatt_blacklist.txt")) as f:
            return any(line.strip() == "%s %s" % (sat, day) for line in f)
    except OSError:
        return False


def load_posatt(sat, ymd, span):
    """位姿里有位置解**且地心距落在本星轨道窄带内**的行：(met, |偶极磁纬|)。

    只筛 NaN 不够。全队实测：POSATT 里有 `X = Y = Z = 0` 一类的坏行，一天 1565 行；
    **按 `|r|` 非零过滤仍放过 36 行**（地心距 1362–9060 km），造出 |磁纬| 87° 的假点。
    判据要用**物理半径带**——地心距必须落在该星轨道的窄带内。天格是太阳同步、倾角
    97–98°，"纬度超过倾角"这条自检不灵敏（极区本来就到得了），只能看高度。

    两层：先夹死物理上不可能的（LEO 硬带），再按本文件高度中位数 ±100 km 掐掉离群行
    （圆轨道，一次过境内高度变化只有几十 km）。
    """
    hits = sorted(glob.glob(ARCH.format(sat=sat) + "/fits8/%s/posatt_*/*%s*" % (ymd, span)))
    if not hits:
        return None
    try:
        d = fits.getdata(hits[-1], extname="ORBIT_ATTITUDE")
        t = np.asarray(d["TIME"], dtype=np.float64)
        lat = np.asarray(d["Latitude"], dtype=np.float64)
        lon = np.asarray(d["Longitude"], dtype=np.float64)
        alt_km = np.asarray(d["Altitude"], dtype=np.float64) / 1000.0
    except Exception:
        return None
    ok = (np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt_km)
          & (alt_km >= ALT_BAND_KM[0]) & (alt_km <= ALT_BAND_KM[1])
          & (np.abs(lat) <= 90.0) & (np.abs(lon) <= 360.0))
    if ok.sum() < 2:
        return None
    med = np.median(alt_km[ok])
    ok &= np.abs(alt_km - med) <= ALT_SPREAD_KM
    if ok.sum() < 2:
        return None
    return t[ok], np.abs(dipole_lat(lat[ok], lon[ok]))


def load_orbit_fit(sat, day):
    """拟合轨道表 `<GRID_ORBIT_FIT_DIR>/<sat>/<YYYYMMDD>.csv`：time,lon,lat,alt_m。"""
    d = os.environ.get("GRID_ORBIT_FIT_DIR")
    if not d:
        return None
    path = os.path.join(d, sat, day.replace("-", "") + ".csv")
    if not os.path.exists(path):
        return None
    try:
        a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2), ndmin=2)
    except Exception:
        return None
    if a.shape[0] < 2:
        return None
    return a[:, 0], np.abs(dipole_lat(a[:, 2], a[:, 1]))


def mlat_at(pos, q):
    """在采样密度正常的段内插值；落在 > MAX_GAP_S 的空档里返回 NaN（不连线）。"""
    t, m = pos
    if q < t[0] or q > t[-1]:
        return float("nan")
    i = min(max(np.searchsorted(t, q) - 1, 0), t.size - 2)
    if t[i + 1] - t[i] > MAX_GAP_S:
        return float("nan")
    w = (q - t[i]) / (t[i + 1] - t[i])
    return float(m[i] + w * (m[i + 1] - m[i]))


def position_for(sat, day, ymd, span):
    """位置来源按管线口径择一：位姿（未被黑名单抹掉）→ 拟合轨道表 → 没有。"""
    if not blacklisted(sat, day):
        pos = load_posatt(sat, ymd, span)
        if pos is not None:
            return pos, "posatt"
    pos = load_orbit_fit(sat, day)
    if pos is not None:
        return pos, "orbit_fit"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk", type=int)
    ap.add_argument("nchunk", type=int)
    ap.add_argument("outdir")
    ap.add_argument("--sat", default="GRID-03B")
    ap.add_argument("--seg", type=float, default=10.0, help="分段秒数，用来配磁纬")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    all_days = sorted(glob.glob(ARCH.format(sat=args.sat) + "/fits7/*/*/*"))
    days = all_days[args.chunk::args.nchunk]
    n_days = n_passes = 0
    nb = len(MLAT_EDGES) - 1
    acc_ev = np.zeros(nb)
    acc_in3 = np.zeros(nb)
    acc_k3 = np.zeros(nb)
    acc_s = np.zeros(nb)
    no_pos_s = 0.0

    fh = open(os.path.join(args.outdir, "trip_%02d.csv" % args.chunk), "w")
    fh.write("sat,day,version,pass_file,gti_start,gti_stop,dur_s,n_kept,rate_cps,"
             "k3,k4,r3_per_s,frac_ev_in3,max_mult,"
             "mlat_min,mlat_max,mlat_at_maxfrac,pos_src,pos_seconds,note\n")
    for daydir in days:
        p = daydir.split("/")
        day = "%s-%s-%s" % (p[-3], p[-2], p[-1])
        ymd = "%s/%s/%s" % (p[-3], p[-2], p[-1])
        vers = sorted(glob.glob(daydir + "/evt_v*"))
        if not vers:
            continue
        ver = os.path.basename(vers[-1])
        n_days += 1
        for path in sorted(glob.glob(vers[-1] + "/*.fits")):
            n_passes += 1
            base = os.path.basename(path)
            try:
                with fits.open(path, memmap=True) as hd:
                    g = hd["GTI"].data
                    gs = float(np.asarray(g["START"])[0])
                    ge = float(np.asarray(g["STOP"])[0])
                    emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
                    nch = emin.size
                    # E_MIN 随道号单调（97 道对数分布），所以"沉积 ≥ 30 keV"等价于
                    # "道号 ≥ ch_lo"。这样就不必为整列事例造一个 float64 的能量数组
                    # ——一个过境上亿个事例，那一个数组就是几百 MB。先断言单调性。
                    if np.any(np.diff(emin) <= 0):
                        raise ValueError("E_MIN 非单调，道号下界这条捷径不成立")
                    above = np.flatnonzero(emin >= ETH)
                    ch_lo = int(above[0]) + 1 if above.size else nch  # PI 从 1 起
                    T = []
                    for i in range(4):
                        d = hd["EVENTS%d" % i].data
                        pi = np.asarray(d["PI"])
                        ty = np.asarray(d["EVT_TYPE"])
                        keep = (ty == 1) & (pi >= ch_lo) & (pi < nch)
                        del pi, ty
                        T.append(np.asarray(d["TIME"], dtype=np.float64)[keep])
                        del keep, d
            except Exception as ex:
                fh.write("%s,%s,%s,%s,,,,,,,,,,,,,,,,ERR %s\n"
                         % (args.sat, day, ver, base, str(ex)[:60].replace(",", ";")))
                continue
            # 一次过境到 1e8 个事例，临时数组就是几百 MB 一个。concatenate 之后立刻
            # 放掉逐探头的那四个、原地排序（np.sort 会再复制一份）。
            t = np.concatenate(T)
            del T
            t.sort()
            dur = max(ge - gs, 1e-9)
            if t.size < 100:
                fh.write("%s,%s,%s,%s,%.3f,%.3f,%.1f,%d,,,,,,,,,,,,too few events\n"
                         % (args.sat, day, ver, base, gs, ge, dur, t.size))
                continue
            # 分块转时戳格：`np.rint(t * TICK).astype(np.int64)` 会先造一个和 t 一样大的
            # float64 临时数组，再造一个 int64，1e8 个事例就是多占 1.6 GB。
            tk = np.empty(t.size, dtype=np.int64)
            for a in range(0, t.size, TICK_CHUNK):
                b = min(a + TICK_CHUNK, t.size)
                tk[a:b] = np.rint(t[a:b] * TICK)
            _, sz = cluster_arrays(tk)
            big = sz >= TRIPLE
            k3 = int(big.sum())
            k4 = int((sz >= 4).sum())
            frac = float(sz[big].sum()) / tk.size

            # 事例文件名里的时段串，用来配同一次过境的位姿文件
            parts = base.split("_")
            span = "%s_%s" % (parts[2], parts[3]) if len(parts) >= 4 else ""
            pos, src = position_for(args.sat, day, ymd, span) if span else (None, "none")

            mlo = mhi = mbest = float("nan")
            best_frac = -1.0
            pos_s = 0.0
            edges = np.arange(gs, ge + args.seg, args.seg)
            # t 已排序 ⇒ 用 searchsorted 定段边界。先前对每一段都做整数组布尔掩码，
            # 那是 O(段数 × 事例数)：一次 2 小时的过境 720 段 × 1e8 个事例，跑不完。
            lo_arr = edges[:-1]
            hi_arr = np.minimum(edges[1:], ge)
            b0 = np.searchsorted(t, lo_arr, side="left")
            b1 = np.searchsorted(t, hi_arr, side="left")   # 与 (t >= lo) & (t < hi) 逐个等价
            for i in range(len(edges) - 1):
                lo, hi = lo_arr[i], hi_arr[i]
                if hi <= lo:
                    continue
                a0, a1 = int(b0[i]), int(b1[i])
                nev = a1 - a0
                if nev <= 0:
                    continue
                _, szx = cluster_arrays(tk[a0:a1])
                bx = szx >= TRIPLE
                ml = mlat_at(pos, 0.5 * (lo + hi)) if pos is not None else float("nan")
                if not np.isfinite(ml):
                    no_pos_s += hi - lo
                    continue
                b = min(int(ml / 2.0), nb - 1)
                acc_ev[b] += nev
                acc_in3[b] += int(szx[bx].sum())
                acc_k3[b] += int(bx.sum())
                acc_s[b] += hi - lo
                pos_s += hi - lo
                mlo = ml if not np.isfinite(mlo) else min(mlo, ml)
                mhi = ml if not np.isfinite(mhi) else max(mhi, ml)
                f = int(szx[bx].sum()) / nev
                if f > best_frac:
                    best_frac, mbest = f, ml

            def fmt(v):
                return "%.2f" % v if np.isfinite(v) else ""

            fh.write("%s,%s,%s,%s,%.3f,%.3f,%.1f,%d,%.1f,%d,%d,%.4f,%.6f,%d,%s,%s,%s,%s,%.1f,\n"
                     % (args.sat, day, ver, base, gs, ge, dur, t.size, t.size / dur,
                        k3, k4, k3 / dur, frac, int(sz.max()),
                        fmt(mlo), fmt(mhi), fmt(mbest), src, pos_s))
            fh.flush()
    fh.close()

    with open(os.path.join(args.outdir, "mlat_%02d.csv" % args.chunk), "w") as mf:
        mf.write("mlat_lo,mlat_hi,seconds,n_events,n_in3,n_clusters\n")
        for i in range(nb):
            if acc_s[i] == 0:
                continue
            mf.write("%.0f,%.0f,%.1f,%d,%d,%d\n"
                     % (MLAT_EDGES[i], MLAT_EDGES[i + 1], acc_s[i],
                        int(acc_ev[i]), int(acc_in3[i]), int(acc_k3[i])))
        mf.write("# no_position_seconds,%.1f\n" % no_pos_s)

    # 完成标记：worker 跑完才写。汇总入口要硬断言 Σdays_done == days_expected，
    # 不能等非零退出——"队列空 ≠ 跑完"，残缺输入上跑完整条链不会报错。
    with open(os.path.join(args.outdir, "done_%02d.txt" % args.chunk), "w") as df:
        df.write("sat,%s\nchunk,%d\nnchunk,%d\n"
                 "days_expected,%d\ndays_done,%d\npasses,%d\nno_position_seconds,%.1f\n"
                 % (args.sat, args.chunk, args.nchunk, len(days), n_days, n_passes, no_pos_s))
    print("chunk %d done: %d/%d days, %d passes" % (args.chunk, n_days, len(days), n_passes))


if __name__ == "__main__":
    main()
