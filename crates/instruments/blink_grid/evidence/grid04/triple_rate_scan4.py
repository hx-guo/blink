"""逐过境扫 `frac_ev_in3`（共帧星 02/04/07），并给出**纯本底期望**做分母。

## 为什么共帧星不能照搬 03B 的阈

`frac_ev_in3` = 落在 ≥3 重同戳簇里的事例占比。同一个数在两种读出下量的不是一件事：

- **GRID-03B**（四路独立、逐探头死时间 20 tick）：同探头从不同戳，所以 ≥3 重同戳
  **必然**是跨探头真符合。纯本底下的偶然期望量级 1e-9，实测 8e-4–1.7e-2 高出六个
  数量级 ⇒ 量的是物理（贯穿粒子），与速率解耦。
- **GRID-02/04/07**（共用一条读出链）：任一路击中开一帧，帧开头 w ≈ 4 µs 内其余各路的
  击中并进本帧、共用触发时戳。于是"一帧里有 ≥3 路"**不需要任何粒子**就会造出 ≥3 重
  同戳簇，而并进概率随速率单调上升——所以先验上它可能退化成速率的代理量，那样直接
  卡阈就等于**偷偷加了一道磁纬门**（速率随磁纬陡变）。

  **实测结果是这个担心不成立，记在这里免得后人重复**：GRID-04 上帧偶然项算出来是
  4e-6–1e-4，而实测 `frac_ev_in3` 是 1.2e-4–8e-2，**高出 8–670 倍**。所以共帧星上
  这个量仍然由真实贯穿粒子主导，不是速率代理。但**绝对阈值照旧不能跨星搬**，
  分母要各星自己算，比值 `ratio_frame` 才是判别量。

## 两个分母，都算，并排报

1. **自标定期望** `frac_exp_self`：用**这一片自己的二重占比**反推单路并进概率
   q = (E[m] − 1)/3，再给出纯偶然下的三重／四重占比
   `[3·3q²(1−q) + 4q³] / E[m]`。它不依赖任何模型参数，**天然把速率依赖除掉**。
   贯穿粒子的四重／三重比远高于偶然（实例：一次坏过境 9428 个四重对 60 个三重），
   所以这个分母虽然保守（真粒子也抬高二重），判别力仍在。
2. **解析帧模型期望** `frac_exp_frame`：四路泊松 + 帧内其余三路在 w 内有击中，
   w 取该星实测值。两个分母对不上本身就是结果。

## 位置口径（三处坑，都按管线来）

- 位姿成段缺失时 `np.interp` 会把两头连成直线，几十秒的轨道弧被当成直线——**比丢掉
  更糟，因为给出一个看着正常的磁纬值**。按 `io/posatt.rs::MAX_SAMPLE_GAP_SECONDS`
  取 30 s 门槛，空档里返回 NaN。
- `$GRID_ORBIT_FIT_DIR/posatt_blacklist.txt` 里的日子位姿本身是错的
  （GRID-07 2024-02-08/09 错约 380 s，会把辐射带峰灌进低磁纬）。
- 2024-02 起 `POS_TYPE = 0`，退回拟合轨道表；逐过境记 `pos_src`，**不静默合并**。
- 坏行判据是**物理半径带**而不是 `|r| > 0`：`|r| > 0` 拦不住 GECAM 那种 1362–9060 km
  的坏行（造出 |磁纬| 87° 的假高纬点）。天格地心距应在 6800–6960 km。

用法: python3 triple_rate_scan4.py <SAT> <chunk> <nchunk> <outdir> [起始日] [结束日]
"""

import glob
import math
import os
import sys

import numpy as np
from astropy.io import fits

ARCH = "/gecamfs/Exchange/GSDC/missions/GRID/%s"
ETH = 30.0
TICK = 2.0**22
TRIPLE = 3
SLICE_S = 10.0
MAX_GAP_S = 30.0                       # io/posatt.rs::MAX_SAMPLE_GAP_SECONDS
R_MIN_KM, R_MAX_KM = 6800.0, 6960.0    # 天格 SSO 的地心距窄带（高度 430–580 km）
POLE_LAT, POLE_LON = math.radians(80.7), math.radians(-72.7)
# 帧长与有效并进窗，逐星从数据实测（`frame_probe.py` / `readout_window.py`）。
# GRID-04：帧成本 dead(s) = 120 + 37(s−1) tick，并进窗 w ≈ 3.8 µs。
FRAME_TICK = {"GRID-02": 120, "GRID-04": 120, "GRID-07": 120}
JOIN_W_US = {"GRID-02": 3.8, "GRID-04": 3.8, "GRID-07": 3.8}


def dipole_lat(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    s = (np.sin(lat) * math.sin(POLE_LAT)
         + np.cos(lat) * math.cos(POLE_LAT) * np.cos(lon - POLE_LON))
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def blacklisted(sat, day):
    d = os.environ.get("GRID_ORBIT_FIT_DIR")
    if not d:
        return False
    try:
        with open(os.path.join(d, "posatt_blacklist.txt")) as f:
            return any(line.strip() == "%s %s" % (sat, day) for line in f)
    except OSError:
        return False


def load_posatt(sat, ymd, span):
    """位姿里位置可用的行：(met, |偶极磁纬|)。坏行按地心距窄带滤，不是按 |r| > 0。"""
    hits = sorted(glob.glob(ARCH % sat + "/fits8/%s/posatt_*/*%s*" % (ymd, span)))
    if not hits:
        return None
    try:
        d = fits.getdata(hits[-1], extname="ORBIT_ATTITUDE")
        t = np.asarray(d["TIME"], dtype=np.float64)
        lat = np.asarray(d["Latitude"], dtype=np.float64)
        lon = np.asarray(d["Longitude"], dtype=np.float64)
        x = np.asarray(d["X_WGS84"], dtype=np.float64)
        y = np.asarray(d["Y_WGS84"], dtype=np.float64)
        z = np.asarray(d["Z_WGS84"], dtype=np.float64)
    except Exception:
        return None
    r = np.sqrt(x * x + y * y + z * z)
    if np.nanmedian(r) < 1e5:          # 有的产品以 km 写，有的以 m 写
        r_km = r
    else:
        r_km = r / 1000.0
    ok = (np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
          & np.isfinite(r_km) & (r_km > R_MIN_KM) & (r_km < R_MAX_KM))
    if ok.sum() < 2:
        return None
    return t[ok], np.abs(dipole_lat(lat[ok], lon[ok]))


def load_orbit_fit(sat, day):
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
    """采样正常的段内插值；落在 > MAX_GAP_S 的空档里返回 NaN（绝不连线）。"""
    t, m = pos
    if q < t[0] or q > t[-1]:
        return float("nan")
    i = min(max(np.searchsorted(t, q) - 1, 0), t.size - 2)
    if t[i + 1] - t[i] > MAX_GAP_S:
        return float("nan")
    w = (q - t[i]) / (t[i + 1] - t[i])
    return float(m[i] + w * (m[i + 1] - m[i]))


def position_for(sat, day, ymd, span):
    if not blacklisted(sat, day):
        pos = load_posatt(sat, ymd, span)
        if pos is not None:
            return pos, "posatt"
    pos = load_orbit_fit(sat, day)
    if pos is not None:
        return pos, "orbit_fit"
    return None, "none"


def load_pass(path):
    """按 `Event::keep` 的三条准入读一次过境。返回 (gs, ge, 合并时刻, 合并能量, 计数)。"""
    with fits.open(path, memmap=False) as hd:
        g = hd["GTI"].data
        gs, ge = float(np.asarray(g["START"])[0]), float(np.asarray(g["STOP"])[0])
        eb = hd["EBOUNDS"].data
        emin = np.asarray(eb["E_MIN"], dtype=np.float64)
        emax = np.asarray(eb["E_MAX"], dtype=np.float64)
        egeo = np.sqrt(emin * emax)     # 道中心用几何平均：四星道—能对应不同，道号不可比
        nch = emin.size
        T, E, n_all, n_t2 = [], [], 0, 0
        for i in range(4):
            d = hd["EVENTS%d" % i].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            pi = np.asarray(d["PI"], dtype=np.int32)
            ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
            n_all += t.size
            n_t2 += int((ty == 2).sum())
            ok = (pi >= 1) & (pi < nch)
            e = np.zeros(t.size)
            e[ok] = emin[pi[ok] - 1]
            sel = (ty == 1) & ok & (e >= ETH)
            T.append(t[sel])
            E.append(egeo[pi[sel] - 1])
    counts = [x.size for x in T]
    t = np.concatenate(T)
    e = np.concatenate(E)
    o = np.argsort(t, kind="stable")
    return gs, ge, t[o], e[o], n_all, n_t2, counts


def stats(times, energies, dur, tau_s, w_s):
    """一段上的同戳簇统计与两个纯本底期望。"""
    n = times.size
    if n < 50:
        return None
    tk = np.rint(times * TICK).astype(np.int64)
    ed = np.flatnonzero(np.diff(tk) != 0)
    st = np.concatenate(([0], ed + 1))
    sz = np.concatenate((ed + 1, [tk.size])) - st
    big = sz >= TRIPLE
    frac = float(sz[big].sum()) / n
    em = float(sz.mean())
    # 自标定：用本片自己的 E[m] 反推单路并进概率，再给三重／四重的偶然期望
    q = max((em - 1.0) / 3.0, 0.0)
    p3 = 3 * q * q * (1 - q)
    p4 = q ** 3
    frac_self = (3 * p3 + 4 * p4) / em if em > 0 else float("nan")
    # 解析帧模型：由观测率反推入射率，q = 1 − exp(−λ_d w)
    lam = n / dur / 4.0
    lam_inc = lam / max(1.0 - (n / dur) * tau_s / max(em, 1e-9), 1e-3)
    qf = 1.0 - math.exp(-lam_inc * w_s)
    emf = 1 + 3 * qf
    frac_frame = (3 * (3 * qf * qf * (1 - qf)) + 4 * qf ** 3) / emf
    e_q4 = float("nan")
    if (sz >= 4).any():
        idx = np.concatenate([np.arange(s, s + z) for s, z in zip(st[sz >= 4], sz[sz >= 4])])
        e_q4 = float(np.median(energies[idx]))
    return dict(n=n, rate=n / dur, em=em, k3=int(big.sum()), k4=int((sz >= 4).sum()),
                frac=frac, r3=big.sum() / dur, max_mult=int(sz.max()),
                frac_self=frac_self, frac_frame=frac_frame,
                ratio_self=frac / frac_self if frac_self > 0 else float("nan"),
                ratio_frame=frac / frac_frame if frac_frame > 0 else float("nan"),
                e_med=float(np.median(energies)), e_q4=e_q4)


def main():
    sat, chunk, nchunk, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    lo = sys.argv[5] if len(sys.argv) > 5 else "0000-00-00"
    hi = sys.argv[6] if len(sys.argv) > 6 else "9999-99-99"
    tau_s = FRAME_TICK.get(sat, 120) / TICK
    w_s = JOIN_W_US.get(sat, 3.8) * 1e-6
    os.makedirs(outdir, exist_ok=True)
    days = sorted(glob.glob(ARCH % sat + "/fits7/*/*/*"))
    days = [d for d in days if lo <= "-".join(d.split("/")[-3:]) <= hi][chunk::nchunk]
    fp = open(os.path.join(outdir, "trip_pass_%02d.csv" % chunk), "w")
    fp.write("sat,day,pass_file,ver,dur_s,n_kept,rate_cps,r0,r1,r2,r3d,type2_frac,em,"
             "k3,k4,r3_per_s,frac_ev_in3,frac_self,ratio_self,frac_frame,ratio_frame,"
             "max_mult,e_med,e_q4,pos_src,mlat_med,mlat_min,mlat_max,frac_t_mlat40\n")
    fs = open(os.path.join(outdir, "trip_slice_%02d.csv" % chunk), "w")
    fs.write("sat,day,pass_file,islice,dur_s,n,rate_cps,em,k3,k4,frac_ev_in3,"
             "frac_self,ratio_self,max_mult,e_q4,pos_src,mlat\n")
    for daydir in days:
        p = daydir.split("/")
        day = "%s-%s-%s" % (p[-3], p[-2], p[-1])
        ymd = "/".join(p[-3:])
        vers = sorted(glob.glob(daydir + "/evt_v*"))
        if not vers:
            continue
        ver = os.path.basename(vers[-1])
        for path in sorted(glob.glob(vers[-1] + "/*.fits")):
            base = os.path.basename(path)
            span = "_".join(base.split("_")[2:4])
            try:
                gs, ge, t, e, n_all, n_t2, counts = load_pass(path)
            except Exception as ex:
                fp.write("%s,%s,%s,%s" % (sat, day, base, ver) + "," * 24
                         + "ERR %s\n" % str(ex)[:40])
                continue
            dur = max(ge - gs, 1e-9)
            if t.size < 100:
                continue
            rates = [c / dur for c in counts]
            s_all = stats(t, e, dur, tau_s, w_s)
            pos, src = position_for(sat, day, ymd, span)
            nsl = int(math.ceil(dur / SLICE_S))
            mids = gs + SLICE_S * np.arange(nsl) + SLICE_S / 2
            ml = np.array([mlat_at(pos, q) for q in mids]) if pos else np.full(nsl, np.nan)
            aml = ml[np.isfinite(ml)]
            fp.write("%s,%s,%s,%s,%.1f,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.5f,%.5f,"
                     "%d,%d,%.4f,%.6e,%.6e,%.3f,%.6e,%.3f,%d,%.1f,%.1f,%s,%s,%s,%s,%s\n"
                     % (sat, day, base, ver, dur, t.size, s_all["rate"],
                        rates[0], rates[1], rates[2], rates[3], n_t2 / max(n_all, 1),
                        s_all["em"], s_all["k3"], s_all["k4"], s_all["r3"], s_all["frac"],
                        s_all["frac_self"], s_all["ratio_self"], s_all["frac_frame"],
                        s_all["ratio_frame"], s_all["max_mult"], s_all["e_med"], s_all["e_q4"],
                        src,
                        "%.2f" % np.median(aml) if aml.size else "",
                        "%.2f" % aml.min() if aml.size else "",
                        "%.2f" % aml.max() if aml.size else "",
                        "%.4f" % (aml > 40).mean() if aml.size else ""))
            bounds = np.searchsorted(t, gs + SLICE_S * np.arange(nsl + 1))
            for i in range(nsl):
                a, z = bounds[i], bounds[i + 1]
                if z - a < 50:
                    continue
                d_i = min(SLICE_S, ge - (gs + SLICE_S * i))
                si = stats(t[a:z], e[a:z], d_i, tau_s, w_s)
                if si is None:
                    continue
                fs.write("%s,%s,%s,%d,%.1f,%d,%.1f,%.5f,%d,%d,%.6e,%.6e,%.3f,%d,%.1f,%s,%s\n"
                         % (sat, day, base, i, d_i, si["n"], si["rate"], si["em"],
                            si["k3"], si["k4"], si["frac"], si["frac_self"],
                            si["ratio_self"], si["max_mult"], si["e_q4"], src,
                            "%.2f" % ml[i] if np.isfinite(ml[i]) else ""))
            fp.flush()
            fs.flush()
    fp.close()
    fs.close()
    print("%s chunk %d done" % (sat, chunk))


if __name__ == "__main__":
    main()
