"""GECAM-B：没找回的已发表 TGF，逐个归因。**每一个都要有确定答案，不留"原因未知"。**

做法是把 `snapshot_stepping::search_new` 在 TGF 附近**逐位复刻**一遍，而不是估。
复刻的是单组（`coincidence = 1`、`group_number = 1`）那条路径：

    窗    = [t[cursor], t[cursor+step]]，窗内计数 = step + 1
    本底  = 事例落在 [t[cursor] − 0.5 s, t[cursor+step] + 0.5 s) 内的个数
            减去空窗 [t[cursor] − 5 ms, t[cursor+step] + 5 ms) 内的个数
    时长  = 上面两个窗与 GTI 交集的时长之差
    λ     = 本底计数 × (窗长 / 本底时长)
    sf    = P(X > 计数 | λ)，  fa = sf × (一年秒数 / 窗长)
    触发条件：计数 ≥ 8、窗长 ≤ 1 ms、fa < 20

有了逐位复刻，归因就不是推测：对每个没找回的 TGF 报出**它附近最好的那个窗**
的 (count, λ, fa)，再逐条过否决判据。同一套复刻还顺带给出三个对照口径：

  * `dedup=dead`  —— 当前 main 的死时间去重（3c017a2）
  * `dedup=exact` —— 旧的时戳相等去重（e34f560），用来判"是不是去重改动压掉的"
  * `dedup=none`  —— 完全不去重，给出上界

用法: python3 gb_missing.py <recall.csv> <输出 CSV>
"""

import csv
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_feat import (ARCHIVE, EPOCH, MIN_CHANNEL, OVERFLOW_CHANNEL, NORMAL_EVT_TYPE,
                     dedupe_one_detector, gti_overlap, hour_file, inside_gti, met, read_gti)

NEIGHBOR, HOLLOW = 1.0, 0.010     # 本底窗全宽 / 空窗全宽（秒）
MAX_DURATION, MIN_NUMBER = 1e-3, 8
MAX_DETECTOR_FRACTION = 0.8
FPY = 20.0
YEAR = 3600.0 * 24.0 * 365.2425   # DAYS_PER_YEAR
PROBE = 5e-3                      # 在 UT 两侧这么多秒内逐事例当 cursor 试


def load_local(iso_hour, t_lo, t_hi):
    """读一小时 GRD，准入 + GTI 过滤，返回三种去重口径下的局部事例流。"""
    path = hour_file(iso_hour + ":00:00", "grd")
    if path is None:
        return None
    times, dets, keep_dead, keep_exact = [], [], [], []
    with fits.open(path, memmap=True) as hdus:
        gti = read_gti(hdus)
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            pi = np.asarray(data["PI"])
            k = (np.asarray(data["EVT_TYPE"]) == NORMAL_EVT_TYPE) & (pi >= MIN_CHANNEL) & (pi < OVERFLOW_CHANNEL)
            t = np.asarray(data["TIME"], float)[k]
            ins = inside_gti(t, gti)
            t = t[ins]
            g = np.asarray(data["GAIN_TYPE"])[k][ins].astype(np.int8)
            dead = np.asarray(data["DEAD_TIME"], np.float32)[k][ins]
            o = np.argsort(t, kind="stable")
            t, g, dead = t[o], g[o], dead[o]
            # 只留局部（本底窗要 ±0.5 s，留 1.5 s 余量）
            m = (t >= t_lo) & (t <= t_hi)
            drop_dead = dedupe_one_detector(t, g, dead)[m]
            # 旧口径（e34f560）：同探头同时戳的一串里只留一条，留 gain_type 最大
            # （低增益）那条，同档留先出现的。一路一张表，表内同时戳天然同探头。
            tm, gm = t[m], g[m]
            drop_exact = np.zeros(tm.size, bool)
            if tm.size > 1:
                head = np.flatnonzero(np.diff(tm) != 0) + 1
                run_id = np.zeros(tm.size, np.int64)
                run_id[head] = 1
                run_id = np.cumsum(run_id)
                # 每串里留 (gain_type 最大, 下标最小) 的那条
                order = np.lexsort((np.arange(tm.size), -gm.astype(np.int16), run_id))
                first = np.ones(tm.size, bool)
                first[1:] = run_id[order][1:] != run_id[order][:-1]
                drop_exact[order[~first]] = True
            times.append(tm)
            dets.append(np.full(tm.size, int(hdu.name[-2:]), np.int8))
            keep_dead.append(~drop_dead)
            keep_exact.append(~drop_exact)
    if not times:
        return None
    time = np.concatenate(times)
    det = np.concatenate(dets)
    order = np.lexsort((det, time))
    return {
        "time": time[order], "det": det[order],
        "dead": np.concatenate(keep_dead)[order],
        "exact": np.concatenate(keep_exact)[order],
        "none": np.ones(time.size, bool),
        "gti": gti,
    }


def best_window(time, det, gti, span, t_center, probe=PROBE):
    """在 t_center 两侧 probe 秒内逐位复刻 search_new，返回最好（fa 最小）的那个窗。"""
    n = time.size
    if n == 0:
        return None
    lo = int(np.searchsorted(time, t_center - probe, "left"))
    hi = int(np.searchsorted(time, t_center + probe, "right"))
    best = None
    for cursor in range(lo, hi):
        t0 = time[cursor]
        step = MIN_NUMBER - 1
        while cursor + step < n:
            t1 = time[cursor + step]
            duration = t1 - t0
            if duration >= MAX_DURATION or t1 >= span[1]:
                break
            count = step + 1
            # 与 Rust 的窗口维护逐位等价：左端 t >= t0 − 半宽（含），
            # 右端 t < t1 + 半宽（不含）—— 见 search_new 里两个 snapshot 的推进条件
            mean_n = int(np.searchsorted(time, t1 + NEIGHBOR / 2, "left")
                         - np.searchsorted(time, t0 - NEIGHBOR / 2, "left"))
            hollow_n = int(np.searchsorted(time, t1 + HOLLOW / 2, "left")
                           - np.searchsorted(time, t0 - HOLLOW / 2, "left"))
            mean_live = gti_overlap(t0 - NEIGHBOR / 2, t1 + NEIGHBOR / 2, gti)
            hollow_live = gti_overlap(t0 - HOLLOW / 2, t1 + HOLLOW / 2, gti)
            pure_dur = mean_live - hollow_live
            if pure_dur > 0 and duration > 0:
                lam = (mean_n - hollow_n) * (duration / pure_dur)
                sf = float(poisson.sf(count, lam)) if lam > 0 else 1.0
                fa = sf * (YEAR / duration)
                share = np.bincount(det[cursor:cursor + step + 1]).max() / count
                if best is None or fa < best["fa"]:
                    best = {"t0": t0, "t1": t1, "count": count, "lam": lam, "sf": sf,
                            "fa": fa, "det_frac": share, "dur_us": duration * 1e6}
            step += 1
    return best


def verify(rows, n_check=12):
    """先验复刻器：拿**找回的**那些 TGF 跑一遍，(count, λ, fa) 必须与搜索报的逐条相等。

    这一步不是装饰。复刻器要是有偏差，后面对没找回者的每一条归因都不成立——
    "回到事例流重算候选窗内的量之前先用搜索报的 count 对账"就是这个意思。
    """
    ok = [r for r in rows if r["matched"] == "1" and r["fa"]]
    pick = ok[:: max(len(ok) // n_check, 1)][:n_check]
    print(f"=== 复刻器自检：拿 {len(pick)} 个已找回的 TGF 对账 ===")
    good = 0
    for r in pick:
        ut = met(r["UT"])
        ev = load_local(r["hour"], ut - 1.5, ut + 1.5)
        if ev is None:
            print(f"  {r['UT']}: 读不到文件")
            continue
        span0 = met(r["hour"] + ":00:00.0")
        keep = ev["dead"]
        best = best_window(ev["time"][keep], ev["det"][keep], ev["gti"],
                           (span0, span0 + 3600.0), ut + float(r["dt_ms"]) * 1e-3)
        want_fa, want_n = float(r["fa"]), int(r["count"])
        hit = best is not None and best["count"] == want_n and abs(best["fa"] / want_fa - 1) < 1e-6
        good += hit
        print(f"  {r['UT']}  搜索 count={want_n:3d} fa={want_fa:.4e}   复刻 "
              f"count={best['count'] if best else -1:3d} fa={best['fa'] if best else float('nan'):.4e}  "
              f"{'一致' if hit else '不一致'}")
    print(f"  对账 {good}/{len(pick)}\n")
    return good == len(pick)


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    if not verify(rows):
        print("！复刻器与搜索对不上，先修口径，下面的归因不算数")
    missed = [r for r in rows if r["matched"] == "0"]
    print(f"没找回 {len(missed)} 个，逐个复刻搜索")
    out = []
    for r in missed:
        iso_hour = r["hour"]
        ut = met(r["UT"])
        span0 = met(iso_hour + ":00:00.0")
        span = (span0, span0 + 3600.0)
        ev = load_local(iso_hour, ut - 1.5, ut + 1.5)
        if ev is None:
            print(f"  {r['UT']}: 读不到 GRD 文件")
            continue
        in_gti = bool(inside_gti(np.array([ut]), ev["gti"])[0])
        rec = {"UT": r["UT"], "hour": iso_hour, "Duration_us": r["Duration_us"],
               "NetCounts": r["NetCounts"], "n_cand_hour": r["n_cand_hour"],
               "ut_in_gti": int(in_gti)}
        print(f"\n  {r['UT']}  Duration {float(r['Duration_us']):.1f} µs  "
              f"NetCounts {float(r['NetCounts']):.1f}  UT 在 GTI 内: {in_gti}")
        for mode in ("dead", "exact", "none"):
            keep = ev[mode]
            t, d = ev["time"][keep], ev["det"][keep]
            # 目录窗内的裸计数
            a = int(np.searchsorted(t, ut, "left"))
            b = int(np.searchsorted(t, ut + float(r["Duration_us"]) * 1e-6, "right"))
            best = best_window(t, d, ev["gti"], span, ut)
            rec[f"n_cat_{mode}"] = b - a
            if best:
                rec[f"fa_{mode}"] = f"{best['fa']:.4e}"
                rec[f"count_{mode}"] = best["count"]
                rec[f"lam_{mode}"] = f"{best['lam']:.3f}"
                rec[f"dur_{mode}"] = f"{best['dur_us']:.3f}"
                rec[f"detfrac_{mode}"] = f"{best['det_frac']:.3f}"
                rec[f"dtms_{mode}"] = f"{(best['t0'] - ut) * 1e3:.4f}"
                print(f"    {mode:5s} 目录窗内 {b - a:3d} 计数；最好的窗 "
                      f"count={best['count']:3d} λ={best['lam']:.3f} 窗长={best['dur_us']:7.2f} µs "
                      f"fa={best['fa']:.3e} 单路占比={best['det_frac']:.3f} "
                      f"dt={(best['t0'] - ut) * 1e3:+.4f} ms → "
                      f"{'越线' if best['fa'] < FPY else '不越线'}"
                      f"{'，但被单路占比否决' if best['fa'] < FPY and best['det_frac'] > MAX_DETECTOR_FRACTION else ''}")
            else:
                rec[f"fa_{mode}"] = ""
                print(f"    {mode:5s} 目录窗内 {b - a:3d} 计数；±{PROBE * 1e3:.0f} ms 内凑不出 "
                      f"{MIN_NUMBER} 计数的窗（min_number 否决）")
        out.append(rec)

    if out:
        with open(sys.argv[2], "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
