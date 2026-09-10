"""量每个显著候选的真实时长与真实硬度。

搜索给出的 `dur_us` 是窗长，顶在 1 ms 的搜索上限上，不是时长；窗内能道中位数同样
受窗长影响。这里在 ±15 ms 里先用 1 ms 格框出暴的范围，再对扣除本底的累积计数取
5%–95% 得到 T90，然后在 T90 区间内重新量沉积能量中位数与本底之比。

用法: python3 grid_t90.py <候选 CSV: sat,start,...> <输出 CSV>
"""
from astropy.io import fits
from scipy.stats import poisson
import glob, os, csv, sys, numpy as np, datetime as dt

G = "/gecamfs/Exchange/GSDC/missions/GRID"
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)


def met(iso):
    b = iso.rstrip("Z")
    h, f = b.split(".")
    return (dt.datetime.strptime(h + "." + (f + "000000")[:6], "%Y-%m-%dT%H:%M:%S.%f")
            .replace(tzinfo=dt.timezone.utc) - REF).total_seconds()


def pass_files(sat, t0):
    out = []
    for back in (0, 1):
        day = (REF + dt.timedelta(seconds=t0) - dt.timedelta(days=back)).strftime("%Y/%m/%d")
        dd = f"{G}/{sat}/fits7/{day}"
        if not os.path.isdir(dd):
            continue
        v = sorted(os.listdir(dd))[-1]
        out += sorted(glob.glob(f"{dd}/{v}/*.fits"))
    return out


rows = list(csv.DictReader(open(sys.argv[1])))
w = csv.writer(open(sys.argv[2], "w", newline=""))
w.writerow(["sat", "start", "dur_search_us", "t90_us", "excess", "n_t90", "rate_bkg",
            "pi_med_t90", "pi_med_bkg", "hardness_t90", "n_det_t90", "det_frac_t90"])
n_ok = 0
for r in rows:
    sat, t0 = r["sat"], met(r["start"])
    written = False
    for f in pass_files(sat, t0):
        with fits.open(f) as h:
            g = h["GTI"].data
            s0, s1 = float(g["START"][0]), float(g["STOP"][0])
            if not (s0 <= t0 <= s1):
                continue
            T, P, D = [], [], []
            for k in range(4):
                x = h[f"EVENTS{k}"].data
                t = np.asarray(x["TIME"], float)
                m = (np.asarray(x["EVT_TYPE"]) == 1) & (t >= t0 - 1.0) & (t <= t0 + 1.0)
                T.append(t[m]); P.append(np.asarray(x["PI"])[m]); D.append(np.full(int(m.sum()), k))
            T = np.concatenate(T); P = np.concatenate(P); D = np.concatenate(D)
            o = np.argsort(T); T, P, D = T[o], P[o], D[o]
            live = min(t0 + 1.0, s1) - max(t0 - 1.0, s0)
            far = np.abs(T - t0) > 0.02
            rate = far.sum() / live

            edges = t0 + np.arange(-15.0, 15.01, 1.0) * 1e-3
            cnt, _ = np.histogram(T, bins=edges)
            sig = poisson.sf(cnt - 1, rate / 1000.0) < 1e-4
            peak = int(np.argmax(cnt))
            lo = hi = peak
            while lo - 1 >= 0 and sig[lo - 1]:
                lo -= 1
            while hi + 1 < len(sig) and sig[hi + 1]:
                hi += 1
            b0, b1 = edges[max(lo - 1, 0)], edges[min(hi + 2, len(edges) - 1)]
            ts = np.sort(T[(T >= b0) & (T < b1)])
            if len(ts) < 3:
                break
            cum = np.arange(1, len(ts) + 1) - rate * (ts - b0)
            total = cum[-1]
            if total <= 0:
                break
            t05 = float(np.interp(0.05 * total, cum, ts))
            t95 = float(np.interp(0.95 * total, cum, ts))
            inside = (T >= t05) & (T <= t95)
            t90 = (t95 - t05) * 1e6
            excess = inside.sum() - rate * (t95 - t05)
            pi_in = float(np.median(P[inside])) if inside.any() else np.nan
            pi_bk = float(np.median(P[far])) if far.any() else np.nan
            dets, cnts = np.unique(D[inside], return_counts=True)
            w.writerow([sat, r["start"], r["dur_us"], f"{t90:.0f}", f"{excess:.1f}", int(inside.sum()),
                        f"{rate:.0f}", f"{pi_in:.0f}", f"{pi_bk:.0f}",
                        f"{pi_in / pi_bk:.3f}" if pi_bk else "", len(dets),
                        f"{cnts.max() / inside.sum():.2f}" if inside.any() else ""])
            n_ok += 1; written = True
            break
    if not written:
        w.writerow([sat, r["start"], r["dur_us"]] + [""] * 9)
print("measured:", n_ok, "of", len(rows))
