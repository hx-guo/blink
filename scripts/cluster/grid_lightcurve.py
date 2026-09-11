"""导出候选前后 ±50 ms 的逐事例（时间/能道/探头），供本地画光变。

事例准入与搜索一致（`blink_grid::types::Event::keep`）：EVT_TYPE==1、道号在
[1, n_channels) 内（最高道是溢出道）、且该道下限能量 >= 30 keV。只筛 EVT_TYPE 会把
溢出道和 30 keV 以下的道也画进去，逐事例能量图会多出一层假的低能本底。

同时把该星的 EBOUNDS 写成 `ebounds_<卫星>.csv`（道号、E_MIN、E_MAX），画图时用它把
道号换成能量——四颗星的道—能对应各不相同，不能共用一张表。

用法: python3 grid_lightcurve.py <候选 CSV: sat,start,...> <输出目录>"""
from astropy.io import fits
import glob, os, csv, sys, numpy as np, datetime as dt

G = "/gecamfs/Exchange/GSDC/missions/GRID"
REF = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc)
HALF = 0.05  # 导出半宽 50 ms
BKG = 1.0    # 本底窗半宽 1 s
ENERGY_THRESHOLD_KEV = 30.0   # 与 blink_grid::types::event::ENERGY_THRESHOLD_KEV 同步


def met(iso):
    """ISO 时刻 → MET 秒。小数秒不能截到微秒：搜索产物的时刻带纳秒，候选窗的
    两端都是事例本身的时刻，截断把窗口整体左移不到 1 µs 就足以把落在窗末端的
    那个事例（连同与它同戳的几个）挤出窗外。整秒交给 datetime，小数部分按
    浮点单独加。"""
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (stamp - REF).total_seconds() + (float("0." + frac) if frac else 0.0)


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
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

for r in rows:
    sat, t0 = r["sat"], met(r["start"])
    tag = f"{sat}_{r['start'][:23].replace(':', '').replace('-', '').replace('.', '')}"
    done = False
    for f in pass_files(sat, t0):
        with fits.open(f) as h:
            g = h["GTI"].data
            s0, s1 = float(g["START"][0]), float(g["STOP"][0])
            if not (s0 <= t0 <= s1):
                continue
            eb = h["EBOUNDS"].data
            emin = np.asarray(eb["E_MIN"], float)
            emax = np.asarray(eb["E_MAX"], float)
            n_ch = len(emin)
            with open(f"{outdir}/ebounds_{sat}.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["ch", "e_min", "e_max"])
                for i in range(n_ch):
                    wr.writerow([i + 1, f"{emin[i]:.4f}", f"{emax[i]:.4f}"])
            T, P, D = [], [], []
            for k in range(4):
                x = h[f"EVENTS{k}"].data
                t = np.asarray(x["TIME"], float)
                pi = np.asarray(x["PI"])
                # 与搜索同一道准入：非溢出道 + 道下限能量 >= 30 keV
                ok = (pi >= 1) & (pi < n_ch)
                good = np.zeros(len(pi), bool)
                good[ok] = emin[pi[ok] - 1] >= ENERGY_THRESHOLD_KEV
                m = (np.asarray(x["EVT_TYPE"]) == 1) & good
                keep = m & (t >= t0 - BKG) & (t <= t0 + BKG)
                T.append(t[keep])
                P.append(pi[keep])
                D.append(np.full(int(keep.sum()), k))
            T = np.concatenate(T); P = np.concatenate(P); D = np.concatenate(D)
            o = np.argsort(T); T, P, D = T[o], P[o], D[o]
            # 本底谱/本底率用整个 ±1 s（GTI 内），光变只导出 ±50 ms
            live = min(t0 + BKG, s1) - max(t0 - BKG, s0)
            near = np.abs(T - t0) <= HALF
            with open(f"{outdir}/{tag}.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["dt_ms", "pi", "det"])
                for tt, pp, dd_ in zip(T[near], P[near], D[near]):
                    wr.writerow([f"{(tt - t0) * 1e3:.6f}", int(pp), int(dd_)])
            far = np.abs(T - t0) > 0.005
            with open(f"{outdir}/{tag}_bkg.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["live_s", "n_bkg", "pi"])
                wr.writerow([f"{live:.3f}", int(far.sum()), ""])
                for pp in P[far]:
                    wr.writerow(["", "", int(pp)])
            print(tag, "events", int(near.sum()), "bkg_rate", f"{far.sum() / live:.1f}")
            done = True
            break
    if not done:
        print(tag, "NOT FOUND")
