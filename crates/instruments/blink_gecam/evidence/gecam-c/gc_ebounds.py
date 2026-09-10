"""GECAM-C：把 PI 道号换成能量，并算已发表 TGF 目录用的那个硬度比口径。

`pi_ratio`（窗内中位道号 / 本底中位道号）是**道号之比**，而 GECAM 的能量梯是对数的
（每道约 1.34%），所以道号之比 1.27 对应的能量之比要大得多。拿它跟别的仪器（天格 3.0）
比是没有意义的——两台仪器的道刻度不同。这里用真实 EBOUNDS 做换算，并直接算
"200 keV 以上计数占比"，与已发表目录的 HardnessRatio_200keV 同口径。
"""

import csv
import datetime as dt
import glob
import json
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
EPOCH = (2021, 1, 1)


def met(iso):
    body = iso.rstrip("Z")
    head, _, frac = body.partition(".")
    stamp = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    ref = dt.datetime(*EPOCH, tzinfo=dt.timezone.utc)
    return (stamp - ref).total_seconds() + (float("0." + frac) if frac else 0.0)


def hour_file(iso):
    directory = f"{ROOT}/{iso[:10].replace('-', '/')}/GRD_EVT"
    stem = f"gcg_evt_{iso[2:4]}{iso[5:7]}{iso[8:10]}_{iso[11:13]}_v"
    files = sorted(glob.glob(f"{directory}/{stem}*.fits"))
    return files[-1] if files else None


signals = json.load(open(sys.argv[1]))
out = sys.argv[2]

# 先取一份 EBOUNDS
path = hour_file(signals[0]["start"])
with fits.open(path) as hdus:
    eb = hdus["EBOUNDS"].data
    emin = np.asarray(eb["E_MIN"], float)
    emax = np.asarray(eb["E_MAX"], float)
n = 448
centre = np.sqrt(emin[:n] * emax[:n])
print("EBOUNDS: ch0 %.2f–%.2f keV, ch54 %.2f–%.2f, ch447 %.1f–%.1f"
      % (emin[0], emax[0], emin[54], emax[54], emin[447], emax[447]))
ch200 = int(np.searchsorted(emin[:n], 200.0)) - 1
ch500 = int(np.searchsorted(emin[:n], 500.0)) - 1
print("200 keV 落在道 %d（%.1f–%.1f keV）；500 keV 落在道 %d（%.1f–%.1f keV）"
      % (ch200, emin[ch200], emax[ch200], ch500, emin[ch500], emax[ch500]))

writer = csv.writer(open(out, "w", newline=""))
writer.writerow(["start", "n_core", "e_med_win", "e_med_bkg", "e_ratio",
                 "frac_gt200_win", "frac_gt200_bkg", "hr200_win", "hr200_bkg",
                 "frac_gt500_win", "n_bkg"])

cache = {}
written = mismatch = 0
for signal in signals:
    iso = signal["start"]
    key = iso[:13]
    if key not in cache:
        cache.clear()
        p = hour_file(iso)
        if p is None:
            cache[key] = None
        else:
            times, chans = [], []
            with fits.open(p, memmap=True) as hdus:
                for hdu in hdus:
                    if not hdu.name.startswith("EVENTS"):
                        continue
                    d = hdu.data
                    pi = np.asarray(d["PI"])
                    keep = (np.asarray(d["EVT_TYPE"]) == 1) & (pi >= 54) & (pi < 448)
                    times.append(np.asarray(d["TIME"], float)[keep])
                    chans.append(pi[keep])
            t = np.concatenate(times)
            o = np.argsort(t, kind="stable")
            cache[key] = (t[o], np.concatenate(chans)[o])
        print("  载入", key, flush=True)
    got = cache[key]
    if got is None:
        continue
    time, chan = got
    t0 = met(iso) + signal["delay"]
    t1 = t0 + signal["bin_size_best"]
    lo, hi = np.searchsorted(time, t0, "left"), np.searchsorted(time, t1, "right")
    if hi - lo == 0:
        continue
    if hi - lo != signal["count"]:
        mismatch += 1
    win = chan[lo:hi].astype(int)
    a = np.searchsorted(time, t0 - 1.0, "left")
    b = np.searchsorted(time, t1 + 1.0, "right")
    near = np.concatenate([chan[a:lo], chan[hi:b]]).astype(int)
    ew, eb_ = centre[win], centre[near] if near.size else np.array([np.nan])
    fw = (win >= ch200).mean()
    fb = (near >= ch200).mean() if near.size else np.nan
    writer.writerow([iso[:23], hi - lo, "%.1f" % np.median(ew), "%.1f" % np.median(eb_),
                     "%.3f" % (np.median(ew) / np.median(eb_)) if near.size else "",
                     "%.4f" % fw, "%.4f" % fb,
                     "%.4f" % (fw / (1 - fw)) if fw < 1 else "inf",
                     "%.4f" % (fb / (1 - fb)) if near.size and fb < 1 else "",
                     "%.4f" % (win >= ch500).mean(), near.size])
    written += 1

print("写出 %d；n_core 与 count 对不上 %d" % (written, mismatch))
