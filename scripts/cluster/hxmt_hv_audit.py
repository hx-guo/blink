#!/usr/bin/env python3
"""批量查候选时刻的高压状态：机箱级不均匀到底是不是高压瞬变造成的。

对输入表的每一行，读 HE-HV 的 HV_PHODet_0..17（1 Hz），取候选前后 ±10 s，
给出：
  hv_now      候选那一秒 18 路高压的中位（绝对值）
  hv_ref      这一小时的中位（正常工作点）
  hv_drop     |hv_now| / |hv_ref|
  hv_slope    候选前后 ±3 s 内 |HV| 的最大逐秒变化（V/s）—— 瞬变的直接判据
  hv_spread   候选那一秒三个机箱各自 |HV| 中位的 max/min，机箱之间是否不同步

用法: hv_audit.py <in.csv> <out.csv>   （输入需要 start 列）
"""
import csv, os, sys, collections
from datetime import datetime, timedelta, timezone
import numpy as np
from astropy.io import fits

MET_EPOCH = datetime(2012, 1, 1, tzinfo=timezone.utc)
K1 = os.environ.get("HXMT_1K_DIR", "/hxmt/work/HXMT-DATA/1K")
LAUNCH = datetime(2017, 6, 15, tzinfo=timezone.utc)


def met_of(iso):
    iso = iso.rstrip("Z")
    head, _, frac = iso.partition(".")
    base = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (base - MET_EPOCH).total_seconds() + (float("0." + frac) if frac else 0.0)


def pick(folder_path, prefix):
    if not os.path.isdir(folder_path):
        return None
    best, bestv = None, -1
    for name in os.listdir(folder_path):
        if name.startswith(prefix):
            try:
                v = int(name[len(prefix):len(prefix) + 1])
            except ValueError:
                v = 0
            if v > bestv:
                best, bestv = name, v
    return os.path.join(folder_path, best) if best else None


def load(hour):
    num = (hour.replace(hour=0) - LAUNCH).days + 1
    f = "%s/Y%04d%02d/%04d%02d%02d-%04d" % (K1, hour.year, hour.month,
                                            hour.year, hour.month, hour.day, num)
    stem = "HXMT_%04d%02d%02dT%02d_HE-HV_FFFFFF_V" % (hour.year, hour.month, hour.day, hour.hour)
    p = pick(f, stem)
    if not p:
        return None
    try:
        with fits.open(p) as h:
            d = h["HE_HV_PHODet"].data
            t = np.array(d["Time"], float)
            v = np.abs(np.stack([np.array(d["HV_PHODet_%d" % i], float) for i in range(18)], 1))
    except Exception as exc:
        print("  bad %s: %s" % (p, exc), flush=True)
        return None
    return t, v


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    mets = [met_of(r["start"]) for r in rows]
    by_hour = collections.defaultdict(list)
    for i, m in enumerate(mets):
        by_hour[(MET_EPOCH + timedelta(seconds=m)).replace(minute=0, second=0, microsecond=0)].append(i)
    print("%d rows, %d hours" % (len(rows), len(by_hour)), flush=True)

    res = [None] * len(rows)
    for hour, items in sorted(by_hour.items()):
        data = load(hour)
        if data is None:
            continue
        t, v = data
        ref = np.median(v, 0)                      # 这一小时的工作点，逐路
        for i in items:
            m = mets[i]
            k = int(np.argmin(np.abs(t - m)))
            if abs(t[k] - m) > 2.0:
                continue
            now = v[k]
            lo, hi = max(0, k - 3), min(len(t), k + 4)
            seg = v[lo:hi]
            slope = np.abs(np.diff(seg, axis=0)).max() if len(seg) > 1 else 0.0
            box = np.array([np.median(now[0:6]), np.median(now[6:12]),
                            np.median(now[12:18])])
            res[i] = (np.median(now), np.median(ref),
                      np.median(now) / max(np.median(ref), 1e-9),
                      slope, box.max() / max(box.min(), 1e-9))

    hdr = list(rows[0].keys()) + ["hv_now", "hv_ref", "hv_drop", "hv_slope", "hv_spread"]
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r, x in zip(rows, res):
            w.writerow(list(r.values()) + ([""] * 5 if x is None else ["%.4f" % y for y in x]))
    print("resolved %d/%d -> %s" % (sum(x is not None for x in res), len(rows), sys.argv[2]))


if __name__ == "__main__":
    main()
