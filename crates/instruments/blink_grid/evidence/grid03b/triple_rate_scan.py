"""逐过境扫 `frac_ev_in3`：找 2022-03-19 那种「大半事例落在同戳簇里」的坏过境。

2022-03-19T06:05–06:19 那次过境实测：keep 后 3362 c/s、三重及以上簇率 **599 个/s**、
四重簇 463 个/s、**64% 的事例落在 ≥3 重同戳簇里**。对照 38 次正常过境（速率 141–6318
c/s）：r₃ 平在 1.0–2.0 个/s、事例占比 8×10⁻⁴–1.7×10⁻²。**不是速率效应**——速率 5722
与 6318 c/s 的两次过境 r₃ 只有 1.97 和 1.69。

463 次四路同戳/秒穿过一颗立方星不合理，形态更像读出/解码把四路的时戳写成同一个值。
这次过境产出了一个 v11 的新显著候选（`2022-03-19T06:16:38`，count = 8、窗 8.3 µs、
四路同戳、f₃ 正好 0.500），而速率门 `RATE_CEILING = 5000 c/s` 放它过去了。

本脚本逐过境算两个量，好定一道与速率门互补的过境级门：
    r3           = ≥3 重同戳簇的个数 / 过境时长
    frac_ev_in3  = 落在 ≥3 重同戳簇里的事例数 / 事例数

只读 `TIME` / `PI` / `EVT_TYPE`，准入与 `Event::keep` 一致。

用法（农场 N 分片）：
    python3 triple_rate_scan.py <chunk> <nchunk> <outdir> [起始日 YYYY-MM-DD] [结束日]
"""

import glob
import os
import sys

import numpy as np
from astropy.io import fits

BASE = "/gecamfs/Exchange/GSDC/missions/GRID/GRID-03B/fits7"
ETH = 30.0
TICK = 2.0**22
TRIPLE = 3


def main():
    chunk, nchunk, outdir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    lo = sys.argv[4] if len(sys.argv) > 4 else "0000-00-00"
    hi = sys.argv[5] if len(sys.argv) > 5 else "9999-99-99"
    os.makedirs(outdir, exist_ok=True)
    days = sorted(glob.glob(BASE + "/*/*/*"))
    days = [d for d in days if lo <= "-".join(d.split("/")[-3:]) <= hi][chunk::nchunk]
    fh = open(os.path.join(outdir, "trip_%02d.csv" % chunk), "w")
    fh.write("day,pass_file,dur_s,n_kept,rate_cps,k3,k4,r3_per_s,frac_ev_in3,max_mult\n")
    for daydir in days:
        p = daydir.split("/")
        day = "%s-%s-%s" % (p[-3], p[-2], p[-1])
        vers = sorted(glob.glob(daydir + "/evt_v*"))
        if not vers:
            continue
        for path in sorted(glob.glob(vers[-1] + "/*.fits")):
            try:
                with fits.open(path, memmap=False) as hd:
                    g = hd["GTI"].data
                    gs = float(np.asarray(g["START"])[0])
                    ge = float(np.asarray(g["STOP"])[0])
                    emin = np.asarray(hd["EBOUNDS"].data["E_MIN"], dtype=np.float64)
                    nch = emin.size
                    T = []
                    for i in range(4):
                        d = hd["EVENTS%d" % i].data
                        t = np.asarray(d["TIME"], dtype=np.float64)
                        pi = np.asarray(d["PI"], dtype=np.int32)
                        ty = np.asarray(d["EVT_TYPE"], dtype=np.int32)
                        ok = (pi >= 1) & (pi < nch)
                        e = np.zeros(t.size)
                        e[ok] = emin[pi[ok] - 1]
                        T.append(t[(ty == 1) & ok & (e >= ETH)])
            except Exception as ex:
                fh.write("%s,%s,,,,,,,,ERR %s\n" % (day, os.path.basename(path), str(ex)[:40]))
                continue
            t = np.sort(np.concatenate(T))
            dur = max(ge - gs, 1e-9)
            if t.size < 100:
                continue
            tk = np.rint(t * TICK).astype(np.int64)
            edge = np.flatnonzero(np.diff(tk) != 0)
            st = np.concatenate(([0], edge + 1))
            en = np.concatenate((edge + 1, [tk.size]))
            sz = en - st
            big = sz >= TRIPLE
            k3 = int(big.sum())
            k4 = int((sz >= 4).sum())
            fh.write("%s,%s,%.1f,%d,%.1f,%d,%d,%.4f,%.5f,%d\n"
                     % (day, os.path.basename(path), dur, t.size, t.size / dur,
                        k3, k4, k3 / dur, sz[big].sum() / tk.size, int(sz.max())))
            fh.flush()
    fh.close()
    print("chunk %d done" % chunk)


if __name__ == "__main__":
    main()
