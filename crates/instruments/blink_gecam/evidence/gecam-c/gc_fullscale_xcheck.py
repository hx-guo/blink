"""GECAM-C：EBOUNDS 溢出段每一行的 `E_MIN` 就是那条 (探头, 增益) 支路的满量程能量。

两条互相独立的路子对同一个量：

* **声明值**——EBOUNDS 道 448 及以上每行一个 (探头, 增益)，`E_MAX` 一律 20000（哨兵），
  而 `E_MIN` 是一个几百 keV 到几 MeV 的具体数。把它钉到 (探头, 增益) 不靠猜：事例表里
  `EVT_TYPE == 2` 的事例，其 PI **就是该支路的那一行行号**（逐路逐档只有一个取值）。
* **实测值**——从 `EVT_TYPE == 1` 的 PI 直方图求最后一个 ≥10 计数的道（`edge`），
  再用道 0..447 那把梯子折成能量。

两者若吻合，「满量程道」这件事就不必查 CALDB：**EBOUNDS 自己带着这张表，而且逐小时
随文件走，标定再更新一次也自动跟上**。

用法: gc_fullscale_xcheck.py [<YYYY/MM/DD> <hour> ...]
"""

import glob
import sys

import numpy as np
from astropy.io import fits

ROOT = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily"
LADDER = 448
MIN_COUNTS = 10

EPOCHS = [("2023/06/15", 12), ("2024/03/10", 2), ("2025/02/05", 2)]


def run(day, hour):
    path = sorted(glob.glob(f"{ROOT}/{day}/GRD_EVT/*_{hour:02d}_v*.fits"))[-1]
    with fits.open(path, memmap=True) as hdus:
        eb = hdus["EBOUNDS"].data
        lo = np.asarray(eb["E_MIN"], float)
        hi = np.asarray(eb["E_MAX"], float)
        centre = np.sqrt(lo[:LADDER] * hi[:LADDER])

        sentinel, edges = {}, {}
        for hdu in hdus:
            if not hdu.name.startswith("EVENTS"):
                continue
            data = hdu.data
            if data is None or len(data) == 0:
                continue
            det = int(hdu.name[-2:])
            pi = np.asarray(data["PI"]).astype(int)
            gain = np.asarray(data["GAIN_TYPE"]).astype(int)
            evt = np.asarray(data["EVT_TYPE"]).astype(int)
            for g in (0, 1):
                over = (gain == g) & (evt == 2)
                good = (gain == g) & (evt == 1)
                if over.sum() > 100:
                    sentinel[(det, g)] = (np.unique(pi[over]), int(over.sum()))
                if good.sum() > 1000:
                    hist = np.bincount(np.clip(pi[good], 0, 511), minlength=512)
                    nz = np.flatnonzero(hist >= MIN_COUNTS)
                    edges[(det, g)] = int(nz[-1]) if nz.size else -1

    name = path.rsplit("/", 1)[-1]
    print(f"\n=== {day} {hour:02d}h  {name} ===")
    print("探头 档  哨兵道  唯一?   声明满量程 keV   实测 edge 道   实测 keV   实测/声明")
    declared, measured, gains = [], [], []
    for key in sorted(sentinel):
        det, g = key
        values, _ = sentinel[key]
        channel = int(values[0])
        d_kev = lo[channel]
        edge = edges.get(key, -1)
        e_kev = centre[edge] if 0 <= edge < LADDER else float("nan")
        declared.append(d_kev)
        measured.append(e_kev)
        gains.append(g)
        print(
            f"d{det:02d} g{g}   ch{channel:3d}   {str(len(values) == 1):5s}   "
            f"{d_kev:12.1f}   {edge:11d}   {e_kev:9.1f}   {e_kev / d_kev:9.3f}"
        )

    declared = np.array(declared)
    measured = np.array(measured)
    gains = np.array(gains)
    for tag, g in (("高增益", 0), ("低增益", 1)):
        sel = (gains == g) & np.isfinite(measured)
        if sel.sum() < 4:
            continue
        r = np.corrcoef(declared[sel], measured[sel])[0, 1]
        ratio = measured[sel] / declared[sel]
        print(
            f"  {tag}: n={int(sel.sum())}  r={r:+.4f}  "
            f"实测/声明 中位 {np.median(ratio):.3f}  极差 {np.ptp(ratio):.3f}"
        )


def main():
    if len(sys.argv) > 1:
        pairs = [(sys.argv[i], int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)]
    else:
        pairs = EPOCHS
    for day, hour in pairs:
        run(day, hour)


if __name__ == "__main__":
    main()
