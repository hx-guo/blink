#!/usr/bin/env python3
"""Single-chunk HV extractor — used as a hep_sub worker.

Each job processes a small range of dates and writes its own CSV part.
No resume logic, no header line (concat handles header).

Usage:  python3 extract_hv_chunk.py START_DATE END_DATE OUT_CSV
"""
import sys
import os
import glob
import numpy as np
from astropy.io import fits
from datetime import date as Date, timedelta

ARCH_1K = "/hxmt/work/HXMT-DATA/1K"


def dirname_for(d_str):
    y, m, d = int(d_str[:4]), int(d_str[4:6]), int(d_str[6:8])
    delta = (Date(y, m, d) - Date(2017, 6, 15)).days + 1
    return f"{d_str}-{delta:04d}"


def date_range(start_str, end_str):
    s = Date(int(start_str[:4]), int(start_str[4:6]), int(start_str[6:8]))
    e = Date(int(end_str[:4]), int(end_str[4:6]), int(end_str[6:8]))
    cur = s
    while cur <= e:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def main():
    start = sys.argv[1]
    end   = sys.argv[2]
    out_path = sys.argv[3]
    fmt = "%s,%d" + ",%.1f" * 18 + "\n"
    n_files = n_rows = 0
    with open(out_path, "w", buffering=1) as f:
        for d_str in date_range(start, end):
            ym = d_str[:6]
            dirpath = f"{ARCH_1K}/Y{ym}/{dirname_for(d_str)}"
            if not os.path.isdir(dirpath):
                continue
            for fpath in sorted(glob.glob(f"{dirpath}/HXMT_{d_str}T*_HE-HV_FFFFFF_V1_1K.FITS")):
                try:
                    with fits.open(fpath, memmap=False) as fe:
                        d = fe["HE_HV_PHODet"].data
                        t = d["Time"].astype(np.int64)
                        cols = [d[f"HV_PHODet_{j}"].astype(np.float32) for j in range(18)]
                        for i in range(len(t)):
                            f.write(fmt % (d_str, int(t[i]),
                                            cols[0][i], cols[1][i], cols[2][i], cols[3][i],
                                            cols[4][i], cols[5][i], cols[6][i], cols[7][i],
                                            cols[8][i], cols[9][i], cols[10][i], cols[11][i],
                                            cols[12][i], cols[13][i], cols[14][i], cols[15][i],
                                            cols[16][i], cols[17][i]))
                        n_files += 1
                        n_rows += len(t)
                except Exception as e:
                    print(f"WARN {fpath}: {e}", file=sys.stderr)
    print(f"chunk {start}-{end}: {n_files} files, {n_rows} rows → {out_path}")


if __name__ == "__main__":
    main()
