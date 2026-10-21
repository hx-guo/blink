#!/usr/bin/env python3
"""Extract HE PMT HV per-(date, sec, det) from 1K HE-HV FITS archive.
Run this ON THE IHEP SERVER (lxlogin).

Usage:
    python3 extract_hv_server.py [start_date] [end_date]
    e.g.   python3 extract_hv_server.py 20170102 20260517

Output: hv_table_full.csv.gz   (same format as hv_table_partial.csv.gz:
        columns date, met_sec, hv0, hv1, ..., hv17)
"""
from pathlib import Path
import glob
import sys
import gzip
import os
import time
import numpy as np
from astropy.io import fits
from datetime import date as Date, timedelta

ARCH_1K = "/hxmt/work/HXMT-DATA/1K"
OUT_TXT = "hv_table_full.csv"      # plain CSV during extraction (append-safe)
OUT     = "hv_table_full.csv.gz"   # gzipped at end


def dirname_for(d_str: str) -> str:
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
    start = sys.argv[1] if len(sys.argv) > 1 else "20170102"
    end   = sys.argv[2] if len(sys.argv) > 2 else "20260517"
    # Resume from plain CSV (append-safe, line-atomic, last partial line dropped)
    mode = "w"
    if os.path.exists(OUT_TXT) and os.path.getsize(OUT_TXT) > 100:
        import subprocess
        try:
            # tail -1 of plain CSV; if truncated, we'll drop the partial line
            last_line = subprocess.check_output(
                ["bash", "-c", f"tail -1 {OUT_TXT}"]).decode().strip()
            parts = last_line.split(",")
            # Validate: must be 20 fields (date + met_sec + 18 hv), date must parse
            if len(parts) == 20 and not last_line.startswith("date"):
                last_date_str = parts[0]
                _ = Date(int(last_date_str[:4]), int(last_date_str[4:6]),
                          int(last_date_str[6:8]))   # raises if invalid
                start_d = Date(int(last_date_str[:4]), int(last_date_str[4:6]),
                                int(last_date_str[6:8])) + timedelta(days=1)
                start = start_d.strftime("%Y%m%d")
                mode = "a"
                # Truncate any partial last line — keep only up to last newline
                with open(OUT_TXT, "rb+") as fh:
                    fh.seek(-min(4096, os.path.getsize(OUT_TXT)), 2)
                    tail = fh.read()
                    last_nl = tail.rfind(b"\n")
                    if last_nl >= 0:
                        fh.seek(-(len(tail) - last_nl - 1), 2)
                        fh.truncate()
                print(f"RESUMING: last date in {OUT_TXT} = {last_date_str}; "
                      f"continuing from {start}", flush=True)
        except Exception as e:
            print(f"Resume check failed ({e}); starting from {start}", flush=True)

    dates = list(date_range(start, end))
    print(f"Extracting HV for {len(dates)} dates: {start} → {end}  "
          f"(mode={mode})", flush=True)

    header = "date,met_sec," + ",".join(f"hv{i}" for i in range(18)) + "\n"
    fmt = "%s,%d" + ",%.1f" * 18 + "\n"

    t0 = time.time()
    n_files = n_rows = n_missing_dir = n_missing_files = 0
    with open(OUT_TXT, mode, buffering=1) as f:        # line-buffered
        if mode == "w":
            f.write(header)
        for di, d_str in enumerate(dates):
            ym = d_str[:6]
            dirname = dirname_for(d_str)
            dirpath = f"{ARCH_1K}/Y{ym}/{dirname}"
            if not os.path.isdir(dirpath):
                n_missing_dir += 1
                continue
            pattern = f"{dirpath}/HXMT_{d_str}T*_HE-HV_FFFFFF_V1_1K.FITS"
            files = sorted(glob.glob(pattern))
            if not files:
                n_missing_files += 1
                continue
            for fpath in files:
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
                        n_rows += len(t)
                        n_files += 1
                except Exception as e:
                    print(f"  [WARN] {fpath}: {e}", file=sys.stderr, flush=True)
            if (di + 1) % 100 == 0 or (di + 1) == len(dates):
                elapsed = time.time() - t0
                rate = (di + 1) / elapsed
                eta = (len(dates) - (di + 1)) / rate
                print(f"  {di+1}/{len(dates)}  "
                      f"files={n_files} rows={n_rows:,}  "
                      f"missing dir={n_missing_dir} no-files={n_missing_files}  "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                      flush=True)

    # Gzip the final plain CSV
    print(f"\nExtraction loop done. Gzipping...", flush=True)
    import subprocess
    subprocess.run(["gzip", "-6", "-f", OUT_TXT], check=True)
    sz = os.path.getsize(OUT) / 1e6
    print(f"Done. {n_files} files, {n_rows:,} rows → {OUT} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
