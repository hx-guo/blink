#!/usr/bin/env python3
"""Convert the big HV gz CSV into per-date parquet partitions.

Memory-safe: streams chunked CSV reads, groups by date, writes one parquet
per date (one date per file → ~3300 files, ~1-2 MB each compressed).
No per-second resolution lost.

Input:  n_below_study/hv_table_full.csv.gz   (5.8 GB compressed)
Output: n_below_study/hv_by_date_parquet/YYYYMMDD.parquet  (~3300 files)
"""
from pathlib import Path
import time
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("n_below_study/hv_table_full.csv.gz")
OUT_DIR = Path("n_below_study/hv_by_date_parquet")

DTYPE = {"date":"string","met_sec":"int64",
         **{f"hv{i}":"float32" for i in range(18)}}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Converting {SRC} → {OUT_DIR}/  (per-date parquet partitions)")

    # Buffer per date: when we finish a date (chunk no longer contains rows
    # for that date), flush it. Since rows are sorted by date in the source,
    # we can detect date transitions.
    #
    # But our source HV CSV is concatenated from many per-chunk-job files,
    # NOT sorted by date globally — some dates may appear in non-contiguous
    # chunks. So we open a parquet writer per date and append.

    writers = {}        # date → ParquetWriter
    schemas = {}        # date → first schema seen
    n_chunks = 0
    n_rows_in = 0
    t0 = time.time()

    for chunk in pd.read_csv(SRC, chunksize=5_000_000, dtype=DTYPE):
        n_chunks += 1
        n_rows_in += len(chunk)
        # Group by date within chunk
        for date_str, sub in chunk.groupby("date", observed=True):
            sub = sub.drop(columns=["date"])
            table = pa.Table.from_pandas(sub, preserve_index=False)
            path = str(OUT_DIR / f"{date_str}.parquet")
            if date_str not in writers:
                writers[date_str] = pq.ParquetWriter(path, table.schema,
                                                       compression="zstd")
            writers[date_str].write_table(table)
        print(f"  chunk {n_chunks}: in={n_rows_in:,}  "
              f"open_writers={len(writers)}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        del chunk
        gc.collect()

    print(f"\nClosing {len(writers)} parquet writers...")
    for w in writers.values():
        w.close()

    sizes = sorted(p.stat().st_size for p in OUT_DIR.glob("*.parquet"))
    total = sum(sizes)
    print(f"Wrote {len(sizes)} parquet files, total {total/1e6:.1f} MB")
    print(f"  per-date size: min={sizes[0]/1024:.1f}KB  "
          f"median={sizes[len(sizes)//2]/1024:.1f}KB  "
          f"max={sizes[-1]/1024:.1f}KB")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
