"""Concatenate clean_partials/year_{YYYY}/*.parquet → clean_relaxed/clean_relaxed_{YYYY}.parquet.

Usage:  python3 concat_clean_year.py YYYY [YYYY ...]

Streams via ParquetWriter to avoid loading entire year into memory.
Idempotent: if final exists and is non-empty, skip the year.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pyarrow.parquet as pq


def concat_year(year: str) -> None:
    partial_dir = Path("clean_partials") / f"year_{year}"
    out = Path("clean_relaxed") / f"clean_relaxed_{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        print(f"{year}: skip (output exists, {out.stat().st_size/1e6:.1f} MB)")
        return

    files = sorted(glob.glob(str(partial_dir / "*.parquet")))
    if not files:
        print(f"{year}: no partials found in {partial_dir}")
        return

    schema = pq.read_schema(files[0])
    n_rows = 0
    tmp = out.with_suffix(out.suffix + ".tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for f in files:
            t = pq.read_table(f)
            if t.schema != schema:
                t = t.cast(schema)
            w.write_table(t)
            n_rows += t.num_rows
    tmp.rename(out)
    print(f"{year}: {len(files)} partials → {n_rows:,} rows, {out.stat().st_size/1e6:.1f} MB")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: concat_clean_year.py YYYY [YYYY ...]", file=sys.stderr)
        return 2
    for year in sys.argv[1:]:
        concat_year(year)
    return 0


if __name__ == "__main__":
    sys.exit(main())
