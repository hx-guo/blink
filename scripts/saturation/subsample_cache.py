#!/usr/bin/env python3
"""Uniformly subsample a parquet cache via streaming (memory-safe for large caches).

For caches too big to fit in vmem (e.g. server with ulimit -v 48GB and a 12 GB cache),
this streams batches through and writes a downsampled cache without loading everything.

Per (box, det) we get ~1M rows at 5% (from 414M total) — plenty for per-det stats and
density scatter (which are visually saturated at much smaller N anyway).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input parquet path")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument("--fraction", type=float, default=0.05, help="Sample fraction (default 0.05)")
    p.add_argument("--batch-size", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(input_path)
    rng = np.random.RandomState(args.seed)
    writer = None
    total_in = 0
    total_out = 0

    print(f"Streaming {input_path} → {output_path}  (sample fraction = {args.fraction})")
    for i, batch in enumerate(pf.iter_batches(batch_size=args.batch_size)):
        n = batch.num_rows
        total_in += n
        keep = rng.random(n) < args.fraction
        sub = batch.filter(pa.array(keep))
        if writer is None:
            writer = pq.ParquetWriter(output_path, sub.schema, compression="zstd")
        writer.write_batch(sub)
        total_out += sub.num_rows
        if i % 50 == 0:
            print(f"  batch {i+1}: in={total_in:,}  kept={total_out:,}  ratio={total_out/max(total_in,1):.3%}")

    if writer:
        writer.close()
    size_mb = output_path.stat().st_size / 1e6
    print(f"Done. Input rows: {total_in:,}  Output rows: {total_out:,}  ({total_out/total_in:.3%})  size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
