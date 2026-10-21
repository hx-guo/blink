"""Day-level wrapper for hep_sub: run process_one_day for exactly one date.

Usage:  python3 build_clean_one_day.py YYYYMMDD

Writes clean_partials/year_{YYYY}/{YYYYMMDD}.parquet (idempotent).
Final per-year concat is a separate post-processing step (concat_clean_year.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `import build_clean_cache` from same dir
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".",)]  # sibling and shared script modules
from build_clean_cache import BurstCatalog, process_one_day  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2 or len(sys.argv[1]) != 8 or not sys.argv[1].isdigit():
        print("usage: build_clean_one_day.py YYYYMMDD", file=sys.stderr)
        return 2

    date = sys.argv[1]
    year = date[:4]
    partial_dir = Path("clean_partials") / f"year_{year}"
    partial_dir.mkdir(parents=True, exist_ok=True)

    cat = BurstCatalog.fetch_or_load(
        Path("n_below_study/gbm_triggers.parquet"),
        window_sec=300,
        allow_fetch=False,
    )
    result = process_one_day(
        date,
        Path("per_sec_parquet"),
        partial_dir,
        cat,
        relax=True,
    )
    print(f"{date}: {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
