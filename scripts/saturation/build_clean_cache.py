#!/usr/bin/env python3
"""Build clean PHO-verification cache from per_sec_parquet.

See docs/superpowers/specs/2026-05-20-clean-pho-cache-design.md
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ============================================================
# MET conversion (GBM ISO time -> HXMT MET)
# ============================================================

# HXMT MET epoch: 2012-01-01T00:00:00 UTC. We compute the offset once via astropy
# so leap-seconds are handled correctly.
_HXMT_EPOCH_ISO = "2012-01-01T00:00:00"


def gbm_iso_to_hxmt_met(iso_string: str) -> int:
    """Convert a GBM trigger time (ISO 8601 UTC) to HXMT MET seconds (int)."""
    from astropy.time import Time
    t = Time(iso_string, format="isot", scale="utc")
    epoch = Time(_HXMT_EPOCH_ISO, format="isot", scale="utc")
    return int(round((t - epoch).sec))


def _fetch_gbm_triggers_from_heasarc():
    """Query HEASARC fermigtrig via ADQL, return DataFrame with 'trigger_met_hxmt'.

    Network call; runs on a node with HTTPS access to HEASARC. Uses astroquery's
    TAP/ADQL path (query_region is broken in 0.4.11 — auto-generated query is rejected
    by HEASARC TAP). The fermigtrig table exposes trigger_time as MJD (UTC, float).
    """
    import pandas as pd
    from astroquery.heasarc import Heasarc
    from astropy.time import Time

    heasarc = Heasarc()
    tab = heasarc.query_tap(
        query="SELECT trigger_name, trigger_time FROM fermigtrig"
    ).to_table()
    # trigger_time is MJD (UTC). Vectorised astropy conversion to HXMT MET.
    mjd = tab["trigger_time"].data  # float64 array
    t = Time(mjd, format="mjd", scale="utc")
    epoch = Time(_HXMT_EPOCH_ISO, format="isot", scale="utc")
    metsec = ((t - epoch).sec).round().astype("int64")
    return pd.DataFrame({"trigger_met_hxmt": metsec})


# ============================================================
# Module 1: BurstCatalog
# ============================================================


@dataclass
class BurstCatalog:
    """In-memory store of burst trigger MET-seconds, with ±window membership test."""

    triggers_met: np.ndarray   # int64, sorted ascending
    window_sec: int            # half-width of exclusion zone (e.g. 300 = ±5 min)

    @classmethod
    def from_array(cls, met_array: np.ndarray, window_sec: int) -> "BurstCatalog":
        sorted_met = np.sort(met_array.astype(np.int64))
        return cls(triggers_met=sorted_met, window_sec=int(window_sec))

    @classmethod
    def fetch_or_load(cls, cache_path, window_sec: int, allow_fetch: bool = True) -> "BurstCatalog":
        """Load triggers from a cached parquet file. If missing and allow_fetch=True,
        fetch from HEASARC and write the cache first.

        cache_path: pathlib.Path to a parquet file with one int64 column 'trigger_met_hxmt'.
        """
        import pandas as pd
        from pathlib import Path
        cache_path = Path(cache_path)

        if not cache_path.exists():
            if not allow_fetch:
                raise FileNotFoundError(
                    f"GBM trigger cache not found: {cache_path}. "
                    f"Re-run with allow_fetch=True (must have HEASARC HTTPS access)."
                )
            df = _fetch_gbm_triggers_from_heasarc()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, compression="zstd")

        df = pd.read_parquet(cache_path)
        arr = df["trigger_met_hxmt"].to_numpy().astype(np.int64)
        return cls.from_array(arr, window_sec=window_sec)

    def any_within(self, query_met: np.ndarray) -> np.ndarray:
        q = np.asarray(query_met, dtype=np.int64)
        if self.triggers_met.size == 0:
            return np.zeros(q.size, dtype=bool)
        idx = np.searchsorted(self.triggers_met, q)
        left = np.clip(idx - 1, 0, self.triggers_met.size - 1)
        right = np.clip(idx, 0, self.triggers_met.size - 1)
        dist_left = np.abs(q - self.triggers_met[left])
        dist_right = np.abs(q - self.triggers_met[right])
        return (dist_left <= self.window_sec) | (dist_right <= self.window_sec)


# ============================================================
# Module 2: apply_filters (Stages 1-5)
# ============================================================

# Constants from spec
L_CYCLES_MIN = 50_000           # Stage 1: livetime > 0.8s
HV_LO, HV_HI = -1100.0, -900.0  # Stage 1: detector operating range (exclusive both sides)
LAT_MAX_ABS = 3.0               # Stage 3: equatorial belt half-width
# HXMT 1K Orbit Lon is [0, 360). SAA box [-90, +30] in [-180, +180] convention
# maps to [270, 360) ∪ [0, 30]. Keep window is the complement: (30, 270).
SAA_LON_KEEP_LO, SAA_LON_KEEP_HI = 30.0, 270.0

_RAW_COUNTERS = [
    "PHO", "OOC", "Wide", "Large", "Dt",
    "Sci_094", "Sci_pure_094", "Sci_ACD1_094", "Sci_ACDN_094",
    "Sci_1s", "Sci_pure_1s", "Sci_ACD1_1s", "Sci_ACDN_1s",
]


def _apply_stage1_detector_state(df):
    """Keep rows with L_cycles > 50_000 AND HV strictly inside (-1100, -900)."""
    mask = (df["L_cycles"] > L_CYCLES_MIN) & (df["HV"] > HV_LO) & (df["HV"] < HV_HI)
    return df.loc[mask].copy()


def _apply_stage2_integrity(df):
    """Drop NaN in HV/Lat/Lon, negative counters, Sci-partition violations."""
    # NaN check
    mask = df[["HV", "Lat", "Lon"]].notna().all(axis=1)
    # Non-negative counters
    for c in _RAW_COUNTERS:
        mask &= (df[c] >= 0)
    # Sci breakdown invariant for both windows
    for w in ("094", "1s"):
        lhs = df[f"Sci_pure_{w}"] + df[f"Sci_ACD1_{w}"] + df[f"Sci_ACDN_{w}"]
        mask &= (lhs == df[f"Sci_{w}"])
    return df.loc[mask].copy()


def _apply_stage3_spatial(df):
    """Keep rows in equatorial belt AND outside SAA Lon box (Lon in (30, 270))."""
    mask = (df["Lat"].abs() < LAT_MAX_ABS) & (df["Lon"] > SAA_LON_KEEP_LO) & (df["Lon"] < SAA_LON_KEEP_HI)
    return df.loc[mask].copy()


def _apply_stage4_burst(df, burst_catalog):
    """Drop rows whose met_sec is within ±window_sec of any GBM trigger."""
    times = df["met_sec"].to_numpy().astype(np.int64)
    drop_mask = burst_catalog.any_within(times)
    return df.loc[~drop_mask].copy()


def _apply_stage5_completeness(df):
    """Keep only (date, met_sec) groups where all 18 (3 boxes × 6 dets) rows are present."""
    counts = df.groupby(["date", "met_sec"]).size()
    full_keys = counts[counts == 18].index
    df_idx = df.set_index(["date", "met_sec"])
    keep = df_idx.index.isin(full_keys)
    return df_idx.loc[keep].reset_index()


def apply_filters(df, burst_catalog):
    """Run all 5 filter stages in order; return (filtered_df, counts_dict).

    counts_dict keys: start, after_stage1, after_stage2, after_stage3, after_stage4,
    after_stage5 — for logging row counts at each stage.
    """
    counts = {"start": len(df)}
    df = _apply_stage1_detector_state(df)
    counts["after_stage1"] = len(df)
    df = _apply_stage2_integrity(df)
    counts["after_stage2"] = len(df)
    df = _apply_stage3_spatial(df)
    counts["after_stage3"] = len(df)
    df = _apply_stage4_burst(df, burst_catalog)
    counts["after_stage4"] = len(df)
    df = _apply_stage5_completeness(df)
    counts["after_stage5"] = len(df)
    return df, counts


# ============================================================
# Module 3: (no derivations — cache is raw counts only)
# ============================================================
#
# By design, this cache stores ONLY raw counts and instantaneous samples.
# All normalisations (rates, dead-time corrections, sums like Sci_ACD =
# Sci_ACD1 + Sci_ACDN) are downstream concerns.
#
# Convenient downstream conventions (use per-row `length`, NOT hardcoded 0.94):
#   length    = L_cycles × 16e-6   (engineering cycle wallclock per row; ≈0.94s nominal but varies ±0.7%; NOT livetime)
#   dt_frac   = Dt / L_cycles      (dead-time fraction within the cycle)
#   live_frac = 1 - dt_frac
#   pho_rate  = PHO / length       (events / 1s wallclock; same for OOC/Wide/Large)
#   sci_rate_094 = Sci_094 / length  (events / 1s wallclock from 0.94s event window)
#   sci_rate_1s  = Sci_1s  / 1.0   (events / 1s wallclock from 1s event window; the 1s
#                                    window is fixed by extract spec, not cycle-aligned)
#
# Dead-time correction (PHO/Large unaffected; Sci/Wide affected):
#   To compare front-end vs eventizer counts on the same livetime footing,
#   scale PHO and Large by live_frac, then compare to Sci + Wide.


# ============================================================
# Module 4: process_one_day
# ============================================================

def process_one_day(date_str, input_dir, partial_dir, burst_catalog):
    """Load one day's per_sec_parquet, run filters + derives, write a partial.

    Args:
        date_str: 'YYYYMMDD'
        input_dir: Path to directory containing {date_str}.parquet
        partial_dir: Path where {date_str}.parquet partial is written
        burst_catalog: BurstCatalog instance

    Returns: Path to written partial, or None if no rows survived / input missing.
    """
    from pathlib import Path
    import pandas as pd

    input_dir = Path(input_dir)
    partial_dir = Path(partial_dir)
    in_path = input_dir / f"{date_str}.parquet"

    if not in_path.exists():
        print(f"[process_one_day] {date_str}: input parquet missing, skipping")
        return None

    df = pd.read_parquet(in_path)
    df, counts = apply_filters(df, burst_catalog)
    print(f"[process_one_day] {date_str}: {counts}")

    if len(df) == 0:
        return None

    out_path = partial_dir / f"{date_str}.parquet"
    df.to_parquet(out_path, compression="zstd")
    return out_path


# ============================================================
# Module 5: run_build (main pipeline) + CLI
# ============================================================

from functools import partial as _functools_partial


def _process_one_day_for_pool(date_str, input_dir, partial_dir, gbm_cache, window_sec):
    """Top-level wrapper for multiprocessing.Pool.

    Re-loads BurstCatalog from cache in each worker (fast, ~ms; avoids serialising
    a 3000+ entry array per worker invocation)."""
    cat = BurstCatalog.fetch_or_load(gbm_cache, window_sec=window_sec, allow_fetch=False)
    try:
        return process_one_day(date_str, input_dir, partial_dir, cat)
    except Exception as exc:
        print(f"[worker] {date_str}: FAILED with {type(exc).__name__}: {exc}")
        raise


def run_build(input_dir, output, partial_dir, gbm_cache, dates, workers=8, min_rows=1_000_000):
    """Top-level pipeline: per-day fan-out via multiprocessing.Pool, then concat + assertions."""
    from pathlib import Path
    from multiprocessing import Pool
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_dir = Path(input_dir)
    output = Path(output)
    partial_dir = Path(partial_dir)
    gbm_cache = Path(gbm_cache)
    partial_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    worker = _functools_partial(
        _process_one_day_for_pool,
        input_dir=input_dir,
        partial_dir=partial_dir,
        gbm_cache=gbm_cache,
        window_sec=300,
    )

    print(f"Starting pool with {workers} workers over {len(dates)} dates")
    if workers == 1:
        partials = [worker(d) for d in dates]
    else:
        with Pool(processes=workers) as pool:
            partials = pool.map(worker, dates)

    written = [p for p in partials if p is not None]
    print(f"Day-level done: {len(written)}/{len(dates)} days wrote partials")

    if not written:
        raise AssertionError("No partials written — entire run produced zero rows")

    print("Concatenating partials...")
    tables = [pq.read_table(p) for p in written]
    combined = pa.concat_tables(tables)
    tmp_out = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(combined, tmp_out, compression="zstd")

    n_rows = combined.num_rows
    print(f"Concatenated {n_rows:,} rows")

    df = combined.to_pandas()
    assert n_rows >= min_rows, f"Final row count {n_rows:,} below minimum {min_rows:,}"
    assert (df["Lat"].abs() < LAT_MAX_ABS).all(), "Lat assertion failed"
    assert ((df["Lon"] > SAA_LON_KEEP_LO) & (df["Lon"] < SAA_LON_KEEP_HI)).all(), "Lon assertion failed"
    for w in ("094", "1s"):
        lhs = df[f"Sci_pure_{w}"] + df[f"Sci_ACD1_{w}"] + df[f"Sci_ACDN_{w}"]
        assert (lhs == df[f"Sci_{w}"]).all(), f"Sci invariant failed for window {w}"

    tmp_out.rename(output)
    print(f"Final cache written: {output} ({output.stat().st_size / 1e6:.1f} MB)")
    for p in written:
        p.unlink()
    print(f"Partial files cleaned up")
    return output


def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Dir of per_sec_parquet/{YYYYMMDD}.parquet")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument("--partial-dir", required=True, help="Scratch dir for per-day partials")
    p.add_argument("--gbm-cache", required=True, help="Path to GBM trigger parquet")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2020-06-30")
    p.add_argument("--allow-fetch", action="store_true",
                    help="If GBM cache missing, fetch from HEASARC (needs HTTPS)")
    return p.parse_args()


def main():
    args = _parse_args()
    from pathlib import Path
    gbm_cache = Path(args.gbm_cache)

    # Pre-fetch the GBM cache up front (before forking workers) if requested
    if args.allow_fetch and not gbm_cache.exists():
        print("Fetching GBM triggers from HEASARC...")
        BurstCatalog.fetch_or_load(gbm_cache, window_sec=300, allow_fetch=True)
        print(f"  cached: {gbm_cache}")
    elif not gbm_cache.exists():
        raise FileNotFoundError(
            f"GBM cache missing ({gbm_cache}); pass --allow-fetch to download from HEASARC"
        )

    from datetime import date, timedelta
    d_start = date.fromisoformat(args.start)
    d_end = date.fromisoformat(args.end)
    dates = []
    d = d_start
    while d <= d_end:
        dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    run_build(
        input_dir=args.input_dir,
        output=args.output,
        partial_dir=args.partial_dir,
        gbm_cache=gbm_cache,
        dates=dates,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
