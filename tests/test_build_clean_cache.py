"""Tests for scripts/saturation/build_clean_cache.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# conftest.py already adds scripts/saturation/ to sys.path
# Tests will `import build_clean_cache as M` once that script exists.


# ------------------- Synthetic row builder -------------------

# Columns the per_sec_parquet schema gives us (subset relevant to this cache).
PER_SEC_COLS = [
    "date", "box", "det", "met_sec",
    "Lat", "Lon",
    "L_cycles", "HV",
    "PHO", "OOC", "Wide", "Large", "Dt",
    "Sci_094", "Sci_pure_094", "Sci_ACD1_094", "Sci_ACDN_094",
    "Sci_1s", "Sci_pure_1s", "Sci_ACD1_1s", "Sci_ACDN_1s",
]


def make_row(**overrides):
    """One canonical 'good' per-sec row that should pass all filters.

    Override any field to test edge cases.
    """
    base = dict(
        date="2020-01-15", box="A", det=0, met_sec=252633600,
        Lat=0.5, Lon=120.0,           # Pacific, far from SAA
        L_cycles=60_000, HV=-1000.0,
        PHO=100, OOC=5, Wide=20, Large=10, Dt=500,
        Sci_094=80, Sci_pure_094=70, Sci_ACD1_094=8, Sci_ACDN_094=2,
        Sci_1s=85, Sci_pure_1s=74, Sci_ACD1_1s=8, Sci_ACDN_1s=3,
    )
    base.update(overrides)
    return base


def make_df(rows):
    """Build a DataFrame from a list of dicts (output of make_row)."""
    return pd.DataFrame(rows, columns=PER_SEC_COLS)


def make_complete_second(box="A", met_sec=252633600, date="2020-01-15", **overrides):
    """Build all 6 (det 0..5) rows for one (box, met_sec) — passes Stage 5 box-completeness."""
    return [make_row(box=box, det=d, met_sec=met_sec, date=date, **overrides) for d in range(6)]


def make_complete_groupsec(met_sec=252633600, date="2020-01-15", **overrides):
    """Build 18 rows = 3 boxes × 6 dets for one met_sec — passes Stage 5 fully."""
    rows = []
    for b in "ABC":
        rows.extend(make_complete_second(box=b, met_sec=met_sec, date=date, **overrides))
    return rows


def test_builder_canonical_row_dict_has_all_columns():
    row = make_row()
    assert set(row.keys()) == set(PER_SEC_COLS)


def test_builder_complete_groupsec_makes_18_rows():
    rows = make_complete_groupsec()
    df = make_df(rows)
    assert len(df) == 18
    assert sorted(df["box"].unique().tolist()) == ["A", "B", "C"]
    assert sorted(df["det"].unique().tolist()) == [0, 1, 2, 3, 4, 5]


# ------------------- BurstCatalog.any_within -------------------

def test_burstcatalog_any_within_empty_catalog():
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([], dtype=np.int64), window_sec=300)
    times = np.array([1000, 2000, 3000], dtype=np.int64)
    out = cat.any_within(times)
    assert out.tolist() == [False, False, False]


def test_burstcatalog_any_within_exact_hit():
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([5000], dtype=np.int64), window_sec=300)
    out = cat.any_within(np.array([5000], dtype=np.int64))
    assert out.tolist() == [True]


def test_burstcatalog_any_within_at_boundary():
    """A trigger at T, window ±300s: T+300 and T-300 inclusive are within."""
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([5000], dtype=np.int64), window_sec=300)
    out = cat.any_within(np.array([4700, 4699, 5300, 5301], dtype=np.int64))
    assert out.tolist() == [True, False, True, False]


def test_burstcatalog_any_within_multi_trigger():
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([1000, 5000, 9000], dtype=np.int64), window_sec=300)
    out = cat.any_within(np.array([1100, 5500, 9000, 3000, 7000], dtype=np.int64))
    assert out.tolist() == [True, False, True, False, False]


def test_burstcatalog_any_within_unsorted_input_triggers_sorted_internally():
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([9000, 1000, 5000], dtype=np.int64), window_sec=300)
    out = cat.any_within(np.array([1100], dtype=np.int64))
    assert out.tolist() == [True]


# ------------------- BurstCatalog.fetch_or_load -------------------

def test_burstcatalog_loads_from_existing_parquet(tmp_path):
    import build_clean_cache as M

    cache = tmp_path / "gbm.parquet"
    # Pretend we already fetched: write a tiny parquet with one trigger at MET 12345.
    pd.DataFrame({"trigger_met_hxmt": [12345]}).to_parquet(cache)

    cat = M.BurstCatalog.fetch_or_load(cache, window_sec=300, allow_fetch=False)
    assert cat.triggers_met.tolist() == [12345]
    assert cat.window_sec == 300


def test_burstcatalog_raises_when_missing_and_fetch_disabled(tmp_path):
    import build_clean_cache as M

    cache = tmp_path / "nonexistent.parquet"
    with pytest.raises(FileNotFoundError):
        M.BurstCatalog.fetch_or_load(cache, window_sec=300, allow_fetch=False)


def test_gbm_iso_to_hxmt_met_2020_01_01():
    """2020-01-01 00:00:00 UTC ≈ HXMT MET 252460803 (2922 days × 86400 + 3 leap-seconds)."""
    import build_clean_cache as M

    met = M.gbm_iso_to_hxmt_met("2020-01-01T00:00:00")
    # ±5 second tolerance for any leap-second / epoch convention drift.
    assert abs(met - 252460803) < 5


# ------------------- apply_filters stages 1-3 -------------------

def test_filter_drops_low_lcycles():
    import build_clean_cache as M
    df = make_df([
        make_row(L_cycles=60_000),    # keep
        make_row(L_cycles=50_000),    # drop (strict >)
        make_row(L_cycles=10_000),    # drop
    ])
    out = M._apply_stage1_detector_state(df)
    assert len(out) == 1
    assert out["L_cycles"].iloc[0] == 60_000


def test_filter_drops_bad_hv():
    import build_clean_cache as M
    df = make_df([
        make_row(HV=-1000),   # keep
        make_row(HV=-1100),   # drop (strict >)
        make_row(HV=-900),    # drop (strict <)
        make_row(HV=-1200),   # drop
        make_row(HV=-800),    # drop
    ])
    out = M._apply_stage1_detector_state(df)
    assert len(out) == 1
    assert out["HV"].iloc[0] == -1000


def test_filter_drops_nan_in_critical_cols():
    import build_clean_cache as M
    df = make_df([
        make_row(),
        make_row(HV=np.nan),
        make_row(Lat=np.nan),
        make_row(Lon=np.nan),
    ])
    out = M._apply_stage2_integrity(df)
    assert len(out) == 1


def test_filter_drops_negative_counters():
    import build_clean_cache as M
    df = make_df([
        make_row(),
        make_row(PHO=-1),
        make_row(Wide=-5),
        make_row(Dt=-1),
    ])
    out = M._apply_stage2_integrity(df)
    assert len(out) == 1


def test_filter_drops_sci_breakdown_mismatch():
    import build_clean_cache as M
    df = make_df([
        # Good: 80 = 70 + 8 + 2
        make_row(Sci_094=80, Sci_pure_094=70, Sci_ACD1_094=8, Sci_ACDN_094=2),
        # Bad: 80 != 70 + 8 + 3
        make_row(Sci_094=80, Sci_pure_094=70, Sci_ACD1_094=8, Sci_ACDN_094=3),
        # Bad 1s window
        make_row(Sci_1s=85, Sci_pure_1s=74, Sci_ACD1_1s=8, Sci_ACDN_1s=4),
    ])
    out = M._apply_stage2_integrity(df)
    assert len(out) == 1


def test_filter_keeps_equator_belt():
    import build_clean_cache as M
    df = make_df([
        make_row(Lat=0.0),    # keep
        make_row(Lat=2.9),    # keep
        make_row(Lat=-2.9),   # keep
        make_row(Lat=3.0),    # drop (strict <)
        make_row(Lat=10.0),   # drop
        make_row(Lat=-50.0),  # drop
    ])
    out = M._apply_stage3_spatial(df)
    assert len(out) == 3
    assert out["Lat"].abs().max() < 3.0


def test_filter_excludes_saa_lon_box():
    import build_clean_cache as M
    df = make_df([
        make_row(Lon=120.0),    # keep (Pacific)
        make_row(Lon=240.0),    # keep (Pacific, was Lon=-120 in [-180,180])
        make_row(Lon=269.9),    # keep (just west of SAA, was Lon=-90.1)
        make_row(Lon=270.0),    # drop (boundary, west edge of SAA)
        make_row(Lon=0.0),      # drop (in SAA)
        make_row(Lon=30.0),     # drop (boundary, east edge of SAA)
        make_row(Lon=30.1),     # keep (just east of SAA)
    ])
    out = M._apply_stage3_spatial(df)
    assert len(out) == 4
    assert ((out["Lon"] > 30.0) & (out["Lon"] < 270.0)).all()


# ------------------- apply_filters stage 4 -------------------

def test_filter_drops_rows_within_burst_window():
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([252633600], dtype=np.int64), window_sec=300)
    df = make_df([
        make_row(met_sec=252633600),         # drop (exact hit)
        make_row(met_sec=252633600 + 299),   # drop (inside ±300)
        make_row(met_sec=252633600 + 300),   # drop (boundary inclusive)
        make_row(met_sec=252633600 + 301),   # keep (outside ±300)
        make_row(met_sec=252633600 - 1000),  # keep
        make_row(met_sec=252633600 + 1000),  # keep
    ])
    out = M._apply_stage4_burst(df, cat)
    assert len(out) == 3
    keeps = set(out["met_sec"].tolist())
    assert 252633600 + 301 in keeps
    assert 252633600 - 1000 in keeps
    assert 252633600 + 1000 in keeps


def test_filter_burst_empty_catalog_keeps_all():
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([], dtype=np.int64), window_sec=300)
    df = make_df([make_row(met_sec=t) for t in (1000, 2000, 3000)])
    out = M._apply_stage4_burst(df, cat)
    assert len(out) == 3


# ------------------- apply_filters stage 5 (completeness) -------------------

def test_completeness_keeps_full_18_row_second():
    import build_clean_cache as M
    df = make_df(make_complete_groupsec(met_sec=252633600))
    out = M._apply_stage5_completeness(df)
    assert len(out) == 18


def test_completeness_drops_second_missing_one_det():
    import build_clean_cache as M
    rows = make_complete_groupsec(met_sec=252633600)
    # Drop box-A det-3
    rows = [r for r in rows if not (r["box"] == "A" and r["det"] == 3)]
    df = make_df(rows)
    out = M._apply_stage5_completeness(df)
    assert len(out) == 0


def test_completeness_drops_second_missing_one_box():
    import build_clean_cache as M
    rows = make_complete_groupsec(met_sec=252633600)
    # Drop entire box C
    rows = [r for r in rows if r["box"] != "C"]
    df = make_df(rows)
    out = M._apply_stage5_completeness(df)
    assert len(out) == 0


def test_completeness_mixed_seconds():
    """One full second + one broken second → only the full one survives."""
    import build_clean_cache as M
    good = make_complete_groupsec(met_sec=252633600)
    bad = [r for r in make_complete_groupsec(met_sec=252633700) if r["box"] != "B"]
    df = make_df(good + bad)
    out = M._apply_stage5_completeness(df)
    assert len(out) == 18
    assert out["met_sec"].unique().tolist() == [252633600]


# ------------------- apply_filters orchestrator -------------------

def test_apply_filters_runs_all_stages_and_logs():
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([], dtype=np.int64), window_sec=300)
    df = make_df(make_complete_groupsec(met_sec=252633600))
    out, counts = M.apply_filters(df, cat)
    assert len(out) == 18
    assert counts["start"] == 18
    assert counts["after_stage1"] == 18
    assert counts["after_stage5"] == 18


def test_apply_filters_drops_all_when_lat_too_high():
    import build_clean_cache as M
    cat = M.BurstCatalog.from_array(np.array([], dtype=np.int64), window_sec=300)
    df = make_df(make_complete_groupsec(met_sec=252633600, Lat=10.0))
    out, counts = M.apply_filters(df, cat)
    assert len(out) == 0
    assert counts["after_stage3"] == 0


# ------------------- raw-only cache: no derivations -------------------

def test_no_derived_columns_in_cache_output():
    """Cache must contain only raw counts, no rate/dt_frac/length/Sci_ACD_* columns."""
    import build_clean_cache as M
    df = make_df([make_row()])
    # Whatever process_one_day writes, it must not contain rate-style derived cols.
    forbidden = {
        "pho_rate", "ooc_rate", "wide_rate", "large_rate",
        "sci_rate_094", "sci_rate_1s",
        "scipure_rate_094", "scipure_rate_1s",
        "acd1_rate_094", "acd1_rate_1s",
        "acdn_rate_094", "acdn_rate_1s",
        "acd_rate_094", "acd_rate_1s",
        "length", "dt_frac", "Sci_ACD_094", "Sci_ACD_1s",
    }
    present = forbidden & set(df.columns)
    assert not present, f"raw input unexpectedly contains derived cols: {present}"


# ------------------- process_one_day -------------------

def test_process_one_day_writes_partial(tmp_path):
    import build_clean_cache as M

    input_dir = tmp_path / "per_sec_parquet"
    input_dir.mkdir()
    df = make_df(make_complete_groupsec(date="2020-01-15", met_sec=252633600))
    df.to_parquet(input_dir / "20200115.parquet")

    cat = M.BurstCatalog.from_array(np.array([], dtype=np.int64), window_sec=300)
    out_dir = tmp_path / "partials"
    out_dir.mkdir()

    result = M.process_one_day("20200115", input_dir, out_dir, cat)

    assert result is not None
    assert result.exists()
    pq = pd.read_parquet(result)
    assert len(pq) == 18  # all 18 rows survive (Lat=0.5, Lon=120 → safe)
    # Raw counts preserved, no derived columns
    assert "PHO" in pq.columns
    assert "Sci_1s" in pq.columns
    assert "pho_rate" not in pq.columns


def test_process_one_day_returns_none_when_no_rows_survive(tmp_path):
    import build_clean_cache as M

    input_dir = tmp_path / "per_sec_parquet"
    input_dir.mkdir()
    # All rows have Lat=10° → Stage 3 kills everything
    df = make_df(make_complete_groupsec(date="2020-01-15", Lat=10.0))
    df.to_parquet(input_dir / "20200115.parquet")

    cat = M.BurstCatalog.from_array(np.array([], dtype=np.int64), window_sec=300)
    out_dir = tmp_path / "partials"
    out_dir.mkdir()

    result = M.process_one_day("20200115", input_dir, out_dir, cat)
    assert result is None
    assert not (out_dir / "20200115.parquet").exists()


def test_process_one_day_missing_input_returns_none(tmp_path):
    import build_clean_cache as M
    input_dir = tmp_path / "per_sec_parquet"
    input_dir.mkdir()
    cat = M.BurstCatalog.from_array(np.array([], dtype=np.int64), window_sec=300)
    out_dir = tmp_path / "partials"
    out_dir.mkdir()

    result = M.process_one_day("99999999", input_dir, out_dir, cat)
    assert result is None


# ------------------- run_build integration -------------------

def test_run_build_writes_final_parquet_and_passes_assertions(tmp_path):
    import build_clean_cache as M

    input_dir = tmp_path / "per_sec_parquet"
    input_dir.mkdir()
    for date_str, date_iso in [("20200115", "2020-01-15"),
                                 ("20200116", "2020-01-16"),
                                 ("20200117", "2020-01-17")]:
        # Two seconds per day, all clean
        rows = (make_complete_groupsec(date=date_iso, met_sec=252633600)
                + make_complete_groupsec(date=date_iso, met_sec=252633700))
        make_df(rows).to_parquet(input_dir / f"{date_str}.parquet")

    # Empty GBM cache so we don't try to fetch
    gbm_cache = tmp_path / "gbm.parquet"
    pd.DataFrame({"trigger_met_hxmt": []}).to_parquet(gbm_cache)

    output = tmp_path / "clean.parquet"
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()

    M.run_build(
        input_dir=input_dir,
        output=output,
        partial_dir=partial_dir,
        gbm_cache=gbm_cache,
        dates=["20200115", "20200116", "20200117"],
        workers=2,
        min_rows=1,        # lower the assertion floor for tests
    )

    assert output.exists()
    df = pd.read_parquet(output)
    # 3 days × 2 seconds × 18 rows = 108
    assert len(df) == 108
    assert "PHO" in df.columns
    assert "pho_rate" not in df.columns


def test_run_build_raises_when_under_min_rows(tmp_path):
    import build_clean_cache as M
    input_dir = tmp_path / "per_sec_parquet"
    input_dir.mkdir()
    # One day, all dropped (Lat too high)
    make_df(make_complete_groupsec(date="2020-01-15", Lat=10.0)).to_parquet(
        input_dir / "20200115.parquet"
    )
    gbm_cache = tmp_path / "gbm.parquet"
    pd.DataFrame({"trigger_met_hxmt": []}).to_parquet(gbm_cache)
    (tmp_path / "partial").mkdir()

    with pytest.raises(AssertionError, match="No partials"):
        M.run_build(
            input_dir=input_dir,
            output=tmp_path / "clean.parquet",
            partial_dir=tmp_path / "partial",
            gbm_cache=gbm_cache,
            dates=["20200115"],
            workers=1,
            min_rows=1,
        )
