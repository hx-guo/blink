"""Tests for scripts/saturation/extract_per_sec_day.py."""
from __future__ import annotations

import math
import sys

import extract_per_sec_day as M


def test_module_imports():
    """Sanity check: the worker module is importable."""
    assert M.MET_CORRECTION == 4.0
    assert M.BOX_PORTS == {"A": "0766", "B": "1009", "C": "1781"}
    assert M.BOX_INDEX == {"A": 0, "B": 1, "C": 2}


def test_compute_offset_basic():
    # From HE_Eng row 0 of data/1B/2017/20171001/0766/...
    assert M.compute_offset(utc_last_bdc=181439999, stime_last_bdc=1618548) == 179821451


def test_compute_met_float_basic():
    # Verified: 1618548 + 179821451 + 4.0 = 181440003.0
    out = M.compute_met_float(time_1b=1618548, offset=179821451)
    assert math.isclose(out, 181440003.0, abs_tol=1e-9)


def test_compute_met_float_array():
    import numpy as np
    times = np.array([1618548, 1618549, 1618550], dtype=np.int64)
    out = M.compute_met_float(time_1b=times, offset=179821451)
    np.testing.assert_allclose(out, [181440003.0, 181440004.0, 181440005.0])


def test_count_acd_bits_zero():
    import numpy as np
    acd = np.zeros((1, 18), dtype=bool)
    assert M.count_acd_bits(acd).tolist() == [0]


def test_count_acd_bits_single():
    import numpy as np
    acd = np.zeros((3, 18), dtype=bool)
    acd[0, 0] = True
    acd[1, 5] = True
    acd[2, 17] = True
    assert M.count_acd_bits(acd).tolist() == [1, 1, 1]


def test_count_acd_bits_multi():
    import numpy as np
    acd = np.zeros((2, 18), dtype=bool)
    acd[0, [0, 1, 3]] = True   # 3 bits
    acd[1, :] = True            # 18 bits
    assert M.count_acd_bits(acd).tolist() == [3, 18]


def test_window_indices_basic():
    import numpy as np
    times = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    i_start, i_end = M.window_indices(times, 10.0, 11.5)
    # half-open [10.0, 11.5): indices 0,1,2  → i_start=0, i_end=3
    assert i_start == 0
    assert i_end == 3


def test_window_indices_empty():
    import numpy as np
    times = np.array([1.0, 2.0, 3.0])
    i_start, i_end = M.window_indices(times, 10.0, 11.0)
    assert i_start == 3
    assert i_end == 3   # zero-length window past end


def test_read_he_eng_2017_box_a(require_file):
    from tests.conftest import HE_ENG_2017_BOXA
    require_file(HE_ENG_2017_BOXA)

    d = M.read_he_eng(HE_ENG_2017_BOXA)

    # Expect a dict of numpy arrays, one per kept column.
    assert d["Time"][0] == 1618548
    assert d["UTC_Last_Bdc"][0] == 181439999
    assert d["sTime_Last_Bdc"][0] == 1618548
    assert d["Length_Time_Cycle"][0] == 58799
    assert d["Cnt_PHODet"][0, 0] == 2200       # per-det 2D: (n_sec, 6)
    assert d["Cnt_OOCDet"][0, 0] == 201
    assert d["Cnt_CsI_PHODet"][0, 0] == 196
    assert d["Cnt_LargeEvt"][0, 0] == 532
    assert d["DeadTime_PHODet"][0, 0] == 2496
    # File should have 3600 rows for one hour
    assert len(d["Time"]) == 3600

    # Raw byte fields preserved
    assert d["BUS_Time_Bdc"].shape == (3600, 6)
    assert d["Error_code"].shape == (3600, 4)


def test_read_he_hv_synthetic(tmp_path):
    """Build a fake HE-HV FITS and confirm the reader unpacks it correctly."""
    from astropy.io import fits
    import numpy as np

    n = 10
    cols = [fits.Column(name="Time", format="J",
                        array=np.arange(1000, 1000 + n, dtype=np.int64))]
    for j in range(18):
        cols.append(fits.Column(name=f"HV_PHODet_{j}", format="E",
                                array=(-1000.0 - j) * np.ones(n, dtype=np.float32)))
    hdu = fits.BinTableHDU.from_columns(cols, name="HE_HV_PHODet")
    fpath = tmp_path / "fake_he_hv.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(fpath)

    d = M.read_he_hv(fpath)
    # Expected: dict with 'Time' (n,) and 'HV' (n, 18)
    assert d["Time"].shape == (n,)
    assert d["HV"].shape == (n, 18)
    assert d["Time"][0] == 1000
    assert d["HV"][0, 0] == -1000.0
    assert d["HV"][0, 17] == -1017.0


def test_read_orbit_20260410(require_file):
    from tests.conftest import ORBIT_20260410_HR07
    require_file(ORBIT_20260410_HR07)

    d = M.read_orbit(ORBIT_20260410_HR07)

    # Schema: dict with 10 keys
    for key in ["Time", "X", "Y", "Z", "Vx", "Vy", "Vz", "Lon", "Lat", "Alt"]:
        assert key in d

    assert len(d["Time"]) == 3601   # 1 hour + 1 sample
    assert d["Time"][0] == 450428403   # integer first sample
    # Adjacent samples 1 second apart
    assert d["Time"][1] - d["Time"][0] == 1

    # Sample row 0 ground-truth (rounded to 1 decimal)
    assert abs(d["X"][0]  - (-5008185.1)) < 1.0
    assert abs(d["Lat"][0] - (-36.039)) < 0.01
    assert abs(d["Alt"][0] - 536848.8) < 1.0


def test_read_att_20260410(require_file):
    from tests.conftest import ATT_20260410_HR07
    require_file(ATT_20260410_HR07)

    import numpy as np

    target_secs = np.array([450428403, 450428404, 450428405], dtype=np.int64)
    d = M.read_att(ATT_20260410_HR07, target_secs)

    # Schema: 14 keys (5 pointing + 3 euler + 3 quat + 3 omega)
    expected_keys = {"Ra", "Dec", "Delta_Ra", "Delta_Dec", "Delta",
                      "Euler_Phi", "Euler_Theta", "Euler_Psi",
                      "Q1", "Q2", "Q3",
                      "Omega_X", "Omega_Y", "Omega_Z"}
    assert set(d.keys()) == expected_keys

    # All arrays match the target_secs length
    for k, arr in d.items():
        assert arr.shape == (3,), f"{k} wrong shape: {arr.shape}"

    # Ra/Dec sanity: not NaN at integer seconds covered by the file
    assert np.isfinite(d["Ra"][0])
    assert np.isfinite(d["Dec"][0])


def test_read_he_evt_20260410(require_file):
    from tests.conftest import HE_EVT_20260410_HR07
    require_file(HE_EVT_20260410_HR07)

    d = M.read_he_evt(HE_EVT_20260410_HR07)

    import numpy as np

    # Expected: dict with two arrays
    assert "Time" in d
    assert "Det_ID" in d
    assert "ACD_popcount" in d
    assert len(d["Time"]) == 27_905_540
    assert d["Time"].min() >= 450428403
    assert d["Time"][0] < d["Time"][-1]   # sorted

    # First event ground truth from local file
    assert d["Det_ID"][0] == 0
    assert d["ACD_popcount"][0] == 0
    assert d["ACD_popcount"][1] == 1   # second event has 1 ACD bit


def test_aggregate_he_evt_synthetic():
    """Build a tiny synthetic event stream and verify aggregation."""
    import numpy as np

    # 4 events in 1 second, det_global = 9 (box B, det 3 → global 9)
    # Times in [10.0, 10.5, 10.95, 10.99]   - first 2 in 0.94s window, all 4 in 1.0s
    times      = np.array([10.0, 10.5, 10.95, 10.99], dtype=np.float64)
    det_ids    = np.array([9, 9, 9, 9], dtype=np.int8)       # box B det 3 → global 9
    popcounts  = np.array([0, 1, 2, 0], dtype=np.int8)       # pure, ACD1, ACDN, pure

    evt = {"Time": times, "Det_ID": det_ids, "ACD_popcount": popcounts}
    met_floats = np.array([10.0], dtype=np.float64)

    sci = M.aggregate_he_evt(
        evt, met_floats, box_index=1, det=3,
        window_s_094=0.94, window_s_1s=1.0,
    )
    # 0.94s window [10.0, 10.94) picks events at 10.0, 10.5; ACD breakdown: 1 pure, 1 ACD1, 0 ACDN
    assert sci["Sci_094"][0] == 2
    assert sci["Sci_pure_094"][0] == 1
    assert sci["Sci_ACD1_094"][0] == 1
    assert sci["Sci_ACDN_094"][0] == 0
    # 1.0s window [10.0, 11.0) picks all 4: 2 pure, 1 ACD1, 1 ACDN
    assert sci["Sci_1s"][0] == 4
    assert sci["Sci_pure_1s"][0] == 2
    assert sci["Sci_ACD1_1s"][0] == 1
    assert sci["Sci_ACDN_1s"][0] == 1


def test_find_he_eng_path_local(monkeypatch):
    """The function should glob the local 2017-10-01 layout when 1B root is overridden."""
    from tests.conftest import REPO_ROOT
    monkeypatch.setenv("BLINK_1B_ROOT", str(REPO_ROOT / "data/1B"))
    # No reload needed: root_1b() reads env each call.
    p = M.find_he_eng_path(date="20171001", hour=0, port="0766")
    assert p is not None
    assert p.exists()
    assert p.name == "HXMT_1B_0766_20171001T000000_G002572_000_003.fits"


def test_find_he_eng_path_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("BLINK_1B_ROOT", str(tmp_path))
    assert M.find_he_eng_path(date="20171001", hour=0, port="0766") is None


def test_find_1k_aux_path_local(monkeypatch):
    """Should locate the 2026-04-10 Orbit file."""
    from tests.conftest import REPO_ROOT
    monkeypatch.setenv("BLINK_1K_ROOT", str(REPO_ROOT / "data/1K"))
    p = M.find_1k_aux_path(date="20260410", hour=7, product="Orbit")
    assert p is not None
    assert p.exists()
    assert "Orbit" in p.name


def test_find_1k_aux_path_he_evt(monkeypatch):
    from tests.conftest import REPO_ROOT
    monkeypatch.setenv("BLINK_1K_ROOT", str(REPO_ROOT / "data/1K"))
    p = M.find_1k_aux_path(date="20260410", hour=7, product="HE-Evt")
    assert p is not None
    assert "HE-Evt" in p.name


def test_find_1k_aux_path_picks_highest_version(monkeypatch, tmp_path):
    """When multiple V<n> revisions exist, pick the highest."""
    # Build a fake 1K layout: /Y202604/20260410-0001/HXMT_..._V1_1K.FITS, V2, V3
    root_1k = tmp_path / "1K"
    dir1k = root_1k / "Y202604" / "20260410-0001"
    dir1k.mkdir(parents=True)
    # Create three versions of HE-Evt for hour 07
    for v in [1, 2, 3]:
        (dir1k / f"HXMT_20260410T07_HE-Evt_FFFFFF_V{v}_1K.FITS").touch()
    monkeypatch.setenv("BLINK_1K_ROOT", str(root_1k))

    p = M.find_1k_aux_path(date="20260410", hour=7, product="HE-Evt")
    assert p is not None
    assert "_V3_1K.FITS" in p.name, f"expected V3, got: {p.name}"


def test_extract_day_20260410(monkeypatch, require_file):
    """Integration test on real local 2026-04-10 data (hour 07 only)."""
    from tests.conftest import REPO_ROOT, HE_EVT_20260410_HR07
    require_file(HE_EVT_20260410_HR07)

    monkeypatch.setenv("BLINK_1B_ROOT", str(REPO_ROOT / "data/1B"))
    monkeypatch.setenv("BLINK_1K_ROOT", str(REPO_ROOT / "data/1K"))

    df = M.extract_day("20260410")

    # We expect rows only for hour 07 (only HE_Eng we have for this date).
    # 1 hour × 3 boxes × 6 dets × 3600 sec = 64800 rows
    assert len(df) > 0

    # Schema check — required columns
    required = {"date", "box", "det", "met_sec",
                "time_float", "L_cycles",
                "PHO", "OOC", "Wide", "Large", "Dt",
                "HV",
                "Sci_094", "Sci_pure_094", "Sci_ACD1_094", "Sci_ACDN_094",
                "Sci_1s",  "Sci_pure_1s",  "Sci_ACD1_1s",  "Sci_ACDN_1s",
                "crc_box",
                "X", "Y", "Z", "Vx", "Vy", "Vz", "Lon", "Lat", "Alt",
                "Ra", "Dec", "Delta_Ra", "Delta_Dec", "Delta",
                "Euler_Phi", "Euler_Theta", "Euler_Psi",
                "Q1", "Q2", "Q3",
                "Omega_X", "Omega_Y", "Omega_Z",
                "utc_last_bdc", "stime_last_bdc", "error_code", "bus_time_bdc"}
    assert required.issubset(set(df.columns)), f"missing: {required - set(df.columns)}"

    # All 3 boxes represented
    assert set(df["box"].unique()) == {"A", "B", "C"}
    # All 6 dets per box
    for box in ["A", "B", "C"]:
        assert set(df[df["box"] == box]["det"].unique()) == {0, 1, 2, 3, 4, 5}

    # HE-HV not available locally → HV column should be all NaN
    import numpy as np
    assert df["HV"].isna().all()

    # crc_box is the deferred NaN field
    assert df["crc_box"].isna().all()

    # Lat finite (Orbit file present)
    assert df["Lat"].notna().any()
    # Pointing finite (Att file present)
    assert df["Ra"].notna().any()

    # Partition invariant: Sci = pure + ACD1 + ACDN (for both windows)
    for tag in ["094", "1s"]:
        s = df[f"Sci_{tag}"].fillna(0)
        partition = (df[f"Sci_pure_{tag}"].fillna(0) +
                     df[f"Sci_ACD1_{tag}"].fillna(0) +
                     df[f"Sci_ACDN_{tag}"].fillna(0))
        assert (s == partition).all()


def test_write_parquet_atomic(tmp_path):
    """Verify atomic write: temp file then rename."""
    import pandas as pd
    import numpy as np

    # Create a small DataFrame
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4.0, 5.0, 6.0],
    })

    output_file = tmp_path / "test.parquet"

    M.write_parquet_atomic(df, output_file)

    # File should exist and be readable
    assert output_file.exists()
    df_read = pd.read_parquet(output_file)
    pd.testing.assert_frame_equal(df, df_read)


def test_main_idempotent(monkeypatch, tmp_path, require_file):
    """Verify CLI exits 0 (no-op) if output already exists."""
    from tests.conftest import REPO_ROOT, HE_EVT_20260410_HR07
    require_file(HE_EVT_20260410_HR07)

    monkeypatch.setenv("BLINK_1B_ROOT", str(REPO_ROOT / "data/1B"))
    monkeypatch.setenv("BLINK_1K_ROOT", str(REPO_ROOT / "data/1K"))

    output_dir = tmp_path / "per_sec_parquet"
    output_file = output_dir / "20260410.parquet"

    # First invocation: should create file
    sys.argv = ["extract_per_sec_day.py", "20260410", "--output-dir", str(output_dir)]
    ret = M.main()
    assert ret == 0
    assert output_file.exists()

    # Second invocation: should exit 0 (idempotent, no-op)
    ret = M.main()
    assert ret == 0


def test_main_invalid_date(monkeypatch, capsys):
    """Verify CLI rejects invalid date format."""
    monkeypatch.setenv("BLINK_1B_ROOT", "/nonexistent")
    monkeypatch.setenv("BLINK_1K_ROOT", "/nonexistent")

    sys.argv = ["extract_per_sec_day.py", "2026-04-10"]  # Wrong format
    ret = M.main()
    assert ret == 1
    captured = capsys.readouterr()
    assert "YYYYMMDD" in captured.err


def test_find_he_eng_path_picks_highest_segment(monkeypatch, tmp_path):
    """When multiple segments exist for one hour, pick the highest number."""
    # Build a synthetic 1B directory layout with two segments for hour 3.
    yr = tmp_path / "1B" / "2022" / "20220115" / "0766"
    yr.mkdir(parents=True)
    (yr / "HXMT_1B_0766_20220115T030000_G040183_000_004.fits").touch()
    (yr / "HXMT_1B_0766_20220115T030000_G040183_001_004.fits").touch()
    (yr / "HXMT_1B_0766_20220115T030000_G040183_002_004.fits").touch()

    monkeypatch.setenv("BLINK_1B_ROOT", str(tmp_path / "1B"))

    p = M.find_he_eng_path("20220115", 3, "0766")
    assert p is not None
    assert p.name == "HXMT_1B_0766_20220115T030000_G040183_002_004.fits"


def test_find_he_eng_path_single_segment_unchanged(monkeypatch, tmp_path):
    """When only one segment exists, behaviour unchanged."""
    yr = tmp_path / "1B" / "2022" / "20220115" / "0766"
    yr.mkdir(parents=True)
    (yr / "HXMT_1B_0766_20220115T040000_G040184_000_004.fits").touch()

    monkeypatch.setenv("BLINK_1B_ROOT", str(tmp_path / "1B"))

    p = M.find_he_eng_path("20220115", 4, "0766")
    assert p is not None
    assert p.name == "HXMT_1B_0766_20220115T040000_G040184_000_004.fits"


def test_find_he_eng_path_no_match(monkeypatch, tmp_path):
    """When no segment exists, returns None."""
    yr = tmp_path / "1B" / "2022" / "20220115" / "0766"
    yr.mkdir(parents=True)

    monkeypatch.setenv("BLINK_1B_ROOT", str(tmp_path / "1B"))

    p = M.find_he_eng_path("20220115", 5, "0766")
    assert p is None


def test_effective_offsets_no_outlier():
    """All hours within 10s of median: pass through unchanged."""
    raw = {0: 179821369, 1: 179821368, 2: 179821368, 3: 179821367}
    out = M.effective_offsets(raw, threshold_sec=10)
    assert out == raw


def test_effective_offsets_one_outlier_below():
    """One hour 800s below median: replace with neighbour median."""
    raw = {0: 179821369, 1: 179821369, 2: 179820567, 3: 179821369, 4: 179821368}
    out = M.effective_offsets(raw, threshold_sec=10)
    assert out[0] == 179821369
    assert out[1] == 179821369
    # Hour 2 is the outlier; immediate good neighbours are hours 1 and 3 (both 179821369)
    assert out[2] == 179821369
    assert out[3] == 179821369
    assert out[4] == 179821368


def test_effective_offsets_one_outlier_above():
    """Symmetric: also catches outliers above median."""
    raw = {0: 179821369, 1: 179822500, 2: 179821368}
    out = M.effective_offsets(raw, threshold_sec=10)
    # Hour 1 deviation = 1131 from median 179821369; gets overridden.
    # Neighbours that pass threshold: hours 0 (179821369) and 2 (179821368)
    assert out[1] in (179821368, 179821369)


def test_effective_offsets_no_good_neighbour_uses_median():
    """If immediate neighbours are also outliers, fall back to day median."""
    raw = {0: 179821369, 1: 179821369, 2: 179820567, 3: 179820567, 4: 179821369}
    out = M.effective_offsets(raw, threshold_sec=10)
    # Median is 179821369 (3 good hours vs 2 bad). Hours 2,3 both outliers; no
    # good immediate neighbour for hour 2's neighbour at hour 3, fallback to median.
    assert out[2] == 179821369


def test_effective_offsets_empty():
    """Empty input → empty output (no crash)."""
    assert M.effective_offsets({}, threshold_sec=10) == {}


def test_effective_offsets_single_hour():
    """One hour only → that hour is the median, no override."""
    raw = {5: 179821369}
    out = M.effective_offsets(raw, threshold_sec=10)
    assert out == raw


def test_probe_he_eng_offset(require_file):
    """Probe just reads row 0 to extract offset — should be fast and correct."""
    from tests.conftest import HE_ENG_2017_BOXA
    require_file(HE_ENG_2017_BOXA)

    off = M.probe_he_eng_offset(HE_ENG_2017_BOXA)
    # Matches test_compute_offset_basic: 181439999 - 1618548 = 179821451
    assert off == 179821451


def test_probe_he_eng_offset_returns_none_for_bad_file(tmp_path):
    """Unreadable / non-existent file returns None (extract continues gracefully)."""
    bogus = tmp_path / "not_a_fits.fits"
    bogus.write_bytes(b"\x00" * 100)
    assert M.probe_he_eng_offset(bogus) is None


def test_read_he_eng_default_offset(require_file):
    """Without override: reader returns offset from file header."""
    from tests.conftest import HE_ENG_2017_BOXA
    require_file(HE_ENG_2017_BOXA)

    d = M.read_he_eng(HE_ENG_2017_BOXA)
    # The reader returns the file's own offset baseline.
    assert d["offset"] == 179821451


def test_read_he_eng_override_offset(require_file):
    """With override: reader returns the override value."""
    from tests.conftest import HE_ENG_2017_BOXA
    require_file(HE_ENG_2017_BOXA)

    d = M.read_he_eng(HE_ENG_2017_BOXA, override_offset=179800000)
    assert d["offset"] == 179800000


def test_extract_day_no_dups_for_known_offset_bug(monkeypatch):
    """20220115 box A has a known bad offset in hour 2. After the fix, the
    produced dataframe has zero (box, det, met_sec) duplicates."""
    from pathlib import Path
    src_b = Path("/hxmtfs/data/Archive_tmp/1B/2022/20220115")
    if not src_b.exists():
        import pytest
        pytest.skip("Real 1B archive not mounted")

    monkeypatch.setenv("BLINK_1B_ROOT", "/hxmtfs/data/Archive_tmp/1B")
    monkeypatch.setenv("BLINK_1K_ROOT", "/hxmt/work/HXMT-DATA/1K")

    df = M.extract_day("20220115")
    assert len(df) > 0
    dup_count = df.duplicated(subset=["box", "det", "met_sec"]).sum()
    assert dup_count == 0, f"Expected 0 dups, got {dup_count}"


def test_extract_day_monotonic_after_fix(monkeypatch):
    """Each (box, det) group has strictly increasing met_sec after the fix."""
    import numpy as np
    from pathlib import Path
    src_b = Path("/hxmtfs/data/Archive_tmp/1B/2022/20220115")
    if not src_b.exists():
        import pytest
        pytest.skip("Real 1B archive not mounted")

    monkeypatch.setenv("BLINK_1B_ROOT", "/hxmtfs/data/Archive_tmp/1B")
    monkeypatch.setenv("BLINK_1K_ROOT", "/hxmt/work/HXMT-DATA/1K")

    df = M.extract_day("20220115")
    for (box, det), sub in df.groupby(["box", "det"]):
        s = sub["met_sec"].to_numpy()
        assert (np.diff(s) > 0).all(), f"non-monotonic in ({box}, {det})"
