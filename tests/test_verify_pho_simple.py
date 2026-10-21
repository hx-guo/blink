"""Tests for scripts/saturation/verify_pho_simple.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

import verify_pho_simple as M


def test_residual_formula_dt_zero():
    """With dt=0: residual = PHO/0.94 - Sci_1s - Large/0.94 - Wide/0.94."""
    # Row 1: PHO=94 → 100 /1s; Large=4.7 → 5; Wide=9.4 → 10; Sci_1s=80
    # residual = 100 - 80 - 5 - 10 = 5
    df = pd.DataFrame({
        "PHO":        [94.0, 188.0, 47.0],
        "Sci_1s":     [80.0, 150.0, 30.0],
        "Large":      [4.7,  18.8,  9.4],
        "Wide":       [9.4,  23.5,  4.7],
        "Dt":         [0,    0,     0],
        "L_cycles":   [58750, 58750, 58750],   # = 0.94s wallclock; dt_frac = 0
    })
    out = M.derive_inline(df)
    out["residual_rate"] = M.compute_residual(out)
    np.testing.assert_array_almost_equal(out["residual_rate"].values, [5.0, 5.0, 5.0], decimal=5)


def test_residual_dt_correction_scales_pho_and_large():
    """With dt=0.5: PHO/0.94 × 0.5 = 50; Large/0.94 × 0.5 = 2.5; Wide/0.94 = 10;
    residual = 50 - 80 - 2.5 - 10 = -42.5."""
    df = pd.DataFrame({
        "PHO":        [94.0],
        "Sci_1s":     [80.0],
        "Large":      [4.7],
        "Wide":       [9.4],
        "Dt":         [29375],     # 0.5 × 58750
        "L_cycles":   [58750],
    })
    out = M.derive_inline(df)
    out["residual_rate"] = M.compute_residual(out)
    np.testing.assert_array_almost_equal(out["residual_rate"].values, [-42.5], decimal=5)


def test_summarize_per_group_yields_18_rows():
    rows = []
    for box in "ABC":
        for det in range(6):
            for i in range(10):
                rows.append({"box": box, "det": det, "residual_rate": float(i)})
    df = pd.DataFrame(rows)
    s = M.summarize_per_group(df)
    assert len(s) == 18
    assert set(s["box"].unique()) == {"A", "B", "C"}
    assert set(s["det"].unique()) == set(range(6))
    assert (s["N"] == 10).all()


def test_summarize_returns_expected_stats():
    df = pd.DataFrame({
        "box": ["A"] * 5,
        "det": [0] * 5,
        "residual_rate": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    s = M.summarize_per_group(df)
    assert len(s) == 1
    row = s.iloc[0]
    assert row["N"] == 5
    assert abs(row["mean"] - 3.0) < 1e-9
    assert abs(row["median"] - 3.0) < 1e-9
    assert abs(row["q01"] - 1.04) < 1e-2   # linear interp between 1 and 2
    assert abs(row["q99"] - 4.96) < 1e-2
