"""Unwrap 10-bit Large counter using conservation predictor.

Predicts real Large from PHO/Sci/Wide via the conservation relation
    PHO·(1−dt) ≈ Sci·L + Large·(1−dt) + Wide + C·L
solved for Large:
    predicted_Large = PHO − (Wide + (Sci + C)·L) / (1 − dt)

C ≈ 150 cnt/s comes from H1-strict per-det median fit. The predictor uses
information from THREE observable counters (PHO, Sci, Wide) instead of relying
on a single calibrated ratio r = Large/PHO, which biases low when calibrated
on low-rate data and underestimates wraps at high PHO due to pile-up.

After determining n_wraps via round-to-nearest-1024, a sanity cap enforces
real_Large ≤ PHO − Wide (physically required).

Confidence is reported per-row:
- HIGH: predictor residual |predicted − corrected| < 200 (well within wrap grid)
- MEDIUM: 200-400
- LOW: ≥ 400 OR sanity cap reduced n_wraps OR Wide > PHO (data corruption)
"""
from __future__ import annotations

import numpy as np

L_CYCLES_TO_SEC = 16e-6
C_DEFAULT = 150.0

CONF_HIGH = 2
CONF_MEDIUM = 1
CONF_LOW = 0


def unwrap_large_v2(pho, large, wide, sci, l_cycles, dt, C=C_DEFAULT, return_confidence=False):
    """Conservation-based 10-bit unwrap of Large.

    Args:
        pho, large, wide, sci: per-row raw counts (broadcastable).
        l_cycles: per-row L_cycles count.
        dt: per-row Dt count.
        C: additive constant in cnt/s (default 150 from H1 strict fit). Can be scalar or per-row array.
        return_confidence: if True, returns (large_corrected, confidence) where confidence is
            an int array per row (2=HIGH, 1=MEDIUM, 0=LOW).

    Returns:
        large_corrected: unwrapped Large counts.
        (optionally) confidence: per-row confidence level.
    """
    pho = np.asarray(pho, dtype=np.float64)
    large = np.asarray(large, dtype=np.float64)
    wide = np.asarray(wide, dtype=np.float64)
    sci = np.asarray(sci, dtype=np.float64)
    L = np.asarray(l_cycles, dtype=np.float64) * L_CYCLES_TO_SEC
    lf = 1.0 - np.asarray(dt, dtype=np.float64) / np.asarray(l_cycles, dtype=np.float64)

    predicted = pho - (wide + (sci + C) * L) / lf
    n_wraps_raw = np.round((predicted - large) / 1024.0).astype(int)
    n_wraps_raw = np.maximum(n_wraps_raw, 0)

    max_allowed = pho - wide
    large_corr = large + n_wraps_raw * 1024.0
    over = large_corr > max_allowed
    cap_triggered = over.copy()
    if over.any():
        n_max = np.floor((max_allowed - large) / 1024.0).astype(int)
        n_max = np.maximum(n_max, 0)
        n_wraps = np.where(over, n_max, n_wraps_raw)
        large_corr = large + n_wraps * 1024.0
    else:
        n_wraps = n_wraps_raw

    if not return_confidence:
        return large_corr

    # Confidence assessment
    # 1. Predictor residual: |predicted - large_corr| — should be small (well within 512)
    pred_resid = np.abs(predicted - large_corr)
    # 2. Data sanity: Wide > PHO is impossible (data corruption)
    bad_data = wide > pho
    # 3. Sanity cap was triggered (algorithm couldn't pick closest n_wraps)
    cap_used = cap_triggered

    confidence = np.full(large.shape, CONF_HIGH, dtype=np.int8)
    confidence[pred_resid >= 200] = CONF_MEDIUM
    confidence[(pred_resid >= 400) | cap_used | bad_data] = CONF_LOW

    return large_corr, confidence


def confidence_label(c):
    """Convert int code to human-readable label."""
    return {CONF_HIGH: "HIGH", CONF_MEDIUM: "MEDIUM", CONF_LOW: "LOW"}.get(int(c), "?")
