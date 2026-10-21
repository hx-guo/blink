#!/usr/bin/env python3
"""GRB 250919A engineering-counter 1-s cross-check (paper §6.3, Table 4).

Drives the original per-frame pipeline (per_burst_eng_ratio.compute_ratio,
met_sec join of engineering frames vs reconstructed events) with the
250919A recon cache. Shims the missing `unwrap_large` module with the
identical implementation kept in engineering_prediction._unwrap_large.

Published numbers (2026-07):
  full window T0-30..+70 : median 0.98, sigma_IQR 0.03, 100 bins
  saturated phase +4..+13: median 1.00, sigma_IQR 0.02, 9 bins

Run:  .venv/bin/python scripts/bursts/eng_ratio_250919a.py
"""
import sys, types
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib", "../saturation",)]  # sibling and shared script modules
import numpy as np
from astropy.time import Time
from engineering_prediction import _unwrap_large

shim = types.ModuleType("unwrap_large")
shim.unwrap_large = _unwrap_large
sys.modules["unwrap_large"] = shim
import per_burst_eng_ratio as P

HEP = Time("2012-01-01T00:00:00", scale="utc")
T0 = Time("2025-09-19T00:29:15", scale="utc").unix_tai - HEP.unix_tai
TY = (np.datetime64("2025-09-19") - P.T_REF).astype("timedelta64[D]").astype(float) / 365.25
ORBIT = "data/hxmt_aux/HXMT_20250919T00_Orbit_FFFFFF_V1_1K.FITS"
RECON = "data/recon_cache/250919A_recon.csv"

for label, lo, hi in [("250919A full window", -30, 70),
                      ("250919A saturated phase", 4, 13)]:
    P.compute_ratio(label, "20250919", "000000", trigger_met=T0,
                    t_lo_rel=lo, t_hi_rel=hi, recon_csv=RECON,
                    t_years_const=TY, orbit_path=ORBIT)
