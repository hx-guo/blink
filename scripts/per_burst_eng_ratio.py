#!/usr/bin/env python3
"""Per-burst engineering-counter rate vs event-level Sci_rec ratio at 1s cadence.

For each burst window, compute:
  Sci_eng(t) = engineering-counter rate channel (C25 prediction) summed over 18 dets
  Sci_rec(t) = reconstructed event count from event-level recovery at 1s
  ratio(t)   = Sci_rec / Sci_eng

Reports mean +/- std of ratio over burst bins, N_bins.

Used to fill §5.3 Table 5 RE-VERIFY entries.
"""
from __future__ import annotations
import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, "scripts")
from unwrap_large import unwrap_large

BLINK = Path(__file__).resolve().parent.parent
os.chdir(BLINK)

MET_CORRECTION = 4.0
L_CYC_TO_SEC = 16e-6
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}
BOX_CODE = {"A": "0766", "B": "1009", "C": "1781"}

C25 = json.loads(Path("/tmp/per_det_25param.json").read_text())
A_DET = np.array(C25["a_det"])
ALPHA = C25["alpha"]; MU_M = C25["mu_m"]; K_M = C25["k_m"]
AMP0 = C25["amp0"]; MU_T = C25["mu_t"]; K_T = C25["k_t"]; C0 = C25["C0"]
T_REF = np.datetime64("2017-06-22")

_grid = np.load("n_below_study/aacgm_grid_2020.npz")
MLAT_INTERP = RegularGridInterpolator(
    (_grid["lat_grid"], _grid["lon_grid"]), _grid["mlat"],
    bounds_error=False, fill_value=0.0,
)


def sigm(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def c25_baseline(box_letter, det_local, mlat_abs, t_years):
    """C25 baseline C(i,|m|,t) for one detector."""
    i = BOX_OFFSET[box_letter] + det_local
    A = A_DET[i]
    sm = sigm((mlat_abs - MU_M) / K_M)
    st = sigm((t_years - MU_T) / K_T)
    g = 1.0 + ALPHA * sm
    return A * g * (1.0 - AMP0 * g * st) + C0


def load_aux_for_date(date_str):
    """Load latitude/longitude per second from auxiliary FITS if present.
    Falls back to constant 0,0 (equator) if not available."""
    return None  # MLAT lookup needs aux orbit file; use 0 for now (small effect on bright bursts)


def load_eng(box_letter, fits_path, t_lo_met, t_hi_met, t_years_const, orbit_path=None):
    """Load engineering counters for one box within MET window. Returns
    DataFrame with columns: met_sec, det, Sci_eng (cnt/s per det).

    If orbit_path is supplied, looks up |MLAT| for each met_sec via the
    AACGM grid; otherwise uses MLAT=0 fallback.
    """
    fe = fits.open(fits_path, memmap=True)
    d = fe["HE_Eng"].data
    offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
    met = d["Time"].astype(float) + offset + MET_CORRECTION
    lc_all = d["Length_Time_Cycle"].astype(float)
    mask = (met >= t_lo_met) & (met <= t_hi_met)
    met = met[mask]; lc = lc_all[mask]; L = lc * L_CYC_TO_SEC

    # MLAT per (filtered) eng row
    if orbit_path is not None:
        orb = fits.open(orbit_path)
        orb_t = orb[1].data["Time"].astype(float)
        orb_lat = orb[1].data["Lat"].astype(float)
        orb_lon = orb[1].data["Lon"].astype(float)
        orb.close()
        lat_at = np.interp(met, orb_t, orb_lat)
        lon_at = np.interp(met, orb_t, orb_lon)
        pts = np.column_stack([lat_at, lon_at])
        mlat_abs_per_sec = np.abs(MLAT_INTERP(pts))
        mlat_abs_per_sec = np.where(np.isnan(mlat_abs_per_sec), 0.0, mlat_abs_per_sec)
    else:
        mlat_abs_per_sec = None

    rows = []
    for det_local in range(6):
        det_global = BOX_OFFSET[box_letter] + det_local
        pho = d[f"Cnt_PHODet_{det_global}"].astype(float)[mask]
        wide = d[f"Cnt_CsI_PHODet_{det_global}"].astype(float)[mask]
        large_raw = d[f"Cnt_LargeEvt_{det_global}"].astype(float)[mask]
        dt = d[f"DeadTime_PHODet_{det_global}"].astype(float)[mask]
        large = unwrap_large(pho, large_raw)
        lf_det = 1.0 - dt / lc
        # Sci_eng formula from C25 model. C_per_row is the per-orbit baseline.
        # Use MLAT-resolved value if mlat_abs_per_sec supplied; else fall back to 0.
        # Also compute raw version (without C) for comparison.
        if mlat_abs_per_sec is None:
            raise RuntimeError(
                "orbit_path is required: evaluating C25 at |MLAT|=0 biases "
                "Sci_eng by ~100 cnt/s/det and the burst ratio by 10-17% "
                "(fixed 2026-07: silent fallback removed)")
        C_per = c25_baseline(box_letter, det_local, mlat_abs_per_sec, t_years_const)
        sci_eng_raw = (pho - large) * lf_det / L - wide / L
        sci_eng = sci_eng_raw - C_per
        for i in range(len(met)):
            rows.append({
                "met_sec": int(round(met[i])),
                "box": box_letter, "det": det_local,
                "Sci_eng": sci_eng[i],
                "Sci_eng_raw": sci_eng_raw[i],
            })
    fe.close()
    return pd.DataFrame(rows)


def load_recon_events(csv_path, t_lo_met, t_hi_met):
    """Bin reconstructed events (EVT + FILL_GAP) to 1s cadence, summed over boxes."""
    df = pd.read_csv(csv_path, dtype={"box": "string", "type": "string", "met": "float64"})
    df = df[df["type"].isin(["EVT", "FILL_GAP"])]
    df = df[(df["met"] >= t_lo_met) & (df["met"] < t_hi_met)]
    df["met_sec"] = df["met"].astype("int64")
    counts = df.groupby("met_sec").size().rename("Sci_rec").reset_index()
    return counts


def compute_ratio(burst_label, date_str, hour_str, trigger_met, t_lo_rel, t_hi_rel,
                  recon_csv, t_years_const, orbit_path=None):
    """Main pipeline for one burst."""
    t_lo = trigger_met + t_lo_rel
    t_hi = trigger_met + t_hi_rel

    eng_parts = []
    for box, code in BOX_CODE.items():
        folder = Path(f"data/1B/{date_str[:4]}/{date_str}/{code}")
        prefix = f"HXMT_1B_{code}_{date_str}T{hour_str}"
        matches = sorted(folder.glob(f"{prefix}*.fits"))
        if not matches:
            print(f"  WARN: no FITS at {folder} matching {prefix}")
            continue
        eng_parts.append(load_eng(box, matches[0], t_lo, t_hi, t_years_const, orbit_path))
    eng = pd.concat(eng_parts, ignore_index=True)

    sci_eng_per_sec = eng.groupby("met_sec").agg(
        Sci_eng=("Sci_eng", "sum"),
        Sci_eng_raw=("Sci_eng_raw", "sum"),
    ).reset_index()

    sci_rec_per_sec = load_recon_events(recon_csv, t_lo, t_hi)

    df = sci_eng_per_sec.merge(sci_rec_per_sec, on="met_sec", how="inner")
    df["t_rel"] = df["met_sec"] - trigger_met
    df = df[(df["Sci_eng"] > 0) & (df["Sci_rec"] > 0)].copy()
    df["ratio"] = df["Sci_rec"] / df["Sci_eng"]
    df["ratio_raw"] = df["Sci_rec"] / df["Sci_eng_raw"]

    print(f"\n=== {burst_label} ===")
    print(f"  Window: T0{t_lo_rel:+}s to T0{t_hi_rel:+}s, MET [{t_lo:.0f}, {t_hi:.0f}]")
    print(f"  Eng rows (1s/box/det): {len(eng):,}  →  per-sec aggregate: {len(sci_eng_per_sec)} bins")
    print(f"  Reconstructed rows: 1s bins = {len(sci_rec_per_sec)}")
    print(f"  Merged usable bins: {len(df)}")
    if len(df) == 0:
        return None
    print(f"  Sci_rec range: {df['Sci_rec'].min():.0f} -- {df['Sci_rec'].max():.0f} cnt/s")
    print(f"  Sci_eng range: {df['Sci_eng'].min():.0f} -- {df['Sci_eng'].max():.0f} cnt/s")
    for col, label in [("ratio", "Sci_rec / Sci_eng (with C25)"),
                       ("ratio_raw", "Sci_rec / Sci_eng_raw (no C subtraction)")]:
        print(f"  {label}:")
        print(f"    median = {df[col].median():.3f}")
        print(f"    IQR    = {df[col].quantile(0.75) - df[col].quantile(0.25):.3f}")
        print(f"    sigma_IQR (=IQR/1.349) = {(df[col].quantile(0.75) - df[col].quantile(0.25)) / 1.349:.3f}")
        print(f"    P05/P95 = {df[col].quantile(0.05):.3f} / {df[col].quantile(0.95):.3f}")

    # residual stats in cnt/s
    df["resid"] = df["Sci_rec"] - df["Sci_eng"]
    print(f"  (Sci_rec - Sci_eng) cnt/s: median {df['resid'].median():+.1f}, "
          f"std {df['resid'].std():.1f}, P5/P95 {df['resid'].quantile(0.05):+.1f} / "
          f"{df['resid'].quantile(0.95):+.1f}")
    if len(df) <= 30:
        print(f"  per-bin dump (t_rel | Sci_rec | Sci_eng | ratio):")
        for _, r in df.sort_values("t_rel").iterrows():
            print(f"    {r['t_rel']:+5.0f}  {r['Sci_rec']:>8.0f}  {r['Sci_eng']:>8.0f}  {r['ratio']:.3f}")
    return df


def main():
    bursts = {}

    # GRB 221009A tail: T0+330 to T0+680 s
    T0_221009 = 339945423.0  # leap-second-aware MET of 2022-10-09T13:17:00 UTC
    bursts["221009A_tail"] = compute_ratio(
        "GRB 221009A tail", "20221009", "130000",
        trigger_met=T0_221009, t_lo_rel=330, t_hi_rel=680,
        recon_csv="data/cache_221009a_reconstruct.csv",
        t_years_const=(np.datetime64("2022-10-09") - T_REF).astype("timedelta64[D]").astype(float) / 365.25,
        orbit_path="data/hxmt_aux/HXMT_20221009T13_Orbit_FFFFFF_V1_1K.FITS",
    )

    # FRB 200428: burst-peak 1-s bin is MET 262708467 (43k events). Cache spans
    # MET 262708462 to 262708471 inclusive (the +9s bin at 262708472 has only 3
    # events and is incomplete coverage; drop it).
    T0_200428 = 262708467.0  # peak 1-s bin
    bursts["200428"] = compute_ratio(
        "FRB/XRB 200428", "20200428", "140000",
        trigger_met=T0_200428, t_lo_rel=-5, t_hi_rel=4,
        recon_csv="data/cache_frb200428_reconstruct_3box.csv",
        t_years_const=(np.datetime64("2020-04-28") - T_REF).astype("timedelta64[D]").astype(float) / 365.25,
        orbit_path="data/hxmt_aux/HXMT_20200428T14_Orbit_FFFFFF_V1_1K.FITS",
    )

    # GRB 260226A: trigger UTC 2026-02-26T10:37:50 -> MET 446726273 leap-aware (T10 hour, NOT T13).
    # Burst-peak phase T0+20 to T0+40s per §5.3; analyse window T0-30 to T0+70s for stats.
    recon_260226 = Path("data/cache_260226a_reconstruct.csv")
    if recon_260226.exists():
        T0_260226 = 446726273.0
        bursts["260226A"] = compute_ratio(
            "GRB 260226A", "20260226", "100000",
            trigger_met=T0_260226, t_lo_rel=-30, t_hi_rel=70,
            recon_csv=str(recon_260226),
            t_years_const=(np.datetime64("2026-02-26") - T_REF).astype("timedelta64[D]").astype(float) / 365.25,
            orbit_path="data/hxmt_aux/HXMT_20260226T10_Orbit_FFFFFF_V1_1K.FITS",
        )
    else:
        print("\n=== GRB 260226A ===")
        print(f"  NOTE: {recon_260226} not present; generate first via blink_cli reconstruct")


if __name__ == "__main__":
    main()
