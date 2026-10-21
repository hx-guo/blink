#!/usr/bin/env python3
"""Plot HXMT/HE reconstructed light curve vs INTEGRAL/IBIS-ISGRI + engineering.

Target: SGR 1935+2154 X-ray short burst simultaneous with FRB 200428
(2020-04-28T14:34:24.011 UTC, geocentric).

The HXMT/HE 1B reconstruction adds events to the FIFO-gap region of the
brightest sub-peak. INTEGRAL/IBIS-ISGRI photon events are read directly
from the public Science Window archive; their TIME column is in IJD (days
since 2000-01-01 00:00:00 TT) and we apply a constant ~400 ms light-travel
projection to put them on a common geocentric reference.

Usage:
    python3 scripts/bursts/plot_hxmt_vs_ibis.py --bin 0.005 --before 0.5 --after 1.0
"""

import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from datetime import datetime, timezone

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from engineering_prediction import load_engineering_prediction, T_REF

# ── HXMT config ──
HXMT_TRIGGER_UTC_STR = "2020-04-28T14:34:24.011"
HXMT_CACHE = "data/cache_frb200428_reconstruct_3box.csv"

# ── IBIS config ──
IBIS_FILE = "data/integral_ibis/isgri_events_222200240010.fits.gz"
# Geometry needed to compute the INTEGRAL→geocentric light-travel
# projection at burst time.  SGR 1935+2154 J2000 coordinates and the
# INTEGRAL spacecraft position vector (geocentric, J2000 equatorial) are
# combined to project the photon travel time R·n_source/c onto the
# geocentric reference frame.
SGR1935_RA_DEG = 293.7317   # 19h34m55.598s
SGR1935_DEC_DEG = 21.8966   # +21d53m47.7s
INTEGRAL_ORBIT_FILE = "data/integral_ibis/sc_orbit_param.fits.gz"
HXMT_ORBIT_FILE = "data/hxmt_aux/HXMT_20200428T14_Orbit_FFFFFF_V1_1K.FITS"


def compute_hxmt_light_travel(burst_met, burst_utc, orbit_file):
    """R_HXMT·n_source / c at burst time, in seconds, with frame correction.

    HXMT 1K-format orbit file gives spacecraft position (X, Y, Z in metres)
    in the **ECEF / ITRS** (Earth-fixed) frame, NOT in J2000 / GCRS.
    Source-direction unit vector is naturally in J2000.  To compute the
    R·n_src dot product correctly both must be in the same frame, so we
    transform the HXMT position from ITRS to GCRS at the burst epoch
    using astropy before projecting onto the source direction.  Mixing
    ECEF position with J2000 source direction introduces an error of
    order R·sin(GMST_diff) / c ~ 10 ms — exactly the residual we saw
    against IBIS before this correction.
    """
    ra, dec = np.deg2rad(SGR1935_RA_DEG), np.deg2rad(SGR1935_DEC_DEG)
    n = np.array([np.cos(dec) * np.cos(ra),
                  np.cos(dec) * np.sin(ra),
                  np.sin(dec)])
    with fits.open(orbit_file) as f:
        d = f[1].data
        # X/Y/Z stored in metres (ECEF); interpolate to burst MET.
        x_ecef = np.interp(burst_met, d["Time"].astype(float),
                           d["X"].astype(float))
        y_ecef = np.interp(burst_met, d["Time"].astype(float),
                           d["Y"].astype(float))
        z_ecef = np.interp(burst_met, d["Time"].astype(float),
                           d["Z"].astype(float))
    # ECEF (ITRS) → ECI (GCRS) via astropy at the burst epoch
    from astropy.coordinates import ITRS, GCRS, CartesianRepresentation
    import astropy.units as u
    itrs = ITRS(
        CartesianRepresentation(x_ecef * u.m, y_ecef * u.m, z_ecef * u.m),
        obstime=burst_utc,
    )
    gcrs = itrs.transform_to(GCRS(obstime=burst_utc))
    R_eci_km = np.array([
        gcrs.cartesian.x.to(u.m).value,
        gcrs.cartesian.y.to(u.m).value,
        gcrs.cartesian.z.to(u.m).value,
    ]) / 1000.0
    return float(np.dot(R_eci_km, n) / 299792.458)


def compute_integral_light_travel(burst_ijd, ibis_file, orbit_file):
    """R·n_source / c at burst time, geocentric.

    Returns the time correction in seconds that must be ADDED to IBIS-ISGRI
    photon times to project them onto a geocentric reference (positive
    when INTEGRAL is closer to the source than the Earth centre, since
    INTEGRAL then detects the photon earlier).

    INTEGRAL spacecraft position is interpolated to the burst time using
    the IBIS GTI table (which gives OB_TIME ↔ IJD mapping) to convert
    OB_TIME→IJD for the orbit-parameter file.
    """
    # Unit vector toward SGR 1935+2154 (geocentric equatorial J2000)
    ra, dec = np.deg2rad(SGR1935_RA_DEG), np.deg2rad(SGR1935_DEC_DEG)
    n = np.array([np.cos(dec) * np.cos(ra),
                  np.cos(dec) * np.sin(ra),
                  np.sin(dec)])
    # Build OB_TIME → IJD calibration from the GTI table.  INTEGRAL OBT is
    # a 4-int packed format [revolution, day, ticks_1/16s, ticks_1/(16·65536)s];
    # decoding both the 1/16-s and the sub-1/16-s columns gives µs-level
    # accuracy.  Using col2 alone leaves a 18-32 ms residual which would
    # propagate directly into the light-travel correction.
    def _obt_seconds(obt_arr):
        # obt_arr shape: (n, 4); returns seconds since start of OBT day
        return obt_arr[:, 2].astype(float) / 16.0 + obt_arr[:, 3].astype(float) / (16.0 * 65536.0)

    with fits.open(ibis_file) as fe:
        gti = fe["IBIS-GNRL-GTI"].data
        gti_ob_sec = _obt_seconds(gti["OBT_START"])
        gti_ijd_start = gti["START"].astype(float)
    p = np.polyfit(gti_ob_sec, gti_ijd_start, 1)
    # Interpolate INTEGRAL position to burst_ijd via OBT→IJD fit
    with fits.open(orbit_file) as f:
        d = f["INTL-ORBI-SCP"].data
        ob_sec = _obt_seconds(d["OB_TIME"])
        ob_ijd = np.polyval(p, ob_sec)
        # Interpolate position to burst_ijd
        xp = np.interp(burst_ijd, ob_ijd, d["XPOS"].astype(float))
        yp = np.interp(burst_ijd, ob_ijd, d["YPOS"].astype(float))
        zp = np.interp(burst_ijd, ob_ijd, d["ZPOS"].astype(float))
        R = np.array([xp, yp, zp])
    c_km_s = 299792.458
    return float(np.dot(R, n) / c_km_s)


def load_hxmt_from_cache(cache_path, trigger_met):
    """Read 1B reconstruction CSV, return (obs_t, fill_t) relative to trigger."""
    obs, fill = [], []
    with open(cache_path) as cf:
        for line in cf:
            p = line.strip().split(",")
            if len(p) < 3 or p[0] == "box":
                continue
            met = float(p[2])
            t = met - trigger_met
            if p[1] == "EVT":
                obs.append(t)
            elif p[1] == "FILL_GAP":
                fill.append(t)
    return np.array(obs), np.array(fill)


def load_ibis_events(ibis_file, hxmt_trigger_tt_ijd, light_travel):
    """Load IBIS-ISGRI events, return time relative to HXMT trigger (geocentric).

    IBIS TIME column is in IJD (days from 2000-01-01 00:00:00 TT). We convert
    to seconds relative to the HXMT trigger (also in TT), then apply the
    INTEGRAL→geocentric light-travel projection so the time axis is common
    with HXMT.
    """
    with fits.open(ibis_file, memmap=True) as f:
        ev = f["ISGR-EVTS-ALL"].data
        t_ijd = ev["TIME"].astype(float)
        energy = ev["ISGRI_ENERGY"].astype(float)
    t_rel_tt = (t_ijd - hxmt_trigger_tt_ijd) * 86400.0
    return t_rel_tt + light_travel, energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=float, default=0.005, help="Bin width (s)")
    parser.add_argument("--before", type=float, default=0.5)
    parser.add_argument("--after", type=float, default=1.0)
    parser.add_argument("--bkg", type=float, nargs=4, default=[-0.5, -0.1, 0.7, 1.0],
                        metavar=("T1", "T2", "T3", "T4"))
    parser.add_argument("--scale-range", type=float, nargs=2, default=[0.0, 0.7],
                        metavar=("S1", "S2"))
    parser.add_argument("--energy-lo", type=float, default=20.0,
                        help="IBIS-ISGRI lower energy cut (keV); matches "
                             "Mereghetti 2020 default")
    parser.add_argument("--energy-hi", type=float, default=200.0)
    parser.add_argument("--xlim", type=float, nargs=2, default=None)
    parser.add_argument("--ylim", type=float, default=None)
    parser.add_argument("-o", "--output", default="hxmt_vs_ibis.png")
    args = parser.parse_args()

    # Time alignment.  The 1B reconstruction cache uses astropy-style MET
    # (real elapsed SI seconds since 2012-01-01 UTC, leap-second-aware), so
    # we subtract the astropy trigger MET to put events on a clean
    # trigger-relative axis.  The Python-style MET (no leap) is also kept
    # for the engineering loader, which reads HXMT 1B FITS files whose
    # Time column uses the same Python convention.
    hxmt_trigger_utc = Time(HXMT_TRIGGER_UTC_STR, scale="utc")
    hxmt_epoch_utc = Time("2012-01-01T00:00:00", scale="utc")
    hxmt_trigger_met_astropy = (hxmt_trigger_utc - hxmt_epoch_utc).sec
    hxmt_trigger_met_python = (
        datetime.strptime(HXMT_TRIGGER_UTC_STR.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        .replace(tzinfo=timezone.utc) -
        datetime(2012, 1, 1, tzinfo=timezone.utc)
    ).total_seconds()
    ijd_zero = Time("2000-01-01T00:00:00", scale="tt")
    hxmt_trigger_tt_ijd = (hxmt_trigger_utc.tt - ijd_zero).to("day").value
    print(f"  HXMT trigger UTC: {hxmt_trigger_utc.iso}", file=sys.stderr)
    print(f"  HXMT trigger MET (astropy): {hxmt_trigger_met_astropy:.3f}", file=sys.stderr)
    print(f"  HXMT trigger MET (Python):  {hxmt_trigger_met_python:.3f} "
          f"(off by {hxmt_trigger_met_python - hxmt_trigger_met_astropy:+.1f}s leap)",
          file=sys.stderr)
    print(f"  HXMT trigger IJD (TT):      {hxmt_trigger_tt_ijd:.6f}", file=sys.stderr)

    # HXMT light-travel projection to geocentric reference at burst time.
    hxmt_light_travel = compute_hxmt_light_travel(
        burst_met=hxmt_trigger_met_astropy,
        burst_utc=hxmt_trigger_utc,
        orbit_file=HXMT_ORBIT_FILE,
    )
    print(f"  HXMT→geocentric light-travel: {hxmt_light_travel*1000:+.2f} ms "
          f"(R_HXMT·n_source/c)", file=sys.stderr)

    # HXMT (from cache, astropy MET convention); apply HXMT→geocentric
    # light-travel correction so all traces share a geocentric time axis.
    print("Loading HXMT/HE reconstruction (cache)...", file=sys.stderr)
    hxmt_obs, hxmt_fill = load_hxmt_from_cache(HXMT_CACHE, hxmt_trigger_met_astropy)
    hxmt_obs = hxmt_obs + hxmt_light_travel
    hxmt_fill = hxmt_fill + hxmt_light_travel if len(hxmt_fill) > 0 else hxmt_fill
    hxmt_all = (np.concatenate([hxmt_obs, hxmt_fill])
                if len(hxmt_fill) > 0 else hxmt_obs)

    # Ge 2023 (ApJ 953, 67) reanalysis: gap-averaged-rate fill where the
    # rate is taken from HE engineering counters (P/L/W/D conservation
    # equation), which are independent of the FIFO and already include the
    # PDAU-level deadtime correction.  We use this approach (rather than a
    # naive pre/post-observed average) to match the spirit of their
    # algorithm.  For each detected gap, the total fill = (engineering
    # events in the containing 1-Hz frame) − (HXMT-observed events in the
    # same frame outside the gap), then distributed uniformly across the
    # gap interval.  The engineering rate is loaded later below; we defer
    # this computation until after `eng_rate` is available.
    hxmt_ge_fill = np.array([])
    hxmt_ge_all = hxmt_obs
    print(f"  HXMT: {len(hxmt_obs):,} obs + {len(hxmt_fill):,} fill", file=sys.stderr)

    # IBIS — compute INTEGRAL→geocentric light-travel projection from
    # actual spacecraft position and source direction at the burst time.
    light_travel = compute_integral_light_travel(
        burst_ijd=hxmt_trigger_tt_ijd,
        ibis_file=IBIS_FILE,
        orbit_file=INTEGRAL_ORBIT_FILE,
    )
    print(f"  INTEGRAL→geocentric light-travel: {light_travel:+.4f} s "
          f"(R·n_source/c at burst time)", file=sys.stderr)
    print("Loading INTEGRAL/IBIS-ISGRI events...", file=sys.stderr)
    ibis_t, ibis_e = load_ibis_events(IBIS_FILE, hxmt_trigger_tt_ijd, light_travel)
    if args.energy_lo is not None and args.energy_hi is not None:
        e_mask = (ibis_e >= args.energy_lo) & (ibis_e < args.energy_hi)
        ibis_t = ibis_t[e_mask]
        print(f"  IBIS: {len(ibis_t):,} events in {args.energy_lo:.0f}-{args.energy_hi:.0f} keV",
              file=sys.stderr)
    else:
        print(f"  IBIS: {len(ibis_t):,} events (no energy filter)", file=sys.stderr)

    # Engineering at 1 Hz cadence — load over a wider window so we have several
    # bins on either side of the burst for the step trace to be visible.
    print("Loading engineering-channel prediction...", file=sys.stderr)
    t_years_const = ((np.datetime64("2020-04-28") - T_REF)
                     .astype("timedelta64[D]").astype(float) / 365.25)
    eng_t_raw, eng_rate = load_engineering_prediction(
        date_str="20200428", hour_str="140000",
        trigger_met=hxmt_trigger_met_python,
        before=10.0, after=10.0,
        t_years_const=t_years_const,
        orbit_path="data/hxmt_aux/HXMT_20200428T14_Orbit_FFFFFF_V1_1K.FITS",
    )
    if eng_t_raw is None:
        eng_t = None
        print("  WARN: engineering data missing", file=sys.stderr)
    else:
        # Engineering loader returns t_rel against Python MET trigger; shift
        # by the leap-second delta so it aligns with the astropy-MET cache
        # used in the main panel.
        eng_t = eng_t_raw + (hxmt_trigger_met_python - hxmt_trigger_met_astropy)
        print(f"  Engineering 1-Hz frames: {len(eng_t)} "
              f"(shifted by {hxmt_trigger_met_python - hxmt_trigger_met_astropy:+.1f}s to astropy frame)",
              file=sys.stderr)

    # Compute Ge 2023-style gap fill using engineering rate (deferred from
    # above so we have eng data).  For each detected gap, find the 1-Hz
    # engineering frame containing it, and set the gap fill total to
    # (engineering events in that frame) − (HXMT-observed events in the
    # same frame outside the gap).
    if len(hxmt_fill) > 0 and eng_t is not None:
        f_sorted = np.sort(hxmt_fill)
        gap_thresh = 0.05
        gaps = []
        g_events_buf = [f_sorted[0]]
        for tval in f_sorted[1:]:
            if tval - g_events_buf[-1] < gap_thresh:
                g_events_buf.append(tval)
            else:
                gaps.append((g_events_buf[0], g_events_buf[-1]))
                g_events_buf = [tval]
        gaps.append((g_events_buf[0], g_events_buf[-1]))

        obs_sorted = np.sort(hxmt_obs)
        ge_fill_parts = []
        for g_start, g_end in gaps:
            # Engineering frame [a, a+1) containing the gap_start
            # (eng_t are 1-Hz LEFT edges of frames in our display time)
            cand = np.where((eng_t <= g_start) & (g_start < eng_t + 1.0))[0]
            if len(cand) == 0:
                continue
            frame_idx = cand[0]
            a = eng_t[frame_idx]
            b = a + 1.0
            eng_total = float(eng_rate[frame_idx])  # events in this 1-s frame
            in_frame = ((obs_sorted >= a) & (obs_sorted < b))
            in_gap = (obs_sorted >= g_start) & (obs_sorted < g_end)
            obs_in_frame_outside_gap = int(np.sum(in_frame & ~in_gap))
            n_fill = max(0, int(round(eng_total - obs_in_frame_outside_gap)))
            if n_fill > 0:
                ge_fill_parts.append(np.linspace(g_start, g_end, n_fill))
            print(f"  Ge 2023 gap [{g_start*1000:.0f},{g_end*1000:.0f}] ms: "
                  f"eng_frame_total={eng_total:.0f}, obs_in_frame_outside_gap="
                  f"{obs_in_frame_outside_gap}, gap_fill={n_fill}",
                  file=sys.stderr)
        hxmt_ge_fill = (np.concatenate(ge_fill_parts) if ge_fill_parts
                        else np.array([]))
        hxmt_ge_all = np.concatenate([hxmt_obs, hxmt_ge_fill])
        print(f"  Ge 2023-style fill total: {len(hxmt_ge_fill):,} events",
              file=sys.stderr)

    # ── Binning ──
    bin_w = args.bin
    edges = np.arange(-args.before, args.after + bin_w, bin_w)
    x = edges[:-1]
    t1, t2, t3, t4 = args.bkg

    r_hxmt_obs = np.histogram(hxmt_obs, bins=edges)[0] / bin_w
    r_hxmt_all = np.histogram(hxmt_all, bins=edges)[0] / bin_w
    r_hxmt_ge = np.histogram(hxmt_ge_all, bins=edges)[0] / bin_w
    r_ibis = np.histogram(ibis_t, bins=edges)[0] / bin_w

    bkg_mask = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    bkg_hxmt = r_hxmt_all[bkg_mask].mean() if bkg_mask.any() else 0
    bkg_ibis = r_ibis[bkg_mask].mean() if bkg_mask.any() else 0
    print(f"  HXMT background: {bkg_hxmt:.0f} evt/s ({bkg_mask.sum()} bins)",
          file=sys.stderr)
    print(f"  IBIS background: {bkg_ibis:.0f} cts/s", file=sys.stderr)

    net_hxmt_obs = r_hxmt_obs - bkg_hxmt
    net_hxmt_all = r_hxmt_all - bkg_hxmt
    net_hxmt_ge = r_hxmt_ge - bkg_hxmt
    net_ibis = r_ibis - bkg_ibis

    # Scale IBIS to HXMT in the scale_range window (non-saturated burst segment)
    s1, s2 = args.scale_range
    sm = (x >= s1) & (x < s2)
    sum_hxmt = net_hxmt_all[sm].sum()
    sum_ibis = net_ibis[sm].sum()
    scale = sum_hxmt / sum_ibis if sum_ibis > 0 else 1.0
    net_ibis_scaled = net_ibis * scale
    print(f"  Scale: ×{scale:.1f}", file=sys.stderr)

    # Engineering background (using its OWN 1-s bins)
    if eng_t is not None:
        # Use frames outside [-0.5, +1.0] burst window as background
        eng_bkg_mask = (eng_t < -2.0) | (eng_t > 2.0)
        bkg_eng = eng_rate[eng_bkg_mask].mean() if eng_bkg_mask.any() else 0.0
        net_eng = eng_rate - bkg_eng
        print(f"  Engineering background: {bkg_eng:.0f} evt/s ({eng_bkg_mask.sum()} bins)",
              file=sys.stderr)

    # ── Plot ──
    import colorsys
    import matplotlib.colors as _mc
    _h, _l, _s = colorsys.rgb_to_hls(*_mc.to_rgb("C0"))
    NAVY = colorsys.hls_to_rgb(_h, 0.25, _s)
    SKY_BLUE = colorsys.hls_to_rgb(_h, 0.58, _s)
    CROSS_LW = 1.2

    fig, (ax_lc, ax_ratio) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    ax_lc.fill_between(x, 0, np.nan_to_num(net_hxmt_obs), step="post", alpha=0.55,
                       color="C0", zorder=1)
    ax_lc.fill_between(x, np.nan_to_num(net_hxmt_obs), np.nan_to_num(net_hxmt_all),
                       step="post", alpha=0.30, color="C0", zorder=2)
    ax_lc.step(x, net_hxmt_obs, where="post", color=NAVY, lw=1.0,
               label="HXMT/HE observed", zorder=3)
    ax_lc.step(x, net_hxmt_all, where="post", color=SKY_BLUE, lw=1.0,
               label=f"HXMT/HE + reconstructed (+{len(hxmt_fill):,})", zorder=4)
    # Ge 2023 (ApJ 953, 67) reanalysis — DIGITIZED data points from their
    # Figure 1 (HE 15-250 keV recovered light curve, 5 ms binning).
    # Per their caption, BLACK squares = unaffected (normal) bins and
    # RED circles = saturated bins recovered by their PDAU deadtime
    # algorithm; black+red together form the complete recovered curve.
    # Their T0 = 2020-04-28T14:34:24.4265 UTC (CHIME P1); we shift to our
    # T0 (= 14:34:24.011) by adding +415.5 ms.  Rates are in 10^4 cnts/s.
    #
    # Energy-band calibration: Ge uses 15-250 keV while our HXMT cache has
    # lost the channel information (channel=0 for all reconstructed events,
    # so we cannot energy-filter here).  To compare shapes on a common
    # scale, we derive an energy-band factor from the NON-saturated black
    # squares: for each Ge black point at time t_b we lookup our HXMT
    # observed rate in the corresponding 5 ms bin and take the median of
    # the ratios.  Applying that single factor to all Ge points (red+black)
    # puts them on our energy-band scale; the saturated-region comparison
    # (HXMT reconstructed vs Ge red circles, both rescaled) is then a
    # genuine shape/amplitude test of the two recovery algorithms.
    GE_RED = Path(__file__).resolve().parents[2] / "data" / "ge2023" / "ge2023_fig1_red.csv"
    GE_BLACK = Path(__file__).resolve().parents[2] / "data" / "ge2023" / "ge2023_fig1_black.csv"
    if GE_RED.exists() and GE_BLACK.exists():
        ge_r = np.loadtxt(GE_RED, delimiter=",", skiprows=1)
        ge_b = np.loadtxt(GE_BLACK, delimiter=",", skiprows=1)
        ge_red_t = ge_r[:, 1] / 1000.0
        ge_blk_t = ge_b[:, 1] / 1000.0
        ge_red_rate_raw = ge_r[:, 2] * 1e4  # cnts/s (15-250 keV)
        ge_blk_rate_raw = ge_b[:, 2] * 1e4

        # ---- derive energy-band scale from Ge black squares OUTSIDE the
        # HXMT FIFO gap ---- both detectors must have valid data in the
        # calibration bins, so we exclude any black square that lies inside
        # our FIFO-saturation window.  Without this filter the in-gap black
        # squares (where HXMT_obs ≈ 0) drag the median ratio down.
        if len(gaps) > 0:
            gap_starts = np.array([g[0] for g in gaps])
            gap_ends = np.array([g[1] for g in gaps])
            in_gap = np.zeros_like(ge_blk_t, dtype=bool)
            for g0, g1 in zip(gap_starts, gap_ends):
                in_gap |= (ge_blk_t >= g0) & (ge_blk_t <= g1)
        else:
            in_gap = np.zeros_like(ge_blk_t, dtype=bool)
        outside_gap = ~in_gap
        print(f"  Ge 2023 black squares: {in_gap.sum()} inside FIFO gap, "
              f"{outside_gap.sum()} outside (for energy-band calibration)",
              file=sys.stderr)

        idx = np.searchsorted(edges, ge_blk_t) - 1
        valid = (idx >= 0) & (idx < len(x)) & outside_gap
        if valid.sum() >= 3:
            hxmt_at_blk = r_hxmt_obs[idx[valid]]
            ge_blk_at_valid = ge_blk_rate_raw[valid]
            # need a real signal in both — drop near-zero / noise bins
            usable = (ge_blk_at_valid > 5e3) & (hxmt_at_blk > 5e3)
            if usable.sum() >= 3:
                ratios = hxmt_at_blk[usable] / ge_blk_at_valid[usable]
                eband_scale = float(np.median(ratios))
                print(f"  Ge 2023 energy-band scale (HXMT_all/Ge_15-250): "
                      f"×{eband_scale:.2f} from {usable.sum()} non-saturated points",
                      file=sys.stderr)
            else:
                eband_scale = 1.0
                print(f"  Ge 2023: only {usable.sum()} usable non-saturated bins, "
                      "using ×1.0", file=sys.stderr)
        else:
            eband_scale = 1.0
            print("  Ge 2023: no overlapping non-saturated bins, using ×1.0",
                  file=sys.stderr)

        ge_red_rate = ge_red_rate_raw * eband_scale - bkg_hxmt
        ge_blk_rate = ge_blk_rate_raw * eband_scale - bkg_hxmt
        ax_lc.plot(ge_blk_t, ge_blk_rate, "s", color="black",
                   markersize=4, markerfacecolor="none", markeredgewidth=0.8,
                   alpha=0.85,
                   label=f"Ge 2023 normal (×{eband_scale:.1f} band, {len(ge_blk_t)} pts)",
                   zorder=5)
        ax_lc.plot(ge_red_t, ge_red_rate, "o", color="crimson",
                   markersize=4, markeredgewidth=0, alpha=0.85,
                   label=f"Ge 2023 recovered (×{eband_scale:.1f} band, {len(ge_red_t)} pts)",
                   zorder=6)
    ax_lc.step(x, net_ibis_scaled, where="post", color="C1", lw=CROSS_LW,
               label=(f"INTEGRAL/IBIS-ISGRI {args.energy_lo:.0f}-{args.energy_hi:.0f} keV (×{scale:.1f})") if args.energy_lo is not None else f"INTEGRAL/IBIS-ISGRI (×{scale:.1f})",
               zorder=5)

    # Engineering at 1 Hz cadence — left-edge aligned step.  On the 5 ms
    # main panel this is a wide plateau across the 1-s frame containing the
    # burst with near-zero level elsewhere; the inset below shows a finer
    # 1-s rebinning over ±5 s for a side-by-side comparison.
    if eng_t is not None:
        eng_edges = np.concatenate([eng_t, [eng_t[-1] + 1.0]])
        eng_step_x = np.repeat(eng_edges, 2)[1:-1]
        eng_step_y = np.repeat(net_eng, 2)
        ax_lc.plot(eng_step_x, eng_step_y, color="C2", lw=CROSS_LW,
                   label=r"engineering $\widehat{S}_{\rm rec}^{\rm eng}$ (1 Hz, 18 det)",
                   zorder=6)

    # ── Inset: 1-Hz coarse-binning view including engineering channel.
    if eng_t is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # Anchor slightly inside the axes — enough to clear the main panel's
        # y-tick labels on the left without crowding the title above.
        ax_ins = inset_axes(ax_lc, width="32%", height="36%",
                            loc="upper left",
                            bbox_to_anchor=(0.07, -0.08, 1.0, 1.0),
                            bbox_transform=ax_lc.transAxes,
                            borderpad=0)
        ins_lo, ins_hi = -5.0, 5.0
        ins_edges = np.arange(ins_lo, ins_hi + 1.0, 1.0)
        ins_x = ins_edges[:-1]
        r_hxmt_ins = np.histogram(hxmt_all, bins=ins_edges)[0] / 1.0
        bkg_h_ins = r_hxmt_ins[(ins_x < -2) | (ins_x > 2)].mean()
        r_hxmt_ins_net = r_hxmt_ins - bkg_h_ins
        r_ibis_ins_counts = np.histogram(ibis_t, bins=ins_edges)[0]
        # INTEGRAL ScW has GTI gaps (dither slews) that do NOT align to
        # integer-second boundaries.  Compute the actual GTI overlap per
        # 1-s bin and divide by the effective exposure, not by the nominal
        # 1 s, so partial-GTI bins don't get rate-underestimated.  Mask
        # bins whose GTI coverage is < 0.1 s.
        from astropy.io import fits as _fits
        from astropy.time import Time as _Time
        with _fits.open(IBIS_FILE) as _h:
            _gti = _h["IBIS-GNRL-GTI"].data
        _burst_iso = _Time(HXMT_TRIGGER_UTC_STR, scale="utc")
        gti_start_rel = np.array(
            [(_Time(r["UTC_START"][:23], scale="utc") - _burst_iso).sec
             for r in _gti])
        gti_end_rel = np.array(
            [(_Time(r["UTC_END"][:23], scale="utc") - _burst_iso).sec
             for r in _gti])
        ins_exposure = np.zeros(len(ins_x))
        for i, (b_lo, b_hi) in enumerate(zip(ins_edges[:-1], ins_edges[1:])):
            lo = np.maximum(gti_start_rel, b_lo)
            hi = np.minimum(gti_end_rel, b_hi)
            overlap = np.clip(hi - lo, 0, None).sum()
            ins_exposure[i] = overlap
        # mask bins with negligible GTI coverage
        gti_mask = ins_exposure < 0.1
        with np.errstate(divide="ignore", invalid="ignore"):
            r_ibis_ins = np.where(gti_mask, np.nan,
                                  r_ibis_ins_counts / ins_exposure)
        full_bins = (np.abs(ins_x) >= 2) & (ins_exposure >= 0.9)
        bkg_i_ins = np.nanmean(r_ibis_ins[full_bins]) if full_bins.any() else 0
        r_ibis_ins_net = (r_ibis_ins - bkg_i_ins) * scale
        # Engineering 1-Hz step (left-edge aligned)
        eng_edges_full = np.concatenate([eng_t, [eng_t[-1] + 1.0]])
        eng_step_x = np.repeat(eng_edges_full, 2)[1:-1]
        eng_step_y = np.repeat(net_eng, 2)
        # Shade INTEGRAL dither slews (intervals between GTIs, where IBIS
        # is not taking data) — drawn underneath everything else.
        labelled = False
        for i in range(len(gti_start_rel) - 1):
            slew_lo = gti_end_rel[i]
            slew_hi = gti_start_rel[i + 1]
            if slew_hi < ins_lo or slew_lo > ins_hi: continue
            kw = {"color": "0.7", "alpha": 0.45, "zorder": 0}
            if not labelled:
                kw["label"] = "INTEGRAL slew"
                labelled = True
            ax_ins.axvspan(max(slew_lo, ins_lo), min(slew_hi, ins_hi), **kw)
        ax_ins.step(ins_x, r_hxmt_ins_net, where="post", color=SKY_BLUE, lw=1.0,
                    label="HXMT", zorder=3)
        ax_ins.step(ins_x, r_ibis_ins_net, where="post", color="C1",
                    lw=CROSS_LW, label="IBIS", zorder=3)
        ax_ins.plot(eng_step_x, eng_step_y, color="C2", lw=CROSS_LW,
                    label="eng.", zorder=3)
        ax_ins.axhline(0, color="gray", lw=0.4, ls="--")
        ax_ins.set_xlim(ins_lo, ins_hi)
        ax_ins.tick_params(labelsize=7)
        ax_ins.set_title("1 s rebin (eng cadence)", fontsize=8)
        ax_ins.legend(fontsize=6.5, loc="upper right", framealpha=0.85,
                      handlelength=1.0, borderpad=0.3, labelspacing=0.2)

    ax_lc.set_ylabel("Net count rate (evt/s)")
    ax_lc.legend(loc="upper right", fontsize=9.5)
    ax_lc.axhline(0, color="gray", lw=0.5, ls="--")
    if args.ylim:
        ax_lc.set_ylim(-args.ylim * 0.05, args.ylim)
    ax_lc.set_title(
        f"FRB/XRB 200428: HXMT/HE event-level + engineering vs INTEGRAL/IBIS  "
        f"[{int(bin_w*1000)} ms bins, geocentric]",
        fontweight="bold")

    # Ratio panel
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_ibis = np.nanmax(net_ibis_scaled)
        peak_hxmt = np.nanmax(net_hxmt_all)
        thr_i = max(peak_ibis * 0.05, 100)
        thr_h = max(peak_hxmt * 0.05, 100)
        sig = ((net_ibis_scaled > thr_i) & (net_hxmt_all > thr_h)
               & np.isfinite(net_ibis_scaled) & np.isfinite(net_hxmt_all))
        ratio_ibis = np.where(sig, net_hxmt_all / net_ibis_scaled, np.nan)
    ax_ratio.step(x, ratio_ibis, where="post", color="C1", lw=CROSS_LW,
                  label="HXMT / IBIS")

    if eng_t is not None:
        eng_t_min = int(np.floor(eng_t[0]))
        idx = np.floor(x).astype(int) - eng_t_min
        valid = (idx >= 0) & (idx < len(net_eng))
        eng_up = np.where(valid, net_eng[np.clip(idx, 0, len(net_eng) - 1)], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            peak_eng = np.nanmax(eng_up)
            thr_e = max(peak_eng * 0.05, 100)
            sig_e = ((eng_up > thr_e) & (net_hxmt_all > thr_h)
                     & np.isfinite(eng_up) & np.isfinite(net_hxmt_all))
            ratio_eng = np.where(sig_e, net_hxmt_all / eng_up, np.nan)
        ax_ratio.step(x, ratio_eng, where="post", color="C2", lw=CROSS_LW,
                      label="HXMT / engineering")

    # HXMT / Ge ratio (per-marker scatter), split by color to match the
    # main panel: filled red circles = Ge "recovered" markers,
    # open black squares = Ge "normal" markers.
    ge_ratio_t = np.empty(0); ge_ratio_v = np.empty(0)
    if GE_RED.exists() and GE_BLACK.exists():
        def _ge_ratios(t_arr, rate_arr):
            idx = np.searchsorted(edges, t_arr) - 1
            valid = (idx >= 0) & (idx < len(x)) & (rate_arr > thr_h)
            if not valid.any():
                return np.empty(0), np.empty(0)
            hr = net_hxmt_all[idx[valid]]
            sig = (hr > thr_h) & np.isfinite(hr)
            return t_arr[valid][sig], hr[sig] / rate_arr[valid][sig]

        red_t_r, red_v_r = _ge_ratios(ge_red_t, ge_red_rate)
        blk_t_r, blk_v_r = _ge_ratios(ge_blk_t, ge_blk_rate)
        ge_ratio_t = np.concatenate([red_t_r, blk_t_r])
        ge_ratio_v = np.concatenate([red_v_r, blk_v_r])
        if len(red_t_r):
            ax_ratio.scatter(red_t_r, red_v_r,
                             marker="o", s=12, c="crimson",
                             alpha=0.75, edgecolors="none",
                             label=f"HXMT / Ge recovered ({len(red_t_r)})")
        if len(blk_t_r):
            ax_ratio.scatter(blk_t_r, blk_v_r,
                             marker="s", s=14, facecolors="none",
                             edgecolors="black", linewidths=0.8,
                             alpha=0.8,
                             label=f"HXMT / Ge normal ({len(blk_t_r)})")

    ax_ratio.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax_ratio.set_ylabel("HXMT / ref.")
    ax_ratio.set_ylim(0.5, 1.5)

    # Annotation: median + IQR
    xlim_lo = args.xlim[0] if args.xlim else -args.before
    xlim_hi = args.xlim[1] if args.xlim else args.after
    xm = (x >= xlim_lo) & (x < xlim_hi)

    def _robust(r):
        rv = r[np.isfinite(r)]
        if not len(rv):
            return None
        med = np.median(rv)
        q75, q25 = np.percentile(rv, [75, 25])
        return med, (q75 - q25) / 1.349, len(rv)

    annot = []
    si = _robust(ratio_ibis[xm])
    if si:
        annot.append(f"HXMT/IBIS        = {si[0]:.2f} ± {si[1]:.2f} ({si[2]} bins)")
    if eng_t is not None:
        se = _robust(ratio_eng[xm])
        if se:
            annot.append(f"HXMT/engineering = {se[0]:.2f} ± {se[1]:.2f} ({se[2]} bins)")
    if len(ge_ratio_v):
        in_range = (ge_ratio_t >= xlim_lo) & (ge_ratio_t < xlim_hi)
        sg = _robust(ge_ratio_v[in_range])
        if sg:
            annot.append(f"HXMT/Ge 2023     = {sg[0]:.2f} ± {sg[1]:.2f} ({sg[2]} pts)")
    if annot:
        ax_ratio.text(0.98, 0.92, "\n".join(annot),
                      transform=ax_ratio.transAxes, ha="right", va="top",
                      fontsize=8.5, family="monospace",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))
    ax_ratio.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax_ratio.set_xlabel(f"Time since trigger (s)  [$T_0$ = {HXMT_TRIGGER_UTC_STR} UTC]")
    if args.xlim:
        ax_ratio.set_xlim(args.xlim[0], args.xlim[1])
    else:
        ax_ratio.set_xlim(-args.before, args.after)

    plt.tight_layout()
    # PDF output is vector and ignores dpi; PNG uses dpi for raster
    # resolution.  Bump to 250 so .png output looks sharp on retina /
    # imgcat without needing a separate sips upscale pass.
    plt.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
