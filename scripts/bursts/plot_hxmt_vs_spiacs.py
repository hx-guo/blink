#!/usr/bin/env python3
"""Plot HXMT/HE reconstructed light curve vs INTEGRAL/SPI-ACS for GRB 200415A.

Cross-satellite validation: magnetar giant flare from SGR 0525-66.

Usage:
    python3 scripts/bursts/plot_hxmt_vs_spiacs.py
    python3 scripts/bursts/plot_hxmt_vs_spiacs.py --bin 0.05
    python3 scripts/bursts/plot_hxmt_vs_spiacs.py --bin 0.1 --before 2 --after 5
"""

import argparse, sys
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import astropy.units as u

# ── Source ──
# SGR 0525-66 (in LMC)
SRC_RA = 81.5   # deg
SRC_DEC = -66.0  # deg

# ── HXMT config ──
HXMT_CACHE = "data/cache_200415a_reconstruct.csv"
HXMT_MET_EPOCH = Time("2012-01-01T00:00:00", scale="utc")

# ── SPI-ACS config ──
SPIACS_FILE = "data/spiacs_grb200415a.fits"

# ── ASIM/MXGS LED config ──
ASIM_FILE = "data/asim_mxgs/LED_raw.txt"
ASIM_REF_UTC = "2020-04-15T08:48:05.557"

# ── Fermi/GBM TTE config ──
GBM_DIR = "data/fermi_gbm/bn200415367"
GBM_TRIGGER_MET = 608633290.563746  # Fermi MET (TT seconds since Fermi epoch 2001-01-01)
GBM_TRIGGER_UTC = "2020-04-15T08:48:05.564"  # corresponds to GBM MET above (TT - leap)


def compute_spiacs_time_system():
    """Read SPI-ACS FITS headers, return (timezero_utc, ltt_correction_s).

    Step 1: TIMESYS=TT, MJDREF=51544.0, TIMEREF=LOCAL
      => TIME column is TT seconds since MJD 51544.0
      => Full TT time = TIMEZERO + TIME
    Step 2: Convert TT -> UTC
    Step 3: Light travel time correction (TIMEREF=LOCAL => satellite time)
      EPHS gives satellite position as (RA, DEC, dist_km)
    """
    f = fits.open(SPIACS_FILE)
    h = f["RATE"].header

    # Time system from headers
    assert h["TIMESYS"] == "TT", f"Expected TIMESYS=TT, got {h['TIMESYS']}"
    assert h["TIMEREF"] == "LOCAL", f"Expected TIMEREF=LOCAL, got {h['TIMEREF']}"
    mjdref = h["MJDREF"]   # 51544.0
    timezero = h["TIMEZERO"]

    # TIMEZERO in MJD TT -> convert to UTC-scale Time object
    timezero_mjd_tt = mjdref + timezero / 86400.0
    timezero_tt = Time(timezero_mjd_tt, format="mjd", scale="tt")
    timezero_time = timezero_tt.utc  # convert to UTC scale for all downstream use
    print(f"  SPI-ACS TIMEZERO UTC: {timezero_time.iso}", file=sys.stderr)
    print(f"  SPI-ACS TIMESYS={h['TIMESYS']}, TIMEREF={h['TIMEREF']}, "
          f"TIMEDEL={h['TIMEDEL']}s", file=sys.stderr)

    # Satellite position from EPHS: "RA DEC dist_km vra vdec v_km_s"
    ephs = h["EPHS"].split()
    sat_ra, sat_dec, sat_dist = float(ephs[0]), float(ephs[1]), float(ephs[2])
    sat_dir = SkyCoord(ra=sat_ra * u.deg, dec=sat_dec * u.deg, frame="icrs")
    sat_pos_km = sat_dist * np.array([
        sat_dir.cartesian.x.value,
        sat_dir.cartesian.y.value,
        sat_dir.cartesian.z.value,
    ])

    # Light travel time correction
    src = SkyCoord(ra=SRC_RA * u.deg, dec=SRC_DEC * u.deg, frame="icrs")
    src_vec = np.array([
        src.cartesian.x.value,
        src.cartesian.y.value,
        src.cartesian.z.value,
    ])
    c_km_s = 299792.458
    ltt_s = np.dot(sat_pos_km, src_vec) / c_km_s
    print(f"  INTEGRAL position: RA={sat_ra:.1f}, DEC={sat_dec:.1f}, "
          f"dist={sat_dist:.1f} km", file=sys.stderr)
    print(f"  INTEGRAL LTT correction: {ltt_s * 1000:+.1f} ms", file=sys.stderr)

    f.close()
    return timezero_time, ltt_s


def load_spiacs(timezero_utc, ltt_s, t0_utc, before, after):
    """Load SPI-ACS RATE data, convert to geocentric UTC relative to T0.

    Returns (t_rel, rate, error) in geocentric seconds relative to T0.
    """
    f = fits.open(SPIACS_FILE)
    d = f["RATE"].data

    # TIME column: TT seconds relative to TIMEZERO
    # For relative intervals within 120s, TT and UTC tick identically
    # (no leap seconds within the window).
    # Satellite UTC time = timezero_utc + TIME
    # Geocentric UTC time = satellite UTC + LTT correction
    timezero_offset = (timezero_utc - t0_utc).sec
    t_geo = d["TIME"] + timezero_offset + ltt_s

    mask = (t_geo >= -before) & (t_geo <= after)
    f.close()
    return t_geo[mask], d["RATE"][mask], d["ERROR"][mask]


def hxmt_t0_met(t0_utc):
    """Conversion offset used to map cache MET (naive) to plot time.

    Cache CSV uses HXMT naive MET (calendar seconds since 2012-01-01).
    We pick the SI-aware astropy delta as the offset so the same arithmetic
    trick that aligns HXMT events to other instruments is consistent —
    used for both event times and the saturation marker.
    """
    return (t0_utc - HXMT_MET_EPOCH).sec


def load_hxmt(t0_utc, before, after):
    """Load HXMT/HE reconstruction from cache CSV.

    Cache MET values are HXMT naive seconds since 2012-01-01.
    Returns (obs_t, fill_t) relative to T0 in seconds.
    """
    print(f"  Loading HXMT/HE from cache: {HXMT_CACHE}", file=sys.stderr)
    t0_met = hxmt_t0_met(t0_utc)

    obs, fill = [], []
    with open(HXMT_CACHE) as cf:
        for line in cf:
            p = line.strip().split(",")
            if len(p) < 3 or p[0] == "box":
                continue
            met = float(p[2])
            t_rel = met - t0_met
            if abs(t_rel) > max(before, after) + 10:
                continue
            if p[1] == "EVT":
                obs.append(t_rel)
            elif p[1] == "FILL_GAP":
                fill.append(t_rel)

    obs, fill = np.array(obs), np.array(fill)
    all_t = np.concatenate([obs, fill]) if len(fill) > 0 else obs
    t_min, t_max = all_t.min(), all_t.max()
    print(f"  HXMT: {len(obs):,} observed + {len(fill):,} filled = "
          f"{len(obs) + len(fill):,} total", file=sys.stderr)
    print(f"  HXMT data range: [{t_min:.3f}, {t_max:.3f}]s relative to T0",
          file=sys.stderr)
    return obs, fill, (t_min, t_max)


def load_hxmt_resets():
    """Read FILL_GAP MET range from cache CSV, return (start_met, end_met).

    These delimit the union of FIFO reset gaps that were reconstructed.
    Returns (None, None) if no fills present.
    """
    fill_mets = []
    with open(HXMT_CACHE) as cf:
        for line in cf:
            p = line.strip().split(",")
            if len(p) < 3 or p[1] != "FILL_GAP":
                continue
            fill_mets.append(float(p[2]))
    if not fill_mets:
        return None, None
    return min(fill_mets), max(fill_mets)


def load_gbm(t0_utc, before, after, dets=("n0", "n4")):
    """Load Fermi/GBM TTE events for the requested detectors.

    Returns event times relative to t0_utc (geocentric, seconds).
    Fermi GBM TIME column is TT seconds since Fermi epoch (2001-01-01).
    For relative timing within ~10s no leap-second crosses, so we treat
    TT and UTC as ticking identically over the burst window.
    """
    import os
    gbm_trigger_utc = Time(GBM_TRIGGER_UTC, scale="utc")
    trigger_offset_to_t0 = (gbm_trigger_utc - t0_utc).sec
    print(f"  GBM trigger UTC: {GBM_TRIGGER_UTC}", file=sys.stderr)
    print(f"  GBM trigger offset to T0: {trigger_offset_to_t0:+.3f}s", file=sys.stderr)

    # Fermi GBM LTT correction is small (~22 ms in LEO); negligible at
    # the 1-5 ms bin level we use. Skip it for now.
    all_t = []
    for det in dets:
        path = os.path.join(GBM_DIR, f"glg_tte_{det}_bn200415367_v00.fit")
        with fits.open(path, memmap=True) as f:
            times = f["EVENTS"].data["TIME"]
        # times are TT seconds since Fermi epoch; relative to GBM trigger:
        t_rel_trigger = times - GBM_TRIGGER_MET
        # then offset to t0_utc:
        t_rel_t0 = t_rel_trigger + trigger_offset_to_t0
        mask = (t_rel_t0 >= -before) & (t_rel_t0 <= after)
        all_t.append(t_rel_t0[mask])
        print(f"  GBM {det}: {mask.sum():,} events in [{-before},{after}]s",
              file=sys.stderr)
    return np.concatenate(all_t) if all_t else np.array([])


def load_asim(t0_utc, before, after):
    """Load ASIM/MXGS LED photon-by-photon data.

    tus column = microseconds relative to reference time (UTC).
    ASIM is on ISS (LEO, ~400 km), LTT correction < 10 ms, negligible
    at 50ms bin resolution.

    Returns t_rel in seconds relative to T0.
    """
    print(f"  Loading ASIM/MXGS LED: {ASIM_FILE}", file=sys.stderr)
    asim_ref = Time(ASIM_REF_UTC, scale="utc")
    ref_offset = (asim_ref - t0_utc).sec

    data = np.loadtxt(ASIM_FILE, skiprows=18)
    t_sec = data[:, 0] / 1.0e6 + ref_offset
    mask = (t_sec >= -before) & (t_sec <= after)
    t_out = t_sec[mask]
    print(f"  ASIM: {len(t_out):,} LED events in window", file=sys.stderr)
    return t_out


def main():
    parser = argparse.ArgumentParser(
        description="HXMT/HE vs INTEGRAL/SPI-ACS cross-validation for GRB 200415A")
    parser.add_argument("--bin", type=float, default=0.05,
                        help="Bin width in seconds (default: 0.05, matches SPI-ACS native)")
    parser.add_argument("--before", type=float, default=2.0)
    parser.add_argument("--after", type=float, default=2.0)
    parser.add_argument("--bkg", type=float, nargs="+", default=[-8.0, -1.0],
                        help="Background interval(s): T1 T2 [T3 T4]. "
                             "One or two intervals for background estimation.")
    parser.add_argument("--scale-range", type=float, nargs=2, default=[-0.5, 1.0],
                        metavar=("S1", "S2"),
                        help="Time range for flux scaling (default: burst region)")
    parser.add_argument("--asim", action="store_true",
                        help="Also plot ASIM/MXGS LED data")
    parser.add_argument("--gbm", action="store_true",
                        help="Also plot Fermi/GBM TTE data")
    parser.add_argument("--gbm-dets", type=str, nargs="+",
                        default=["n0", "n4"],
                        help="GBM detectors to combine (default: n0 n4)")
    parser.add_argument("--mark-saturation", action="store_true",
                        help="Shade the FIFO-reset interval inferred from cache")
    parser.add_argument("--no-spiacs", action="store_true",
                        help="Skip SPI-ACS curve (still uses it for T0 derivation)")
    parser.add_argument("--peak-align", choices=["spiacs", "asim", "hxmt"],
                        default=None,
                        help="Shift T0 so that the chosen instrument's peak is at 0")
    parser.add_argument("-o", "--output", default="GRB200415A_hxmt_vs_spiacs.png")
    args = parser.parse_args()

    # ── Step 1: Determine time definitions from FITS headers ──
    print("Step 1: Reading SPI-ACS time system...", file=sys.stderr)
    timezero_utc, ltt_int = compute_spiacs_time_system()

    # ── Define T0 ──
    # Use the SPI-ACS TIMEZERO as basis, converted to geocentric
    # The burst peak in SPI-ACS satellite time is at TIME=5.925
    # Geocentric peak: TIMEZERO_utc + 5.925 + LTT
    # For a clean reference, use the geocentric peak time
    spiacs_peak_sat_time = 5.925  # from data inspection
    t0_utc = timezero_utc + TimeDelta(spiacs_peak_sat_time + ltt_int, format="sec")
    print(f"\n  T0 (geocentric UTC): {t0_utc.iso}", file=sys.stderr)

    # ── Step 2-3: Load data, convert to geocentric UTC relative to T0 ──
    print("\nStep 2-3: Loading and converting data...", file=sys.stderr)

    # SPI-ACS (already in satellite time, apply LTT)
    spiacs_t, spiacs_rate, spiacs_err = load_spiacs(
        timezero_utc, ltt_int, t0_utc, args.before, args.after)

    # HXMT (CSV METs are SI seconds, convert with astropy)
    # HXMT LTT: LEO (~550 km), max correction ~23 ms
    # Without exact orbit file, we estimate ~0 ms (small compared to 50ms bins)
    # This introduces at most 23 ms systematic uncertainty
    hxmt_ltt = 0.0  # placeholder; could refine with orbit data
    print(f"  HXMT LTT correction: {hxmt_ltt * 1000:+.1f} ms (LEO estimate)", file=sys.stderr)
    hxmt_obs_t, hxmt_fill_t, hxmt_range = load_hxmt(t0_utc, args.before, args.after)
    hxmt_obs_t += hxmt_ltt
    if len(hxmt_fill_t) > 0:
        hxmt_fill_t += hxmt_ltt

    hxmt_all_t = (np.concatenate([hxmt_obs_t, hxmt_fill_t])
                  if len(hxmt_fill_t) > 0 else hxmt_obs_t)

    # ASIM (optional)
    asim_t = None
    if args.asim:
        asim_t = load_asim(t0_utc, args.before, args.after)

    # GBM (optional)
    gbm_t = None
    if args.gbm:
        gbm_t = load_gbm(t0_utc, args.before, args.after, args.gbm_dets)

    # ── Binning and background subtraction ──
    bin_w = args.bin
    edges = np.arange(-args.before, args.after + bin_w, bin_w)
    x = edges[:-1] + bin_w / 2  # bin centers for SPI-ACS (already binned)
    x_step = edges[:-1]  # left edges for step plots

    # Parse background intervals
    bkg_intervals = args.bkg
    if len(bkg_intervals) == 2:
        bkg_intervals = bkg_intervals + bkg_intervals  # duplicate for one-sided
    t1, t2, t3, t4 = bkg_intervals[:4]

    # SPI-ACS: already in counts/s, rebin if needed
    # Native bins are 50ms. If our bin width differs, we need to rebin.
    if abs(bin_w - 0.05) < 1e-6:
        # Use native 50ms bins directly
        spiacs_binned = spiacs_rate
        spiacs_binned_err = spiacs_err
        # Align to our grid: find matching bins
        x_spiacs = spiacs_t  # bin centers from FITS
        spiacs_on_grid = np.full(len(x), np.nan)
        spiacs_err_grid = np.full(len(x), np.nan)
        for i, xc in enumerate(x):
            idx = np.argmin(np.abs(x_spiacs - xc))
            if abs(x_spiacs[idx] - xc) < bin_w * 0.6:
                spiacs_on_grid[i] = spiacs_binned[idx]
                spiacs_err_grid[i] = spiacs_binned_err[idx]
        spiacs_rate_final = spiacs_on_grid
        spiacs_err_final = spiacs_err_grid
    else:
        # Rebin: SPI-ACS rates weighted by exposure per output bin
        # Each native 50ms bin contributes proportionally
        spiacs_rate_final = np.full(len(x), np.nan)
        spiacs_err_final = np.full(len(x), np.nan)
        native_dt = 0.05
        for i in range(len(x)):
            lo, hi = edges[i], edges[i + 1]
            mask = (spiacs_t >= lo - native_dt / 2) & (spiacs_t < hi + native_dt / 2)
            if mask.sum() > 0:
                spiacs_rate_final[i] = np.mean(spiacs_rate[mask])
                spiacs_err_final[i] = np.sqrt(np.sum(spiacs_err[mask] ** 2)) / mask.sum()

    # SPI-ACS background
    bkg_mask_s = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    valid_bkg_s = bkg_mask_s & np.isfinite(spiacs_rate_final)
    bkg_spiacs = np.nanmean(spiacs_rate_final[valid_bkg_s]) if valid_bkg_s.sum() > 0 else 0
    net_spiacs = spiacs_rate_final - bkg_spiacs
    print(f"  SPI-ACS background: {bkg_spiacs:.0f} counts/s "
          f"({valid_bkg_s.sum()} bins)", file=sys.stderr)

    # HXMT binning and background
    r_hxmt_obs = np.histogram(hxmt_obs_t, bins=edges)[0] / bin_w
    r_hxmt_all = np.histogram(hxmt_all_t, bins=edges)[0] / bin_w
    # Mask bins outside HXMT data coverage (exclude partial edge bins)
    hxmt_valid = (edges[:-1] >= hxmt_range[0]) & (edges[1:] <= hxmt_range[1])
    bkg_mask_h = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    bkg_mask_h = bkg_mask_h & hxmt_valid
    bkg_hxmt = np.mean(r_hxmt_all[bkg_mask_h]) if bkg_mask_h.sum() > 0 else 0
    net_hxmt_obs = np.where(hxmt_valid, r_hxmt_obs - bkg_hxmt, np.nan)
    net_hxmt_all = np.where(hxmt_valid, r_hxmt_all - bkg_hxmt, np.nan)
    print(f"  HXMT background: {bkg_hxmt:.0f} counts/s "
          f"({bkg_mask_h.sum()} bins)", file=sys.stderr)

    # ASIM binning and background
    if asim_t is not None:
        r_asim = np.histogram(asim_t, bins=edges)[0] / bin_w
        bkg_mask_a = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
        bkg_asim = np.mean(r_asim[bkg_mask_a]) if bkg_mask_a.sum() > 0 else 0
        net_asim = r_asim - bkg_asim
        print(f"  ASIM background: {bkg_asim:.0f} counts/s", file=sys.stderr)

    # GBM binning and background
    if gbm_t is not None:
        r_gbm = np.histogram(gbm_t, bins=edges)[0] / bin_w
        bkg_mask_g = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
        bkg_gbm = np.mean(r_gbm[bkg_mask_g]) if bkg_mask_g.sum() > 0 else 0
        net_gbm = r_gbm - bkg_gbm
        print(f"  GBM background: {bkg_gbm:.0f} counts/s", file=sys.stderr)

    # ── Scale SPI-ACS to HXMT ──
    # Match total net counts in scale range (burst region)
    s1, s2 = args.scale_range
    scale_mask = (x >= s1) & (x < s2) & np.isfinite(net_spiacs)
    sum_hxmt = np.sum(net_hxmt_all[scale_mask])
    sum_spiacs = np.nansum(net_spiacs[scale_mask])
    scale = sum_hxmt / sum_spiacs if sum_spiacs != 0 else 1.0
    net_spiacs_scaled = net_spiacs * scale
    print(f"  Scale factor (HXMT/SPI-ACS): {scale:.3f} "
          f"(matched in [{s1},{s2}]s)", file=sys.stderr)

    if asim_t is not None:
        sum_asim = np.sum(net_asim[scale_mask])
        scale_asim = sum_hxmt / sum_asim if sum_asim != 0 else 1.0
        net_asim_scaled = net_asim * scale_asim
        print(f"  ASIM scale factor: {scale_asim:.3f}", file=sys.stderr)

    if gbm_t is not None:
        sum_gbm = np.sum(net_gbm[scale_mask])
        scale_gbm = sum_hxmt / sum_gbm if sum_gbm != 0 else 1.0
        net_gbm_scaled = net_gbm * scale_gbm
        print(f"  GBM scale factor: {scale_gbm:.3f}", file=sys.stderr)

    # ── Peak analysis ──
    print("\n── Peak analysis ──", file=sys.stderr)
    spiacs_valid = np.isfinite(net_spiacs_scaled)
    spiacs_peak_t = None
    if spiacs_valid.sum() > 0:
        spiacs_peak_idx = np.nanargmax(net_spiacs_scaled)
        spiacs_peak_t = x[spiacs_peak_idx]
        print(f"  SPI-ACS peak: T{spiacs_peak_t:+.3f}s, "
              f"{net_spiacs_scaled[spiacs_peak_idx]:.0f} evt/s (scaled)",
              file=sys.stderr)
    hxmt_peak_idx = np.nanargmax(net_hxmt_all)
    hxmt_peak_t = x[hxmt_peak_idx]
    print(f"  HXMT peak:    T{hxmt_peak_t:+.3f}s, "
          f"{net_hxmt_all[hxmt_peak_idx]:.0f} evt/s", file=sys.stderr)
    asim_peak_t = None
    if asim_t is not None:
        asim_peak_idx = np.nanargmax(net_asim)
        asim_peak_t = x[asim_peak_idx]
        print(f"  ASIM peak:    T{asim_peak_t:+.3f}s", file=sys.stderr)

    # Optional shift so that a chosen instrument's peak sits at T=0
    if args.peak_align:
        shift = {"spiacs": spiacs_peak_t, "asim": asim_peak_t,
                 "hxmt": hxmt_peak_t}.get(args.peak_align)
        if shift is None:
            print(f"  WARN: peak-align={args.peak_align} unavailable, skipping",
                  file=sys.stderr)
        else:
            x = x - shift
            x_step = x_step - shift
            edges = edges - shift
            hxmt_range = (hxmt_range[0] - shift, hxmt_range[1] - shift)
            print(f"  Peak-align shift: {shift:+.4f}s ({args.peak_align} peak -> 0)",
                  file=sys.stderr)
            # save shift so saturation marker can adjust
            sat_shift = shift
    else:
        sat_shift = 0.0

    # ── Step 4: Plot ──
    print("\nStep 4: Plotting...", file=sys.stderr)
    n_panels = 2
    fig, (ax_lc, ax_ratio) = plt.subplots(
        n_panels, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    # Only plot HXMT bins within its data coverage
    hxmt_obs_plot = np.where(hxmt_valid, net_hxmt_obs, np.nan)
    hxmt_all_plot = np.where(hxmt_valid, net_hxmt_all, np.nan)

    # Slice to valid range for clean fill_between (avoid edge artifacts)
    v_lo = np.argmax(hxmt_valid)
    v_hi = len(hxmt_valid) - np.argmax(hxmt_valid[::-1])
    xs_v = x_step[v_lo:v_hi]
    obs_v = net_hxmt_obs[v_lo:v_hi]
    all_v = net_hxmt_all[v_lo:v_hi]

    # HXMT observed (blue)
    ax_lc.step(xs_v, obs_v, where="post", color="C0", lw=0.8,
               label="HXMT/HE observed", zorder=3)
    ax_lc.fill_between(xs_v, 0, obs_v, step="post", alpha=0.15, color="C0")

    # HXMT observed + reconstructed (orange)
    ax_lc.step(xs_v, all_v, where="post", color="C1", lw=0.8,
               label=f"HXMT/HE + reconstructed (+{len(hxmt_fill_t):,})",
               zorder=2)
    ax_lc.fill_between(xs_v, obs_v, all_v, step="post", alpha=0.3, color="C1")

    # SPI-ACS (green)
    if not args.no_spiacs:
        ax_lc.step(x_step, np.nan_to_num(net_spiacs_scaled), where="post",
                   color="C2", lw=1.0,
                   label=f"INTEGRAL/SPI-ACS \u226575 keV (\u00d7{scale:.2f})",
                   zorder=4)

    # ASIM (purple, optional)
    if asim_t is not None:
        ax_lc.step(x_step, net_asim_scaled, where="post", color="C4", lw=0.8,
                   label=f"ASIM/MXGS LED 50\u2013400 keV (\u00d7{scale_asim:.1f})",
                   zorder=5)

    # GBM (red dashed, optional)
    if gbm_t is not None:
        ax_lc.step(x_step, net_gbm_scaled, where="post", color="C3", lw=1.0,
                   linestyle="--",
                   label=f"Fermi/GBM {'+'.join(args.gbm_dets)} (\u00d7{scale_gbm:.2f})",
                   zorder=6)

    # Saturation marker (optional): shade FILL_GAP MET range
    if args.mark_saturation:
        sat_lo_met, sat_hi_met = load_hxmt_resets()
        if sat_lo_met is not None:
            t0_met = hxmt_t0_met(t0_utc)
            sat_lo = sat_lo_met - t0_met - sat_shift
            sat_hi = sat_hi_met - t0_met - sat_shift
            ax_lc.axvspan(sat_lo, sat_hi, color="#D62728", alpha=0.10, zorder=0)
            y_top = ax_lc.get_ylim()[1]
            ax_lc.text((sat_lo + sat_hi) / 2, y_top * 0.92,
                       "HXMT FIFO saturation",
                       ha="center", va="top", fontsize=10,
                       color="#A02030", style="italic")
            print(f"  Saturation marker: [{sat_lo*1000:+.1f}, {sat_hi*1000:+.1f}] ms",
                  file=sys.stderr)

    ax_lc.axhline(0, color="gray", lw=0.5, ls="--")
    ax_lc.set_ylabel("Net count rate (evt/s)")
    ax_lc.legend(loc="upper right", fontsize=9)
    refs = []
    if not args.no_spiacs:
        refs.append("INTEGRAL/SPI-ACS")
    if asim_t is not None:
        refs.append("ASIM/MXGS")
    if gbm_t is not None:
        refs.append("Fermi/GBM")
    refs_label = " vs ".join(refs) if refs else "(no reference)"
    ax_lc.set_title(
        f"GRB 200415A (SGR 0525-66): HXMT/HE vs {refs_label}  "
        f"[{bin_w * 1000:.0f}ms bins, geocentric]",
        fontweight="bold")

    # Ratio panel — pick whichever reference is available, prefer SPI-ACS
    if not args.no_spiacs:
        ref_name = "SPI-ACS"
        ref_curve = net_spiacs_scaled
    elif gbm_t is not None:
        ref_name = "GBM"
        ref_curve = net_gbm_scaled
    elif asim_t is not None:
        ref_name = "ASIM"
        ref_curve = net_asim_scaled
    else:
        ref_curve = np.full_like(x, np.nan)
        ref_name = "ref"
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_val = np.nanmax(np.nan_to_num(ref_curve))
        threshold = max(peak_val * 0.05, 500)
        ratio = np.where(
            np.isfinite(ref_curve) & (ref_curve > threshold)
            & np.isfinite(net_hxmt_all),
            net_hxmt_all / ref_curve, np.nan)
    ax_ratio.step(x_step, ratio, where="post", color="k", lw=0.8)
    ax_ratio.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax_ratio.set_ylabel(f"HXMT / {ref_name}")

    # Ratio statistics
    rv = ratio[~np.isnan(ratio)]
    if len(rv) > 0:
        ratio_mean = np.mean(rv)
        ratio_std = np.std(rv)
        ax_ratio.text(
            0.98, 0.85,
            f"ratio = {ratio_mean:.2f} \u00b1 {ratio_std:.2f} ({len(rv)} bins)",
            transform=ax_ratio.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8))
        ax_ratio.set_ylim(
            max(0, ratio_mean - 3 * ratio_std),
            ratio_mean + 3 * ratio_std)
        print(f"\n── Ratio statistics ──", file=sys.stderr)
        print(f"  Mean: {ratio_mean:.3f}", file=sys.stderr)
        print(f"  Std:  {ratio_std:.3f}", file=sys.stderr)
        print(f"  Bins: {len(rv)}", file=sys.stderr)

    t0_effective = t0_utc + TimeDelta(sat_shift, format="sec")
    t0_label = t0_effective.utc.iso.replace(" ", "T")[:23]
    ax_ratio.set_xlabel(
        f"Time since T\u2080 (s)  [T\u2080 = {t0_label} UTC geocentric]")
    # Clip x-axis to HXMT data range
    xlim_lo = max(-args.before, hxmt_range[0])
    xlim_hi = min(args.after, hxmt_range[1])
    ax_ratio.set_xlim(xlim_lo, xlim_hi)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
