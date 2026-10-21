#!/usr/bin/env python3
"""Plot HXMT/HE reconstructed light curve vs GECAM-C + engineering-channel prediction.

Usage:
    python3 scripts/gecam/plot_hxmt_vs_gecam.py --btime --bin 1.0
    python3 scripts/gecam/plot_hxmt_vs_gecam.py --btime --bin 1.0 --before 50 --after 700

Adds a 4th trace (C2 green step) showing $\\widehat{S}_{rec}^{eng}$ from the C25
model applied to per-second engineering counters, summed over 18 detectors.
Mirrors the Figure 7 design (plot_hxmt_vs_gbm.py).
"""

import argparse, subprocess, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from engineering_prediction import load_engineering_prediction, T_REF

# ── HXMT config ──
HXMT_TRIGGER_UTC = "2022-10-09T13:17:00"
HXMT_EPOCH = "2022-10-09T13"
HXMT_ORBIT_PATH = "data/hxmt_aux/HXMT_20221009T13_Orbit_FFFFFF_V1_1K.FITS"

# ── GECAM-C config ──
GECAM_EVT = "data/gecam_c/gcg_evt_221009_13_v09.fits"
GECAM_BTIME = "data/gecam_c/gcg_btime_221009_13_v12.fits"
GECAM_BTIME_BKG = "data/gecam_c/gcg_btime_221014_13_v06.fits"  # revisited orbit (+5 days)
GECAM_MJDREFI = 59215
GECAM_MJDREFF = 0.00080074074


def compute_time_offset():
    """Compute offset: GECAM T=0 relative to HXMT T=0, both in UTC."""
    hxmt_trigger = Time(HXMT_TRIGGER_UTC, scale='utc')

    # GECAM epoch: TIMESYS=TT, so MJDREFI+MJDREFF is in TT scale
    gecam_epoch = Time(GECAM_MJDREFI + GECAM_MJDREFF, format='mjd', scale='tt')
    gecam_trigger_met = (hxmt_trigger.tt - gecam_epoch).sec

    # HXMT MET system: epoch = 2012-01-01 UTC, events in SI seconds
    hxmt_epoch = Time('2012-01-01T00:00:00', scale='utc')
    hxmt_trigger_met = (hxmt_trigger - hxmt_epoch).sec

    # Python datetime gives wrong MET (ignores leap seconds)
    from datetime import datetime, timezone
    hxmt_trigger_met_python = (
        datetime(2022, 10, 9, 13, 17, 2, tzinfo=timezone.utc) -
        datetime(2012, 1, 1, tzinfo=timezone.utc)
    ).total_seconds()

    hxmt_met_correction = hxmt_trigger_met_python - hxmt_trigger_met

    print(f"  HXMT trigger UTC: {hxmt_trigger.iso}", file=sys.stderr)
    print(f"  HXMT trigger MET (astropy): {hxmt_trigger_met:.3f}", file=sys.stderr)
    print(f"  HXMT trigger MET (python):  {hxmt_trigger_met_python:.3f} "
          f"(off by {hxmt_met_correction:.3f}s due to leap seconds)", file=sys.stderr)
    print(f"  GECAM epoch: {gecam_epoch.iso}", file=sys.stderr)
    print(f"  GECAM trigger MET: {gecam_trigger_met:.3f}", file=sys.stderr)

    return gecam_trigger_met, hxmt_met_correction, hxmt_trigger_met_python


def load_hxmt_reconstruct(before, after):
    """Load HXMT 1B reconstructed events."""
    cmd = ["./target/release/blink", "sat", "reconstruct", HXMT_TRIGGER_UTC,
           "--before", str(before), "--after", str(after)]
    env = os.environ.copy()
    env.setdefault("HXMT_1B_DIR", "data/1B")
    env.setdefault("HXMT_1K_DIR", "data/1K")
    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.stderr:
        for line in proc.stderr.strip().split("\n"):
            print(f"    {line}", file=sys.stderr)

    obs, fill = [], []
    for line in proc.stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 3 or p[0] == "box":
            continue
        met = float(p[2])
        if p[1] == "EVT":
            obs.append(met)
        elif p[1] == "FILL_GAP":
            fill.append(met)
    return np.array(obs), np.array(fill)


def load_gecam_c(gecam_trigger_met, before, after):
    """Load GECAM-C low-gain events."""
    print(f"  Loading GECAM-C: {GECAM_EVT}", file=sys.stderr)
    f = fits.open(GECAM_EVT, memmap=True)

    all_times = []
    for i in range(1, 11):
        hdu_name = f'EVENTS{i:02d}'
        try:
            ev = f[hdu_name].data
            mask = ((ev['GAIN_TYPE'] == 1) &
                    (ev['TIME'] >= gecam_trigger_met - before) &
                    (ev['TIME'] <= gecam_trigger_met + after))
            times = ev['TIME'][mask] - gecam_trigger_met
            all_times.append(times)
            print(f"    {hdu_name}: {mask.sum():,} low-gain events", file=sys.stderr)
        except Exception:
            break

    f.close()
    combined = np.concatenate(all_times)
    print(f"  GECAM-C total: {len(combined):,} events", file=sys.stderr)
    return combined


def load_gecam_btime(gecam_trigger_met, before, after, bin_w, channels="lg",
                     subtract_revisited=True):
    """Load GECAM-C BTIME data (GRD01).

    channels: "hg" (ch0-5, 6-350 keV), "lg" (ch7-12, 0.4-6 MeV),
              "all" (ch0-12, exclude overflow ch6,13)
    """
    ch_slices = {
        "hg": (slice(0, 6), "6-350 keV"),
        "lg": (slice(7, 13), "0.4-6 MeV"),
        "all": (None, "6 keV-6 MeV"),
    }
    ch_slice, ch_label = ch_slices[channels]

    print(f"  Loading GECAM-C BTIME: {GECAM_BTIME}  [{channels}: {ch_label}]",
          file=sys.stderr)
    f_burst = fits.open(GECAM_BTIME)

    dt = 0.05  # native 50 ms bins
    sp_burst = f_burst['SPECTRUM01'].data
    t_burst = sp_burst['STARTTIME']

    if ch_slice is not None:
        counts_burst = sp_burst['COUNTS'][:, ch_slice].sum(axis=1).astype(float)
    else:
        counts_burst = (sp_burst['COUNTS'][:, 0:6].sum(axis=1) +
                        sp_burst['COUNTS'][:, 7:13].sum(axis=1)).astype(float)

    if subtract_revisited:
        f_bkg = fits.open(GECAM_BTIME_BKG)
        sp_bkg = f_bkg['SPECTRUM01'].data
        if ch_slice is not None:
            counts_bkg_raw = sp_bkg['COUNTS'][:, ch_slice].sum(axis=1).astype(float)
        else:
            counts_bkg_raw = (sp_bkg['COUNTS'][:, 0:6].sum(axis=1) +
                              sp_bkg['COUNTS'][:, 7:13].sum(axis=1)).astype(float)
        orbit_offset = f_bkg[0].header['TSTART'] - f_burst[0].header['TSTART']
        t_bkg_aligned = sp_bkg['STARTTIME'] - orbit_offset
        counts_bkg = np.interp(t_burst, t_bkg_aligned, counts_bkg_raw,
                               left=np.nan, right=np.nan)
        rate = (counts_burst - counts_bkg) / dt
        f_bkg.close()
        print(f"    Orbit offset: {orbit_offset:.0f}s ({orbit_offset/86400:.1f} days)",
              file=sys.stderr)
    else:
        rate = counts_burst / dt

    t_rel = t_burst - gecam_trigger_met

    edges = np.arange(-before, after + bin_w, bin_w)
    left = edges[:-1]
    sum_rate, _ = np.histogram(t_rel, bins=edges, weights=rate)
    n_bins, _ = np.histogram(t_rel, bins=edges)
    good = n_bins > 0
    rebinned = np.full(len(left), np.nan)
    rebinned[good] = sum_rate[good] / n_bins[good]

    print(f"    Channels: {channels} ({ch_label})", file=sys.stderr)
    print(f"    Peak: {np.nanmax(rebinned):.0f} cts/s "
          f"at T{left[np.nanargmax(rebinned)]:+.1f}s", file=sys.stderr)

    f_burst.close()
    return left, rebinned, ch_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=float, default=1.0)
    parser.add_argument("--before", type=float, default=50.0)
    parser.add_argument("--after", type=float, default=700.0)
    parser.add_argument("--bkg", type=float, nargs=4, default=[-50, -10, 500, 700],
                        metavar=("T1", "T2", "T3", "T4"))
    parser.add_argument("--xlim", type=float, nargs=2, default=None,
                        metavar=("XMIN", "XMAX"))
    parser.add_argument("--scale-range", type=float, nargs="+", default=None,
                        metavar="T", help="Time ranges for scaling (pairs)")
    parser.add_argument("--scale-override", type=float, default=None,
                        help="Skip fit and apply this exact scale value")
    parser.add_argument("--cache", type=str, default=None,
                        help="Read HXMT reconstruct from cached CSV")
    parser.add_argument("--btime", action="store_true",
                        help="Use BTIME (binned) data instead of events")
    parser.add_argument("--raw", action="store_true",
                        help="With --btime: skip revisited-orbit subtraction")
    parser.add_argument("--channels", choices=["hg", "lg", "all"], default="lg")
    parser.add_argument("--ylim", type=float, default=None)
    parser.add_argument("--mask", type=float, nargs="+", default=None,
                        metavar="T", help="Shutdown intervals to exclude (pairs)")
    parser.add_argument("-o", "--output", default="hxmt_vs_gecam.png")
    parser.add_argument("--title", type=str, default=None,
                        help="Override the main-panel title")
    parser.add_argument("--no-engineering", action="store_true",
                        help="Skip the engineering-channel trace")
    args = parser.parse_args()

    # Time alignment
    print("Computing time alignment...", file=sys.stderr)
    gecam_trigger_met, hxmt_met_correction, hxmt_trigger_met_python = compute_time_offset()

    # Load HXMT
    if args.cache:
        print(f"Loading HXMT/HE from cache: {args.cache}", file=sys.stderr)
        hxmt_obs, hxmt_fill = [], []
        with open(args.cache) as cf:
            for line in cf:
                p = line.strip().split(",")
                if len(p) < 3 or p[0] == "box":
                    continue
                met = float(p[2])
                if p[1] == "EVT":
                    hxmt_obs.append(met)
                elif p[1] == "FILL_GAP":
                    hxmt_fill.append(met)
        hxmt_obs, hxmt_fill = np.array(hxmt_obs), np.array(hxmt_fill)
    else:
        print("Loading HXMT/HE reconstruct...", file=sys.stderr)
        hxmt_obs, hxmt_fill = load_hxmt_reconstruct(args.before, args.after)
    # Convert HXMT MET to time relative to true trigger (correct for leap seconds)
    hxmt_obs_t = hxmt_obs - hxmt_trigger_met_python + hxmt_met_correction
    hxmt_fill_t = (hxmt_fill - hxmt_trigger_met_python + hxmt_met_correction
                   if len(hxmt_fill) > 0 else np.array([]))
    hxmt_all_t = (np.concatenate([hxmt_obs_t, hxmt_fill_t])
                  if len(hxmt_fill_t) > 0 else hxmt_obs_t)
    print(f"  HXMT: {len(hxmt_obs):,} obs + {len(hxmt_fill):,} fill", file=sys.stderr)

    # Load engineering-channel prediction
    if args.no_engineering:
        eng_t = None
        eng_rate = None
        print("Skipping engineering channel (--no-engineering)", file=sys.stderr)
    else:
        print("Loading engineering-channel prediction...", file=sys.stderr)
        t_years_const = (np.datetime64("2022-10-09") - T_REF).astype("timedelta64[D]").astype(float) / 365.25
        eng_t_raw, eng_rate = load_engineering_prediction(
            date_str="20221009", hour_str="130000",
            trigger_met=hxmt_trigger_met_python, before=args.before, after=args.after,
            t_years_const=t_years_const,
            orbit_path=HXMT_ORBIT_PATH if Path(HXMT_ORBIT_PATH).exists() else None,
        )
        if eng_t_raw is None:
            eng_t = None
            print("  ERROR: engineering data missing — skipping that trace", file=sys.stderr)
        else:
            # Apply the same leap-second shift used for HXMT events so engineering
            # aligns to true T0 (matches GECAM time axis).
            eng_t = eng_t_raw + hxmt_met_correction
            print(f"  Engineering 1-Hz frames: {len(eng_t)}", file=sys.stderr)

    # ── Binning ──
    bin_w = args.bin
    edges = np.arange(-args.before, args.after + bin_w, bin_w)
    x = edges[:-1]
    t1, t2, t3, t4 = args.bkg

    # Load GECAM-C
    if args.btime:
        use_revisited = not args.raw
        mode_str = "revisited-orbit bkg" if use_revisited else "raw (flat bkg)"
        print(f"Loading GECAM-C BTIME ({mode_str})...", file=sys.stderr)
        gecam_x, gecam_rate, ch_label = load_gecam_btime(
            gecam_trigger_met, args.before, args.after, bin_w, args.channels,
            subtract_revisited=use_revisited)
        x = gecam_x
        edges = np.append(x, x[-1] + bin_w)
        if args.raw:
            bkg_mask_g = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
            bkg_gecam = np.nanmean(gecam_rate[bkg_mask_g]) if bkg_mask_g.sum() > 0 else 0
            net_gecam = gecam_rate - bkg_gecam
            print(f"  GECAM background (flat): {bkg_gecam:.0f} cts/s", file=sys.stderr)
        else:
            net_gecam = gecam_rate
        gecam_label = f"GECAM-C GRD01 {args.channels.upper()}"
    else:
        gecam_t = load_gecam_c(gecam_trigger_met, args.before, args.after)
        r_gecam = np.histogram(gecam_t, bins=edges)[0] / bin_w
        bkg_mask_g = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
        bkg_gecam = np.mean(r_gecam[bkg_mask_g]) if bkg_mask_g.sum() > 0 else 0
        net_gecam = r_gecam - bkg_gecam
        gecam_label = "GECAM-C low-gain (evt)"
        print(f"  GECAM background: {bkg_gecam:.0f} evt/s", file=sys.stderr)

    # HXMT binning & background
    r_hxmt_obs = np.histogram(hxmt_obs_t, bins=edges)[0] / bin_w
    r_hxmt_all = np.histogram(hxmt_all_t, bins=edges)[0] / bin_w
    bkg_mask = ((x >= t1) & (x < t2)) | ((x >= t3) & (x < t4))
    n_bkg = bkg_mask.sum()
    bkg_hxmt = np.mean(r_hxmt_all[bkg_mask]) if n_bkg > 0 else 0
    print(f"  HXMT background: {bkg_hxmt:.0f} evt/s", file=sys.stderr)

    net_hxmt_obs = r_hxmt_obs - bkg_hxmt
    net_hxmt_all = r_hxmt_all - bkg_hxmt

    # Engineering background subtraction (using 1-Hz bins)
    if eng_t is not None:
        eng_bkg_mask = ((eng_t >= t1) & (eng_t < t2)) | ((eng_t >= t3) & (eng_t < t4))
        bkg_eng = np.mean(eng_rate[eng_bkg_mask]) if eng_bkg_mask.any() else 0.0
        net_eng = eng_rate - bkg_eng
        print(f"  Engineering background: {bkg_eng:.0f} evt/s ({eng_bkg_mask.sum()} bins)",
              file=sys.stderr)

    # Mask shutdown intervals
    if args.mask:
        pairs = list(zip(args.mask[::2], args.mask[1::2]))
        for m_start, m_end in pairs:
            shutdown = (x >= m_start) & (x < m_end)
            net_hxmt_obs[shutdown] = np.nan
            net_hxmt_all[shutdown] = np.nan
            print(f"  Masked shutdown: T+{m_start:.0f} to T+{m_end:.0f} "
                  f"({shutdown.sum()} bins)", file=sys.stderr)

    # Scale GECAM to HXMT
    if args.scale_override is not None:
        scale = args.scale_override
        print(f"  Scale factor: {scale:.2f}  (--scale-override, no fit)",
              file=sys.stderr)
    else:
        if args.scale_range:
            pairs = list(zip(args.scale_range[::2], args.scale_range[1::2]))
            scale_mask = np.zeros(len(x), dtype=bool)
            for s1, s2 in pairs:
                scale_mask |= (x >= s1) & (x < s2)
            scale_mask &= np.isfinite(net_gecam) & np.isfinite(net_hxmt_all)
        else:
            scale_mask = (x >= t2) & (x < t3) & np.isfinite(net_gecam) & np.isfinite(net_hxmt_all)
        sum_hxmt = np.nansum(net_hxmt_all[scale_mask])
        sum_gecam = np.nansum(net_gecam[scale_mask])
        scale = sum_hxmt / sum_gecam if sum_gecam > 0 else 1.0
        print(f"  Scale factor: {scale:.2f}", file=sys.stderr)
    net_gecam_scaled = net_gecam * scale

    # ── Plot ──
    fig, (ax_lc, ax_ratio) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    # Blue family for HXMT/HE event-level: derive dark + light variants from
    # matplotlib C0 so they share hue and saturation, differing only in
    # lightness. Frees C1/C2 for the two equal-weight cross-check references
    # (GECAM, engineering), both plotted with identical line width.
    import colorsys
    import matplotlib.colors as _mc
    _h, _l, _s = colorsys.rgb_to_hls(*_mc.to_rgb("C0"))
    NAVY     = colorsys.hls_to_rgb(_h, 0.25, _s)
    SKY_BLUE = colorsys.hls_to_rgb(_h, 0.58, _s)
    CROSS_LW = 1.2

    # Fills use C0 itself (canonical hue + saturation, mid lightness) so they
    # read unambiguously as "blue"; bottom denser (observed) and the recovery
    # layer above lighter (alpha-modulated).
    ax_lc.fill_between(x, 0, np.nan_to_num(net_hxmt_obs), step="post", alpha=0.55,
                       color="C0", zorder=1)
    ax_lc.fill_between(x, np.nan_to_num(net_hxmt_obs), np.nan_to_num(net_hxmt_all),
                       step="post", alpha=0.30, color="C0", zorder=2)
    ax_lc.step(x, net_hxmt_obs, where="post", color=NAVY, lw=1.0,
               label="HXMT/HE observed", zorder=3)
    ax_lc.step(x, net_hxmt_all, where="post", color=SKY_BLUE, lw=1.0,
               label=f"HXMT/HE + reconstructed (+{len(hxmt_fill):,})", zorder=4)
    ax_lc.step(x, np.nan_to_num(net_gecam_scaled), where="post", color="C1",
               lw=CROSS_LW, label=f"{gecam_label} (×{scale:.1f})", zorder=5)
    if eng_t is not None:
        # 1-Hz step trace, left-edge aligned: each engineering cycle starts at
        # GPS PPS tick N and spans ~0.94 s within [N, N+1].
        eng_edges = np.concatenate([eng_t, [eng_t[-1] + 1.0]])
        eng_step_x = np.repeat(eng_edges, 2)[1:-1]
        eng_step_y = np.repeat(net_eng, 2)
        ax_lc.plot(eng_step_x, eng_step_y, color="C2", lw=CROSS_LW,
                   label=r"engineering $\widehat{S}_{\rm rec}^{\rm eng}$ (1 Hz, summed over 18 det)",
                   zorder=6)

    ax_lc.set_ylabel("Net count rate (evt/s)")
    ax_lc.legend(loc="upper right", fontsize=9.5)
    ax_lc.axhline(0, color="gray", lw=0.5, ls="--")
    if args.ylim:
        ax_lc.set_ylim(-args.ylim * 0.05, args.ylim)
    default_title = (
        f"GRB 221009A tail: HXMT/HE event-level + engineering vs GECAM-C  "
        f"[{bin_w}s bins, geocentric]")
    ax_lc.set_title(args.title or default_title, fontweight="bold")

    # Ratio panel — dual cross-check ratios, colour-matched to upper-panel
    # lines (C1 orange = GECAM, C2 green = engineering). Use 1%-of-peak
    # threshold so the tail also contributes (not just main/secondary peak);
    # outlier-prone bins are tamed by the median+IQR annotation below.
    with np.errstate(divide='ignore', invalid='ignore'):
        peak_gecam = np.nanmax(np.nan_to_num(net_gecam_scaled))
        peak_hxmt = np.nanmax(np.nan_to_num(net_hxmt_all))
        thr_g = max(peak_gecam * 0.01, 100)
        thr_h = max(peak_hxmt * 0.01, 100)
        both_sig_g = (np.isfinite(net_gecam_scaled) & (net_gecam_scaled > thr_g) &
                      np.isfinite(net_hxmt_all) & (net_hxmt_all > thr_h))
        ratio_gecam = np.where(both_sig_g, net_hxmt_all / net_gecam_scaled, np.nan)
    ax_ratio.step(x, ratio_gecam, where="post", color="C1", lw=CROSS_LW,
                  label="HXMT / GECAM")

    if eng_t is not None:
        # Upsample 1-Hz engineering rate to the plot grid by holding net_eng[i]
        # across [eng_t[i], eng_t[i]+1).
        eng_t_min = int(np.floor(eng_t[0]))
        idx = np.floor(x).astype(int) - eng_t_min
        valid = (idx >= 0) & (idx < len(net_eng))
        eng_up = np.where(valid, net_eng[np.clip(idx, 0, len(net_eng) - 1)], np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            peak_eng = np.nanmax(eng_up)
            thr_e = max(peak_eng * 0.01, 100)
            both_sig_e = (np.isfinite(eng_up) & (eng_up > thr_e) &
                          np.isfinite(net_hxmt_all) & (net_hxmt_all > thr_h))
            ratio_eng = np.where(both_sig_e, net_hxmt_all / eng_up, np.nan)
        ax_ratio.step(x, ratio_eng, where="post", color="C2", lw=CROSS_LW,
                      label="HXMT / engineering")

    ax_ratio.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax_ratio.set_ylabel("HXMT / ref.")
    ax_ratio.set_ylim(0.5, 1.5)

    # Annotation block with both ratio statistics, computed within xlim if set.
    # Use median + IQR-derived robust σ to suppress 1-bin outliers from
    # the secondary peak (1 s HXMT bin vs 1 Hz engineering cycle can show
    # transient mismatch at sharp-rise edges).
    xlim_lo = args.xlim[0] if args.xlim else -args.before
    xlim_hi = args.xlim[1] if args.xlim else args.after
    xlim_mask = (x >= xlim_lo) & (x < xlim_hi)

    def _robust_stats(r):
        rv = r[np.isfinite(r)]
        if not len(rv):
            return None
        med = np.median(rv)
        q75, q25 = np.percentile(rv, [75, 25])
        sigma = (q75 - q25) / 1.349  # IQR-derived robust σ estimate
        return med, sigma, len(rv)

    annot_lines = []
    sg = _robust_stats(ratio_gecam[xlim_mask])
    if sg:
        med, sig, n = sg
        annot_lines.append(f"HXMT/GECAM       = {med:.2f} ± {sig:.2f} ({n} bins)")
    if eng_t is not None:
        se = _robust_stats(ratio_eng[xlim_mask])
        if se:
            med, sig, n = se
            annot_lines.append(f"HXMT/engineering = {med:.2f} ± {sig:.2f} ({n} bins)")
    if annot_lines:
        ax_ratio.text(0.98, 0.92, "\n".join(annot_lines),
                      transform=ax_ratio.transAxes, ha="right", va="top",
                      fontsize=8.5, family="monospace",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))
    ax_ratio.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax_ratio.set_xlabel(f"Time since trigger (s)  [$T_0$ = {HXMT_TRIGGER_UTC} UTC]")
    if args.xlim:
        ax_ratio.set_xlim(args.xlim[0], args.xlim[1])
    else:
        ax_ratio.set_xlim(-args.before, args.after)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
