#!/usr/bin/env python3
"""GRB 221009A v2: is the T0+600-700 s rise background or afterglow?  And a
time-dependent HE background model B(t) for the light curve.

Inputs: HE 1K events of the hour (Time, ACD bitmask), 1B HE_Eng frames of the
hour (live fraction, Cnt_LargeEvt), the HXMT orbit file, and the
reconstruction cache (to flag seconds with FIFO loss).

Background tracers (both are recorded through the hour, outside the burst):
  A   1K events with any ACD bit set (charged-particle tag), per second
  LE  Cnt_LargeEvt summed over 18 detectors (events above the upper
      discriminator), per second
Source contamination of each tracer is measured on clean (no FIFO loss)
seconds of the early main emission (T0+175..215) and the flare (440..600)
from the second-to-second covariance of tracer and HE source rate after a
31-s running-median detrend; the ACD tracer is corrected self-consistently,
A_bkg = A - p_A * S.

Model: B_true(t) = quadratic in the 21-s-smoothed tracer, fitted (unweighted
least squares, closed form) to the dead-time-corrected 1K gross rate n/f on
source-free windows: pre-burst T0-1020..-10 s and after the GRB set below
the Earth limb (T0+1700..2580 s; geometric limb crossing at T0+1680 s from
the orbit file). The descending side of the orbital hump (T0+1100..1680 s)
is fitted in the standard calibration too (alternative: excluded, which is
then a prediction there). Seconds with FIFO loss, the main emission
(T0+170..310), the flare peak (T0+498..545) and the HE-off seconds are
bridged by linear interpolation of the smoothed tracer.

Outputs (data/p221009/v2/): bkg_model_1s.csv, bkg_check.txt, bkg_check.png
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent))
from p221009_v2_common import (OUT, T0, HE_1K, ORBIT, X, eng_on_grid,  # noqa: E402
                               he_counts, off_mask)

XH = np.arange(-1020.0, 2580.0, 1.0)          # hour grid, left edges
J0 = int(X[0] - XH[0])                        # index of X[0] in XH
CAL_PRE, CAL_DESC, CAL_OCC = (-1020, -10), (1100, 1680), (1700, 2580)
SMOOTH = 21
MAIN_BRIDGE = (170, 310)
FLARE_BRIDGE = (498, 545)     # flare peak: both tracers visibly follow the source
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def read_1k():
    with fits.open(HE_1K, memmap=True) as f:
        h = f[1]
        off, nrow = h._data_offset, h.header["NAXIS2"]
    dt = np.dtype([("t", ">f8"), ("det", "u1"), ("ch", "u1"), ("pw", "u1"),
                   ("acd", "u1", 3), ("et", "u1"), ("fl", "u1")])
    raw = np.memmap(HE_1K, dtype=dt, mode="r", offset=off, shape=(nrow,))
    t = raw["t"] - T0
    acd = (raw["acd"] != 0).any(1)
    e = np.append(XH, XH[-1] + 1)
    return np.histogram(t, e)[0].astype(float), np.histogram(t[acd], e)[0].astype(float)


def orbit():
    """Sub-satellite lon/lat, dipole geomagnetic latitude and L, GRB elevation above limb."""
    from astropy.coordinates import SkyCoord, ITRS
    from astropy.time import Time, TimeDelta
    import astropy.units as u
    o = fits.open(ORBIT)[1].data
    t = o["Time"] - T0
    r = np.sqrt(o["X"]**2 + o["Y"]**2 + o["Z"]**2)
    # IGRF-13 (2020) dipole north pole: 80.65 N, 72.68 W
    pl, pn = np.radians(80.65), np.radians(-72.68)
    m = np.array([np.cos(pl) * np.cos(pn), np.cos(pl) * np.sin(pn), np.sin(pl)])
    maglat = np.degrees(np.arcsin((o["X"] * m[0] + o["Y"] * m[1] + o["Z"] * m[2]) / r))
    L = (r / 6371.2e3) / np.cos(np.radians(maglat))**2
    ts = Time("2022-10-09T13:17:00", scale="utc") + TimeDelta(t, format="sec")
    c = SkyCoord(ra=288.2645 * u.deg, dec=19.7735 * u.deg, distance=1e9 * u.pc)
    v = c.transform_to(ITRS(obstime=ts)).cartesian.xyz.value
    v = v / np.linalg.norm(v, axis=0)
    cosang = -(o["X"] * v[0] + o["Y"] * v[1] + o["Z"] * v[2]) / r
    elev = np.degrees(np.arccos(cosang)) - np.degrees(np.arcsin(6378.137e3 / r))
    g = lambda y: np.interp(XH + 0.5, t, y)
    return dict(lon=g(o["Lon"]), lat=g(o["Lat"]), maglat=g(maglat), L=g(L), elev=g(elev))


def win(lo, hi):
    return (XH >= lo) & (XH < hi)


def detr(y):
    return y - median_filter(y, size=31, mode="nearest")


def contamination(tr, S, clean, sel):
    """Through-origin slope of detrended tracer on detrended source rate."""
    d1, d2 = detr(tr), detr(S)
    m = sel & clean & np.isfinite(d1) & np.isfinite(d2)
    return np.sum(d1[m] * d2[m]) / np.sum(d2[m]**2), np.corrcoef(d1[m], d2[m])[0, 1], m.sum()


def smooth_bridge(y, bad):
    """21-s running mean over good seconds; linear bridge over bad ones."""
    g = ~bad & np.isfinite(y)
    num = uniform_filter1d(np.where(g, y, 0.0), SMOOTH, mode="nearest")
    den = uniform_filter1d(g.astype(float), SMOOTH, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num / den
    ok = g & (den > 0.5)
    return np.interp(XH, XH[ok], s[ok])


def fit_quad(tr, y, cal):
    c = np.polyfit(tr[cal], y[cal], 2)
    return c, np.std(y[cal] - np.polyval(c, tr[cal]))


def main():
    n1k, a1k = read_1k()
    eng = eng_on_grid(XH)
    f = eng["f"]
    nc = n1k / f
    LE = eng["LE"]
    orb = orbit()

    # burst-window bookkeeping on the hour grid
    cnt = he_counts()
    S_he = np.full(len(XH), np.nan)          # 1B source rate with the constant B, as a driver
    nfill = np.zeros(len(XH))
    sl = slice(J0, J0 + len(X))
    S_he[sl] = cnt["n_all"] / f[sl] - np.mean(cnt["n_all"][X < -10] / f[sl][X < -10])
    nfill[sl] = cnt["n_all"] - cnt["n_obs"]
    hoff = np.zeros(len(XH), bool)
    hoff[sl] = off_mask()
    clean = (nfill == 0) & ~hoff & np.isfinite(f)

    say("== 1K vs 1B on clean seconds (no filler) ==")
    say(f"  1K/1B observed counts, median over clean seconds T0-50..700: "
        f"{np.median((n1k[sl] / cnt['n_obs'])[nfill[sl] == 0]):.4f}")

    say("\n== source contamination of the tracers (d tracer / d source, detrended) ==")
    pA = {}
    for nm, sel in (("early main 175-215", win(175, 215)), ("flare 440-600", win(440, 600)),
                    ("tail 300-440", win(300, 440))):
        sa = contamination(a1k, S_he, clean, sel)
        sle = contamination(LE, S_he, clean, sel)
        pA[nm] = sa
        say(f"  {nm:20s} ACD: p={sa[0]:.4f} (r={sa[1]:.2f}, n={sa[2]})   "
            f"LargeEvt: p={sle[0]:.4f} (r={sle[1]:.2f})")
    p_acd = pA["early main 175-215"][0]
    say(f"  adopted p_ACD = {p_acd:.4f} (best-correlated window); flare value as alternative")

    bad = ~clean | win(*MAIN_BRIDGE) | win(*FLARE_BRIDGE)
    cal_std = win(*CAL_PRE) | win(*CAL_DESC) | win(*CAL_OCC)
    cal_alt = win(*CAL_PRE) | win(*CAL_OCC)
    cal_std &= np.isfinite(f)
    cal_alt &= np.isfinite(f)
    models = {}
    # LargeEvt model (source contamination ignored; see the p values above)
    le_s = smooth_bridge(LE, bad)
    for tag, cal in (("LE", cal_std), ("LE_alt", cal_alt)):
        c, rs = fit_quad(le_s, nc, cal)
        models[tag] = (np.polyval(c, le_s), c, rs)
    # ACD model, source-corrected self-consistently: A_bkg = A - p*S, S = n/f - B(A_bkg)
    for tag, cal, p in (("ACD", cal_std, p_acd), ("ACD_alt", cal_alt, p_acd),
                        ("ACD_pflare", cal_std, pA["flare 440-600"][0])):
        a_s = smooth_bridge(a1k, bad)
        c, rs = fit_quad(a_s, nc, cal)
        for _ in range(30):
            B = np.polyval(c, a_s)
            S = np.where(cal, 0.0, np.clip(nc - B, 0, None))
            S_s = smooth_bridge(S, bad)
            a_s = smooth_bridge(a1k - p * S_s, bad)
            c, rs = fit_quad(a_s, nc, cal)
        models[tag] = (np.polyval(c, a_s), c, rs)
    # orbit-only model: quadratic in dipole L (source independent by construction)
    for tag, cal in (("L", cal_std), ("L_alt", cal_alt)):
        c, rs = fit_quad(orb["L"], nc, cal)
        models[tag] = (np.polyval(c, orb["L"]), c, rs)
    # timing group's model (STANDARD): B_obs = a + b*Cnt_LargeEvt fitted to the 1K gross
    # rate on T0-1020..-10 and T0+1780..2580, bridged over 174-300 and 498-540 s
    from p221009_timing_bkg import background_clean, CAL_WINDOWS, CONTAMINATED
    tl, _, m_obs, _, tcoef, tres = background_clean(chmin=0)
    assert np.array_equal(tl, XH)
    cal_t = (win(*CAL_WINDOWS[0]) | win(*CAL_WINDOWS[1])) & np.isfinite(f)
    cdt = np.sum((1 / f[cal_t] - 1) * nc[cal_t]) / np.sum(nc[cal_t] ** 2)

    def to_true(b):        # observed gross -> dead-time-corrected gross, 1/f-1 = c*B_true
        return b / (1 - cdt * b)
    models["TIMING"] = (to_true(m_obs), tcoef, tres)
    for nm, wins in (("TIMING_pre", [CAL_WINDOWS[0]]), ("TIMING_occ", [CAL_WINDOWS[1]])):
        _, _, mm, _, cc, rr = background_clean(chmin=0, windows=wins)
        models[nm] = (to_true(mm), cc, rr)
    B_std = models["TIMING"][0]
    bridged_t = np.zeros(len(XH), bool)
    for a, b in CONTAMINATED:
        bridged_t |= win(a, b)
    # per-second statistical error of B_std: Poisson of LargeEvt through the slope
    sigB = np.where(bridged_t, tcoef[0] * np.sqrt(np.nanmedian(LE[win(150, 174)]) / 5),
                    tcoef[0] * np.sqrt(np.clip(np.nan_to_num(LE), 0, None)))

    say("\n== background models B_true(t) (dead-time-corrected gross, evt/s) ==")
    say(f"  STANDARD = timing group (p221009_timing_bkg.background_clean, chmin=0): B_obs = "
        f"{tcoef[1]:.1f} + {tcoef[0]:.4f} * Cnt_LargeEvt (rms {tres:.0f}); converted to B_true with "
        f"1/f-1 = {cdt:.3e} * B_true fitted on the same source-free seconds")
    say("  comparison models: quadratic in the 21-s smoothed tracer (ACD-tag, LargeEvt) or in the")
    say("  dipole L; std calibration = pre + descending side + occulted, alt = pre + occulted only")
    for k, (_, c, rs) in models.items():
        say(f"  {k:11s} coeffs {np.array2string(np.asarray(c), precision=5)}  rms resid on calibration {rs:.0f}")
    ALTS = ("ACD", "ACD_alt", "ACD_pflare", "LE_alt", "TIMING_pre", "TIMING_occ")
    say("  LE (std calibration) and the L models are shown but not used for B_sys: their rms on the")
    say("  calibration seconds is several times larger (LargeEvt and the dipole L do not follow one")
    say("  relation on the ascending and descending sides of the orbital hump).")
    say(f"  systematic B_sys(t) = max |B_x - B_std| over x = {', '.join(ALTS)}; it contains the")
    say("  independent ACD-tag proxy, so it measures the extrapolation of the LargeEvt relation")
    say("  beyond its calibrated range (T0+440 s on) as the proxy-to-proxy difference")
    Bsys = np.max([np.abs(models[k][0] - B_std) for k in ALTS], 0)

    say("\n== window means: gross n/f, B models, implied source S = n/f - B ==")
    say("  (n = 1B observed + filler inside T0-50..700, 1K outside; the background models")
    say("   themselves only use 1K on the calibration seconds)")
    nc = nc.copy()
    nc[sl] = cnt["n_all"] / f[sl]
    hdr = f"  {'window':>11} {'n/f':>7} {'B_std':>7} {'B_ACD':>7} {'S_std':>7} {'S_ACD':>7} " \
          f"{'S_ACDalt':>8} {'S_LEalt':>8} {'S_L':>7} {'S_Tpre':>7} {'S_Tocc':>7} {'S_constB':>8} {'GECAMx200':>9}"
    say(hdr)
    Bc = np.mean(nc[win(-50, -10)])
    from p221009_v2_common import gecam_lg
    gdc = np.full(len(XH), np.nan)
    gdc[sl] = gecam_lg()["net_dc"]
    for lo, hi in [(-50, -10), (-10, 10), (100, 170), (300, 340), (340, 380), (380, 440),
                   (440, 600), (570, 600), (600, 650), (650, 700), (700, 750), (750, 900),
                   (900, 1100), (1100, 1400), (1400, 1680), (1700, 2000), (2000, 2580)]:
        w = win(lo, hi) & np.isfinite(nc)
        S = lambda k: (nc - models[k][0])[w].mean()
        say(f"  {lo:5d}-{hi:<5d} {nc[w].mean():7.0f} {B_std[w].mean():7.0f} {models['ACD'][0][w].mean():7.0f} "
            f"{(nc - B_std)[w].mean():7.0f} {S('ACD'):7.0f} {S('ACD_alt'):8.0f} {S('LE_alt'):8.0f} {S('L'):7.0f} "
            f"{S('TIMING_pre'):7.0f} {S('TIMING_occ'):7.0f} "
            f"{(nc - Bc)[w].mean():8.0f} {200 * np.nanmean(gdc[w]) if w.any() else np.nan:9.0f}")
    say(f"  (constant B_true = mean n/f over T0-50..-10 = {Bc:.0f}; the paper's B = 9332 is the "
        f"observed gross, i.e. B_true*f_bkg)")
    say("  GECAMx200 = GRD01 LG net (dead-time corrected) x 200, independent of the HE background;")
    say("  it is unusable before T0+172 s (GECAM-C in the SAA) and after ~T0+600 s (revisit-orbit")
    say("  background over-subtracted, net rate goes negative).")

    # independent check: errors' 1K detector-pattern tracer
    try:
        import pandas as pd
        trc = pd.read_csv("data/p221009/errors/bkg_tracer_1k_30s.csv")
        say("\n== cross-check: 1K detector-pattern tracer (errors group, 30-s, valid windows) ==")
        for lo, hi in [(570, 600), (600, 650), (650, 700), (700, 760)]:
            m = trc["valid"] & (trc["t_mid"] >= lo) & (trc["t_mid"] < hi)
            w = win(lo, hi)
            say(f"  {lo}-{hi}: tracer S = {trc['S'][m].mean():6.0f} +- {trc['sig_S'][m].mean() / np.sqrt(max(m.sum(), 1)):.0f}"
                f" (gross obs {trc['tot'][m].mean():6.0f});  S_std (LargeEvt) = {(nc - B_std)[w].mean():6.0f};"
                f"  S_ACD = {(nc - models['ACD'][0])[w].mean():6.0f}")
    except FileNotFoundError:
        pass

    say("\n== orbit (dipole geomagnetic coordinates; GRB elevation above the geometric limb) ==")
    for tt in (-1000, -500, -10, 200, 400, 600, 700, 900, 1100, 1300, 1500, 1680, 1900, 2200, 2500):
        i = int(tt - XH[0])
        say(f"  T0{tt:+5d} s  lon {orb['lon'][i]:7.2f}  lat {orb['lat'][i]:6.2f}  maglat {orb['maglat'][i]:6.1f}"
            f"  L {orb['L'][i]:5.2f}  GRB elev {orb['elev'][i]:6.1f}  n/f {nc[i - 10:i + 10].mean() if i >= 10 else nc[i]:7.0f}"
            f"  ACD {a1k[max(i-10,0):i + 10].mean():6.0f}")
    occ = XH[np.where(np.diff(np.sign(orb["elev"])) != 0)[0]]
    say(f"  GRB sets below the geometric limb at T0+{occ[0]:.0f} s")
    imax = np.argmax(np.where(XH >= 700, uniform_filter1d(np.nan_to_num(nc), 61), 0))
    say(f"  maximum of the 61-s mean of n/f after T0+700 s at T0+{XH[imax]:.0f} s; |maglat| max at "
        f"T0+{XH[np.argmax(np.abs(orb['maglat']))]:.0f} s; L max at T0+{XH[np.argmax(orb['L'])]:.0f} s")
    r = np.corrcoef(orb["L"][cal_std], nc[cal_std])[0, 1]
    say(f"  correlation of n/f with dipole L on the calibration seconds: r = {r:.3f}")

    import pandas as pd
    df = pd.DataFrame(dict(t=XH, n1k=n1k, n1k_acd=a1k, f_live=f, n_over_f=nc, LE=LE,
                           B_acd=models["ACD"][0], B_le=models["LE"][0],
                           B_acd_alt=models["ACD_alt"][0], B_le_alt=models["LE_alt"][0],
                           B_acd_pflare=models["ACD_pflare"][0], B_L=models["L"][0],
                           B_timing_pre=models["TIMING_pre"][0], B_timing_occ=models["TIMING_occ"][0],
                           B_timing_obs=m_obs, bridged_timing=bridged_t, sig_B=sigB,
                           B_std=B_std, B_sys=Bsys, cal_std=cal_std, cal_alt=cal_alt,
                           bridged=bad, lon=orb["lon"], lat=orb["lat"], maglat=orb["maglat"],
                           L=orb["L"], grb_elev=orb["elev"]))
    df.to_csv(OUT / "bkg_model_1s.csv", index=False, float_format="%.6g")
    le_check(df, S_he, clean, gdc, tcoef, models)
    (OUT / "bkg_check.txt").write_text("\n".join(lines) + "\n")
    plot(df, Bc)


def le_check(df, S_he, clean, gdc, tcoef, models):
    """Does Cnt_LargeEvt respond to the burst?  Detrended covariance with the HE
    source rate, the implied bias of B_std outside the bridged windows, and the
    difference to the source-corrected ACD model; figure largeevt_check.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pubstyle
    LE = df["LE"].to_numpy()
    say("\n== Cnt_LargeEvt source response (needed before using it as the background proxy) ==")
    say("  window     bridged  p_LE=dLE/dS     r      <S>   implied B bias b*p*<S>   <B_std - B_ACD>")
    for lo, hi, br in ((175, 215, True), (215, 229, True), (300, 440, False), (440, 498, False),
                       (498, 540, True), (545, 600, False), (600, 700, False)):
        sel = win(lo, hi)
        p, r, n = contamination(LE, S_he, clean, sel)
        Sm = np.nanmean(S_he[sel & clean]) if (sel & clean).any() else np.nan
        dB = np.mean((df["B_std"] - df["B_acd"]).to_numpy()[sel])
        say(f"  {lo:4d}-{hi:<4d}    {'yes' if br else 'no ':3s}     {p:+.4f}     {r:+.2f}  {Sm:7.0f}"
            f"      {tcoef[0] * p * Sm:+8.0f}            {dB:+7.0f}")
    say("  (S = HE DT-corrected 1B source rate with the constant B, used only as a driver; both")
    say("   detrended with a 31-s running median; clean = no FIFO filler and HE on)")
    pubstyle.apply()
    t = df["t"].to_numpy()
    m = (t >= -50) & (t < 700)
    fig, ax = plt.subplots(2, 1, figsize=(pubstyle.COL_W, 3.6), sharex=True,
                           gridspec_kw={"hspace": 0.06})
    impl = (df["B_timing_obs"].to_numpy() - tcoef[1]) / tcoef[0]
    ax[0].plot(t[m], LE[m], color="C2", lw=0.5, label="Cnt_LargeEvt (18 det)")
    ax[0].plot(t[m], impl[m], color="k", lw=0.9, ls="--", label="as used in B(t) (bridged)")
    for a, b in ((174, 300), (498, 540)):
        ax[0].axvspan(a, b, color="0.5", alpha=0.12, lw=0)
    ax[0].set_ylim(0, 1.6e4)
    ax[0].set_ylabel("Cnt_LargeEvt (/s)")
    ax[0].legend(loc="upper left", fontsize=6.5)
    g = 200 * gdc[m]
    ax[1].plot(t[m], np.where(t[m] >= 172, g, np.nan), color=pubstyle.C_GBM, lw=0.6,
               label=r"GECAM-C LG $\times200$")
    ax[1].plot(t[m], S_he[m], color="k", lw=0.5, label="HE source (const. B)")
    ax[1].set_yscale("symlog", linthresh=1e3, linscale=0.5)
    ax[1].set_ylim(0, 1e7)
    for a, b in ((174, 300), (498, 540)):
        ax[1].axvspan(a, b, color="0.5", alpha=0.12, lw=0)
    ax[1].set_ylabel("rate (evt/s)")
    ax[1].set_xlabel(r"Time since $T_0$ (s)")
    ax[1].legend(loc="upper right", fontsize=6.5)
    ax[1].set_xlim(-50, 700)
    fig.savefig(OUT / "largeevt_check.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

def plot(df, Bc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pubstyle
    pubstyle.apply()
    fig, ax = plt.subplots(3, 1, figsize=(pubstyle.FULL_W, 5.6), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1, 1], "hspace": 0.06})
    t = df["t"]
    r = uniform_filter1d(df["n_over_f"].fillna(0).to_numpy(), 10)
    ax[0].plot(t, r, color="0.3", lw=0.6, label=r"HE gross $n/\bar f_{\rm live}$ (10-s mean)")
    ax[0].plot(t, df["B_acd"], color="C1", lw=0.9, label="B(t): ACD-tag tracer")
    ax[0].plot(t, df["B_L"], color="C4", lw=0.9, label="B(t): dipole-L model")
    ax[0].plot(t, df["B_std"], color="k", lw=1.2, ls="--", label="B(t) standard (LargeEvt, timing)")
    ax[0].axhline(Bc, color="C3", lw=0.8, ls=":", label="constant pre-burst B")
    for lo, hi in (CAL_PRE, CAL_DESC, CAL_OCC):
        ax[0].axvspan(lo, hi, color="C0", alpha=0.08, lw=0)
    ax[0].set_ylim(5e3, 4e4)
    ax[0].set_ylabel("rate (evt/s)")
    ax[0].legend(loc="upper right", ncol=2, fontsize=7)
    ax[1].plot(t, uniform_filter1d(df["n1k_acd"].to_numpy(), 10), color="C1", lw=0.7, label="ACD-tagged 1K")
    ax[1].plot(t, df["LE"], color="C2", lw=0.5, alpha=0.7, label="Cnt_LargeEvt")
    ax[1].set_ylabel("tracer (/s)")
    ax[1].set_ylim(0, 2.0e4)
    ax[1].legend(loc="upper right", fontsize=7)
    ax[2].plot(t, df["maglat"], color="C4", lw=0.9, label="dipole mag. lat (deg)")
    ax[2].plot(t, df["grb_elev"], color="C5", lw=0.9, label="GRB elev. above limb (deg)")
    ax[2].axhline(0, color="0.5", lw=0.5)
    ax[2].set_ylabel("deg")
    ax[2].legend(loc="lower left", fontsize=7)
    ax[2].set_xlabel(r"Time since $T_0$ (s)")
    ax[2].set_xlim(-1020, 2580)
    fig.savefig(OUT / "bkg_check.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
