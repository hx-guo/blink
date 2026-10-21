#!/usr/bin/env python3
"""GRB 221009A v3: HE background B(t) recalibrated strictly after the Earth
occultation of the burst, on three hours of data, plus band-limited
backgrounds for the comparison with the An+2023 HE/CsI afterglow.

Changes with respect to p221009_v2_bkg.py (kept unchanged for v2):
  * inputs from p221009_v3_hours.py: T12+T13+T14 (T0-4620 .. T0+6180 s), so
    the source-free calibration covers the previous orbit (pre-burst
    T0-4220..-10 s, the An+2023 training window) and the whole occultation;
  * occulted window = [T0+1930 s, egress - 50 s]: 1930 s is 46 s after the
    latest occultation time in the literature (1884 s, Zhang+2025, An+2023)
    and 231 s after the geometric limb crossing from our ephemeris
    (1699 s, p221009_v3_geo.py); egress is from our ephemeris, capped at the
    literature 3864 s. v2 used [-1020,-10] + [1780,2580];
  * standard = timing-group linear Cnt_LargeEvt model on those windows, with
    the same bridging (174-300, 498-540 s);
  * comparison models (variants of the occulted start, short hour-13-only
    calibration, the v2 calibration, ACD-tag proxy, LargeEvt quadratic) and
    B_sys(t) = max |B_x - B_std| over them;
  * an afterglow / occultation-time test: joint fit of the source-free model
    plus an afterglow term A*((t-225)/437)^-0.88 switched off at t_occ, for
    t_occ = 1699 and 1884 s;
  * band backgrounds (CsI pw > 75; channel ranges standing in for 0.4-2.95
    and 0.6-3 MeV): B_band = a + b*LE_bridged on the standard windows.

Outputs (data/p221009/v3/): bkg_model_1s.csv (burst hour), bkg_hours_1s.csv,
bkg_bands_1s.csv, bkg_check.txt, bkg_check.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d, median_filter, binary_dilation

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
from p221009_v2_common import X, he_counts, off_mask, gecam_lg  # noqa: E402

OUT = Path("data/p221009/v3")
OUT.mkdir(parents=True, exist_ok=True)
OCC_START, OCC_LIT_EGRESS = 1930.0, 3864.0
CONTAMINATED = [(174.0, 300.0), (498.0, 540.0)]     # as p221009_timing_bkg
MAIN_BRIDGE, FLARE_BRIDGE = (170, 310), (498, 545)
SMOOTH = 21
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def main():
    H = np.load(OUT / "hours_1s.npz")
    XH = H["t"]
    n1k, a1k, f, LE, Lsh, elev = H["n"].astype(float), H["acd"].astype(float), H["f"], H["LE"], H["L"], H["elev"]

    def win(lo, hi):
        return (XH >= lo) & (XH < hi)

    # seconds usable for calibration: complete engineering frame, HE on, not near an SAA passage
    usable = np.isfinite(f) & np.isfinite(LE) & (n1k > 3000)
    usable &= ~binary_dilation(~usable, iterations=60)
    nc1k = n1k / f
    J0 = int(X[0] - XH[0])
    sl = slice(J0, J0 + len(X))
    cnt = he_counts()
    S_he = np.full(len(XH), np.nan)
    S_he[sl] = cnt["n_all"] / f[sl] - np.mean(cnt["n_all"][X < -10] / f[sl][X < -10])
    nfill = np.zeros(len(XH)); nfill[sl] = cnt["n_all"] - cnt["n_obs"]
    hoff = np.zeros(len(XH), bool); hoff[sl] = off_mask()
    clean = (nfill == 0) & ~hoff & np.isfinite(f)

    # occultation egress from our ephemeris (first elev > 0 after the 1699 s ingress)
    after = np.where((XH > 1800) & (elev > 0))[0]
    egress = XH[after[0]] if len(after) else np.inf
    occ_end = min(egress, OCC_LIT_EGRESS) - 50
    pre = win(-4220, -10)
    occ = win(OCC_START, occ_end)
    cal = (pre | occ) & usable
    say("== data and calibration windows ==")
    say(f"  grid T0{XH[0]:+.0f}..{XH[-1] + 1:+.0f} s; usable seconds {usable.sum()} "
        f"(complete eng frame, n1K > 3000/s, >= 60 s from any unusable second)")
    say(f"  GRB occultation from our ephemeris: ingress T0+1699 s (p221009_v3_geo), egress T0+{egress:.0f} s; "
        f"literature 1884-3864 s")
    say(f"  STANDARD calibration: pre-burst [-4220,-10] ({(pre & usable).sum()} s) + occulted "
        f"[{OCC_START:.0f},{occ_end:.0f}] ({(occ & usable).sum()} s)")

    cdt = np.sum((1 / f[cal] - 1) * nc1k[cal]) / np.sum(nc1k[cal] ** 2)

    def to_true(b):
        return b / (1 - cdt * b)

    def bridge_linear(model):
        m = model.copy()
        for a, b in CONTAMINATED:
            ia = int(np.searchsorted(XH, a)); ib = int(np.searchsorted(XH, b))
            ya = model[ia - 5:ia].mean(); yb = model[ib:ib + 5].mean()
            m[ia:ib] = ya + (yb - ya) * (XH[ia:ib] - XH[ia - 3]) / (XH[ib + 2] - XH[ia - 3])
        return m

    def le_linear(y, c):
        ok = c & usable
        b, a = np.polyfit(LE[ok], y[ok], 1)
        return bridge_linear(a + b * np.nan_to_num(LE)), (b, a), np.std(y[ok] - (a + b * LE[ok]))

    def smooth_bridge(y, bad):
        g = ~bad & np.isfinite(y)
        num = uniform_filter1d(np.where(g, y, 0.0), SMOOTH, mode="nearest")
        den = uniform_filter1d(g.astype(float), SMOOTH, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            s = num / den
        ok = g & (den > 0.5)
        return np.interp(XH, XH[ok], s[ok])

    def fit_quad(tr, y, c):
        ok = c & usable
        cc = np.polyfit(tr[ok], y[ok], 2)
        return cc, np.std(y[ok] - np.polyval(cc, tr[ok]))

    def detr(y):
        return y - median_filter(np.nan_to_num(y), size=31, mode="nearest")

    calsets = {
        "STD": cal,
        "occ1750": (pre | win(1750, occ_end)) & usable,
        "occ2000": (pre | win(2000, occ_end)) & usable,
        "hour13": (win(-1020, -10) | win(OCC_START, 2580)) & usable,
        "v2cal": (win(-1020, -10) | win(1780, 2580)) & usable,
        "pre_only": pre & usable,
        "occ_only": occ & usable,
    }
    models = {}
    for k, c in calsets.items():
        m, cc, rr = le_linear(n1k, c)
        models["TIMING" if k == "STD" else "TIMING_" + k] = (to_true(m), cc, rr)
    inwin = (XH >= X[0]) & (XH <= X[-1])
    bad = np.where(inwin, ~clean, ~usable) | win(*MAIN_BRIDGE) | win(*FLARE_BRIDGE)
    m = win(175, 215) & clean
    d1, d2 = detr(a1k), detr(S_he)
    mm = m & np.isfinite(d2)
    p_acd = np.sum(d1[mm] * d2[mm]) / np.sum(d2[mm] ** 2)
    le_s = smooth_bridge(LE, bad)
    cc, rr = fit_quad(le_s, nc1k, cal)
    models["LE_quad"] = (np.polyval(cc, le_s), cc, rr)
    cal_desc = (pre | win(1100, 1650) | occ) & usable
    for tag, c in (("ACD", cal), ("ACD_desc", cal_desc)):
        a_s = smooth_bridge(a1k, bad)
        cc, rr = fit_quad(a_s, nc1k, c)
        for _ in range(30):
            B = np.polyval(cc, a_s)
            S = np.where(c, 0.0, np.clip(nc1k - B, 0, None))
            S[(XH < -10) | (XH > 1000)] = 0.0
            a_s = smooth_bridge(a1k - p_acd * smooth_bridge(S, bad), bad)
            cc, rr = fit_quad(a_s, nc1k, c)
        models[tag] = (np.polyval(cc, a_s), cc, rr)
    cc, rr = fit_quad(Lsh, nc1k, cal)
    models["L"] = (np.polyval(cc, Lsh), cc, rr)

    B_std, tco = models["TIMING"][0], models["TIMING"][1]
    # TIMING_hour13 / TIMING_v2cal (one-hour calibration, LargeEvt range 2.5-4.8k / 2.5-7.5k against
    # 10-13k at T0+600-700 s) are reported but not part of B_sys: they extrapolate where the
    # three-hour calibration interpolates
    ALTS = ("TIMING_occ1750", "TIMING_occ2000", "TIMING_pre_only", "TIMING_occ_only",
            "ACD", "ACD_desc", "LE_quad")
    Bsys = np.max([np.abs(models[k][0] - B_std) for k in ALTS], 0)
    bridged = np.zeros(len(XH), bool)
    for a, b in CONTAMINATED:
        bridged |= win(a, b)
    sigB = np.where(bridged, tco[0] * np.sqrt(np.nanmedian(LE[win(150, 174)]) / 5),
                    tco[0] * np.sqrt(np.clip(np.nan_to_num(LE), 0, None)))

    say(f"\n== models (B_true, dead-time corrected; 1/f-1 = c*B_true, c = {cdt:.3e}) ==")
    say(f"  STANDARD: B_obs = {tco[1]:.1f} + {tco[0]:.4f} * Cnt_LargeEvt (v2: 6064.6 + 1.0844)")
    for k, (_, c, r) in models.items():
        say(f"  {k:16s} coeffs {np.array2string(np.asarray(c), precision=5)}  rms on calibration {r:.0f}")
    say(f"  ACD source-tag probability p = {p_acd:.4f}")
    say(f"  B_sys(t) = max |B_x - B_std| over {', '.join(ALTS)}")
    for k, c in (("STD", cal), ("hour13", calsets["hour13"]), ("v2cal", calsets["v2cal"])):
        say(f"  LargeEvt range on the {k} calibration: {np.nanmin(LE[c]):.0f}-{np.nanmax(LE[c]):.0f}/s")
    for lo, hi in ((300, 440), (600, 700), (700, 900)):
        say(f"  LargeEvt in {lo}-{hi} s: {np.nanmin(LE[win(lo, hi)]):.0f}-{np.nanmax(LE[win(lo, hi)]):.0f}/s")

    nc = nc1k.copy()
    nc[sl] = cnt["n_all"] / f[sl]
    gdc = np.full(len(XH), np.nan)
    gdc[sl] = gecam_lg()["net_dc"]
    say("\n== window means (evt/s): n/f, B_std, S = n/f - B for each model ==")
    keys = ["TIMING"] + [k for k in models if k != "TIMING"]
    say("  window      n/f   B_std | " + " ".join(f"{k.replace('TIMING_', 'T_'):>9s}" for k in keys) + "  B_sys GECAMx200")
    for lo, hi in [(-4220, -2000), (-2000, -10), (-10, 10), (100, 170), (300, 440), (440, 600), (600, 700),
                   (700, 900), (900, 1100), (1100, 1400), (1400, 1699), (1699, 1884), (1884, 1930),
                   (1930, 2600), (2600, 3800), (3900, 5000)]:
        w = win(lo, hi) & np.isfinite(nc) & (usable | ((XH >= X[0]) & (XH <= X[-1])))
        if not w.any():
            continue
        say(f"  {lo:5d}-{hi:<5d} {nc[w].mean():6.0f} {B_std[w].mean():6.0f} | "
            + " ".join(f"{(nc - models[k][0])[w].mean():9.0f}" for k in keys)
            + f"  {Bsys[w].mean():5.0f} {200 * np.nanmean(gdc[w]) if np.isfinite(gdc[w]).any() else np.nan:9.0f}")

    # ---- afterglow / occultation-time test ----
    say("\n== joint fit: source-free model + afterglow A*((t-225)/437)^-0.88 for t < t_occ ==")
    say("  fit seconds: pre-burst [-4220,-10] + [1100, egress-50], usable; tracer: ACD-tag (quadratic, smoothed)")
    a_s = smooth_bridge(a1k, bad)
    jf = (pre | win(1100, occ_end)) & usable
    for tocc in (None, 1699.0, 1884.0):
        g = np.where((XH > 225) & (XH < (tocc or -1e9)), ((XH - 225) / 437) ** -0.88, 0.0)
        A = np.vstack([np.ones(jf.sum()), a_s[jf], a_s[jf] ** 2, g[jf]]).T
        if tocc is None:
            A = A[:, :3]
        coef, res, *_ = np.linalg.lstsq(A, nc1k[jf], rcond=None)
        rms = np.sqrt(np.mean((nc1k[jf] - A @ coef) ** 2))
        say(f"  t_occ = {str(tocc):6s}: afterglow amplitude at 662 s = "
            f"{coef[3] if tocc else 0:7.0f} evt/s, rms {rms:.1f}, chi2-like sum {np.sum((nc1k[jf] - A @ coef) ** 2) / 1e6:.2f}e6")
    for tr_name, tr in (("LargeEvt", np.nan_to_num(LE)),):
        for tocc in (None, 1699.0, 1884.0):
            g = np.where((XH > 225) & (XH < (tocc or -1e9)), ((XH - 225) / 437) ** -0.88, 0.0)
            A = np.vstack([np.ones(jf.sum()), tr[jf], g[jf]]).T
            if tocc is None:
                A = A[:, :2]
            coef, *_ = np.linalg.lstsq(A, nc1k[jf], rcond=None)
            rms = np.sqrt(np.mean((nc1k[jf] - A @ coef) ** 2))
            say(f"  [{tr_name} linear] t_occ = {str(tocc):6s}: amplitude {coef[2] if tocc else 0:7.0f}, rms {rms:.1f}")

    # ---- LargeEvt vs ACD outside bridges ----
    say("\n== <B_std - B_x> by window (evt/s) ==")
    for lo, hi in ((300, 440), (440, 498), (545, 600), (600, 700), (700, 900)):
        w = win(lo, hi)
        say(f"  {lo:4d}-{hi:<4d} " + "  ".join(f"{k}: {np.mean((B_std - models[k][0])[w]):+6.0f}"
                                           for k in ("ACD", "ACD_desc", "LE_quad", "TIMING_v2cal")))

    # ---- bands ----
    ratio_true = B_std / bridge_linear(tco[1] + tco[0] * np.nan_to_num(LE))
    LEc = (bridge_linear(tco[1] + tco[0] * np.nan_to_num(LE)) - tco[1]) / tco[0]
    bands = dict(t=XH, f_live=f)
    say("\n== band backgrounds (1K, CsI pw>75), B_obs = a + b*LE_bridged ==")
    for k in ("csi_all", "csi_0p4_2p95", "csi_0p6_3"):
        y = H[k].astype(float)
        for tag, c in (("", cal), ("_hour13", calsets["hour13"]), ("_occ2000", calsets["occ2000"])):
            ok = c & np.isfinite(LEc)
            b, a = np.polyfit(LEc[ok], y[ok], 1)
            bands[k + "_B" + tag] = (a + b * LEc) * ratio_true
            if not tag:
                say(f"  {k:14s} B_obs = {a:.1f} + {b:.4f} LE (rms {np.std(y[ok] - (a + b * LEc[ok])):.1f})")
        a_q = smooth_bridge(a1k, bad)
        cq, _ = fit_quad(a_q, y / f, cal)
        bands[k + "_B_acd"] = np.polyval(cq, a_q)
        bands[k + "_n"] = y
    pd.DataFrame(bands).to_csv(OUT / "bkg_bands_1s.csv", index=False, float_format="%.6g")

    cols = dict(t=XH, n1k=n1k, n1k_acd=a1k, f_live=f, n_over_f=nc, LE=LE, usable=usable,
                B_std=B_std, B_sys=Bsys, sig_B=sigB, bridged_timing=bridged, cal_std=cal,
                **{"B_" + k.lower(): v[0] for k, v in models.items() if k != "TIMING"},
                lat=H["lat"], lon=H["lon"], L=Lsh, grb_elev=elev)
    df = pd.DataFrame(cols)
    df.to_csv(OUT / "bkg_hours_1s.csv", index=False, float_format="%.6g")
    df[(df.t >= -1020) & (df.t < 2580)].to_csv(OUT / "bkg_model_1s.csv", index=False, float_format="%.6g")
    (OUT / "bkg_check.txt").write_text("\n".join(lines) + "\n")
    plot(df, occ_end)


def plot(df, occ_end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pubstyle
    pubstyle.apply()
    fig, ax = plt.subplots(2, 1, figsize=(pubstyle.FULL_W, 4.4), sharex=True,
                           gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.06})
    t = df["t"].to_numpy()
    y = np.where(df["usable"] | ((t >= -50) & (t < 700)), df["n_over_f"], np.nan)
    ax[0].plot(t, uniform_filter1d(np.nan_to_num(y), 10), color="0.3", lw=0.5,
               label=r"HE gross $n/\bar f_{\rm live}$ (10-s mean)")
    ax[0].plot(t, df["B_std"], color="k", lw=1.0, ls="--", label="B(t) v3 (LargeEvt; pre-burst + occulted)")
    ax[0].plot(t, df["B_timing_v2cal"], color="C3", lw=0.7, label="B(t) with the v2 calibration")
    ax[0].plot(t, df["B_acd"], color="C1", lw=0.7, label="ACD-tag proxy")
    for lo, hi in ((-4220, -10), (OCC_START, occ_end)):
        ax[0].axvspan(lo, hi, color="C0", alpha=0.08, lw=0)
    for v, lab in ((1699, "limb (ephemeris)"), (1884, "1884 s (lit.)")):
        ax[0].axvline(v, color="0.4", lw=0.6, ls=":")
        ax[0].text(v, 3.7e4, lab, rotation=90, fontsize=6, va="top", ha="right")
    ax[0].set_ylim(5e3, 4e4)
    ax[0].set_ylabel("rate (evt/s)")
    ax[0].legend(loc="upper right", fontsize=6.5)
    ax[1].plot(t, df["grb_elev"], color="C5", lw=0.9, label="GRB elevation above limb (deg)")
    ax[1].plot(t, 10 * df["L"], color="C4", lw=0.7, label=r"$10\times$ dipole $L$")
    ax[1].axhline(0, color="0.5", lw=0.5)
    ax[1].set_ylabel("deg")
    ax[1].set_xlabel(r"Time since $T_0$ (s)")
    ax[1].set_xlim(t[0], t[-1])
    ax[1].legend(loc="lower left", fontsize=6.5)
    fig.savefig(OUT / "bkg_check.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
