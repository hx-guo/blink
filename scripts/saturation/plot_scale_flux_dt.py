#!/usr/bin/env python3
"""DT correction overlay — 2-panel cleaner layout."""
import sys, csv, numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../gecam", "../lib",)]  # sibling and shared script modules
from plot_hxmt_vs_gecam import compute_time_offset, load_gecam_btime
from engineering_prediction import BOX_CODE, BOX_OFFSET, MET_CORRECTION

CACHE = "/Users/skyair/Developer/ihep/blink/data/cache_221009a_reconstruct.csv"
TRIGGER_MET_ASTROPY = 339945423.0
TRIGGER_MET_PY = 339945422.0
SAT_THR = 35000


def load_mean_live_fraction(trigger_met_py, before=50, after=700,
                              date_str="20221009", hour_str="130000"):
    t_lo = trigger_met_py - before
    t_hi = trigger_met_py + after
    sec_to_lf = {}
    for box, code in BOX_CODE.items():
        folder = Path(f"data/1B/{date_str[:4]}/{date_str}/{code}")
        matches = sorted(folder.glob(f"HXMT_1B_{code}_{date_str}T{hour_str}*.fits"))
        if not matches: continue
        fe = fits.open(matches[0], memmap=True)
        d = fe["HE_Eng"].data
        offset = d["UTC_Last_Bdc"][0] - d["sTime_Last_Bdc"][0]
        met = d["Time"].astype(float) + offset + MET_CORRECTION
        mask = (met >= t_lo) & (met <= t_hi)
        met_m = met[mask]
        lc = d["Length_Time_Cycle"].astype(float)[mask]
        lc_safe = np.where(lc > 0, lc, np.nan)
        for det_local in range(6):
            det_global = BOX_OFFSET[box] + det_local
            dt = d[f"DeadTime_PHODet_{det_global}"].astype(float)[mask]
            lf_det = 1.0 - dt / lc_safe
            for i, t in enumerate(met_m):
                key = int(round(t))
                sec_to_lf.setdefault(key, []).append(lf_det[i])
        fe.close()
    if not sec_to_lf: return None, None
    secs = sorted(sec_to_lf.keys())
    lf_mean = np.array([np.nanmean(sec_to_lf[s]) for s in secs])
    return np.array(secs, dtype=float) - trigger_met_py, lf_mean


def main():
    # GECAM, HXMT cache, live fraction (same as before)
    g_met, _, _ = compute_time_offset()
    _, g_rate, _ = load_gecam_btime(g_met, 50, 700, 1.0, "lg", True)
    g_rate = np.nan_to_num(g_rate, nan=0.0)

    obs_t, fill_t = [], []
    with open(CACHE) as f:
        r = csv.reader(f); next(r)
        for row in r:
            t = float(row[2]) - TRIGGER_MET_ASTROPY
            (obs_t if row[1] == "EVT" else fill_t).append(t)
    obs_t = np.array(obs_t); fill_t = np.array(fill_t)
    edges = np.arange(-50, 701, 1.0); x = edges[:-1]
    r_obs = np.histogram(obs_t, bins=edges)[0]
    r_all = np.histogram(np.concatenate([obs_t, fill_t]), bins=edges)[0]
    bkg = r_all[x < -10].mean()
    net_all = r_all - bkg

    lf_t, lf = load_mean_live_fraction(TRIGGER_MET_PY, 50, 700)
    lf_t = lf_t + (TRIGGER_MET_PY - TRIGGER_MET_ASTROPY)
    lf_grid = np.interp(x, lf_t, lf, left=np.nan, right=np.nan)

    # Dead-time gates the GROSS rate (source + background), so the corrected
    # net source rate is gross/f_live - bkg = (net + bkg)/f_live - bkg, not
    # net/f_live. The extra term bkg*(1-f_live)/f_live is rate-dependent.
    with np.errstate(divide="ignore", invalid="ignore"):
        net_dt = np.where(lf_grid > 0.01, (net_all + bkg) / lf_grid - bkg, np.nan)

    sig = (net_all > 5e3) & (g_rate > 30) & np.isfinite(g_rate)
    scale_raw = np.where(sig, g_rate / net_all, np.nan)
    scale_dt = np.where(sig & np.isfinite(net_dt), g_rate / net_dt, np.nan)
    sat = sig & (r_obs >= SAT_THR)
    nonsat = sig & (r_obs < SAT_THR)

    bin_edges = [30, 100, 300, 1000, 3000, 10000, 30000]
    centers = [np.sqrt(bin_edges[i]*bin_edges[i+1])
                for i in range(len(bin_edges)-1)]
    def med(arr):
        m, lo, hi, n = [], [], [], []
        for i in range(len(bin_edges)-1):
            msk = sig & (g_rate >= bin_edges[i]) & (g_rate < bin_edges[i+1])
            v = arr[msk]; v = v[np.isfinite(v)]
            if len(v) >= 2:
                m.append(np.median(v))
                a, b = np.percentile(v, [25, 75])
                lo.append(a); hi.append(b); n.append(len(v))
            else:
                m.append(np.nan); lo.append(np.nan); hi.append(np.nan); n.append(0)
        return np.array(m), np.array(lo), np.array(hi), n
    m_r, lo_r, hi_r, n_r = med(scale_raw)
    m_d, lo_d, hi_d, n_d = med(scale_dt)

    # ---- dead-time model (simplest non-paralyzable) --------------------------
    # Dead-time is driven by the GROSS rate (source + background), so the live
    # fraction is f_live = 1/(1 + tau*(R + R_B)), where R is the GECAM net rate
    # and R_B = B/A_REF is the HXMT gross background expressed in GECAM-equivalent
    # cts/s (fixed from the measured background, not fitted). The raw scale is
    # then (1/A_REF)/f_live = (1 + tau*(R+R_B))/A_REF.
    centers_arr = np.asarray(centers)
    lf_med = np.array([
        np.nanmedian(lf_grid[sig & (g_rate >= bin_edges[i])
                              & (g_rate < bin_edges[i + 1])])
        for i in range(len(bin_edges) - 1)])
    A_REF = 200.0  # low-rate asymptote: 1/200 precursor (non-saturated) calibration
    R_B = bkg / A_REF   # gross background in GECAM-equivalent cts/s
    ok = np.isfinite(centers_arr) & np.isfinite(lf_med)
    (tau,), _ = curve_fit(lambda R, t: 1.0 / (1.0 + t * (R + R_B)),
                          centers_arr[ok], lf_med[ok],
                          p0=[2e-4], bounds=(0.0, np.inf))
    R_grid = np.logspace(np.log10(20), np.log10(35000), 400)
    scale_model = (1.0 + tau * (R_grid + R_B)) / A_REF   # (1/A_REF)/f_live
    # Average dead-time per photon. tau is in (GECAM cts/s)^-1; the HXMT gross
    # rate is A_REF*(R+R_B), so f_live = 1/(1+tau_d*G) implies the aggregate
    # per-event dead time is tau_d = tau/A_REF (whole HE, 18 det), and per
    # detector it is 18*tau/A_REF -- comparable to the ~18 us tail-calibrated tau_i.
    tau_d_agg = tau / A_REF
    tau_d_det = tau_d_agg * 18.0
    print(f"\ndead-time fit (gross, Model B): f_live = 1/(1+tau*(R+R_B)),"
          f"  R_B = {R_B:.1f} cts/s (fixed from B={bkg:.0f} evt/s)")
    print(f"  tau = {tau:.3e} (cts/s)^-1,  R(f_live=0.5) = 1/tau - R_B = "
          f"{1.0/tau - R_B:.0f} cts/s")
    print(f"  avg dead-time per photon: {tau_d_agg*1e6:.2f} us/evt (HE aggregate), "
          f"{tau_d_det*1e6:.1f} us/evt/detector")

    # ---- 2-panel plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              sharey=True, gridspec_kw={"wspace": 0.05})

    for ax, scale_arr, m_arr, lo_arr, hi_arr, n_arr, color, label in [
        (axes[0], scale_raw, m_r, lo_r, hi_r, n_r, "C0",
         "no DT correction"),
        (axes[1], scale_dt, m_d, lo_d, hi_d, n_d, "C3",
         "÷ live fraction (DT corrected)"),
    ]:
        ax.scatter(g_rate[nonsat], scale_arr[nonsat], s=26, c=color,
                    alpha=0.55, edgecolors="none",
                    label="non-saturated bins")
        ax.scatter(g_rate[sat], scale_arr[sat], s=32, marker="x", c=color,
                    alpha=0.75, linewidth=1.6, label="HXMT obs ≥ 35k")
        ax.errorbar(centers, m_arr, yerr=[m_arr - lo_arr, hi_arr - m_arr],
                     fmt="o-", color="black", markersize=10, linewidth=2,
                     capsize=5, label="median per log-bin (IQR)", zorder=10)
        for c, m, n in zip(centers, m_arr, n_arr):
            if np.isfinite(m):
                ax.annotate(f"{m*1000:.1f}", xy=(c, m), xytext=(0, 12),
                             textcoords="offset points", fontsize=9,
                             ha="center", fontweight="bold")
        ax.axhline(1/200, color="gray", ls="--", lw=0.8,
                    label="1/200 (precursor non-sat)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(20, 35000); ax.set_ylim(1e-3, 1e-1)
        ax.set_xlabel("GECAM-C GRD01 LG net rate (cts/s)", fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")

    # overlay the dead-time model on the raw (uncorrected) panel
    axes[0].plot(R_grid, scale_model, color="C1", lw=2.4, zorder=9,
                 label=(r"dead-time model  $\frac{1}{200}[1+\tau(R{+}R_B)]$" + "\n"
                        + rf"$\tau_{{\rm det}}\approx{tau_d_det*1e6:.0f}\,\mu$s/evt, "
                        + rf"$R_B={R_B:.0f}$ cts/s"))
    axes[0].legend(fontsize=8.5, loc="upper left")

    axes[0].set_ylabel(r"per-bin scale  =  GECAM / HXMT", fontsize=11)
    if "--no-title" not in sys.argv:
        fig.suptitle("Dead-time correction on the per-bin scale "
                      "(GRB 221009A, GECAM-C/GRD01 LG as reference)",
                      fontsize=12, fontweight="bold", y=1.02)
    out_pdf = "/Users/skyair/Developer/ihep/paper-hxmt-saturation/figures/f9_scale_flux_dt.pdf"
    out_png = "/tmp/f9_scale_flux_dt.png"
    if "-o" in sys.argv:
        out_pdf = sys.argv[sys.argv.index("-o") + 1]
        out_png = str(Path(out_pdf).with_suffix(".png")) if out_pdf.endswith(".pdf") else out_png
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"saved {out_pdf}\n       {out_png}")

    # --- diagnostic only (writes /tmp): does HXMT/GECAM linearize? ---
    figd, axd = plt.subplots(1, 2, figsize=(13, 5.2))
    # (A) reciprocal HXMT/GECAM on log-log -> just a vertical mirror, still curved
    axd[0].scatter(g_rate[nonsat], 1.0 / scale_raw[nonsat], s=18, c="C0",
                   alpha=0.45, edgecolors="none")
    axd[0].scatter(g_rate[sat], 1.0 / scale_raw[sat], s=24, marker="x",
                   c="C0", alpha=0.7, lw=1.3)
    axd[0].errorbar(centers, 1.0 / m_r,
                    yerr=[1.0 / m_r - 1.0 / hi_r, 1.0 / lo_r - 1.0 / m_r],
                    fmt="o-", color="black", ms=8, lw=1.8, capsize=4, zorder=10)
    axd[0].plot(R_grid, A_REF / (1.0 + tau * R_grid), color="C1", lw=2.2,
                label=r"$200/(1+\tau R)$ (hyperbola)")
    axd[0].axhline(A_REF, color="gray", ls="--", lw=0.8, label="200 (precursor)")
    axd[0].set_xscale("log"); axd[0].set_yscale("log")
    axd[0].set_xlim(20, 35000)
    axd[0].set_xlabel("GECAM-C GRD01 LG net rate (cts/s)")
    axd[0].set_ylabel("HXMT / GECAM  (reciprocal)")
    axd[0].set_title("reciprocal on log-log: vertical mirror — still curved")
    axd[0].grid(alpha=0.25, which="both"); axd[0].legend(fontsize=9)
    # (B) GECAM/HXMT on LINEAR axes -> straight line (real non-paralyzable test)
    axd[1].scatter(g_rate[nonsat], scale_raw[nonsat], s=18, c="C3",
                   alpha=0.35, edgecolors="none")
    axd[1].scatter(g_rate[sat], scale_raw[sat], s=24, marker="x",
                   c="C3", alpha=0.7, lw=1.3)
    axd[1].errorbar(centers, m_r, yerr=[m_r - lo_r, hi_r - m_r],
                    fmt="o", color="black", ms=8, lw=1.8, capsize=4, zorder=10)
    axd[1].plot(R_grid, (1.0 + tau * R_grid) / A_REF, color="C1", lw=2.2,
                label=r"$\frac{1}{200}+\frac{\tau}{200}R$ (straight line)")
    axd[1].set_xlim(0, 20000); axd[1].set_ylim(0, 0.03)
    axd[1].set_xlabel("GECAM-C GRD01 LG net rate (cts/s)")
    axd[1].set_ylabel("GECAM / HXMT")
    axd[1].set_title(r"GECAM/HXMT on linear axes: genuinely linear in $R$")
    axd[1].grid(alpha=0.25); axd[1].legend(fontsize=9)
    figd.tight_layout()
    figd.savefig("/tmp/f9_linearity_demo.png", dpi=200, bbox_inches="tight")
    print("saved /tmp/f9_linearity_demo.png  (diagnostic)")
    print(f"\n{'bin':<22} {'lf_med':>7} {'raw scale':>11} {'DT-corr scale':>14}")
    for i, c in enumerate(centers):
        msk = sig & (g_rate >= bin_edges[i]) & (g_rate < bin_edges[i+1])
        lf_med = np.nanmedian(lf_grid[msk]) if msk.any() else np.nan
        print(f"  {bin_edges[i]:>5}-{bin_edges[i+1]:<5} cts/s   "
              f"{lf_med:>7.3f}  {m_r[i]*1000:>9.1f}e-3 {m_d[i]*1000:>9.1f}e-3")


if __name__ == "__main__":
    main()
