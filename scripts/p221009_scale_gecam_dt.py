#!/usr/bin/env python3
"""GRB 221009A: GECAM-C GRD01 LG dead-time correction, made standard.

Live fraction source. The daily event file has a GTI hole over T0-68..T0+330 s,
so the per-event DEAD_TIME column cannot cover the main pulse. The burst-mode
BTIME carries a per-50-ms EXPOSURE for HG and LG; we (1) show that EXPOSURE
equals the DEAD_TIME bookkeeping (4 us per event, ~70-204 us per overflow
event) by comparing both on T0+330..678 s where the event file exists, and
(2) use the BTIME LG EXPOSURE as the LG live fraction everywhere.

Outputs (data/p221009/scale/):
  gecam_lg_reference_1s.csv   t, lg gross/bkg/net, LG live (BTIME), LG live
                              (DEAD_TIME, where available), net dead-time corrected
  gecam_dt_tables.txt         main / flare ratios and scale-flux decade medians
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from p221009_scale_common import (OUT, SCALE, GECAM_EVT, GECAM_T0_MET,
                                  gecam_btime_channels, hxmt_1s, rebin_sum,
                                  robust, saa_mask)

LIVE_BKG = 0.993   # LG live fraction on the background orbit (BTIME, quiet)


def dead_time_live(edges):
    """LG live fraction per bin from the event file DEAD_TIME column."""
    f = fits.open(GECAM_EVT, memmap=True)
    ev = f["EVENTS01"].data
    t = ev["TIME"] - GECAM_T0_MET
    m = (t >= edges[0]) & (t < edges[-1]) & (ev["GAIN_TYPE"] == 1)
    dead = np.histogram(t[m], edges, weights=ev["DEAD_TIME"][m] * 1e-6)[0]
    n = np.histogram(t[m], edges)[0]
    gti = f["GTI"].data
    cov = np.zeros(len(edges) - 1, bool)
    for a, b in zip(gti["START"] - GECAM_T0_MET, gti["STOP"] - GECAM_T0_MET):
        cov |= (edges[:-1] >= a) & (edges[1:] <= b)
    live = np.where(cov, 1 - dead / np.diff(edges), np.nan)
    return live, np.where(cov, n, np.nan)


def main():
    edges = np.arange(-50, 701, 1.0)
    x = edges[:-1]
    tg, c, cb, e = gecam_btime_channels()
    T = rebin_sum(tg, np.ones_like(tg), edges) * 0.05
    G = rebin_sum(tg, c[:, 9:14], edges)     # ch9-12 + overflow ch13 (for checks)
    Bk = rebin_sum(tg, cb[:, 9:14], edges)
    live_bt = rebin_sum(tg, e[:, 1], edges) / T
    lg_gross = G[:, :4].sum(1) / T
    lg_bkg = Bk[:, :4].sum(1) / T
    lg_net = lg_gross - lg_bkg
    lg_net_dc = lg_gross / live_bt - lg_bkg / LIVE_BKG
    live_ev, n_ev = dead_time_live(edges)

    ok = np.isfinite(live_ev) & (x >= 330) & (x < 678)
    d = live_bt[ok] - live_ev[ok]
    # LG event-file counts vs BTIME counts (all LG channels incl. overflow)
    bt_all = rebin_sum(tg, c[:, 7:14].sum(1), edges)
    cnt_ratio = n_ev[ok] / bt_all[ok]
    msg = [f"EXPOSURE(BTIME) - (1 - sum DEAD_TIME) on T0+330..678 s, {ok.sum()} bins: "
           f"median {np.median(d):+.2e}, 1-99% [{np.percentile(d,1):+.2e}, {np.percentile(d,99):+.2e}]",
           f"LG event count / BTIME LG count: median {np.median(cnt_ratio):.4f}, "
           f"range {cnt_ratio.min():.4f}-{cnt_ratio.max():.4f}"]
    for s in msg:
        print(s)

    hdr = "t,lg_gross,lg_bkg,lg_net,live_lg_btime,live_lg_deadtime_col,lg_net_dc"
    np.savetxt(OUT / "gecam_lg_reference_1s.csv",
               np.column_stack([x, lg_gross, lg_bkg, lg_net, live_bt, live_ev, lg_net_dc]),
               delimiter=",", header=hdr, comments="", fmt="%.6g")

    # ---- ratios (make_f10 selection: GECAM net > 50 cts/s, HE net > 0;
    #      plus the HE-off seconds incl. partial ones, see p221009_scale_common) ----
    h = hxmt_1s(edges)
    bad = saa_mask(x)
    out = msg + [""]
    out.append(f"{'window':<22}{'HE':<10}{'GECAM raw':>20}{'GECAM LG-DT corr':>22}")
    for name, lo, hi in [("main 170-310", 170, 310), ("flare 440-610", 440, 610),
                         ("P1 226-229", 226, 229), ("P2 255-267", 255, 267)]:
        for hname, hv in [("rec", h["net"]), ("rec+DT", h["net_dt"])]:
            sel = (x >= lo) & (x < hi) & (lg_net > 50) & (h["net"] > 0) & ~bad
            a = robust(hv[sel] / (SCALE * lg_net[sel]))
            b = robust(hv[sel] / (SCALE * lg_net_dc[sel]))
            out.append(f"{name:<22}{hname:<10}{a[0]:>10.3f} ± {a[1]:.3f}   {b[0]:>10.3f} ± {b[1]:.3f}  (n={a[2]})")
    out.append("")
    # ---- scale-flux decade medians, as f4 (sig: HE net > 5e3, GECAM net > 30) ----
    dec = [30, 100, 300, 1000, 3000, 10000, 30000]
    out.append("scale-flux decade medians (x1e-3), decades on the respective GECAM rate")
    out.append(f"{'decade':<14}{'raw/LGraw':>10}{'DT/LGraw':>10}{'n':>4}   {'raw/LGdc':>10}{'DT/LGdc':>10}{'n':>4}")
    for lo, hi in zip(dec[:-1], dec[1:]):
        row = f"{lo:>5}-{hi:<8}"
        for g in (lg_net, lg_net_dc):
            sig = (h["net"] > 5e3) & (g > 30) & np.isfinite(g) & ~bad
            w = sig & (g >= lo) & (g < hi)
            r1 = np.nanmedian(g[w] / h["net"][w]) * 1e3
            r2 = np.nanmedian(g[w] / h["net_dt"][w]) * 1e3
            row += f"{r1:>10.2f}{r2:>10.2f}{w.sum():>4}   "
        out.append(row)
    txt = "\n".join(out) + "\n"
    (OUT / "gecam_dt_tables.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
