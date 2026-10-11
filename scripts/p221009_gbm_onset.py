"""Step 3c: energy-resolved GBM light curves of the main-emission onset
(T0+140...215 s, TTE event by event), for the downstream onset-vs-energy
analysis.  Deadtime-corrected rates in NaI and BGO bands at 64 ms and 256 ms,
per-band background (linear over 145-172 s; quadratic over 100-118 + 150-172 s
as an alternative), event lists 160-200 s, and quick onset estimators
(first Bayesian-block change point after 165 s; first 0.256 s bin with net rate
> 5 sigma above background, followed by two more > 3 sigma).
Outputs $GBM_OUT/onset/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks
import p221009_gbm_common as C

od = C.outdir("onset")
NAI_BANDS = [(8, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 900)]
BGO_BANDS = [(300, 700), (700, 1500), (1500, 3000), (3000, 10000), (10000, 38000)]
DETS = ["n4", "n8", "n7", "n3", "n6", "b1", "b0"]
TR = (100.0, 215.0)
BKG = {"lin": ([(145.0, 172.0)], 1), "quad": ([(100.0, 118.0), (150.0, 172.0)], 2)}


def wls(t, y, var, win, order):
    m = np.zeros(len(t), bool)
    for a, b in win:
        m |= (t >= a) & (t < b)
    A = np.stack([((t - 170.0) / 10.0) ** k for k in range(order + 1)], -1)
    w = 1.0 / np.maximum(var[m], 1e-12)
    cov = np.linalg.inv(A[m].T @ (A[m] * w[:, None]))
    p = cov @ (A[m].T @ (y[m] * w))
    return A @ p, np.sqrt(np.einsum("ij,jk,ik->i", A, cov, A))


lc = {0.064: {}, 0.256: {}}
onset_rows, events = [], {}
for det in DETS:
    tte = C.open_tte(det, (TR[0] - 1, TR[1] + 1))
    t = np.asarray(tte.data.times); pha = np.asarray(tte.data.channels)
    eb = tte.ebounds
    emid = np.sqrt(np.asarray(eb.low_edges()) * np.asarray(eb.high_edges()))
    sel = (t >= 160) & (t < 200)
    events[det + "_t"] = t[sel]; events[det + "_ch"] = pha[sel].astype(np.uint8)
    events[det + "_emin"] = np.asarray(eb.low_edges()); events[det + "_emax"] = np.asarray(eb.high_edges())
    bands = BGO_BANDS if det.startswith("b") else NAI_BANDS
    for dt in lc:
        edges = np.arange(TR[0], TR[1] + 1e-6, dt)
        tc = 0.5 * (edges[:-1] + edges[1:])
        lc[dt]["t"] = tc
        for e1, e2 in bands:
            lo, hi = C.chan_range(eb, e1, e2)
            cnt, expo = C.lightcurve(t, pha, edges, lo, hi)
            key = "%s_%g_%g" % (det, eb.low_edges()[lo], eb.high_edges()[hi])
            rate = cnt / expo; var = np.maximum(cnt, 1) / expo ** 2
            lc[dt][key + "_cnt"] = cnt; lc[dt][key + "_exp"] = expo
            for bn, (win, order) in BKG.items():
                # fit the background on 1.024 s-equivalent statistics
                b, be = wls(tc, rate, var, win, order)
                lc[dt][key + "_bkg_" + bn] = b
                if dt == 0.256:
                    net = rate - b; sig = net / np.sqrt(var + be ** 2)
                    i5 = None
                    for i in np.where((tc > 165) & (sig > 5))[0]:
                        if i + 2 < len(sig) and (sig[i + 1:i + 3] > 3).all():
                            i5 = i; break
                    row = dict(det=det, band=key, emin=eb.low_edges()[lo], emax=eb.high_edges()[hi], bkg=bn,
                               t_5sigma=edges[i5] if i5 is not None else np.nan)
                    # net rate in 176-186 s, for the band's S/B
                    m = (tc > 176) & (tc < 186)
                    row["net_rate_176_186"] = net[m].mean(); row["bkg_rate"] = b[m].mean()
                    onset_rows.append(row)
    # Bayesian blocks on the events of each band
    for e1, e2 in bands:
        lo, hi = C.chan_range(eb, e1, e2)
        key = "%s_%g_%g" % (det, eb.low_edges()[lo], eb.high_edges()[hi])
        tb = t[(t > 160) & (t < 195) & (pha >= lo) & (pha <= hi)]
        cp = bayesian_blocks(tb, fitness="events", p0=0.01)
        after = cp[(cp > 166) & (cp < 194)]
        for r in onset_rows:
            if r["band"] == key:
                r["t_bb_first"] = after[0] if len(after) else np.nan
                r["bb_changepoints"] = " ".join("%.3f" % x for x in after[:8])
    print(det, "done", flush=True)

for dt, d in lc.items():
    C.write_csv(pd.DataFrame(d), "onset", "onset_lc_%dms.csv" % round(dt * 1000))
O = pd.DataFrame(onset_rows)
O["t_5sigma_hxmt"] = O.t_5sigma - C.HXMT_DT
O["t_bb_first_hxmt"] = O.t_bb_first - C.HXMT_DT
C.write_csv(O, "onset", "onset_quick.csv")
np.savez_compressed(os.path.join(od, "onset_events_160_200.npz"), T0_MET=C.T0, **events)

# ---------------------------------------------------------------- figure
D = pd.DataFrame(lc[0.256])
keys = [k[:-4] for k in D.columns if k.endswith("_cnt") and (k.startswith("n8") or k.startswith("b1"))]
fig, axs = plt.subplots(len(keys), 1, figsize=(7, 1.3 * len(keys)), sharex=True)
for ax, k in zip(axs, keys):
    r = D[k + "_cnt"] / D[k + "_exp"]
    ax.step(D.t, r, where="mid", lw=0.6, color="k")
    ax.plot(D.t, D[k + "_bkg_lin"], color="C1", lw=0.8)
    ax.axvline(174.2 + C.HXMT_DT, color="C3", ls=":", lw=0.8)
    ax.text(0.01, 0.75, k, transform=ax.transAxes, fontsize=7)
    m = (D.t > 160) & (D.t < 200)
    ax.set_ylim(np.percentile(r[m], 0.5) * 0.9, np.percentile(r[m], 99.8) * 1.1)
axs[-1].set_xlim(165, 200); axs[-1].set_xlabel("t - T0(GBM) (s); dotted: HE onset 174.2 s (HXMT T0)")
fig.tight_layout(); fig.savefig(os.path.join(od, "onset_lc.png"), dpi=130)
print("done")
