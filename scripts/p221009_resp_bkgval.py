"""Background-method validation for the precursor spectrum (GRB 221009A
response study).

The precursor has S/B ~0.12 in HE/CsI, so the background spectrum decides the
fitted index.  For 9-s null windows before the burst, each method is fitted
only on data that end 5 s before the window (as for the real precursor) and
its predicted per-group counts are compared with the observed ones.

Methods: tlin1 / tlin2 (polynomial in time on [-400, w0-5]), tloc1 (linear
on [w0-100, w0-5]), proxy_pre (Cnt_LargeEvt-linear on [-1020, w0-5]),
proxy_occ (same plus the Earth-occulted window 1900-2580 s).

    python p221009_resp_bkgval.py [OUTDIR]"""
import sys

import numpy as np

from p221009_resp_spec import GROUP, largeevt, load_events

BASE = "/scratchfs2/gecam/guohx/p221009"
NULLS = [(a, a + 9) for a in range(-300, -20, 15)]


def predict(h, tc, x, cal, win, kind, order=1):
    wsel = (tc >= win[0]) & (tc < win[1])
    if kind == "time":
        V = np.vander(tc[cal], order + 1)
        Vt = np.vander(tc[wsel], order + 1)
    else:
        V = np.column_stack([x[cal], np.ones(cal.sum())])
        Vt = np.column_stack([x[wsel], np.ones(wsel.sum())])
    coef = np.linalg.lstsq(V, h[:, cal].T, rcond=None)[0]
    return (Vt @ coef).sum(0), h[:, wsel].sum(1)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/resp/out"
    t, ch = load_events()
    gi = np.digitize(ch, GROUP) - 1
    ng = GROUP.size - 1
    edges = np.arange(-1020, 2581, 1.0)
    tc = edges[:-1] + 0.5
    h = np.zeros((ng, tc.size))
    s = (t >= edges[0]) & (t < edges[-1]) & (gi >= 0) & (gi < ng)
    np.add.at(h, (gi[s], np.floor(t[s] - edges[0]).astype(int)), 1)
    lm = largeevt()
    x = np.array([lm.get(int(k), np.nan) for k in edges[:-1]])
    ok = np.isfinite(x) & (h.sum(0) > 0)
    methods = {
        "tlin1": lambda w: (((tc >= -400) & (tc < w[0] - 5)) & ok, "time", 1),
        "tlin2": lambda w: (((tc >= -400) & (tc < w[0] - 5)) & ok, "time", 2),
        "tloc1": lambda w: (((tc >= w[0] - 100) & (tc < w[0] - 5)) & ok, "time", 1),
        "proxy_pre": lambda w: (((tc >= -1020) & (tc < w[0] - 5)) & ok, "proxy", 1),
        "proxy_occ": lambda w: ((((tc >= -1020) & (tc < w[0] - 5)) | ((tc >= 1900) & (tc < 2580))) & ok, "proxy", 1),
    }
    lines = []
    lo = GROUP[:-1] < 53
    for name, mk in methods.items():
        res = []
        for w in NULLS:
            cal, kind, order = mk(w)
            pred, obs = predict(h, tc, x, cal, w, kind, order)
            res.append(obs - pred)
        res = np.array(res)                       # n_null x ng
        tot = res.sum(1)
        mean_g = res.mean(0)
        se_g = res.std(0, ddof=1) / np.sqrt(len(NULLS))
        lines.append(f"{name:10s}: total net per 9 s mean {tot.mean():+.0f} rms {tot.std(ddof=1):.0f} "
                     f"(Poisson {np.sqrt(h[:, ok].sum(0).mean() * 9):.0f}); low-ch (<53) mean {res[:, lo].sum(1).mean():+.0f}, "
                     f"high-ch mean {res[:, ~lo].sum(1).mean():+.0f}; chi2(mean_g=0) {np.sum((mean_g / se_g) ** 2):.1f}/{ng}")
        # the same methods applied to the precursor (0-9 s): net per group
        cal, kind, order = mk((0, 9))
        pred, obs = predict(h, tc, x, cal, (0, 9), kind, order)
        net = obs - pred
        lines.append(f"{'':10s}  precursor 0-9 s: net {net.sum():.0f} (low {net[lo].sum():.0f}, high {net[~lo].sum():.0f}), "
                     f"hi/lo {net[~lo].sum() / net[lo].sum():.3f}")
        np.savetxt(f"{out}/bkgval_{name}.txt", np.column_stack([GROUP[:-1], mean_g, se_g, net]),
                   fmt="%d %.2f %.2f %.1f", header="ch_lo null_mean null_se precursor_net")
    with open(f"{out}/bkgval_summary.txt", "w") as f:
        for l in lines:
            print(l, flush=True)
            print(l, file=f)


if __name__ == "__main__":
    main()
